"""32-query bridge and frozen causal-LM adapter; no model weights bundled."""
import torch
from torch import nn

from .losses import multiple_positive_alignment
from .reporting import category


class QueryBridge(nn.Module):
    def __init__(self, graph_dim=256, llm_dim=4096, queries=32, heads=8, layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, graph_dim) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(graph_dim, heads,
                     dim_feedforward=4 * graph_dim, dropout=0.0, batch_first=True)
                     for _ in range(layers)])
        self.project = nn.Linear(graph_dim, llm_dim)

    def forward(self, nodes, covered):
        if not covered.any(1).all():
            raise ValueError("No bridge output for subjects with no covered regions")
        # Boolean True means ignore this memory key in torch attention.
        memory = torch.where(covered[..., None], nodes, 0.0)
        query = self.queries[None].expand(len(nodes), -1, -1)
        for layer in self.layers:
            query = layer(query, memory, memory_key_padding_mask=~covered)
        return self.project(query)


def categorical_positives(values, valid):
    """All region/modality/category entries agree; both invalid counts as absent."""
    signatures = []
    for row, mask in zip(values, valid):
        signatures.append(tuple(category(float(v)) if ok else 'absent'
                                for v, ok in zip(row.flatten(), mask.flatten())))
    return torch.tensor([[a == b for b in signatures] for a in signatures],
                        dtype=torch.bool, device=values.device)


class FrozenLanguageAdapter(nn.Module):
    def __init__(self, llm, bridge):
        super().__init__()
        self.llm, self.bridge = llm, bridge
        self.llm.requires_grad_(False)
        self.llm.eval()

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()
        return self

    def objective(self, nodes, covered, input_ids, attention_mask, positives,
                  align_weight=0.5, generation_weight=1.0):
        """Caller freezes graph (detach nodes) during manuscript stage 3.

        Autograd stays enabled through frozen LM operations to train the prefix.
        Token IDs should include EOS; padding is excluded from both objectives.
        """
        visual = self.bridge(nodes, covered)
        text = self.llm.get_input_embeddings()(input_ids)
        visual = visual.to(text.dtype)
        with torch.no_grad():
            pooled_text = (text * attention_mask[..., None]).sum(1) / attention_mask.sum(1).clamp_min(1)[:, None]
        alignment = multiple_positive_alignment(visual.mean(1), pooled_text, positives)
        inputs = torch.cat([visual, text], 1)
        prefix_mask = attention_mask.new_ones(len(nodes), visual.shape[1])
        mask = torch.cat([prefix_mask, attention_mask], 1)
        labels = input_ids.masked_fill(~attention_mask.bool(), -100)
        labels = torch.cat([labels.new_full(prefix_mask.shape, -100), labels], 1)
        output = self.llm(inputs_embeds=inputs, attention_mask=mask, labels=labels, use_cache=False)
        return align_weight * alignment + generation_weight * output.loss

    @torch.no_grad()
    def generate(self, nodes, covered, prompt_ids, attention_mask, max_new_tokens=512, **kwargs):
        visual = self.bridge(nodes, covered)
        text = self.llm.get_input_embeddings()(prompt_ids)
        inputs = torch.cat([visual.to(text.dtype), text], 1)
        prefix_mask = attention_mask.new_ones(len(nodes), visual.shape[1])
        return self.llm.generate(inputs_embeds=inputs,
            attention_mask=torch.cat([prefix_mask, attention_mask], 1),
            max_new_tokens=max_new_tokens, do_sample=False, **kwargs)
