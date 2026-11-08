import pytest
import torch


def test_dual_image_backbone_to_region_graph_without_downloads():
    pytest.importorskip('torchvision')
    from lg_gnn.backbone import ImageLGGNN
    from lg_gnn.model import LGGNN
    torch.set_num_threads(1)
    graph = LGGNN(2, feature_dim=256, hidden=16, heads=4, steps=1)
    model = ImageLGGNN(graph, pretrained=False).eval()
    # 64px images produce a 16px feature grid; both synthetic regions are covered.
    assignment = torch.zeros(1, 1, 16, 16, dtype=torch.long)
    assignment[:, :, 8:] = 1
    with torch.no_grad():
        result = model(torch.randn(1, 1, 1, 64, 64), torch.randn(1, 1, 1, 64, 64),
                       assignment, torch.zeros(2, 2))
    assert result.logits.shape == (1, 3)
    assert result.concepts.shape == (1, 2, 2)
    assert torch.isfinite(result.logits).all()
