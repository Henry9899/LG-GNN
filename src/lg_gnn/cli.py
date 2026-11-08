import argparse
import json
from pathlib import Path

from .data import synthetic_cohort
from .experiment import ExperimentConfig, run_experiment, write_json
from .reporting import EvidenceRecord, generate_checked, template_summary


def main():
    parser = argparse.ArgumentParser(description='LG-GNN reference implementation')
    sub = parser.add_subparsers(dest='command', required=True)
    synthetic = sub.add_parser('synthetic', help='Generate artificial execution-check data')
    synthetic.add_argument('--output', type=Path, required=True)
    synthetic.add_argument('--subjects-per-class', type=int, default=30)
    synthetic.add_argument('--seed', type=int, default=17)
    train = sub.add_parser('train', help='Nested CV on external-frozen regional descriptors')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--config', type=Path, required=True)
    prediction = sub.add_parser('predict', help='Checkpoint inference; labels and measurements not required')
    prediction.add_argument('--data', type=Path, required=True)
    prediction.add_argument('--fold', type=Path, required=True)
    prediction.add_argument('--output', type=Path, required=True)
    prediction.add_argument('--device', default='cpu')
    report = sub.add_parser('report', help='Render checked evidence-only summaries')
    report.add_argument('--evidence', type=Path, required=True, help='JSON: records plus calibrated probabilities')
    report.add_argument('--output', type=Path, required=True)
    report.add_argument('--model', help='Optional locally available Hugging Face causal LM path')
    report.add_argument('--device', default='cpu')
    report.add_argument('--max-new-tokens', type=int, default=2048)
    args = parser.parse_args()
    if args.command == 'synthetic':
        if args.subjects_per_class < 20:
            parser.error('Use at least 20 subjects per class for the nested smoke protocol')
        synthetic_cohort(args.output, args.subjects_per_class, args.seed)
        print(f'Artificial data written to {args.output}')
    elif args.command == 'train':
        config = ExperimentConfig(**json.loads(args.config.read_text(encoding='utf-8')))
        result = run_experiment(args.data, args.output, config)
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == 'predict':
        from .inference import predict
        print(json.dumps(predict(args.data, args.fold, args.output, args.device)))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = json.loads(args.evidence.read_text(encoding='utf-8'))
        records = [EvidenceRecord(**record) for record in payload['records']]
        probabilities = payload['probabilities']
        if args.model:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            llm = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to(args.device).eval()

            def generator(prompt):
                import torch
                if tokenizer.chat_template:
                    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                           tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
                with torch.no_grad():
                    tokens = llm.generate(**inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id)
                return tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            result = generate_checked(generator, records, probabilities)
        else:
            result = template_summary(records, probabilities)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(f'Checked summary written to {args.output}')


if __name__ == '__main__':
    main()
