"""Independent-process seed experiment; run with python -m scripts.check_seed_reproducibility."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def worker(args):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from src.model import MetaLoRA, is_trainable_parameter_name
    from src.losses import qa_loss, objective_total
    from utils.config import MemoryConfig
    from utils import seed_everything

    torch.set_num_threads(2)
    seed_everything(args.seed, deterministic=args.deterministic)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=256,
                        tie_word_embeddings=True)
    config._attn_implementation = 'sdpa'
    backbone = Qwen3ForCausalLM(config).to(dtype=getattr(torch, args.dtype))
    model = MetaLoRA(backbone, rank=4, alpha=8, memory_length=8,
                     decoder_hidden_size=32, decoder_heads=4, max_context_tokens=64,
                     token_recon=True, embedding_recon=False, ae_lm=True,
                     slot_attention='causal', init_mode='token_embed', init_seed=0,
                     dropout=0.1).to(args.device).train()
    for name, p in model.named_parameters():
        p.requires_grad_(is_trainable_parameter_name(name))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    objective = MemoryConfig(embedding_recon_weight=0, token_recon_weight=1,
                             causal_recon_weight=1, distill_weight=0.3,
                             token_recon_positions=8, causal_recon_positions=8,
                             distill_positions=8, distill_entropy_weight=True)
    # Inputs stay identical even in the different-seed control.
    generator = torch.Generator().manual_seed(123)
    batches = [[torch.randint(2, 128, (2, length), generator=generator).to(args.device)
                for length in (32, 8, 8)] for _ in range(args.steps)]
    def snapshot(values):
        return torch.cat([p.detach().float().cpu().reshape(-1) for p in values])
    initial = snapshot(params)
    trajectory = []
    for context, question, answer in batches:
        embedding = model.qwen.get_input_embeddings()
        out = model(embedding(context), torch.ones_like(context, dtype=torch.bool),
                    embedding(question), torch.ones_like(question, dtype=torch.bool),
                    embedding(answer), torch.ones_like(answer, dtype=torch.bool), answer,
                    torch.arange(2, device=args.device), context_ids=context,
                    objective_config=objective)
        loss = objective_total(qa_loss(out.logits, out.labels), out.objective_terms, objective)
        loss.backward()
        gradients = snapshot([p.grad for p in params if p.grad is not None])
        optimizer.step()
        trajectory.append({'loss': loss.detach().cpu(), 'gradients': gradients,
                           'parameters': snapshot(params)})
        optimizer.zero_grad(set_to_none=True)
    torch.save({'initial': initial, 'trajectory': trajectory,
                'deterministic': torch.are_deterministic_algorithms_enabled()}, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--output-dir', default='outputs/seed_repro_check')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    runs, failures = {}, {}
    for mode in (False, True):
        for repeat, seed in enumerate([42, 42, 42, 43]):
            name = f'{"on" if mode else "off"}-{repeat}'
            output = folder / f'{name}.pt'
            command = [sys.executable, '-m', 'scripts.check_seed_reproducibility', '--worker',
                       '--device', args.device, '--dtype', args.dtype, '--steps', str(args.steps),
                       '--seed', str(seed), '--output', str(output)]
            if mode:
                command.append('--deterministic')
            env = os.environ.copy()
            env.pop('CUBLAS_WORKSPACE_CONFIG', None)
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            (folder / f'{name}.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                failures[name] = result.stderr[-3000:]
            else:
                runs[name] = torch.load(output, weights_only=True)
            print(f'{name}: {"failed" if result.returncode else "completed"}', flush=True)
    def compare(a, b):
        return {'initial_equal': torch.equal(a['initial'], b['initial']), **{
            key: {'all_steps_equal': all(torch.equal(x[key], y[key]) for x, y in zip(a['trajectory'], b['trajectory'])),
                  'max_abs_diff': max(float((x[key] - y[key]).abs().max()) for x, y in zip(a['trajectory'], b['trajectory']))}
            for key in ('loss', 'gradients', 'parameters')}}
    pairs = [(f'{mode}-0', f'{mode}-{i}') for mode in ('off', 'on') for i in (1, 2, 3)]
    pairs.append(('off-0', 'on-0'))
    summary = {'torch': torch.__version__, 'device': args.device, 'dtype': args.dtype,
               'steps': args.steps, 'dropout': 0.1, 'failures': failures,
               'comparisons': {f'{a} vs {b}': compare(runs[a], runs[b]) for a, b in pairs if a in runs and b in runs}}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
