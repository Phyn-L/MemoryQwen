from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shlex
from datetime import datetime
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evaluation.test_all_suite import DATASETS, plans

from src.icl_baseline import (
    ICLExample,
    aggregate_metrics,
    example_metrics,
    load_jsonl,
    parse_prediction,
    render_prompt,
    sample_jsonl,
    limit_examples,
)
from utils.config import dtype_from_name, expand_env
from utils.icl_config import load_icl_defaults
from utils.model_paths import machine_paths, resolve_model_path
from utils.ddp import barrier, init_distributed, is_main_process
from utils.machines import fill_missing, machine_environ, machine_names, resolve_machine

# Both defaults go through the same ${VAR:-default} mechanism the YAML configs use, so a
# machine with a different model directory or data tree overrides them with MODEL_ROOT /
# DATA_ROOT instead of editing this file. `--machine h200` (or MACHINE=h200, or the hostname)
# picks those two values out of utils/machines.py, exactly like the training entry does.
def _machine_overrides(machine: str | None = None) -> dict[str, str]:
    """Environment overrides for this machine, from utils/machines.py."""
    resolved, authoritative = resolve_machine(machine)
    values = machine_environ(resolved)
    return values if authoritative else fill_missing(values, os.environ)


def _default_paths(overrides=None):
    model_root = expand_env("${MODEL_ROOT:-/data/lz/hf_cache/hub}", overrides)
    data_root = expand_env("${DATA_ROOT:-/data/lz/contexts/aggregated}", overrides)
    return (
        f"{model_root}/models--Qwen--Qwen3-1.7B/"
        "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        Path(data_root),
    )


# The data default is the same tree the training pipeline uses (data.root in
# configs/*/train.yaml), and the only supported one: `src.icl_baseline.iter_examples` reads the
# aggregated context schema and rejects anything else. See the README section "The SQuAD
# evaluation set, in detail" for what that file actually contains.
DEFAULT_MODEL, DEFAULT_DATA = _default_paths(_machine_overrides())


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--machine")
    known, _ = bootstrap.parse_known_args(argv)
    default_model, default_data = _default_paths(_machine_overrides(known.machine))
    defaults = load_icl_defaults(known.config, known.machine) if known.config else {}
    parser = argparse.ArgumentParser(
        description="Multi-GPU Qwen family ICL baseline"
    )
    parser.add_argument(
        "--machine", choices=machine_names(), default=None,
        help="Machine whose paths (utils/machines.py) this run uses; without it the "
             "MACHINE variable or the hostname decides.",
    )
    parser.add_argument("--config", help="ICL YAML; controls shot count, decoding, split and output")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--source-version")
    parser.add_argument("--data-root")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--model", default=default_model)
    parser.add_argument(
        "--datasets", nargs="+", default=["squad", "race"]
    )
    parser.add_argument(
        "--squad-validation-file",
        default=str(default_data / "squad/validation.jsonl"),
    )
    parser.add_argument(
        "--squad-train-file", default=str(default_data / "squad/train.jsonl")
    )
    parser.add_argument(
        "--race-test-file", default=str(default_data / "race/test.jsonl")
    )
    parser.add_argument(
        "--race-train-file", default=str(default_data / "race/train.jsonl")
    )
    parser.add_argument("--output-dir", default="outputs/icl_baseline/qwen3-1.7b")
    parser.add_argument("--num-shots", type=int, default=4)
    parser.add_argument("--batch-size", "--bs", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--squad-max-new-tokens", type=int, default=32)
    parser.add_argument("--race-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Debug-only cap applied independently to each dataset (in *examples*, i.e. QA rows)",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        help="Keep only the first N distinct contexts of the file, the unit the memory "
        "evaluator's data.validation_max_samples uses",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="Drop examples whose context does not fit in N tokens, mirroring the memory "
        "side's data.max_context_tokens + filter_long_context",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse completed rows in rank shard files"
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.num_shots < 0 or args.batch_size <= 0 or args.max_input_tokens <= 0:
        parser.error(
            "num-shots must be non-negative and batch/token limits must be positive"
        )
    if not args.datasets or any(not name or "/" in name or name in (".", "..") for name in args.datasets):
        parser.error("datasets must be directory names under the data root")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    for name in ("max_samples", "max_contexts", "max_context_tokens"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if args.config and args.split == "train" and args.num_shots:
        parser.error("few-shot evaluation on train risks demonstration leakage; use validation/test")
    args.model = resolve_model_path(args.model, args.machine)
    if args.data_root is None:
        args.data_root = str(_default_paths(_machine_overrides(args.machine))[1])
    if args.output_dir is None:
        if args.resume:
            parser.error("--resume requires --output-dir pointing to the existing run")
        from datetime import datetime
        from src.pipeline import model_run_label
        args.output_dir = f"outputs/icl/{model_run_label(args.model)}_{datetime.now():%Y%m%d_%H%M%S}"
    return args


def dataset_paths(args):
    root = Path(args.data_root)
    paths = {}
    for dataset in args.datasets:
        if args.split is not None:
            paths[dataset] = (str(root / dataset / f"{args.split}.jsonl"), str(root / dataset / "train.jsonl"))
        elif dataset == "squad":
            paths[dataset] = (args.squad_validation_file, args.squad_train_file)
        elif dataset == "race":
            paths[dataset] = (args.race_test_file, args.race_train_file)
        else:
            raise ValueError(f"Specify --split for dataset {dataset}")
    return paths


def distributed_context():
    init_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("This inference entrypoint requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def load_model(args, device):
    # One name->torch.dtype map for the whole repo (utils.config.dtype_from_name).
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return tokenizer, model


def collate_records(rows):
    indices, records = zip(*rows)
    return list(indices), list(records)


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["index"]))
    return completed


@torch.inference_mode()
def evaluate_dataset(
    args,
    dataset,
    records,
    demos,
    tokenizer,
    model,
    rank,
    world_size,
    device,
    output_dir,
):
    shard_path = output_dir / f"{dataset}.rank-{rank:05d}-of-{world_size:05d}.jsonl"
    done = completed_indices(shard_path) if args.resume else set()
    indexed = [
        (index, record)
        for index, record in enumerate(records)
        if index % world_size == rank and index not in done
    ]
    loader = DataLoader(
        indexed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_records,
    )
    mode = "a" if args.resume else "w"
    max_new_tokens = (
        args.max_new_tokens or (args.race_max_new_tokens if dataset == "race" else args.squad_max_new_tokens)
    )
    with shard_path.open(mode, encoding="utf-8") as handle:
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"{dataset} rank {rank}",
            unit="batch",
            disable=not is_main_process(),
        )
        for indices, examples in progress:
            prompts = [
                render_prompt(tokenizer, example, demos, not args.no_chat_template)
                for example in examples
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
                add_special_tokens=False,
            ).to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[:, encoded.input_ids.shape[1] :]
            raw_predictions = tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            for index, example, raw_prediction in zip(
                indices, examples, raw_predictions
            ):
                prediction, predicted_letter = parse_prediction(example, raw_prediction)
                metrics = example_metrics(prediction, example.references)
                row = {
                    "index": index,
                    "id": example.id,
                    "dataset": dataset,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction.strip(),
                    "references": list(example.references),
                    **metrics,
                }
                if dataset == "race":
                    row.update(
                        {
                            "predicted_letter": predicted_letter,
                            "answer_letter": example.answer_letter,
                        }
                    )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    barrier()
    if is_main_process():
        merge_and_score(dataset, output_dir, world_size)
    barrier()


def merge_and_score(dataset: str, output_dir: Path, world_size: int):
    rows = []
    for rank in range(world_size):
        path = output_dir / f"{dataset}.rank-{rank:05d}-of-{world_size:05d}.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: row["index"])
    prediction_path = output_dir / f"{dataset}.predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = aggregate_metrics(rows)
    (output_dir / f"{dataset}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{dataset}: {json.dumps(metrics, ensure_ascii=False)}", flush=True)


def experiment_metadata(args, world_size, demonstrations):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "arguments": vars(args),
        "protocol": {
            "decoding": "greedy",
            "datasets": dataset_paths(args),
            "answer_filter": "Skip QA rows with no reference answer",
            "em_f1_normalization": "official SQuAD style",
            "multi_reference_reduction": "maximum per example",
            "bleu_4": "corpus BLEU-4, closest reference length, add-one smoothing",
            "rouge_l": "sentence-level LCS F1, macro average",
        },
        "distributed": {"world_size": world_size},
        "demonstrations": {
            name: [asdict(example) for example in values]
            for name, values in demonstrations.items()
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "git_revision": revision,
        },
    }


def worker_main():
    args = parse_args()
    rank, world_size, device = distributed_context()
    seed_everything(args.seed, rank)
    if dist.is_initialized():
        shared_output = [args.output_dir if rank == 0 else None]
        dist.broadcast_object_list(shared_output, src=0)
        args.output_dir = shared_output[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(args)
    demonstrations = {
        dataset: sample_jsonl(paths[dataset][1], dataset, args.num_shots, args.seed)
        for dataset in args.datasets
    }
    if is_main_process():
        (output_dir / "run_config.json").write_text(
            json.dumps(
                experiment_metadata(args, world_size, demonstrations),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    barrier()
    tokenizer, model = load_model(args, device)
    for dataset in args.datasets:
        records = load_jsonl(paths[dataset][0], dataset, args.source_version)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        records = limit_examples(
            records,
            max_contexts=args.max_contexts,
            max_context_tokens=args.max_context_tokens,
            tokenizer=tokenizer,
        )
        if not records:
            raise SystemExit(f"{dataset}: no examples left after the subset filters")
        evaluate_dataset(
            args,
            dataset,
            records,
            demonstrations[dataset],
            tokenizer,
            model,
            rank,
            world_size,
            device,
            output_dir,
        )
    if dist.is_initialized():
        dist.destroy_process_group()






def suite_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=machine_names(), default=None)
    parser.add_argument('--model', default='Qwen3-8B')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for key in ('bs', 'max_input_tokens', 'max_new_tokens', 'max_samples'):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f'{key} must be positive')
    machine, paths = machine_paths(args.machine)
    model = resolve_model_path(args.model, machine)
    root = Path(args.data_root or paths['DATA_ROOT'])
    dataset_plans = list(plans(root, args.datasets))
    out = Path(args.output_dir or f'outputs/icl_suite/{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    print(f'Model: {args.model} -> {model}', flush=True)
    for plan in dataset_plans:
        destination = out / plan['name']
        command = [sys.executable, '-m', 'utils.launcher', 'icl',
                   '--config', 'configs/icl_zeroshot.yaml', '--model', model,
                   '--num-shots', '0', '--datasets', plan['dataset'], '--split', plan['split'],
                   '--data-root', str(root), '--output-dir', str(destination),
                   '--bs', str(args.bs), '--max-input-tokens', str(args.max_input_tokens),
                   '--max-new-tokens', str(args.max_new_tokens)]
        if machine:
            command += ['--machine', machine]
        if plan['version']:
            command += ['--source-version', plan['version']]
        if args.max_samples:
            command += ['--max-samples', str(args.max_samples)]
        print(shlex.join(command), flush=True)
        row = dict(dataset=plan['name'], split=plan['split'], model=model,
                   num_shots=0, command=command, status='planned')
        if not args.dry_run:
            # Inherit stdout/stderr so tqdm progress is visible in the terminal.
            result = subprocess.run(command)
            row['status'] = 'ok' if result.returncode == 0 else f'failed:{result.returncode}'
            metrics_file = destination / f"{plan['dataset']}.metrics.json"
            if result.returncode == 0:
                if metrics_file.is_file():
                    row['metrics'] = json.loads(metrics_file.read_text())
                    print(f"{plan['name']}: {row['metrics']}", flush=True)
                else:
                    row['status'] = 'failed:missing_metrics'
        results.append(row)
        (out / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('\nDataset | Split | Status | Count | EM | F1 | ROUGE-L | Accuracy')
    for row in results:
        metrics = row.get('metrics', {})
        values = [f'{100 * metrics[k]:.2f}' if k in metrics else '-' for k in ('em', 'f1', 'rouge_l', 'accuracy')]
        print(' | '.join([row['dataset'], row['split'], row['status'], str(metrics.get('count', '-')), *values]))
    print(f'Results: {out}')
    if any(row['status'].startswith('failed') for row in results):
        raise SystemExit(1)



if __name__ == '__main__':
    if '--worker' in sys.argv:
        sys.argv.remove('--worker')
        worker_main()
    else:
        suite_main()
