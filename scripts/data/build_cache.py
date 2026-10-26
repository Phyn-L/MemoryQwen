#!/usr/bin/env python3
"""Filter the raw 1QA meta splits into the cache a training run reads.

The rules live in ``src/data/cache.py``; this is the command line around them.

Usage::

    # manual build with a training config (the usual case)
    python scripts/data/build_cache.py --config configs/train_baseline.yaml

    # only one split, another data root, or a look-before-you-write
    python scripts/data/build_cache.py --config configs/train_baseline.yaml --splits validation
    python scripts/data/build_cache.py --config configs/train_baseline.yaml --data-root /mnt/x/contexts
    python scripts/data/build_cache.py --config configs/train_baseline.yaml --dry-run

``scripts/train.py`` builds missing or stale cache files itself, so this command is for
building them ahead of time (or for a data root you only filter once).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cache import (  # noqa: E402
    CACHE_SPLITS,
    DEFAULT_BUILD_WORKERS,
    build,
    ensure_cache,
    write_lengths_for_cache,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="training YAML the filters come from")
    parser.add_argument("--data-root", default=None, help="override data.files.root")
    parser.add_argument("--splits", nargs="+", choices=CACHE_SPLITS, default=list(CACHE_SPLITS))
    parser.add_argument("--dry-run", action="store_true",
                        help="Filter and report the counts without keeping the files")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even when the cache is already fresh")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_BUILD_WORKERS,
                        help="Worker processes for the row filter (1 = in-process)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Do not draw the per-split filtering progress bar")
    parser.add_argument("--no-lengths", action="store_true",
                        help="Do not record the context lengths the filter measured "
                             "(the loader then measures them itself, once per run)")
    parser.add_argument("--lengths-only", action="store_true",
                        help="Only (re)write the context-length sidecars of existing cache "
                             "files; no re-filtering")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from utils.config import TrainConfig

    cfg = TrainConfig.from_file(Path(args.config), machine=os.environ.get("MACHINE"))
    cfg.validate()
    if args.data_root:
        cfg.data.root = args.data_root

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path, local_files_only=True, trust_remote_code=False
    )
    splits = list(args.splits)
    progress = not args.no_progress
    if args.lengths_only:
        for split in splits:
            if not Path(cfg.data.cache_file(split)).is_file():
                print(f"[cache] {split}: no cache file to measure", flush=True)
                continue
            written = write_lengths_for_cache(cfg.data, tokenizer, split, quiet=args.quiet)
            if written is None and not args.quiet:
                print(f"[cache] {split}: nothing recorded", flush=True)
        return 0
    if args.dry_run:
        build(cfg.data, tokenizer, splits=splits, quiet=args.quiet, dry_run=True,
              num_workers=args.num_workers, progress=progress,
              write_lengths=not args.no_lengths)
        return 0
    if args.force:
        result = build(cfg.data, tokenizer, splits=splits, quiet=args.quiet,
                       num_workers=args.num_workers, progress=progress,
                       write_lengths=not args.no_lengths)
    else:
        built = ensure_cache(cfg.data, tokenizer, splits=splits, quiet=args.quiet,
                             num_workers=args.num_workers, progress=progress)
        if not built and not args.quiet:
            print("[cache] up to date; nothing to do (use --force to rebuild)", flush=True)
        result = {"splits": {split: {"file": cfg.data.cache_file(split)} for split in splits}}
    if not args.quiet:
        print(json.dumps(
            {split: {k: v for k, v in info.items() if k != "fingerprint"}
             for split, info in result["splits"].items()},
            ensure_ascii=False, indent=2, default=str,
        ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
