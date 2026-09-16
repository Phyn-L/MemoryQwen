"""Prove the vectorised masks in src/model.py are exactly equivalent to the originals.

The reference implementations are taken verbatim from git HEAD (the row-by-row versions).
Run: PYTHONPATH=. python .tmp_analysis/verify_masks.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/data/lz/MemoryQwen")
from src.model import build_block_causal_mask, build_continuation_mask  # noqa: E402


def reference_source():
    """Recreate the pre-vectorisation builders from git."""
    old = subprocess.run(["git", "-C", "/data/lz/MemoryQwen", "show", "HEAD:src/model.py"],
                         capture_output=True, text=True, check=True).stdout
    start = old.index("def build_block_causal_mask(")
    end = old.index("class MetaLoRA(")
    namespace = {"torch": torch}
    exec(old[start:end], namespace)
    return namespace["build_block_causal_mask"], namespace["build_continuation_mask"]


ref_block, ref_cont = reference_source()

torch.manual_seed(0)
ok_block = ok_cont = True
for trial in range(30):
    b = torch.randint(1, 5, (1,)).item()
    length = torch.randint(1, 40, (1,)).item()
    memory = torch.randint(1, 6, (1,)).item()
    qlen = torch.randint(1, 12, (1,)).item()
    alen = torch.randint(1, 12, (1,)).item()
    c = torch.rand(b, length) > 0.35
    q = torch.rand(b, qlen) > 0.4
    a = torch.rand(b, alen) > 0.4
    for row in range(b):
        q[row, 0] = True
        a[row, 0] = True
    ok_block &= bool(torch.equal(ref_block(c.clone(), memory, q.clone(), a.clone(), torch.bfloat16),
                                 build_block_causal_mask(c, memory, q, a, torch.bfloat16)))
    ok_cont &= bool(torch.equal(ref_cont(q.clone(), a.clone(), memory, torch.bfloat16),
                                build_continuation_mask(q, a, memory, torch.bfloat16)))
print("exact equality over 30 random batches -> block:", "PASS" if ok_block else "FAIL",
      "| continuation:", "PASS" if ok_cont else "FAIL")
assert ok_block and ok_cont

# also check the fully-masked (-inf) rows still have at least one allowed key
q = torch.zeros(2, 5, dtype=torch.bool)
a = torch.zeros(2, 3, dtype=torch.bool)
q[0, 0] = True
a[0, 0] = True
m = build_continuation_mask(q, a, 4, torch.bfloat16)
assert not torch.isinf(m).all(dim=-1).any(), "a query row is fully masked"
print("no fully-masked query rows: PASS")
