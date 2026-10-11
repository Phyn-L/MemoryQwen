from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def compute_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list).encode()
    ).hexdigest()


def tokenizer_fingerprint(tokenizer) -> str | None:
    """Identify tokenization behavior independently of the model repository path."""
    if tokenizer is None:
        return None
    payload: dict[str, Any] = {
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        payload["backend"] = hashlib.sha256(
            backend.to_str().encode()
        ).hexdigest()
    else:
        # Slow/custom tokenizers have no serialized backend. Keep the fallback
        # conservative so unrelated implementations do not share a cache.
        payload["name"] = str(
            getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        )
    return compute_fingerprint(payload)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Serialize cache writers that share a cache directory."""
    with path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
