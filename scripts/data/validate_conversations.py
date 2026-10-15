#!/usr/bin/env python3
"""Validate a conversation JSONL/JSON file: schema, roles and non-empty content.

Used as a pre-flight check before converting or training on a new corpus. Fails loudly on the
first malformed row instead of letting the filter silently drop it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    """Yield rows from a ``.jsonl`` file, or from a SHINE JSON array for anything else."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    from convert_conversations_json import iter_array

    yield from iter_array(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--rows", type=int, default=1000, help="Rows to check (0 = all)")
    args = parser.parse_args()

    checked = assistant_turns = multi_turn_rows = 0
    for row in rows(args.path):
        context = row.get("context")
        assert isinstance(context, str) and context.strip(), "row has no context"
        conversations = row.get("conversations")
        assert isinstance(conversations, list) and conversations, "row has no conversations"
        roles = [message.get("role") for message in conversations]
        assert all(role in {"system", "user", "assistant"} for role in roles), roles
        assert all(
            isinstance(message.get("content"), str) and message["content"].strip()
            for message in conversations
        ), "every message needs non-empty content"
        assert "assistant" in roles, "row has no assistant turn"

        turns = sum(role == "assistant" for role in roles)
        checked += 1
        assistant_turns += turns
        multi_turn_rows += int(turns > 1)
        if args.rows and checked >= args.rows:
            break

    print(
        json.dumps(
            {
                "path": str(args.path),
                "rows_checked": checked,
                "assistant_turns": assistant_turns,
                "multi_turn_rows": multi_turn_rows,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
