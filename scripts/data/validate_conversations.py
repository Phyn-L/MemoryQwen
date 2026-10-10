#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def rows(path):
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    else:
        from convert_conversations_json import iter_array
        yield from iter_array(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('path', type=Path); p.add_argument('--rows', type=int, default=1000); a=p.parse_args()
    n=turns=multi=0
    for row in rows(a.path):
        assert isinstance(row.get('context'), str) and row['context'].strip()
        conv=row.get('conversations'); assert isinstance(conv, list) and conv
        roles=[m.get('role') for m in conv]
        assert all(r in {'system','user','assistant'} for r in roles)
        assert all(isinstance(m.get('content'), str) and m['content'].strip() for m in conv)
        assert 'assistant' in roles
        n += 1; turns += sum(r == 'assistant' for r in roles); multi += int(sum(r == 'assistant' for r in roles) > 1)
        if n >= a.rows: break
    print(json.dumps({'path': str(a.path), 'rows_checked': n, 'assistant_turns': turns, 'multi_turn_rows': multi}, ensure_ascii=False))

if __name__ == '__main__': main()
