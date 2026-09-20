#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -f scripts/env.local.sh ]; then source scripts/env.local.sh; fi
exec python scripts/evaluation/test_all_suite.py "$@"
