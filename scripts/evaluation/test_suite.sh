#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$ROOT/scripts/env.local.sh" ] && source "$ROOT/scripts/env.local.sh"
MACHINE=4090; CKPT=""; BS=1; QA_BS=4; OUT_ROOT=""; DATASETS=(squad ms_marco hotpotqa race)
while (($#)); do case "$1" in
 --machine) MACHINE=$2; shift 2;; --ckpt|--checkpoint) CKPT=$2; shift 2;; --bs|--batch-size) BS=$2; shift 2;; --qa-batch-size) QA_BS=$2; shift 2;; --output-root) OUT_ROOT=$2; shift 2;; --datasets) IFS=',' read -r -a DATASETS <<< "$2"; shift 2;;
 -h|--help) echo "Usage: $0 --ckpt PATH [--datasets squad,ms_marco,hotpotqa,race] [--machine 4090|h200] [--bs N] [--qa-batch-size N] [--output-root PATH]"; exit 0;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
[ -n "$CKPT" ] || { echo '--ckpt is required' >&2; exit 2; }
OUT_ROOT=${OUT_ROOT:-"$(dirname "$CKPT")/suite_$(date +%Y%m%d_%H%M%S)"}; mkdir -p "$OUT_ROOT"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
run_eval() { local name=$1 dataset=$2 split=$3 version=${4:-}; local dir="$OUT_ROOT/$name"; mkdir -p "$dir"; echo "[suite] $name split=$split ${version:+version=$version}"; python -m utils.launcher test --machine "$MACHINE" --ckpt "$CKPT" --datasets "$dataset" --split "$split" --bs "$BS" --qa-batch-size "$QA_BS" ${version:+--source-version "$version"} > >(tee "$dir/run.log") 2>&1; }
for name in "${DATASETS[@]}"; do case "$name" in
 squad) run_eval squad squad validation;;
 ms_marco) run_eval ms_marco_v1 ms_marco validation ms_marco_v1_1; run_eval ms_marco_v1_test ms_marco test ms_marco_v1_1; run_eval ms_marco_v2_validation ms_marco validation ms_marco_v2_1; run_eval ms_marco_v2_test ms_marco test ms_marco_v2_1;;
 hotpotqa) run_eval hotpotqa hotpotqa validation;; race) run_eval race race test;; *) echo "unsupported dataset: $name" >&2; exit 2;; esac; done
python - <<'PY' "$OUT_ROOT"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); rows=[]
for p in root.rglob('eval_*.json'):
 try:
  d=json.loads(p.read_text()); m=d.get('autoregressive',{}); rows.append({'report':str(p),'datasets':d.get('datasets'),'split':d.get('split'),'count':m.get('count'),'em':m.get('em'),'f1':m.get('f1'),'rouge_l':m.get('rouge_l')})
 except Exception: pass
(root/'summary.json').write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows,indent=2))
PY
