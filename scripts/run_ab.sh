#!/bin/bash
# Run one clean reader A/B: the ON arm and then the OFF arm, back to back.
#
# The two configs differ only in the six reader switches and the output directory
# (asserted by tests/test_ab_configs.py), and both are scheduled for a single epoch, so
# the pair answers "do the reader switches help at this shape and this budget".
#
# Every knob here is fixed for BOTH arms -- a schedule that drifts between the arms makes
# the comparison unreadable -- and all of them are documented in docs/AB_H200.md.
#
# Usage:
#   NUM_PROCESSES=4 bash scripts/run_ab.sh          # 4 ranks x batch 8 = global 32
#   NUM_PROCESSES=8 bash scripts/run_ab.sh          # 8 ranks, halve training.batch_size
#   DRYRUN=1 bash scripts/run_ab.sh                 # print what would run, launch nothing
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# The H200 checkout has the working conda env outside the default PATH. Prepending it when
# it exists keeps this script usable on a machine where it does not.
H200_ENV=/home/lijie/proj2/.conda/envs/shine/bin
if [ -d "$H200_ENV" ]; then
    PATH="$H200_ENV:$PATH"
    export PATH
fi

NUM_PROCESSES="${NUM_PROCESSES:-4}"
ON_CONFIG="${ON_CONFIG:-configs/qwen-1.7b/ab_h200_on.yaml}"
OFF_CONFIG="${OFF_CONFIG:-configs/qwen-1.7b/ab_h200_off.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
DRYRUN="${DRYRUN:-0}"

# Contexts kept by train_datasets=all at max_context_tokens=1024 (see the config header).
# Used only to print the expected step budget, which is what every cadence is measured
# against; the training loop echoes the real number as its first line.
CONTEXTS=252465

for file in scripts/train.sh "$ON_CONFIG" "$OFF_CONFIG"; do
    if [ ! -f "$file" ]; then
        echo "run_ab.sh: '$file' not found -- run this from a complete checkout." >&2
        exit 1
    fi
done

case "$NUM_PROCESSES" in
    ''|*[!0-9]*) echo "run_ab.sh: NUM_PROCESSES='$NUM_PROCESSES' is not a number" >&2; exit 1 ;;
esac
[ "$NUM_PROCESSES" -ge 1 ] || { echo "run_ab.sh: NUM_PROCESSES must be >= 1" >&2; exit 1; }

# training.batch_size is per rank, so the global batch is what decides the step budget.
BATCH="$(awk '/^training:/{found=1} found && /^  batch_size:/{print $2; exit}' "$ON_CONFIG")"
if [ -z "${BATCH:-}" ]; then
    echo "run_ab.sh: could not read training.batch_size from $ON_CONFIG" >&2
    exit 1
fi
GLOBAL_BATCH=$((BATCH * NUM_PROCESSES))
EXPECTED_STEPS=$(( (CONTEXTS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))

echo "=== reader A/B on H200 ==="
echo "checkout      : $ROOT"
echo "HEAD          : $(git log --oneline -1 2>/dev/null || echo 'not a git checkout')"
echo "ranks x batch : $NUM_PROCESSES x $BATCH = global batch $GLOBAL_BATCH"
echo "expected      : 1 epoch = ceil($CONTEXTS / $GLOBAL_BATCH) = $EXPECTED_STEPS steps"
if [ "$EXPECTED_STEPS" -ne 7890 ]; then
    echo "NOTE: the cadences in both configs are written for 7890 steps. At $EXPECTED_STEPS" >&2
    echo "      steps per epoch they are not wrong, just coarser/finer than designed:" >&2
    echo "      warmup 400 = $(awk -v w=400 -v s="$EXPECTED_STEPS" 'BEGIN{printf "%.1f", 100*w/s}')% of the run," \
         "TF every 500 = $((EXPECTED_STEPS / 500)) points." >&2
fi
echo "order         : ON then OFF (sequential; the pair is only comparable if both run)"
echo

if [ "$DRYRUN" != "0" ]; then
    for arm in on off; do
        echo "would run: CONFIG=configs/qwen-1.7b/ab_h200_${arm}.yaml NUM_PROCESSES=$NUM_PROCESSES bash scripts/train.sh"
    done
    echo "DRYRUN OK (nothing launched)"
    exit 0
fi

mkdir -p "$LOG_DIR" || exit 1
STAMP="$(date +%Y%m%d_%H%M%S)"

for arm in on off; do
    if [ "$arm" = "on" ]; then
        config="$ON_CONFIG"
    else
        config="$OFF_CONFIG"
    fi
    log="$LOG_DIR/ab_${arm}_${STAMP}.log"
    echo "=== arm $arm: CONFIG=$config -> $log ==="
    CONFIG="$config" NUM_PROCESSES="$NUM_PROCESSES" bash scripts/train.sh 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        # A half-run pair cannot be read as an A/B, and a failed arm usually means OOM --
        # exactly the failure mode the OFF-arm run exposed before. Stop and point at the log.
        echo "run_ab.sh: arm '$arm' exited with status $status; stopping. Last 40 lines:" >&2
        tail -n 40 "$log" >&2
        echo "run_ab.sh: full log: $log" >&2
        exit "$status"
    fi
    echo "=== arm $arm finished ==="
    echo
done

echo "=== both arms finished ==="
echo "checkpoints : outputs/ab_h200_on, outputs/ab_h200_off"
echo "wandb       : $(ls -dt wandb/offline-run-*-*  2>/dev/null | head -2 | tr '\n' ' ')"
echo "logs        : $LOG_DIR/ab_on_${STAMP}.log, $LOG_DIR/ab_off_${STAMP}.log"
