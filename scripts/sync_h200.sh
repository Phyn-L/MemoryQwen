#!/usr/bin/env bash
# Local is authoritative; outputs, secrets and Git metadata stay machine-local.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
git ls-files -co --exclude-standard -z > "$LIST"
# Save overwritten remote files outside the checkout for recovery.
BACKUP="../MemoryQwen-sync-backups/$(date +%Y%m%d_%H%M%S)"
rsync -rcliv --backup --backup-dir="$BACKUP" --from0 --files-from="$LIST" \
  -e 'ssh -o BatchMode=yes' "$ROOT/" h200:~/proj2/xmu/lz/MemoryQwen/
rsync -rclni --from0 --files-from="$LIST" \
  -e 'ssh -o BatchMode=yes' "$ROOT/" h200:~/proj2/xmu/lz/MemoryQwen/
