#!/usr/bin/env bash
# Pull the GitHub main branch on A800 without overwriting local work.
set -euo pipefail
REMOTE_HOST="${A800_SSH_HOST:-A800}"
REMOTE_REPO="${A800_REPO:-~/MemoryQwen}"
ssh "$REMOTE_HOST" "REPO=$REMOTE_REPO bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$REPO"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: remote checkout has uncommitted changes; refusing to pull: $PWD" >&2
  git status --short
  exit 2
fi
git fetch origin main
git merge --ff-only origin/main
echo "synced: $(git rev-parse --short HEAD) $PWD"
REMOTE_SCRIPT
