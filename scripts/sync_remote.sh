#!/usr/bin/env bash
# 4090's committed HEAD is authoritative; compute hosts need only SSH access.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
case "${1:-}" in
  h200) targets=(h200);;
  a800|A800) targets=(a800);;
  all) targets=(h200 a800);;
  *) echo "Usage: bash scripts/sync_remote.sh h200|a800|all"; exit 2;;
esac
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'ERROR: commit local changes first (including YAML); nothing has been synced.' >&2
  git status --short
  exit 2
fi
local_head=$(git rev-parse HEAD)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)

sync_one() (
  set -euo pipefail
  target=$1
  case "$target" in
    h200) host=${H200_SSH_HOST:-h200}; repo=${H200_REPO:-/home/lijie/proj2/xmu/lz/MemoryQwen};;
    a800) host=${A800_SSH_HOST:-A800}; repo=${A800_REPO:-/mnt/beegfs-storage-a8002/seanlz/MemoryQwen};;
  esac
  # Restrict interpolated remote paths to shell-safe absolute paths.
  [[ "$repo" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'ERROR: repo must be an absolute shell-safe path' >&2; exit 2; }
  remote_head=$(ssh "${ssh_opts[@]}" "$host" "cd $repo && git rev-parse HEAD")
  [[ "$remote_head" =~ ^[0-9a-f]{40,64}$ ]] || { echo 'ERROR: invalid remote HEAD' >&2; exit 2; }
  if [[ "$remote_head" != "$local_head" ]]; then
    git cat-file -e "$remote_head^{commit}" 2>/dev/null || {
      echo "ERROR: $target HEAD $remote_head is unknown locally; inspect remote history first." >&2; exit 2;
    }
    git merge-base --is-ancestor "$remote_head" "$local_head" || {
      echo "ERROR: $target has divergent/ahead history; refusing to overwrite it." >&2; exit 2;
    }
    git bundle create "$work/$target.bundle" "$remote_head..$local_head" HEAD
  fi
  remote_tmp=$(ssh "${ssh_opts[@]}" "$host" 'mktemp -d /tmp/memoryqwen-sync.XXXXXXXX')
  [[ "$remote_tmp" =~ ^/tmp/memoryqwen-sync\.[a-zA-Z0-9]+$ ]] || exit 2
  trap 'ssh "${ssh_opts[@]}" "$host" "rm -rf $remote_tmp" >/dev/null 2>&1 || true' EXIT
  if [[ "$remote_head" != "$local_head" ]]; then
    ssh "${ssh_opts[@]}" "$host" "cat > $remote_tmp/update.bundle" < "$work/$target.bundle"
  fi
  ssh "${ssh_opts[@]}" "$host" "bash -s -- $repo $remote_head $local_head $remote_tmp" <<'REMOTE'
set -euo pipefail
cd "$1"
# Serialize this script's updates and recheck state after transfer.
exec 9>"$(git rev-parse --git-path codex-sync.lock)"
flock -n 9 || { echo 'ERROR: another sync is running' >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$2" ]] || { echo 'ERROR: remote HEAD changed during sync' >&2; exit 2; }
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'ERROR: compute-host changes must be reconciled on 4090 first; refusing to overwrite:' >&2
  git status --short
  exit 2
fi
if [[ "$2" != "$3" ]]; then
  git bundle verify "$4/update.bundle"
  git fetch "$4/update.bundle" HEAD
  [[ "$(git rev-parse FETCH_HEAD)" == "$3" ]]
  git merge --ff-only FETCH_HEAD
fi
[[ "$(git rev-parse HEAD)" == "$3" ]]
echo "synced: $(git rev-parse --short HEAD) $PWD"
REMOTE
)
failed=0
for target in "${targets[@]}"; do
  echo "=== Sync $target to $local_head ==="
  # Keep errexit active inside sync_one while still trying the next target.
  set +e
  sync_one "$target"
  result=$?
  set -e
  if ((result != 0)); then echo "FAILED: $target ($result)" >&2; failed=1; fi
done
exit "$failed"
