#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bizhaoh200}"
REMOTE_PTO_CU="${REMOTE_PTO_CU:-}"
SYNC_TREE=0

usage() {
  cat <<'USAGE' >&2
Usage:
  run-remote-cuda.sh [--host HOST] [--remote-dir DIR] [--sync] -- <command...>

Environment:
  REMOTE_HOST     SSH host, default bizhaoh200.
  REMOTE_PTO_CU   Remote PTO checkout. Required unless --remote-dir is passed.

Examples:
  REMOTE_PTO_CU=/work/pto-cu run-remote-cuda.sh -- nvidia-smi
  REMOTE_PTO_CU=/work/pto-cu run-remote-cuda.sh --sync -- \
    .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py -q --platform cuda
USAGE
}

while (($#)); do
  case "$1" in
    --host)
      test "$#" -ge 2 || { usage; exit 2; }
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-dir)
      test "$#" -ge 2 || { usage; exit 2; }
      REMOTE_PTO_CU="$2"
      shift 2
      ;;
    --sync)
      SYNC_TREE=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

test -n "$REMOTE_PTO_CU" || { usage; exit 2; }
test "$#" -gt 0 || { usage; exit 2; }

if [[ "$SYNC_TREE" == "1" ]]; then
  rsync -a --delete \
    --exclude=.venv --exclude=.venv-* --exclude=build --exclude=tmp \
    --exclude=__pycache__ --exclude=.pytest_cache \
    ./ "${REMOTE_HOST}:${REMOTE_PTO_CU}/"
fi

remote_command=$(printf '%q ' "$@")
ssh "$REMOTE_HOST" \
  "cd $(printf '%q' "$REMOTE_PTO_CU") && CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:\$PATH PYTHONPATH=\$PWD:\$PWD/python ${remote_command}"
