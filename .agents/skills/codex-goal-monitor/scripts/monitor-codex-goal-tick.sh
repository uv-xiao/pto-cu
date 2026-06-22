#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  monitor-codex-goal-tick.sh --session-id <uuid> --pane <tmux-pane> --worktree <path> --out-dir <dir> [--transcript <path>] [--tail-lines <n>]

Runs one read-only monitor tick and writes bounded captures plus latest/summary.md.
USAGE
}

session_id=""
pane=""
worktree=""
out_dir=""
transcript=""
tail_lines="120"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session-id)
      session_id="${2:-}"
      shift 2
      ;;
    --pane)
      pane="${2:-}"
      shift 2
      ;;
    --worktree)
      worktree="${2:-}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:-}"
      shift 2
      ;;
    --transcript)
      transcript="${2:-}"
      shift 2
      ;;
    --tail-lines)
      tail_lines="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 64
      ;;
  esac
done

if [ -z "$session_id" ] || [ -z "$pane" ] || [ -z "$worktree" ] || [ -z "$out_dir" ]; then
  usage
  exit 64
fi

case "$tail_lines" in
  ''|*[!0-9]*)
    echo "tail-lines must be a positive integer" >&2
    exit 64
    ;;
esac

if [ -z "$transcript" ]; then
  transcript="$(find "$HOME/.codex/sessions" -name "rollout-${session_id}.jsonl" -print | sort | tail -1)"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${out_dir%/}/runs/$timestamp"
latest_dir="${out_dir%/}/latest"
mkdir -p "$run_dir" "$latest_dir"

summary="$run_dir/summary.md"
pane_capture="$run_dir/tmux-tail.txt"
transcript_tail="$run_dir/transcript-tail.jsonl"
status_file="$run_dir/git-status.txt"
log_file="$run_dir/git-log.txt"
diffstat_file="$run_dir/git-diff-stat.txt"

pane_status="ok"
if ! tmux display-message -p -t "$pane" '#{session_name}:#{window_index}.#{pane_index}' >"$run_dir/pane-address.txt" 2>"$run_dir/pane-error.txt"; then
  pane_status="missing"
fi

if [ "$pane_status" = "ok" ]; then
  if ! tmux capture-pane -p -t "$pane" -S "-$tail_lines" >"$pane_capture" 2>>"$run_dir/pane-error.txt"; then
    pane_status="missing"
    : >"$pane_capture"
  fi
else
  : >"$pane_capture"
fi

transcript_status="ok"
if [ -z "$transcript" ] || [ ! -f "$transcript" ]; then
  transcript_status="missing"
  : >"$transcript_tail"
else
  tail -n "$tail_lines" "$transcript" >"$transcript_tail"
fi

worktree_status="ok"
if [ ! -d "$worktree/.git" ] && ! git -C "$worktree" rev-parse --git-dir >/dev/null 2>&1; then
  worktree_status="missing"
  : >"$status_file"
  : >"$log_file"
  : >"$diffstat_file"
else
  git -C "$worktree" status --short --branch >"$status_file"
  git -C "$worktree" log -1 --oneline --decorate >"$log_file"
  git -C "$worktree" diff --stat >"$diffstat_file"
fi

dirty_count="unknown"
if [ "$worktree_status" = "ok" ]; then
  dirty_count="$(git -C "$worktree" status --porcelain=v1 | wc -l | tr -d ' ')"
fi

{
  echo "# Codex Goal Monitor Tick"
  echo
  echo "- timestamp_utc: $timestamp"
  echo "- session_id: $session_id"
  echo "- pane: $pane"
  echo "- pane_status: $pane_status"
  echo "- transcript: ${transcript:-missing}"
  echo "- transcript_status: $transcript_status"
  echo "- worktree: $worktree"
  echo "- worktree_status: $worktree_status"
  echo "- dirty_count: $dirty_count"
  echo
  echo "## Files"
  echo
  echo "- tmux_tail: $pane_capture"
  echo "- transcript_tail: $transcript_tail"
  echo "- git_status: $status_file"
  echo "- git_log: $log_file"
  echo "- git_diff_stat: $diffstat_file"
  echo
  echo "## Git Status"
  echo
  if [ -s "$status_file" ]; then
    sed -n '1,40p' "$status_file"
  else
    echo "(empty)"
  fi
  echo
  echo "## Latest Commit"
  echo
  if [ -s "$log_file" ]; then
    sed -n '1,20p' "$log_file"
  else
    echo "(empty)"
  fi
} >"$summary"

cp "$summary" "$latest_dir/summary.md"
cp "$status_file" "$latest_dir/git-status.txt"
cp "$log_file" "$latest_dir/git-log.txt"
cp "$diffstat_file" "$latest_dir/git-diff-stat.txt"
cp "$pane_capture" "$latest_dir/tmux-tail.txt"
cp "$transcript_tail" "$latest_dir/transcript-tail.jsonl"

echo "$summary"
