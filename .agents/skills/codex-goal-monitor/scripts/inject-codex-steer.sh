#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  inject-codex-steer.sh verify-target <tmux-pane>
  inject-codex-steer.sh type <tmux-pane> <payload-file>
  inject-codex-steer.sh submit <tmux-pane>
  inject-codex-steer.sh send <tmux-pane> <payload-file>
USAGE
}

require_pane() {
  local pane="$1"
  tmux display-message -p -t "$pane" '#{session_name}:#{window_index}.#{pane_index}' >/dev/null
}

paste_payload() {
  local pane="$1"
  local payload_file="$2"

  test -f "$payload_file"
  require_pane "$pane"
  tmux load-buffer "$payload_file"
  tmux paste-buffer -t "$pane"
}

submit_input() {
  local pane="$1"

  require_pane "$pane"
  tmux send-keys -t "$pane" Enter
}

cmd="${1:-}"
case "$cmd" in
  verify-target)
    test "$#" -eq 2 || { usage; exit 2; }
    require_pane "$2"
    ;;
  type)
    test "$#" -eq 3 || { usage; exit 2; }
    paste_payload "$2" "$3"
    ;;
  submit)
    test "$#" -eq 2 || { usage; exit 2; }
    submit_input "$2"
    ;;
  send)
    test "$#" -eq 3 || { usage; exit 2; }
    paste_payload "$2" "$3"
    submit_input "$2"
    ;;
  *)
    usage
    exit 2
    ;;
esac
