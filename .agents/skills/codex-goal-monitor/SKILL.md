---
name: codex-goal-monitor
description: Monitor a separate Codex /goal session through tmux, audit it read-only, and inject approved steering prompts through a hardened helper.
---

# Codex Goal Monitor

Use this skill when one Codex session must supervise another Codex `/goal`
session. This is the Codex-to-Codex adaptation of the upstream
`monitor-codex-goal` source, not a direct copy.

## Required Inputs

- Codex session id: the UUID in `~/.codex/sessions/**/rollout-*.jsonl`.
- tmux pane: exact target pane such as `session:window.pane`.
- Cadence or one-shot mode: run a manual audit once, or arm an external
  scheduler for recurring ticks. Codex does not expose Claude Code's
  `CronCreate`/`PushNotification` tools in this repo environment, so recurring
  ticks are implemented with `cron`, `systemd --user`, or a tmux shell loop.
- Parent dispatch log: the goal log that records the child worker's launch,
  branch, scope, and expected PR.

If either locator is uncertain, discover first:

```bash
find ~/.codex/sessions -name 'rollout-*.jsonl' -print | sort | tail -20
tmux list-panes -a \
  -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_title}'
```

Use `~` for Codex transcript paths in notes. Do not write private absolute
paths, user names, or hostnames into committed docs.

## Worker Launch Record

When monitoring a launched Codex worker, first find or create the worker entry
in the parent dispatch log. It should contain:

- worker id and objective;
- parent goal and child slice;
- exact Codex command or script invocation;
- branch name, starting commit, and planned PR slot;
- allowed files or artifact families;
- expected verification commands;
- transcript path or session id, if known;
- tmux pane target, if the worker is running in a pane;
- current status: launched, running, steered, blocked, PR opened, merged, or
  abandoned.

If the transcript or pane was not recorded at launch, discover it before
auditing and append the locator to the dispatch log. Do not rely on private
terminal scrollback as the only record of a worker.

Helpful discovery commands:

```bash
# Recent Codex transcripts, newest last.
find ~/.codex/sessions -name 'rollout-*.jsonl' -print | sort | tail -20

# Candidate tmux panes with stable pane targets.
tmux list-panes -a \
  -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_title}'

# Recent pane text when matching a pane to a worker prompt.
tmux capture-pane -p -t '<session:window.pane>' -S -200

# Basic transcript orientation without editing it.
tail -n 80 ~/.codex/sessions/**/rollout-*.jsonl
```

## Safety Contract

- The monitor is read-only by default: use `git status`, `git diff`,
  `git log`, `rg`, `sed`, `find`, `jq`, `tail`, `head`, `wc`, and
  `tmux capture-pane` only for inspection.
- Scheduler ticks may write only monitor-owned artifacts under `tmp/` or
  another explicitly named scratch directory. They must not edit the target
  repository, target transcript, or target tmux pane.
- Do not edit the target repository, target transcript, or target tmux pane
  while auditing.
- If the target is the parent dispatcher, treat direct implementation as drift
  unless it is recorded emergency stabilization of the dispatch system.
- No raw `tmux send-keys` from the agent. The only allowed injection path is
  `scripts/inject-codex-steer.sh` after `verify-target`.
- Every steer requires human approval unless it only reinforces a previously
  stated human constraint about no fabricated evidence, no fake/stub
  implementation, or no false completion claim.
- If the transcript, pane, or current `/goal` state is ambiguous, stop and ask.

## Recurring Scheduler

This section adapts the upstream `monitor-codex-goal` cron behavior to Codex.
The upstream version uses Claude Code `CronCreate` and phone push tools. Codex
does not provide those builtins here, so the monitor uses an external scheduler
that runs a read-only tick script and records concise snapshots for the parent
session to inspect.

Use the bundled tick helper:

```bash
.agents/skills/codex-goal-monitor/scripts/monitor-codex-goal-tick.sh \
  --session-id '<codex-session-id>' \
  --pane '<session:window.pane>' \
  --worktree '<target-worktree>' \
  --out-dir tmp/codex-goal-monitor/<worker-name>
```

The helper verifies the pane, captures a bounded tmux tail, tails the transcript
without loading it whole, records `git status` / `git log -1` / optional dirty
diff stats, and writes a compact `latest/summary.md`. The parent should read
`summary.md` first and open larger captures only when the summary reports a
finding, terminal state, dirty worktree, or missing locator.

To arm a recurring tick, prefer one of these patterns:

```bash
# tmux loop, easy to stop by killing the monitor-loop pane.
while :; do
  .agents/skills/codex-goal-monitor/scripts/monitor-codex-goal-tick.sh \
    --session-id '<codex-session-id>' \
    --pane '<session:window.pane>' \
    --worktree '<target-worktree>' \
    --out-dir tmp/codex-goal-monitor/<worker-name>
  sleep 3600
done

# cron entry; use absolute paths in a wrapper script for readability.
17 * * * * '<repo>/tmp/codex-goal-monitor/<worker-name>/tick.sh'
```

The wrapper should `cd` to the repo and run the helper with absolute
`--worktree` / `--out-dir` paths, appending stdout/stderr to `cron.log`.

Do not run frequent captures. Default to 30-60 minutes for long-running workers,
or 10-15 minutes only while a worker is near a handoff. Each parent review
should inspect at most the latest summary unless there is a concrete reason to
open full tmux or transcript captures.

## Audit Loop

1. Resolve the transcript under `~/.codex/sessions` and verify the tmux pane:

   ```bash
   .agents/skills/codex-goal-monitor/scripts/inject-codex-steer.sh \
     verify-target '<session:window.pane>'
   ```

2. If a scheduler is armed, read only the latest monitor `summary.md` first.
   Open the full transcript or pane captures only when the summary flags a
   reason to inspect them.
3. Read recent transcript entries line-by-line. Extract the latest human
   steering, current goal status, completion claims, tool failures, and blocked
   state. Do not treat agent summaries as evidence unless commands or artifacts
   back them.
4. Inspect the target worktree read-only. Compare claims against `git status`,
   `git diff`, tests, benchmark outputs, and docs mentioned by the target.
5. Classify findings:
   - drift from the human's latest constraints;
   - worker doing dispatcher-owned decomposition or launching nested workers;
   - direct long-running implementation in the dispatcher session;
   - child work not recorded in the dispatch log;
   - progress claimed from private scrollback or unmerged local changes;
   - branch, PR, or allowed-scope mismatch;
   - fabricated data or unsupported performance claims;
   - fake/stub implementation described as working;
   - broken project invariant or skipped verification;
   - repeated tool failure or blocker without a blocked-status handoff.
6. If clean, report the transcript id, pane, checked commit, and evidence read.
7. If steering is needed, write a narrow prompt to a scratch file. It must name
   the desired outcome, verification surface, constraints that must not regress,
   and when the target should mark itself blocked instead of complete.
8. After human approval, inject:

   ```bash
   .agents/skills/codex-goal-monitor/scripts/inject-codex-steer.sh \
     send '<session:window.pane>' /path/to/steer.txt
   ```

## Steering Criteria

Steer only when the evidence shows one of these conditions:

- the worker is outside its allowed files, branch, or child-slice objective;
- the worker is about to claim completion without fresh verification evidence;
- the worker is fabricating, guessing, or omitting benchmark/test evidence;
- the worker is dispatching nested workers or creating unlogged child work;
- the worker is blocked but continues to churn without recording a handoff;
- the worker ignores newer human or dispatcher constraints;
- the dispatcher session starts direct implementation instead of recording and
  dispatching a PR-sized child slice;
- the dispatcher claims goal progress without a PR, dispatch-log entry,
  handoff, verification record, or merge decision;
- the worker needs a narrow reminder to update the dispatch log, PR
  description, or handoff file before stopping.

Do not steer for style preferences, speculative improvements, or work that is
merely slow but still inside the recorded objective. If the right next action
is unclear, report the ambiguity instead of injecting a prompt.

## Helper Commands

```bash
# Capture one bounded recurring-monitor tick.
scripts/monitor-codex-goal-tick.sh \
  --session-id '<codex-session-id>' \
  --pane '<session:window.pane>' \
  --worktree '<target-worktree>' \
  --out-dir tmp/codex-goal-monitor/<worker-name>

# Verify the pane exists and can be addressed.
scripts/inject-codex-steer.sh verify-target '<session:window.pane>'

# Paste text into the Codex input box without submitting it.
scripts/inject-codex-steer.sh type '<session:window.pane>' /path/to/steer.txt

# Submit whatever is currently in the input box.
scripts/inject-codex-steer.sh submit '<session:window.pane>'

# Paste and submit a reviewed steer in one command.
scripts/inject-codex-steer.sh send '<session:window.pane>' /path/to/steer.txt
```

## Handoff

Record monitor runs in the parent goal's dispatch log or a named handoff file:
transcript path, pane, target commit, findings, injected steer text path,
approval mode, and verification evidence. Do not rely on terminal scrollback.
