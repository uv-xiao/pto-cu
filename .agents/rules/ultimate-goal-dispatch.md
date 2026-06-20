# Ultimate Goal Dispatch Rules

Use this rule when a goal is too large for one PR or one Codex session.

When creating a new ultimate goal, start from
`.agents/templates/ultimate-goal.md`. The template describes how to write the
umbrella goal file, dispatch log, worker prompt, child PR lifecycle, and
completion pattern.

## Default Operating Pattern

Ultimate-goal work defaults to a dispatcher/worker split:

- the current long-running `/goal` session is the dispatcher and repository
  monitor;
- implementation happens in child Codex sessions or bounded worker agents;
- every child launch, branch, scope change, verification result, PR, merge, and
  handoff is recorded in the dispatch log;
- each child slice uses one PR-sized branch unless the dispatcher records a
  narrower dependency PR first;
- the dispatcher does not perform direct long-running implementation work.

Dispatcher-owned work should be limited to state audits, dispatch-log updates,
worker prompts, shared contracts, review, merge decisions, progress reports,
and short emergency stabilization. Emergency stabilization means a small,
bounded change needed to unblock or preserve the dispatch system itself; record
the reason, diff scope, and verification in the dispatch log.

## Modes

- `pr-sized`: Default mode. One branch, one PR, one coherent review slice.
- `ultimate-goal-dispatcher`: Owns the umbrella goal, repository monitoring,
  decomposition, worker dispatch, logs, cross-PR sequencing, review decisions,
  and final promotion into stable docs. It does not own direct feature
  implementation except for recorded emergency stabilization.
- `worker`: Owns one child slice. A worker may open and merge its own PR when
  verification and review requirements are satisfied, but it must not dispatch
  nested workers.
- `dependency-pr`: A small unblocking PR for reusable framework behavior,
  shared contracts, or tooling needed by one or more workers.

## Dispatcher Responsibilities

The dispatcher must:

- keep an umbrella note under `docs/in_progress/`;
- split the goal into child slices by ownership boundary, not convenience;
- prefer runtime/platform/docs/evaluation vertical slices for CUDA backend
  work;
- record every dispatch, transcript or pane locator, PR, merge, abandoned
  branch, and handoff in a dispatch log;
- keep the parent goal description, child PR descriptions, progress reports,
  and design docs in sync;
- decide whether a child slice is ready to merge based on recorded local
  verification and review evidence;
- monitor launched child Codex sessions read-only unless a narrow, approved
  steer is required;
- avoid taking over a child slice directly. If a child stalls, record the
  blocker, steer or stop the worker, and dispatch a new PR-sized child slice;
- launch only one mutable worker at a time unless the dispatch log records an
  explicit isolation strategy.

## Worker Responsibilities

A worker must:

- start from the parent dispatch prompt and its own `/goal`;
- read the parent umbrella note, dispatch log, and any slice-specific
  preparation files;
- work on one feature branch and one child PR unless the dispatcher explicitly
  re-scopes it;
- record scope changes, local verification, PR URL, merge state, and follow-up
  gaps in the dispatch log or in a worker handoff file named by the
  dispatcher;
- avoid nested dispatch. If the worker finds more parallel work, it records a
  proposed child slice for the dispatcher instead of launching another worker.

## Starting Codex Workers

The default worker launch shape is:

```bash
codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C <repo-root> \
  "<worker prompt beginning with /goal ...>"
```

The dispatcher may wrap this command in a script when a goal needs repeatable
worker startup. The script or command must be copied into the dispatch log
before or immediately after launch.

If the worker is launched in tmux or another persistent terminal, the
dispatcher must also record the monitor locators needed by
`.agents/skills/codex-goal-monitor/SKILL.md`: transcript/session id when
known, pane target when available, starting commit, branch, and expected PR
slot. If those locators are discovered after launch, append them to the same
dispatch entry.

## Dispatch Log Requirements

Each ultimate goal must have a dispatch log near its artifacts, for example
`docs/in_progress/<goal_slug>/dispatch_log.md`.

Every dispatch entry must include:

- timestamp and dispatcher session;
- worker id and objective;
- exact Codex command or script invocation;
- transcript/session id and tmux pane when the worker is monitorable;
- branch name, PR URL or planned PR slot, and parent goal;
- allowed scope and files;
- dependencies and blocked assumptions;
- verification commands and results;
- merge decision and merge commit, when applicable;
- handoff summary and remaining gaps.

The log is review evidence. Do not rely on private terminal scrollback or
unstated session memory.
