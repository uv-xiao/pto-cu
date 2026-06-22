# CUDA Eval Script Selection

This note records the intended active script surface for
`.agents/skills/cuda-backend-eval/`. The restart goal asked for useful agent
guidance, not a permanent pile of prior capture machinery.

## Active Script Surface

- `run-remote-cuda.sh`: generic SSH/rsync wrapper for running arbitrary CUDA
  commands on a configured remote checkout.

This is currently the only committed script in the skill. Future workers
should use it for remote H200 commands and keep task-specific probes under
`tmp/` unless a reviewed PR promotes them into durable examples, tests, or
docs.

## Retired Script Classes

Do not recreate capture-specific scripts such as paired smoke runners,
benchmark matrix generators, validator dumps, scheduler wrappers, or report
generators inside the skill. If a slice needs one, start with an explicit
command or temporary helper under `tmp/`, then promote only the reusable
contract through a reviewable PR.

## Next Cleanup

If a future PR adds a second script to this skill, it must update this note and
the skill manual in the same commit. The default should remain a small,
general remote-command surface rather than historical capture accumulation.
