# Serving Command Artifact Guard

## Code And Data Changed

- Tightened serving command-plan validation so every declared
  `raw_artifact` must live under the plan's `tmp/` artifact root and be
  referenced by the generated command.
- Updated the VDCores Qwen3 decode command builder to capture stdout/stderr
  logs with `tee`, because the current VDCores script does not expose a
  machine-readable output JSON flag.
- Regenerated the sharded serving command-plan records so VDCores planned
  artifacts are `.log` files that the commands actually write.

## Architecture Quality

Serving command plans now have a direct code-to-artifact contract: every raw
artifact path visible in the benchmark viewer is either produced by a command
argument or by an explicit capture redirection in the command.

## Evaluation Run

Focused verification passed:

- `validate_benchmark_viewer_data.py`
- `validate_nvidia_changelog.py`
- `check_nvidia_review_ready.py`
- `pytest -q -k 'paper_serving_command_plan_generates_policy_commands or
  benchmark_viewer_schema_validator_passes'`
- `git diff --check`

## Remaining Gaps

VDCores still needs a machine-readable serving benchmark output path before
the log capture can become a paper result import.
