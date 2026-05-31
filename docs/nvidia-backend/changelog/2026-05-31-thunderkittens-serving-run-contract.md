# 2026-05-31 ThunderKittens Serving Run Contract

## Code And Data Changed

- Added `thunderkittens_decode_attention_tile` as the ThunderKittens
  serving-family run for the LLM-serving paper claim.
- Extended `paper_serving_command_plan.py` with ThunderKittens
  decode-attention commands and limited serving command planning to
  LLM-serving paper-baseline runs.
- Updated run-readiness generation so repo-local capture wrappers under
  `.agents/` can be checked alongside Python entrypoints in baseline source
  trees.
- Regenerated `paper_baseline_run_readiness.json` and
  `paper_readiness_audit.json`.

## Architecture Quality

The LLM-serving audit no longer hides ThunderKittens behind a missing-run
blocker. ThunderKittens now has the same planned-run, readiness, command-plan,
and required-metric contract shape as the other paper-family baselines, while
remaining clearly labeled as a controlled serving-equivalent kernel baseline.

## Evaluation Run

The focused tests first failed because the LLM-serving audit lacked a
ThunderKittens run-readiness record and the command planner still emitted only
MPK, VDCores, vLLM, and SGLang rows. After the update, the focused audit and
command-plan tests passed.

## Remaining Gaps

This is a run contract and readiness update, not a measured performance result.
The H200 ThunderKittens decode-attention serving-family rows still need raw
captures and viewer import, alongside the MPK, VDCores, vLLM, and SGLang
serving baseline captures.
