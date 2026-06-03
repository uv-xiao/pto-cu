# 2026-06-03 PTO Qwen Live-Session Command

## Code And Data Changed

- Added `--single-context-live-session` to generated PTO persistent-device
  Qwen full-serving runner commands.
- Regenerated `serving_command_plan.json` so all PTO Qwen full-serving command
  rows open the live token, KV-cache, resident-weight, and activation-workspace
  owners needed by resource-backed execution.
- Added a regression assertion for the live-session flag in the serving command
  plan test.

## Architecture Quality

The PTO full-serving handoff command now matches the runner contract. Without
the live-session flag, `qwen_decode_loop_runner.py --run-resource-backed-smoke`
returns `single_context_live_session_required` and writes a non-executed
artifact. With the flag, the command reaches CUDA execution and can expose real
prefill/decode correctness failures.

## Evaluation Run

The pre-fix reproduction wrote
`tmp/cuda-backend/repro-no-single-context/qwen-runner.json` with:

- `resource_backed_execution.status=not_run`;
- `reason=single_context_live_session_required`.

The post-fix one-layer MPK smoke command was:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-prefill-prompt --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 1 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 2048 \
  --resource-backed-projection-active-cols 2048 \
  --output-json tmp/cuda-backend/repro-single-context-prefill/qwen-runner.json
```

Result: `resource_backed_execution.status=pass`,
`prompt_prefill.status=prompt_prefill_executed`, 18 prompt positions executed,
zero prompt-prefill scheduler errors, and a passing readout-only first decode
packet.

Focused regression:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_serving_command_plan_generates_policy_commands \
  -q
```

Result: one test passed after first failing on the missing live-session flag.

## Remaining Gaps

This removes a handoff blocker and proves the command reaches live CUDA
execution. PTO still needs model-equivalent prompt prefill, token/logit
agreement against Hugging Face, and full-serving MPK/VDCores rows with
latency and throughput before the paper claim can close.
