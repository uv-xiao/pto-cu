# 2026-06-03 Qwen Layer-8 Prefix Evidence

## Code And Data Changed

- Added no new runtime code in this report.
- Captured fresh A100 resource-backed MPK evidence after the MLP residual
  binding fix, scaling the verified layer-prefix boundary from one layer to
  eight layers with full projection and full logits windows.
- Updated the Qwen full-serving remaining-gap status so review readers can
  distinguish passing layer-prefix evidence from still-open full-prefix
  36-layer correctness.

## Architecture Quality

The layer-prefix sweep exercises the same live resource path as the full DAG:
real token pointers, resident safetensors, activation workspace, KV cache,
generated persistent-device task bodies, and logits readout. Keeping these
artifacts outside the viewer import path preserves the strict promotion gate:
prefix diagnostics can narrow kernel and launch-packet bugs, but only
full-serving MPK and VDCores rows with Hugging Face token/logit agreement can
be imported as paper-readiness evidence.

## Evaluation Run

Two-layer MPK prefix:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --workspace-cuda-live \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 --resource-backed-repeat-runs 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 2 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols full \
  --resource-backed-logits-active-cols full --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-layer2-after-mlp-residual-fix.json
```

Result: 17/17 tasks completed, zero scheduler errors, no row-0 non-finite
activations, full finite logits, populated top-k, and a passing diagnostic
logits reference.

Four-layer MPK prefix:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --workspace-cuda-live \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 --resource-backed-repeat-runs 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 4 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols full \
  --resource-backed-logits-active-cols full --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-layer4-after-mlp-residual-fix.json
```

Result: 31/31 tasks completed, zero scheduler errors, no row-0 non-finite
activations, full finite logits, populated top-k, and a passing diagnostic
logits reference.

Eight-layer MPK prefix:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --workspace-cuda-live \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 --resource-backed-repeat-runs 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 8 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols full \
  --resource-backed-logits-active-cols full --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-layer8-after-mlp-residual-fix.json
```

Result: 59/59 tasks completed, zero scheduler errors, no row-0 non-finite
activations, full finite logits, populated top-k, and a passing diagnostic
logits reference with `max_abs_error=3.47e-06`.

## Remaining Gaps

The 36-layer full-prefix MPK run still exceeded the local 180-second bounded
attempt window after the MLP residual binding fix. Full-serving correctness
and paper-readiness remain gated on MPK and VDCores policy-length rows that
match the Hugging Face Qwen/Qwen3-8B token/logit reference and include
latency/throughput metrics.
