# 2026-06-03 Qwen MLP Residual Binding

## Code And Data Changed

- Fixed `qwen_mlp_down` launch-packet residual binding for layer-prefixed
  descriptors. `layer_0_mlp_down` now resolves `layer_0_input_norm` and binds
  the original layer input activation through `tensor_args[1]` instead of
  falling back to token IDs.
- Preserved the two-argument scalar contract for `unit_math_full_rmsnorm`
  RMSNorm callables so generated full-reduction branches can observe the
  zero scale sentinel in `scalar_args[1]`.
- Added activation row-finiteness evidence to resource-backed Qwen runs. The
  summary reports the first non-finite task, row-local column, NaN/Inf counts,
  and finite magnitude bounds for row 0.

## Architecture Quality

The old prefix lookup only stripped `_post_attention_norm`, so MLP down tasks
could not find their matching input norm descriptor. For layer 0 that fallback
bound token storage as a float residual tensor, which corrupted the residual
stream before final norm and logits. Matching `_mlp_down` to the same layer
prefix restores the intended activation-chain contract without adding graph
tasks or changing kernel ABI.

## Evaluation Run

RED: `test_launch_packet_binds_mlp_down_residual_source` failed with
`tensor_args[1] == 0x3000` for the layer-0 MLP down task, proving the launch
packet was binding token IDs instead of the embedding activation. GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  -q -k mlp_down_residual_source
```

Result: `1 passed, 37 deselected`.

Broader Qwen regressions passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: `63 passed`.

- A100 artifact:
  `tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-layer1-after-mlp-residual-fix.json`
  completed the one-layer-plus-logits MPK workload with 10/10 tasks completed,
  zero scheduler errors, no row-0 non-finite activations, full finite logits,
  populated row-0 top-k, and a passing diagnostic logits reference.

The artifact was produced with:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --workspace-cuda-live \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 --resource-backed-repeat-runs 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 1 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols full \
  --resource-backed-logits-active-cols full --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-layer1-after-mlp-residual-fix.json
```

## Remaining Gaps

This is first-layer resource-backed correctness evidence. Full-prefix
36-layer MPK and VDCores policy-length full-serving rows still require
full-model token/logit agreement against the Hugging Face reference before
viewer import or paper-readiness promotion.
