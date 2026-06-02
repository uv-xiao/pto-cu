# 2026-06-03 Persistent Tensor-Arg Helper Capacity

## Code And Data Changed

- Updated generated persistent-device CUDA source so
  `pto_cuda_tensor_arg_f32` uses the same five-slot tensor-argument capacity
  as `PtoCudaPersistentDagTask`.
- Added a generated-source regression that rejects the stale four-slot helper
  guard.
- Kept benchmark-viewer data unchanged. The live run below is retained only as
  raw implementation evidence under `tmp/`.

## Architecture Quality

The persistent task ABI, Python launch packet, descriptor materializer, and
generated device helper now agree on five tensor slots. This is required for
Qwen task bodies that bind `tensor_args[4]`, such as QK-norm's runtime KV page
table.

## Evaluation Run

- Failed before the source-template fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_persistent_codegen.py \
    -q -k generic_argument_slots
  ```

- Passed Qwen generated-source coverage:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/persistent-helper-capacity-mpk-2026-06-03
  PYTHONPATH=$PWD:$PWD/python timeout 480 .venv/bin/python \
    examples/cuda/qwen_decode_loop_runner.py --mode mock \
    --single-context-live-session --run-resource-backed-smoke \
    --resource-backed-task-selection first_layer_with_logits \
    --resource-backed-workload mpk_offline_decode \
    --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
    --resource-backed-worker-blocks 10 \
    --resource-backed-logits-check-policy final_step \
    --resource-backed-logits-active-cols full \
    --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
    --device 0 --arch compute_80 \
    --cache-root $ARTIFACT/cache \
    --output-json $ARTIFACT/qwen-decode-loop-runner.json
  ```

- Result: generated dispatch rebuilt (`cache_hit=false`), `mpk_offline_decode`
  completed 10 resource-backed task functions with zero scheduler errors,
  checked 2,430,976 logits elements, sampled token `63690`, and passed the
  diagnostic projection reference with `max_abs_error=1.2e-07`.

## Remaining Gaps

PTO still needs full Qwen numerical comparison and full-serving MPK/VDCores
rows before the LLM-serving paper claim can pass.
