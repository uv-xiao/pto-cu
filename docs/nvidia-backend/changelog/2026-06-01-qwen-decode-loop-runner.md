# 2026-06-01 Qwen Decode Loop Runner Plan

## Code And Data Changed

- Added `examples/cuda/qwen_decode_loop_runner.py` plus
  `examples/cuda/qwen_decode_loop_runner_impl/`.
- The artifact composes token pointer, KV-cache, and resident-weight lifecycle
  artifacts into persistent DAG submission plans for the MPK and VDCores
  serving policies.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Added a runner-owned unit-math live bridge so the decode-loop runner entry
  point can execute the repeated diagnostic unit-math DAG.
- Added `--token-cuda-live` so the decode-loop runner can open the
  process-scoped token pointer-table owner in CUDA-live mode while KV-cache
  and resident weights remain dry-run.
- Added `--kv-cuda-live` so the decode-loop runner can also open the full
  planned KV-cache owner in CUDA-live mode.
- Added `--resident-cuda-live` so the decode-loop runner can invoke the
  resident weight-table owner in CUDA-live mode.
- Added `cuda_live_submission_descriptor_contract`, which maps the live
  token, KV-cache, and resident-weight resources to Qwen persistent task
  function ids 7100 through 7109 and records the planned `run_prepared`
  repetitions.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-decode-loop-2026-06-01/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

The runner artifact makes the host-side lifetime boundary explicit:
token pointers, KV-cache pointers, and resident weight pointers are opened
before decode and weight argument materialization, remain live through
persistent DAG submission, and close after submission. It also records output
token accounting for each serving policy. The unit-math bridge keeps the
live CUDA evidence diagnostic, but proves the runner can attach an executed
CUDA sub-DAG summary instead of only pointing at adjacent artifacts.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline \
  --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-2026-06-01/\
qwen-decode-loop-runner.json
```

Result: `status=decode_loop_runner_plan_ready` with 1088 planned decode
iterations across `mpk_offline_decode` and `vdcores_offline_decode`.

Additional bridge command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-unit-math-live --device 0 --arch compute_80 \
  --repeat-runs 3 --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-unit-math-bridge-2026-06-01/\
qwen-decode-loop-runner.json
```

Result: `unit_math_live_bridge_contract.status=diagnostic_bridge_executed`,
`repeat_runs=3`, `total_completed_count=12`, `total_error_count=0`, and
`max_abs_error=0.0`.

Additional partial resource-owner command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --token-cuda-live --device 0 --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-token-live-2026-06-01/\
qwen-decode-loop-runner.json
```

Result: `mode=partial_cuda_live_submission_plan`,
`cuda_live_resource_owners=["token_pointer_table"]`, and
`resource_lifecycle_modes.token_pointer_table=cuda_live`.

Additional token plus KV resource-owner command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --token-cuda-live --kv-cuda-live --device 0 --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-token-kv-live-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `mode=partial_cuda_live_submission_plan`,
`cuda_live_resource_owners=["token_pointer_table","kv_cache"]`, and
`resource_lifecycle_modes.kv_cache=cuda_live`.

Additional token, KV, and resident resource-owner command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --token-cuda-live --kv-cuda-live --resident-cuda-live \
  --device 0 --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-token-kv-resident-live-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `mode=partial_cuda_live_submission_plan`,
`cuda_live_resource_owners=["token_pointer_table","kv_cache",`
`"resident_weight_table"]`, and
`resource_lifecycle_modes.resident_weight_table=cuda_live`.

Additional resource-backed descriptor command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --token-cuda-live --kv-cuda-live --resident-cuda-live \
  --device 0 --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-submission-descriptors-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `cuda_live_submission_descriptor_contract.status=`
`resource_backed_descriptors_ready`, `execution_status=not_executed`,
and descriptor rows for `mpk_offline_decode` and `vdcores_offline_decode`
with Qwen function ids 7100 through 7109.

## Remaining Gaps

- Generate Qwen kernel bodies that consume token, KV-cache, and weight fields.
- Run the resource-backed descriptors through `run_prepared`, validate Qwen
  numerical correctness, and import full-serving viewer rows.
