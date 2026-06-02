# 2026-06-03 Qwen Full Descriptor Full Vocab

## Code And Data Changed

- Added live evaluation evidence for the existing resource-backed Qwen runner
  using the full materialized descriptor chain instead of the first-layer
  diagnostic selector.
- Kept the raw run artifact under
  `tmp/cuda-backend/qwen-full-descriptor-full-vocab-1step-2026-06-03/` instead
  of committing another benchmark-viewer data shard.

## Architecture Quality

The run exercises the CUDA persistent-device scheduler across all 255 Qwen
task descriptors generated for the 36-layer Qwen3-8B decode graph:
embedding, 36 repeated layer blocks, final norm, and logits. This proves the
current descriptor materialization, launch-packet construction, device-side
task dispatch, dynamic RoPE refresh, full RMSNorm reduction mode, weighted
elementwise branches, full-vocab logits, and device token-feedback path can
execute as one ordered persistent-device DAG.

The result is still diagnostic, not full-serving evidence. It uses mock
resource values and a one-step MPK-policy decode, so it does not satisfy the
full Qwen numerical correctness or full-serving row-import gates.

## Evaluation Run

The first A100 run covered a one-step MPK-policy decode:

```bash
ARTIFACT=tmp/cuda-backend/qwen-full-descriptor-full-vocab-1step-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 600 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection prefix \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
  --output-json $ARTIFACT/qwen-decode-loop-runner.json
```

Result: `resource_backed_execution.status=pass`. The run completed all 255
persistent tasks with zero scheduler errors. The final logits task wrote and
checked the full 2,430,976-element logits buffer, covering 16 rows of the
151,936-column vocabulary. Diagnostic reference checking passed across 3,904
sampled elements with `max_abs_error=1.236e-05` under tolerance `2e-05`.

Device token feedback was observed for the generated token path:
`sampled_token_id=64036`, with policy
`device_commits_diagnostic_sampled_token_for_next_step`.

A second A100 run used the same full descriptor and full-vocab settings for
two MPK-policy decode steps:

```bash
ARTIFACT=tmp/cuda-backend/qwen-full-descriptor-full-vocab-2step-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 900 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection prefix \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 2 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy every_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
  --output-json $ARTIFACT/qwen-decode-loop-runner.json
```

Result: both decode steps passed. The run completed 510 total persistent tasks
with zero scheduler errors, checked the full logits buffer on both steps, and
observed device-committed feedback for sampled tokens `64036` then `107152`.
Each step checked 3,904 reference elements across 16 logits rows with
`max_abs_error=1.236e-05`.

A third A100 run covered the VDCores-policy workload with the same full
descriptor and full-vocab settings:

```bash
ARTIFACT=tmp/cuda-backend/qwen-full-descriptor-full-vocab-vdcores-1step-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 900 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection prefix \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
  --output-json $ARTIFACT/qwen-decode-loop-runner.json
```

Result: `vdcores_offline_decode` passed for one decode step. The run completed
255 persistent tasks with zero scheduler errors, checked the full logits
buffer, and observed device-committed token feedback for sampled token `64036`.
Reference checking again covered 3,904 elements across 16 logits rows with
`max_abs_error=1.236e-05`.

The next A100 runs moved from mock resources to the offline Qwen/Qwen3-8B
tokenizer and safetensors already staged under `tmp/`. Both runs kept all four
owners in one CUDA context: token pointer table, KV cache, resident weight
table, and activation workspace.

```bash
ARTIFACT=tmp/cuda-backend/qwen-real-resource-full-vocab-1step-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 1500 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection prefix \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
  --output-json $ARTIFACT/qwen-decode-loop-runner.json
```

Result: `mpk_offline_decode` passed for one decode step with real tokenizer,
real resident safetensors, and full-vocab logits. The run completed 255
persistent tasks with zero scheduler errors, checked the full 2,430,976-element
logits buffer, and observed device-committed token feedback for sampled token
`105397`. Reference checking covered 3,904 elements across 16 logits rows with
`max_abs_error=1.096e-05`.

```bash
ARTIFACT=tmp/cuda-backend/qwen-real-resource-full-vocab-1step-vdcores-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 1500 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection prefix \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
  --output-json $ARTIFACT/qwen-decode-loop-runner.json
```

Result: `vdcores_offline_decode` also passed for one decode step with the same
real-resource and full-vocab settings. It completed 255 persistent tasks with
zero scheduler errors, observed token feedback for sampled token `105397`, and
matched the same 3,904-element diagnostic reference with
`max_abs_error=1.096e-05`.

## Remaining Gaps

- Compare generated Qwen/Qwen3-8B tokens/logits against a full model reference
  instead of the current in-run diagnostic projection reference.
- Run policy-length full-serving captures for both `mpk_offline_decode` and
  `vdcores_offline_decode`.
- Import only rows that satisfy `serving_coverage=full_serving` and
  `correctness_scope=full_qwen_numerical_correctness`.
