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

## Remaining Gaps

- Replace mock resource values with full Qwen/Qwen3-8B numerical correctness
  against a model reference.
- Run policy-length full-serving captures for both `mpk_offline_decode` and
  `vdcores_offline_decode`.
- Import only rows that satisfy `serving_coverage=full_serving` and
  `correctness_scope=full_qwen_numerical_correctness`.
