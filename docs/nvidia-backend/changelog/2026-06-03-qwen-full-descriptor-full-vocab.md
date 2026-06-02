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

The full-model reference path is now runnable locally from staged artifacts.
The temporary HF-style model directory links the staged safetensors and stores
small metadata files under `tmp/sources/qwen3-8b-local-hf-reference/`. Python
is run with user-site packages disabled so the local broken `torchvision`
package does not block importing `Qwen3ForCausalLM`.

```bash
PYTHONPATH=$PWD:$PWD/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout 1800 .venv/bin/python -s - <<'PY'
# Load tmp/sources/qwen3-8b-local-hf-reference with AutoModelForCausalLM,
# feed runtime input ids from runtime-input-binding.json, and compare the
# last active prompt-token logits against PTO output.
PY
```

Result: the HF full-model reference ran and produced
`tmp/cuda-backend/qwen-full-model-reference-mpk-1step-2026-06-03/reference.json`.
The comparison artifact at
`tmp/cuda-backend/qwen-full-model-reference-mpk-1step-2026-06-03/comparison.json`
currently fails. The reference top token at the last active prompt position is
`151667`, while both PTO real-resource rows select token `105397`. This keeps
`full_qwen_numerical_correctness` open and narrows the next implementation
work to kernel and launch-state fidelity rather than missing reference
infrastructure.

The runner now carries active prompt-token semantics separately from padded
serving-policy buffer shape. `qwen_runtime_input_binding.py` records both the
runtime prompt length and the active prompt length. The persistent decode args
use the last active prompt token as the first logits position, while
submission-plan output accounting keeps the padded serving-policy start
position. The generated `qwen_embedding_lookup` task reads
`input_ids[row * prompt_stride + decode_position]`, using `a_batch_stride` for
the padded prompt stride and `scalar_args[2]` for the current logits position.

```bash
ARTIFACT=tmp/cuda-backend/qwen-active-prompt-token-lookup-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
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

Result: the fresh A100 run passed scheduler execution with zero device-side
errors and used `first_decode_position=17`, `runtime_prompt_tokens=64`,
`active_prompt_tokens=18`, `a_batch_stride=64`, and RoPE
`decode_position=17`. This proves the first resource-backed step now consumes
the last active prompt token position instead of row 0 token 0. The top token
changed from the earlier real-resource `105397` to `116324`, but the compact
comparison artifact at
`tmp/cuda-backend/qwen-active-prompt-token-lookup-mpk-2026-06-03/comparison.json`
still fails against the HF full-model reference token `151667`. The remaining
gap is therefore true prompt-prefill/KV state and remaining task-math fidelity,
not padded prompt-position plumbing.

The bounded decode feedback path now writes sampled tokens into the prompt slot
that the next bounded step will read. Host-side feedback observation and the
generated `qwen_logits` device body both derive `next_input_index` from
`decode_position + 1`, guarded by the padded prompt stride. This replaces the
old diagnostic behavior that always wrote `input_ids[0]`.

```bash
ARTIFACT=tmp/cuda-backend/qwen-next-token-feedback-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
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

Result: the fresh A100 two-step real-resource run passed with 510 completed
persistent tasks and zero scheduler errors. Step 0 used RoPE position `17`,
sampled token `116324`, wrote `output_ids[0]`, and wrote the same token to
`input_ids[18]`. Step 1 used RoPE position `18`, sampled token `109379`,
wrote `output_ids[1]`, and wrote the same token to `input_ids[19]`. The compact
comparison artifact at
`tmp/cuda-backend/qwen-next-token-feedback-mpk-2026-06-03/comparison.json`
still fails against the HF reference first-step token `151667`, so full Qwen
numerical correctness remains open.

## Remaining Gaps

- Fix PTO kernel and launch-state fidelity so generated Qwen/Qwen3-8B
  tokens/logits match the full HF model reference.
- Run policy-length full-serving captures for both `mpk_offline_decode` and
  `vdcores_offline_decode`.
- Import only rows that satisfy `serving_coverage=full_serving` and
  `correctness_scope=full_qwen_numerical_correctness`.
