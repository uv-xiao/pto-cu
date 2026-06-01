# 2026-06-01 Qwen Serving Lifecycle Plan

## Code And Data Changed

- Added `examples/cuda/qwen_serving_lifecycle_plan.py`, an executable
  PTO-owned lifecycle-plan artifact for persistent-device Qwen/Qwen3-8B
  serving.
- Extended `persistent_qwen_serving_scaffold.py` so it embeds the lifecycle
  plan and marks the KV-cache lifecycle as `partial` instead of undocumented.
- Updated `pto_serving_preflight.py` so PTO serving readiness checks include
  the lifecycle-plan contract.
- Added the lifecycle plan to `examples/cuda/manifest.json` and
  `examples/cuda/README.md`.
- Refreshed paper-readiness data so the LLM-serving PTO work item points at
  the lifecycle-plan, scaffold, and preflight artifacts.

## Architecture Quality

The PTO full-serving blocker now has a reviewable bridge between the shared
serving policies and future CUDA runtime code. The lifecycle plan records the
Qwen3-8B model shape, MPK and VDCores serving policy batch ladders, KV-cache
capacity formula, token-position lifecycle, weight-binding categories, and the
planned persistent-device callable roles.

This remains a partial runtime plan, not a full-serving implementation. The
remaining runtime gaps are tokenizer integration, safetensors weight loading,
CUDA allocation and task-argument binding, generated Qwen kernel bodies,
decode-loop execution, and viewer-result import.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_serving_lifecycle_plan.py \
    --output-json tmp/cuda-backend/pto-serving-lifecycle-e3c977f8/qwen-serving-lifecycle-plan.json
```

Result: `status=partial_runtime_plan`, with implemented contracts for
`qwen3_8b_model_shape`, `serving_policy_to_kv_cache_plan`, and
`persistent_device_task_mapping`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_qwen_serving_scaffold.py \
    --output-json tmp/cuda-backend/pto-serving-scaffold-e3c977f8/qwen-serving-scaffold.json
```

Result: `status=partial`; the `kv_cache_lifecycle` stage is now `partial`
with evidence in `examples/cuda/qwen_serving_lifecycle_plan.py`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-e3c977f8/pto-serving-preflight.json
```

Result: `status=partial`, with `qwen_serving_lifecycle_plan=pass` and full
Qwen/Qwen3-8B serving rows still absent.

## Remaining Gaps

- Bind the planned KV-cache layout to real CUDA allocations and
  persistent-device task arguments.
- Implement tokenizer, safetensors loading, generated Qwen kernel bodies, and
  decode-loop execution.
- Import persistent-device Qwen/Qwen3-8B full-serving rows for
  `mpk_offline_decode` and `vdcores_offline_decode`.
