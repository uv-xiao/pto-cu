# 2026-06-01 PTO Qwen Serving Scaffold

## Code And Data Changed

- Added `examples/cuda/persistent_qwen_serving_scaffold.py`, an executable
  PTO-owned lifecycle scaffold for persistent-device Qwen/Qwen3-8B serving.
- Added the scaffold to `examples/cuda/manifest.json` and `examples/cuda/README.md`.
- Updated `pto_serving_preflight.py` so the preflight embeds the scaffold
  stage state and reports missing stage IDs.
- Refreshed paper-readiness data so the LLM-serving PTO work item points at
  the scaffold and latest preflight artifacts.

## Architecture Quality

The top PTO full-serving blocker now has an explicit repo-owned stage model:
serving workload contract, persistent-device task ABI, persistent DAG codegen,
Qwen tokenizer, Qwen weight loader, KV-cache lifecycle, decode-loop runner, and
viewer result import. The first three stages currently pass from existing code
or viewer data; the remaining stages stay marked `missing`.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_qwen_serving_scaffold.py \
    --output-json tmp/cuda-backend/pto-serving-scaffold-76d4fca4/qwen-serving-scaffold.json
```

Result: `status=partial`, with missing stage IDs for tokenizer, weight loader,
KV-cache lifecycle, decode-loop runner, and viewer result import.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-76d4fca4/pto-serving-preflight.json
```

Result: `status=partial`, embedding the same lifecycle scaffold.

## Remaining Gaps

- Implement the missing PTO serving host/runtime stages.
- Import persistent-device Qwen/Qwen3-8B full-serving rows for
  `mpk_offline_decode` and `vdcores_offline_decode`.
