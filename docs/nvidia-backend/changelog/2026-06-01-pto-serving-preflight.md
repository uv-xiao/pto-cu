# 2026-06-01 PTO Serving Preflight

## Code And Data Changed

- Added `.agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py`
  to capture the current PTO persistent-device full-serving readiness without
  importing a performance result.
- Added the raw preflight artifact reference to
  `paper_evaluation_matrix.json` and regenerated the derived review data.
- Tightened the LLM-serving missing-evidence action so the PTO blocker names
  Qwen model loading, tokenization, KV-cache management, and decode-loop
  execution.

## Architecture Quality

The preflight separates the existing persistent-device runtime capability from
the missing full-serving application layer. It records that the current ABI has
DAG task descriptors plus generic tensor/scalar slots, and that source codegen
can emit persistent DAG task bodies. It also records that those capabilities
are not yet a repo-owned Qwen/Qwen3-8B serving path.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-26c38df3/pto-serving-preflight.json
```

Result: `status=partial`. The artifact proves the viewer already has a PTO
controlled serving-equivalent attention-tile proxy row, but it has no PTO
Qwen/Qwen3-8B full-serving row for `mpk_offline_decode` or
`vdcores_offline_decode`.

## Remaining Gaps

- Implement or import the PTO persistent-device Qwen/Qwen3-8B model-loading,
  tokenization, KV-cache, and decode-loop path.
- Import PTO full-serving rows for both shared serving workload policies before
  promoting the LLM-serving paper-baseline claim.
