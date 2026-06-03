# 2026-06-03 Paper Queue Qwen HF Mismatch

## Code And Data Changed

- Updated `paper_evaluation_matrix.json` so the PTO full-serving missing
  evidence item records the latest full-prefix MPK evidence and the current
  Hugging Face token mismatch.
- Regenerated `paper_readiness_audit.json` and
  `paper_readiness_work_queue.json` from the source matrix.
- Added raw artifact references for:
  `tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-full-prefix-after-mlp-residual-fix-420s.json`
  and
  `tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-full-prefix-hf-token-comparison.json`.

## Architecture Quality

The paper queue now carries the same boundary as the Qwen status page: the
full-prefix MPK finite-logits blocker is closed, but PTO full-serving rows
remain blocked by token/logit disagreement with the Hugging Face reference.
That keeps the viewer-backed review queue aligned with the strict import gate
instead of leaving reviewers to infer the active blocker from prose-only
status docs.

## Evaluation Run

The generated queue now states that the comparison is diagnostic rather than
model-equivalent: prompt prefill was not executed, PTO selected token `220`,
and the Hugging Face reference selected token `151667` at decode position 17.
It also records the finite full-prefix evidence: 255/255 tasks completed,
zero scheduler errors, no row-0 non-finite activations, full finite logits,
populated top-k, device feedback for token `220`, and a passing diagnostic
reference.

Regeneration command:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
```

## Remaining Gaps

PTO persistent-device Qwen3-8B paper rows still require Hugging Face token and
logit agreement before importing MPK and VDCores full-serving rows with
latency, throughput, and full-Qwen correctness details.
