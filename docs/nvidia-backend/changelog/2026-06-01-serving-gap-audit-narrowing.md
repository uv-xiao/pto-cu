# 2026-06-01 Serving Gap Audit Narrowing

## Code And Data Changed

- Added explicit LLM-serving matrix evidence refs for the imported H200 vLLM
  and SGLang result rows.
- Narrowed the LLM-serving missing-evidence text so it no longer lists vLLM
  and SGLang as absent after their repeated H200 rows were imported.
- Added shape-aware `viewer_result` evidence matching so the audit can
  distinguish vLLM's MPK-policy rows from SGLang's VDCores-policy rows.
- Updated goal-progress wording to describe the remaining queued serving gaps:
  MPK persistent, VDCores full serving, SGLang MPK-policy,
  ThunderKittens-family full serving, and PTO full serving.
- Regenerated the paper-readiness audit, work queue, run readiness, environment
  plans, and goal progress from the updated matrix.

## Architecture Quality

The paper-readiness queue now matches the committed evidence surface. vLLM and
SGLang remain part of the final paper comparison, but the blocker text no
longer asks reviewers to import rows that already exist in `results.json`.
The remaining LLM-serving blocker now separates true full-serving gaps from
proxy or bring-up evidence, and it does not treat a method-level result as
coverage for every serving policy.

## Evaluation Run

The H200 vLLM repeated rows are already imported under:

```text
tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-qwen3-8b-repeats-eb75a235/
```

The H200 SGLang repeated rows are already imported under:

```text
tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-repeats-eb75a235/
```

This slice changed review data only; it did not run new GPU benchmarks.

## Remaining Gaps

- MPK persistent-kernel serving rows are still not imported for the Qwen3-8B
  paper workload.
- VDCores full serving rows remain blocked by the gated Llama path or a
  documented public-model replacement.
- SGLang MPK-policy rows remain absent; current SGLang rows cover the
  VDCores-shaped `128/64` policy.
- ThunderKittens-family and PTO rows remain proxy or partial serving-equivalent
  evidence rather than full Qwen3-8B serving rows.
