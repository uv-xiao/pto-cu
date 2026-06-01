# 2026-06-01 ThunderKittens Dense Reference Policy

## Code And Data Changed

- Promoted `tensor_core_tile_baselines` to `ready_for_paper_claim`.
- Added `evidence_policy_exceptions` to the paper-evaluation matrix and
  generated paper-readiness audit.
- Added the accepted
  `thunderkittens_dense_pytorch_12288_oom_policy` exception for the official
  ThunderKittens H100 MHA dense PyTorch reference cells at sequence length
  12288.
- Updated the HTML benchmark viewer so reviewers can see accepted evidence
  exceptions beside current evidence, blockers, and next actions.
- Tightened `validate_benchmark_viewer_data.py` so accepted exceptions must
  carry scope, decision, rationale, review rule, and concrete artifact links.

## Architecture Quality

The remaining ThunderKittens gap is no longer hidden as an implicit judgment.
It is now a first-class evidence policy in the same JSON contract that drives
the viewer and audit. The exception is intentionally narrow: it applies only to
the dense PyTorch reference oracle rows that still OOM at sequence length
12288 on H200 after isolated fresh-process capture.

Measured FA3, ThunderKittens, PTO, cuBLAS/CUTLASS, and Triton rows remain
normal evidence. The policy only prevents an infeasible reference oracle from
blocking the tensor-core tile claim or from being misreported as a measured
timing row.

## Evaluation Run

No new H200 command was required for this policy slice. It relies on the
previous isolated-reference artifacts:

```text
tmp/cuda-backend/paper-baselines/thunderkittens/pt-reference-isolated-60d797cd/
tmp/cuda-backend/paper-baselines/thunderkittens/upstream-benchmark-fa3-7371626c/
```

Those artifacts show all selected 6144-token dense PyTorch reference cells
passed in fresh processes, all selected 12288-token dense PyTorch reference
cells still OOMed, and FA3 comparator rows completed through sequence length
12288.

## Remaining Gaps

- LLM-serving remains not paper-ready. The active work queue now contains PTO
  full-serving Qwen3-8B rows, VDCores Qwen3-8B full serving after correctness
  repair, ThunderKittens-family full-serving rows, and the latest VDCores
  execution-attempt blocker.
- Paper tables must report the accepted 12288-token dense PyTorch reference
  cells only as OOM/not-applicable footnotes.
