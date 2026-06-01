# 2026-06-01 ThunderKittens Isolated PyTorch Reference Cells

## Code And Data Changed

- Added `thunderkittens_mha_h100_pt_reference_isolated_h200` to the paper
  baseline execution-attempt data.
- Extended the `thunderkittens_full_sweep` run contract with the isolated
  PyTorch reference artifacts under the raw artifact directory:

```text
tmp/cuda-backend/paper-baselines/thunderkittens/pt-reference-isolated-60d797cd/
```

- Refreshed the paper-readiness audit, work queue, run readiness, environment
  plans, and goal-progress data from the updated viewer state.
- Updated the tensor-core work queue so the remaining ThunderKittens
  official-sweep blocker is selected 12288-token dense PyTorch reference
  capacity, not selected 6144-token references.
- Added unit-test coverage for the isolated-reference execution attempt.

## Architecture Quality

This capture keeps the official ThunderKittens benchmark unpatched and records
the PyTorch reference cells as evaluation artifacts. Running each selected
reference cell in a fresh H200 process separates allocator fragmentation and
monolithic benchmark sequencing from true dense-reference memory capacity.

The result narrows the tensor-core blocker without changing the runtime design:
6144-token PyTorch references are now evidence-backed as feasible on H200, and
12288-token dense references remain an explicit paper-policy or capacity
question.

## Evaluation Run

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/thunderkittens/pt-reference-isolated-60d797cd/
```

The H200 run launched one Python process per selected official-reference cell:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python run_pt_reference_cell.py \
    --mode <fwd|bwd> --causal|--no-causal --seqlen <6144|12288>
```

Passing cells:

- `fwd_ctrue_n6144`
- `fwd_cfalse_n6144`
- `bwd_ctrue_n6144`
- `bwd_cfalse_n6144`

Failing cells:

- `fwd_ctrue_n12288`
- `fwd_cfalse_n12288`
- `bwd_ctrue_n12288`
- `bwd_cfalse_n12288`

## Remaining Gaps

- Selected 12288-token dense PyTorch reference cells still OOM on the H200
  capture host.
- [2026-06-01 ThunderKittens dense reference policy](2026-06-01-thunderkittens-reference-policy.md)
  accepts those cells only as OOM/not-applicable footnotes, so they no longer
  block the tensor-core tile claim.
