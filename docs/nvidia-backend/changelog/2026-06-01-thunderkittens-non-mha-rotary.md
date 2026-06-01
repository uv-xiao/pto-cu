# 2026-06-01 ThunderKittens Non-MHA Rotary Capture

## Code And Data Changed

- Added `thunderkittens_rotary_capture.py` as the repo-owned capture wrapper
  for the upstream ThunderKittens `kernels/rotary` extension.
- Added `rotary_flops` as an optional paper-baseline viewer metric so the raw
  capture preserves the rotary operator's work estimate.
- Added the `thunderkittens_non_mha_rotary` run contract, run-readiness row,
  execution attempt, and two imported viewer result rows.
- Narrowed the tensor-core paper-readiness blocker by removing the previous
  missing non-MHA ThunderKittens coverage item.
- Refreshed the paper-readiness audit, work queue, environment plans, and goal
  progress data from the updated viewer state.
- Added unit-test coverage for the rotary capture record builder, run contract,
  attempt metadata, readiness row, and imported viewer rows.

## Architecture Quality

The ThunderKittens tensor-core evidence now has a distinct non-MHA operator
path. MHA coverage remains represented by the bounded MHA capture, selected
full-sweep capture, and official upstream MHA logs. Rotary coverage is kept as
a separate run id so reviewers can distinguish attention-kernel evidence from
non-attention operator evidence instead of treating all ThunderKittens rows as
one undifferentiated baseline.

The remaining tensor-core blocker is now specific to the official upstream MHA
sweep environment. This report originally inherited the first-probe FA3
absence, which is now superseded by the FA3 comparator capture. The current
remaining blocker is further narrowed by the isolated PyTorch reference
capture: all selected 6144-token cells now pass, while 12288-token dense
reference cells still OOM.

## Evaluation Run

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/thunderkittens/non-mha-h200-rotary-ae922a2a/
```

The H200 run built `kernels/rotary` with CUDA 12.8 and SM90, installed the
missing `einops` dependency into the remote project venv, ran the upstream
`test_correctness.py`, and then captured two rotary shapes through the new
repo-owned wrapper:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/thunderkittens_rotary_capture.py \
    --baseline-dir tmp/baselines/thunderkittens/kernels/rotary \
    --output tmp/cuda-backend/paper-baselines/thunderkittens/non-mha-h200-rotary-ae922a2a/capture.json \
    --machine bizhaoh200 --pto-commit ae922a2a --cuda-toolkit 12.8 \
    --shape 2,16,1024,64 --shape 4,16,2048,64 --warmup 5 --repeats 20
```

Both imported rows passed correctness against the upstream torch rotary
formula and include p50 CUDA-event device time, sample count, max absolute
error, throughput, and `rotary_flops`.

## Remaining Gaps

- FA3 bindings are now covered by
  [2026-06-01 ThunderKittens FA3 comparator capture](2026-06-01-thunderkittens-fa3-comparator.md).
- PyTorch reference rows still OOM at selected 12288-token shapes in the
  official ThunderKittens MHA benchmark after isolated 6144-token reference
  cells passed.
