# 2026-05-31 ThunderKittens Bounded Capture

## Code And Data Changed

Added `.agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py`
to capture bounded ThunderKittens MHA correctness and latency evidence on the
selected `kernels/attention/mha_h100` path. The script emits the existing
paper-baseline raw JSON contract consumed by
`paper_baseline_viewer_export.py`.

The benchmark viewer now includes two imported H200 `tensor_core_tile` /
`thunderkittens` records from
`tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/capture.json`.
The ThunderKittens paper-baseline run status moved from `setup_ready` to
`imported_to_viewer`.

## Architecture Quality

The capture is intentionally a separate repo-owned wrapper instead of editing
the cloned ThunderKittens source under `tmp/baselines/`. That keeps upstream
source immutable, records the exact benchmark shape and repeat policy in raw
JSON, and uses the same importer path as other paper baselines.

The viewer row still names this as paper-baseline capture evidence, not a
paper-ready final result. The matrix now distinguishes imported bounded MHA
evidence from the missing full upstream correctness and benchmark sweeps.

## Evaluation Run

The H200 command used the existing ThunderKittens build, CUDA 12.8, and the
remote project venv:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py \
  --baseline-dir tmp/baselines/thunderkittens/kernels/attention/mha_h100 \
  --output tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/capture.json \
  --machine <h200-host> --pto-commit 5915346e --cuda-toolkit 12.8 \
  --shape 1,1,768,64 --shape 1,4,1536,64 \
  --warmup 5 --repeats 20 --causal
```

Both BF16 causal MHA shapes passed correctness against PyTorch
scaled-dot-product attention. The `b=1,h=1,n=768,d=64` row captured twenty
timed CUDA-event samples with p50 `36864 ns`, p90 `38752 ns`, p99 `42784 ns`,
and max absolute difference `0.001953125`. The
`b=1,h=4,n=1536,d=64` row captured twenty timed samples with p50 `49279 ns`,
p90 `50303 ns`, p99 `153152 ns`, and max absolute difference `0.00390625`.

The raw output was converted with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py \
  tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/capture.json \
  --artifact-root tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/ \
  --output \
  tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/viewer-result-records.json
```

## Remaining Gaps

This is imported bounded evidence, not a complete ThunderKittens paper
baseline. Full promotion still requires the upstream correctness and benchmark
sweeps, repeat statistics for the selected paper shapes, and an explicit shape
mapping between ThunderKittens MHA and PTO tensor-core workloads.
