# 2026-05-31 ThunderKittens Selected Sweep

## Code And Data Changed

- Added `thunderkittens_full_sweep_capture.py`, a repo-owned wrapper around the
  pinned ThunderKittens H100 MHA extension.
- Imported two H200 `tensor_core_tile` / `thunderkittens` rows from
  `tmp/cuda-backend/paper-baselines/thunderkittens/full-sweep-4277aa73/`.
- Preserved FLOP-based `throughput` and `attention_flops` fields in the
  paper-baseline viewer exporter.
- Promoted the selected ThunderKittens sweep run record to
  `imported_to_viewer` and regenerated the audit, work queue, and goal
  progress data.

## Architecture Quality

The wrapper does not edit ThunderKittens upstream files under `tmp/baselines/`.
It imports the already-built extension, runs selected BF16 causal MHA shapes,
checks correctness against PyTorch scaled-dot-product attention, and writes the
same normalized raw JSON schema used by other paper baselines. The matrix-level
full-upstream-suite blocker remains, because this selected sweep is stronger
than the bounded smoke capture but is not the complete upstream benchmark
suite.

## Evaluation Run

H200 command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/thunderkittens_full_sweep_capture.py \
  --baseline-dir tmp/baselines/thunderkittens/kernels/attention/mha_h100 \
  --output-dir tmp/cuda-backend/paper-baselines/thunderkittens/full-sweep-4277aa73 \
  --machine <h200-host> --pto-commit 4277aa73 --cuda-toolkit 12.8 \
  --shape 1,1,768,64 --shape 1,4,1536,64 \
  --warmup 5 --repeats 20 --causal
```

Both rows passed correctness. The `b=1,h=1,n=768,d=64` row reported p50
device time `39423 ns`, max absolute error `0.001953125`, and throughput
`1915061563046` FLOP/s. The `b=1,h=4,n=1536,d=64` row reported p50 device
time `52159 ns`, max absolute error `0.00390625`, and throughput
`23159177744972` FLOP/s.

## Remaining Gaps

The paper-readiness work queue drops from 12 to 11 items. The tensor-core claim
still keeps the matrix-level blocker for the complete upstream ThunderKittens
correctness and benchmark suite, and the broader goal still needs MPK,
VDCores, vLLM, SGLang, and full PTO serving captures.
