# 2026-05-31 ThunderKittens Quick Smoke

## Code And Data Changed

Recorded the first H200 ThunderKittens raw capture for the selected
`kernels/attention/mha_h100` path. The benchmark viewer now includes a
`tensor_core_tile` / `thunderkittens` H200 result row imported from
`tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-67c5c655/quick-smoke.json`.
The ThunderKittens paper-baseline run status moved from `planned_not_run` to
`setup_ready`.

## Architecture Quality

The result row is generated through the existing paper-baseline importer
contract rather than a one-off viewer schema. The captured row is deliberately
marked as a single quick smoke, keeping it separate from the future full
correctness and benchmark sweeps needed for paper-ready claims.

## Evaluation Run

H200 setup installed the missing project-venv dependencies and verified CUDA:

```bash
ssh bizhaoh200 'cd /data/shibizhao/pto-cu && \
  .venv/bin/python -m pip install pybind11 tqdm && \
  .venv/bin/python -m pip install torch==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128'
```

The selected ThunderKittens extension then built with `sm_90a`, and a custom
quick smoke ran `tk.mha_forward` for
`b=1,h=1,n=768,d=64,causal=True` on H200. It compared the output with PyTorch
scaled-dot-product attention and passed with max absolute difference
`0.001953125`; the captured device event time was `14116607 ns`.

## Remaining Gaps

This is setup-ready evidence, not a paper-ready ThunderKittens baseline. Full
promotion still requires running the selected correctness and benchmark sweeps,
capturing repeat statistics, and deciding how the MHA workload maps to the PTO
tensor-core tile comparison.
