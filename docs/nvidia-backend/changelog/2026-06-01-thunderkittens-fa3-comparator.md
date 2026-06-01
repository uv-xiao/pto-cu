# 2026-06-01 ThunderKittens FA3 Comparator Capture

## Code And Data Changed

- Added `thunderkittens_mha_h100_fa3_comparator_h200` to
  `paper_baseline_execution_attempts.json`.
- Extended the `thunderkittens_full_sweep` run contract with the FA3 build,
  API probe, official benchmark, correctness, and shim artifacts.
- Narrowed the tensor-core paper-readiness blocker from missing FA3 bindings
  plus PyTorch OOM to PyTorch reference OOM after FA3 comparator capture.
- Updated the paper-ready baseline survey, evaluation plan, dispatch log, and
  review artifact tests to reflect the FA3 run.

## Architecture Quality

The FA3 integration remains an evaluation artifact, not a source patch to
ThunderKittens or FlashAttention. The H200 run used a `PYTHONPATH` shim under
`tmp/` to adapt the current FlashAttention-3 API to the tuple return shape that
the unmodified ThunderKittens benchmark expects.

## Evaluation Run

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/flash-attention/fa3-hopper-build-narrow-7371626c/
tmp/cuda-backend/paper-baselines/thunderkittens/upstream-benchmark-fa3-7371626c/
```

The narrowed FlashAttention-3 Hopper build exited `0`. The FA3 API probe
showed `return_attn_probs=True` returns `(out, lse)`. The official
ThunderKittens benchmark exited `0` and produced FA3 forward/backward rows for
causal and non-causal modes at sequence lengths 768, 1536, 3072, 6144, and
12288. The official correctness script also exited `0`.

## Remaining Gaps

The tensor-core claim is still not paper-ready. PyTorch reference rows OOM in
selected large official H100 MHA benchmark cells, so the next step is either a
fresh-process PyTorch reference capture for those cells or an explicit paper
policy that excludes infeasible dense PyTorch reference cells while preserving
FA2, FA3, TK, PTO, cuBLAS, CUTLASS, and Triton comparisons.

This remaining-gap statement is partially superseded by
[2026-06-01 ThunderKittens isolated PyTorch reference cells][isolated]:
fresh-process H200 runs recovered every selected 6144-token PyTorch
reference cell and left only selected 12288-token dense references OOM.

[isolated]: 2026-06-01-thunderkittens-isolated-pytorch-reference.md
