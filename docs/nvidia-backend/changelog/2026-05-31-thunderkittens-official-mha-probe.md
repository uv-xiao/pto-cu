# 2026-05-31 ThunderKittens Official MHA Probe

## Code And Data Changed

- Added an H200 execution attempt for the official ThunderKittens
  `kernels/attention/mha_h100` benchmark and correctness scripts.
- Added the official-probe raw artifacts to the
  `thunderkittens_full_sweep` expected artifact contract so reviewers can see
  both the repo-owned JSON capture and the upstream harness logs.
- Narrowed the tensor-core paper-readiness blocker from a generic missing
  ThunderKittens sweep to the remaining observed gaps: missing FlashAttention
  3 bindings, PyTorch reference OOM at selected 12288-token shapes, and
  missing non-MHA ThunderKittens kernel coverage.
- Refreshed the paper-readiness audit, work queue, and goal-progress data from
  the updated matrix and run contract.

## Architecture Quality

The branch now separates three ThunderKittens evidence layers:

- bounded repo-owned JSON capture imported into viewer result records;
- selected full-sweep JSON capture for comparable tensor-core rows;
- official upstream `benchmark.py` and `test_correctness.py` logs for the H100
  MHA extension.

This prevents the viewer from overclaiming. The official upstream scripts ran,
but the matrix remains blocked on the concrete comparator and coverage gaps
that the run exposed.

## Evaluation Run

On the H200 host, baseline push URLs under `tmp/baselines/` were disabled
before running the probe. The first build attempt intentionally captured the
system-Python failure mode. The successful build used the project venv:

```bash
cd tmp/baselines/thunderkittens/kernels/attention/mha_h100
PATH=$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH make
```

The official benchmark and correctness scripts then ran with the compiled
extension on `PYTHONPATH`:

```bash
python tmp/baselines/thunderkittens/kernels/attention/mha_h100/benchmark.py
python tmp/baselines/thunderkittens/kernels/attention/mha_h100/test_correctness.py
```

The benchmark completed forward/backward causal and non-causal tables. The
ThunderKittens rows completed for sequence lengths 768, 1536, 3072, 6144, and
12288. FlashAttention 3 rows failed in this first probe because
`flash_attn_interface` was not available. PyTorch reference rows OOMed at
selected largest shapes. The
correctness script completed output, backward, and all-mode error graph
generation.

## Remaining Gaps

- Superseded by
  [2026-06-01 ThunderKittens FA3 comparator capture](2026-06-01-thunderkittens-fa3-comparator.md):
  FA3 bindings were built and the FA3 comparator rows now run on H200.
- PyTorch reference rows still OOM at selected 12288-token shapes in the
  official benchmark script.
- ThunderKittens kernels outside `attention/mha_h100` are not yet captured for
  the tensor-core paper claim.
