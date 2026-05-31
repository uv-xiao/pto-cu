# 2026-05-31 VDCores Extension Readiness

## Code And Data Changed

- Built the VDCores `dae.runtime` extension on the H200 host without modifying
  the VDCores source checkout.
- Refreshed `paper_baseline_run_readiness.json` and the generated paper
  readiness audit so VDCores run-readiness now records the compiled extension
  as present.
- Documented the H200 build command in the CUDA backend evaluation skill.

## Architecture Quality

The build keeps dependency workarounds outside upstream source. `CPATH` points
at the pinned CUTLASS checkout under `tmp/baselines/cutlass`, and `NVCC='nvcc
-include cfloat'` supplies the missing `FLT_MAX` declaration for this nvcc
path. The generated extension remains a `tmp/` artifact and is not committed.

## Evaluation Run

The H200 build used CUDA 12.8 and the project venv:

```bash
CPATH=$PWD/tmp/baselines/cutlass/include \
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH \
make -C tmp/baselines/vdcores clean pyext NVCC="nvcc -include cfloat"
```

The first local attempt proved two setup issues before the H200 build:
missing CUTLASS headers without `CPATH`, and missing `FLT_MAX` without
`-include cfloat`. Local extension installation then hit a PyTorch/CUDA
version mismatch, while the H200 venv matched CUDA 12.8 and completed the
build.

## Remaining Gaps

VDCores persistent and serving runs remain partial because `HF_TOKEN` is not
available for the selected gated model commands. The compiled-extension blocker
is cleared in the current readiness data, so the next required step is model
access or a documented non-gated bring-up path.
