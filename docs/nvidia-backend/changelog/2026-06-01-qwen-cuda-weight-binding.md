# 2026-06-01 Qwen CUDA Weight Binding

## Code And Data Changed

- Added `examples/cuda/qwen_cuda_weight_binding.py`, which turns validated
  Qwen/Qwen3-8B safetensors metadata into stable CUDA binding slots, file byte
  ranges, binding groups, and readonly persistent-device argument roles.
- Added an optional bounded CUDA copy probe. On the local A100 capture it used
  the existing CUDA host runtime C API to allocate device memory, copy 16 small
  norm tensors, and free the allocations.
- Wired the binding artifact into the Qwen serving scaffold, PTO serving
  preflight, CUDA examples manifest, example README, benchmark-viewer matrix,
  and paper-readiness docs.

## Architecture Quality

The PTO Qwen serving path now separates three different weight-loader states:
shape/dtype inventory, safetensors data-offset binding, and full runtime
device residency. The binding artifact is reviewable without CUDA through
`--no-cuda-probe`, but can also exercise the existing CUDA runtime C API when
hardware is available.

The scaffold and preflight now mark CUDA weight binding-plan evidence as
passing while keeping `qwen_weight_loader` partial until resident device
pointers are connected to persistent-device task arguments and consumed by
generated Qwen kernels.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_weight_binding.py \
  --output-json tmp/cuda-backend/pto-serving-weight-binding-35f713e9/qwen-cuda-weight-binding.json
```

Result: `status=binding_plan_ready`, `planned_binding_count=399`,
`total_weight_bytes=16381470720`, and `cuda_probe.status=pass` with 16 copied
small tensors.

## Remaining Gaps

This is not a full Qwen model loader yet. The runtime still needs full CUDA
weight residency, persistent-device task-argument pointer binding, Qwen kernel
weight consumption, token-ID binding, KV-cache allocation/binding, decode-loop
execution, and viewer-result import before PTO `Qwen/Qwen3-8B` full-serving
rows can be claimed.
