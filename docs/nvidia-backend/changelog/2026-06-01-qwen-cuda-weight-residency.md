# 2026-06-01 Qwen CUDA Weight Residency

## Code And Data Changed

- Extended `examples/cuda/qwen_cuda_weight_binding.py` with
  `--cuda-probe-mode full`.
- The full probe copies safetensors tensors to CUDA in bounded host chunks,
  keeps all device allocations live until every Qwen/Qwen3-8B tensor is
  resident, verifies selected small tensors by copy-back, then frees all
  allocations.
- Updated CUDA example docs, the CUDA evaluation skill, benchmark-viewer
  evidence references, in-progress paper-readiness docs, dispatch log, and
  review-artifact tests.

## Architecture Quality

The CUDA weight path now has separate evidence for binding-slot planning,
bounded copy smoke, and full device residency. The full probe still avoids
claiming production model-loader completion: it proves allocation and copy
capacity through the runtime C API, while persistent-device task-argument
pointer binding and kernel consumption remain explicit runtime work.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_weight_binding.py \
  --cuda-probe-mode full \
  --device 4 \
  --verify-tensors 16 \
  --output-json \
  tmp/cuda-backend/pto-serving-weight-residency-1ae913c9/qwen-cuda-weight-residency.json
```

Result: `cuda_probe.status=pass`, `resident_tensor_count=399`,
`resident_bytes=16381470720`, `freed_tensor_count=399`, and
`verified_tensor_count=16`.

## Remaining Gaps

- Bind resident weight pointers into persistent-device task arguments.
- Generate Qwen kernels that consume the bound weights.
- Bind runtime token IDs, allocate and bind the KV cache, execute the decode
  loop, and import full-serving viewer rows for `Qwen/Qwen3-8B`.
