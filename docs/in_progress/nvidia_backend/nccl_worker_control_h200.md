# NCCL Worker-Control H200 Evidence

This note records the first H200 evidence for CUDA host-runtime NCCL
operations through the descriptor-backed runtime boundary and
`CTRL_COMM_OP` worker transport.

## Command

The run used tree sync to a temporary remote checkout and rebuilt the editable
package before execution:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-cuda-comm-nccl-worker-control \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    pip install --no-build-isolation -e . >/tmp/pto-cu-pip-install.log && \
    NCCL_DEBUG=WARN .venv/bin/python \
      examples/cuda/nccl_worker_control_ops.py \
      --device-ids 6,7 --tensor-numel 1024 --build --require-cuda'
```

The remote checkout needed one-time venv provisioning before this exact command
could run: `.venv` was absent after tree sync, then Python 3.12 needed
`scikit-build-core`, `nanobind`, `cmake`, `ninja`, and `nvidia-nccl-cu12`
installed into the project-local venv.

## Environment

- Machine class: NVIDIA H200 host.
- Devices: `NVIDIA H200 NVL` ids `6,7`.
- Driver reported by `nvidia-smi`: `580.126.20`.
- Runtime transport: `worker_control`.
- Runtime boundary: CUDA chip children receive compact
  `CudaCommDeviceDescriptor` bytes and dispatch float32 operations through
  `CTRL_COMM_OP`.
- NCCL library: discovered from the venv package path
  `.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2` and passed
  to the C++ runtime via `PTO_CUDA_NCCL_LIBRARY`.

## Result

The command exited with status `0`.

```text
status: passed
backend: nccl
world_size: 2
device_ids: [6, 7]
tensor_numel: 1024
elapsed_s: 16.6029314994812
capability_id: nccl:rank0->cuda6,rank1->cuda7
operations: [all_reduce, reduce_scatter, all_gather, send_recv]
```

All operation checks reported `passed: true` with `max_abs_error: 0.0`:

- `all_reduce`
- `reduce_scatter`
- `all_gather`
- `send_recv`

## Diagnostic Note

The first H200 attempt failed before communicator initialization because
`dlopen("libnccl.so.2")` could not see NCCL in the dynamic linker path even
though PyTorch had a bundled NCCL wheel. The runtime now exposes
`comm_last_error()` for chip-worker diagnostics, accepts
`PTO_CUDA_NCCL_LIBRARY` as an explicit NCCL path, and the example discovers
the standard `nvidia/nccl/lib/libnccl.so.2` wheel location before workers fork.
