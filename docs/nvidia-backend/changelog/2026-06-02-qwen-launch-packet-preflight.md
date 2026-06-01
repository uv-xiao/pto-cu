# 2026-06-02 Qwen Launch-Packet Preflight

## Code And Data Changed

- Split Qwen decode-loop diagnostic bridge contracts out of the runner
  lifecycle module to keep review files short.
- Added `launch_packet_preflight` under each resource-backed graph workload.
- Updated example evidence symbols, manifest metadata, README text, and the
  LLM-serving paper evaluation matrix with the new raw artifact.

## Architecture Quality

The Qwen runner now separates three states that were previously easy to
confuse: descriptor materialization, host-side launch-packet packing, and
actual `run_prepared` execution. The preflight packs a host
`CudaPersistentDagTask` array from live token, KV-cache, and resident-weight
pointers, then explicitly refuses to promote that state to a runnable full
decode loop until intermediate activation buffers, a float logits or sampling
output path, and final Qwen kernels exist.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-submission-smoke --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --device 0 --arch compute_80 \
  --output-json \
  tmp/cuda-backend/pto-serving-launch-packet-preflight-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: both `mpk_offline_decode` and `vdcores_offline_decode` report
`resource_backed_launch_packet_preflight_ready`, 255 task records,
46,920 host task-packet bytes, queue capacity 256, and the expected
non-launch blockers.

## Remaining Gaps

- Allocate and bind intermediate activation buffers plus a float logits or
  sampling output path.
- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Execute the resource-backed graph through `run_prepared`.
