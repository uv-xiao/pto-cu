# 2026-06-01 VDCores Logits Schedule Introspection

## Code And Data Changed

- Added a VDCores execution-attempt record for the Qwen3-1.7B logits schedule
  introspection diagnostic.
- Refreshed the paper-readiness audit and work queue so the latest VDCores
  blocker points at the logits GEMV instruction window.
- Added focused review assertions for the new viewer data.

Raw artifacts remain under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-logits-schedule-introspection-e5d6786f/
```

## Architecture Quality

The VDCores blocker is now narrower than "logits stage fails." The build-only
probe monkey-patches `dae_app`, materializes instructions, and avoids
`launch_dae`. It shows the failing descriptor sequence is inside the direct
logits GEMV created by `GemvLayer`, not argmax or restore scheduling.

The descriptor map used by the viewer is:

- `desc32`: logits GEMV `loadA`, `matLogitsW[epoch]`
- `desc33`: logits GEMV `loadB`, `matRMSHidden`
- `desc34`: logits GEMV `storeC`, `matLogits[epoch]`

For SM64, the probe captured the logits-era PC window with 12 desc32 loads,
3 desc33 loads, and 2 desc34 stores. This matches the prior launch log's
`desc_idx=33/32/34` sequence and localizes the next diagnostic to VDCores
memory-slot allocation and `RepeatM` loop handling.

## Evaluation Run

Remote host: `bizhaoh200`, H200, `CUDA_VISIBLE_DEVICES=7`.

The remote checkout could not be cleanly refreshed because it has local
viewer-data modifications, so this diagnostic used the allowed tmp probe-copy
path. No upstream repository was edited or pushed.

Command shape:

```bash
cd tmp/baselines/vdcores
CUDA_VISIBLE_DEVICES=7 \
HF_HOME=<remote-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=6 \
VDC_LOGITS_SCHEDULE_OUTPUT=<artifact-root>/logits-schedule-summary.json \
<project-venv-python> <artifact-root>/vdcores_logits_schedule_probe.py
```

Result:

```text
logits-schedule-probe-status.txt: 0
logits_epoch=3
logits_slice=50688
vocab_size=151936
SM64 first logits PC=38
```

## Remaining Gaps

- VDCores still has no imported queue/resource-policy timing for paper
  comparison.
- The next VDCores diagnostic should inspect the runtime memory-slot
  allocator and `RepeatM` loop behavior for the direct desc32/33/34 logits
  GEMV sequence.
