# 2026-06-01 VDCores Qwen3 8B Preflight

## Code And Data Changed

- Retargeted the VDCores serving-baseline run contract from the older
  Llama demo path to `app/python/qwen3/sched.py`, whose source declares
  `MODEL_NAME = "Qwen/Qwen3-8B"`.
- Updated the VDCores serving command plan to use the Qwen3 path, offline
  Hugging Face cache, and explicit `HF_TOKEN=` environment key required by
  the source.
- Added a H200 execution-attempt record for
  `vdcores_qwen3_8b_decode_preflight`.
- Refreshed run-readiness, paper-readiness audit, work queue, and goal
  progress data.

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-preflight-b6df049f/
```

## Architecture Quality

The LLM-serving matrix now points at the actual Qwen3-8B VDCores source path
instead of treating the Llama demo as the paper-target row. The run-readiness
record distinguishes public model access from the source-level requirement
that `HF_TOKEN` exist in the environment.

The new blocker is also more precise: Qwen3-8B model load succeeds from the
offline H200 cache, but the compiled `dae.runtime` does not include the
Qwen3-8B compute-operator set needed by the launcher.

## Evaluation Run

The bounded H200 preflight ran:

```bash
cd tmp/baselines/vdcores
HF_TOKEN= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python app/python/qwen3/sched.py \
  --hf-cache-dir <shared-hf-cache>/hub \
  --correctness
```

The run loaded all five `Qwen/Qwen3-8B` checkpoint shards and reached
`dae.launch()`. It then failed before correctness or serving timing because
the launcher reported missing compiled operators, including:

```text
OP_RMS_NORM_F16_K_4096_SMEM
OP_SILU_MUL_SHARED_BF16_K_64_SW128
```

The normalized attempt summary is:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-preflight-b6df049f/qwen3-8b-preflight-summary.json
```

## Remaining Gaps

- Rebuild `dae.runtime` with the full Qwen3-8B `DAE_COMPUTE_OPS` set reported
  by the launcher.
- Re-run VDCores Qwen3-8B correctness after the operator rebuild.
- Add a paper-serving harness that records the actual scheduled request count,
  batch policy, latency, throughput, and raw outputs before importing the
  final VDCores serving row.
