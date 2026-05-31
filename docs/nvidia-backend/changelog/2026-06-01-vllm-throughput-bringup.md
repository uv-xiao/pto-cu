# 2026-06-01 vLLM Offline Throughput Bring-Up

## Code And Data Changed

- Captured the first local A100 vLLM offline-throughput run from the isolated
  vLLM environment instead of leaving the serving baseline at environment-only
  readiness.
- Recorded the run as a partial execution attempt in
  `paper_baseline_execution_attempts.json` without importing it into
  `results.json`, because it is not yet a complete serving result.
- Refreshed `paper_readiness_audit.json`, `paper_readiness_work_queue.json`,
  and `goal_progress.json` so the viewer shows the partial vLLM attempt as a
  current blocker and next action.
- Regenerated `paper_baseline_environment_plans.json` and
  `paper_baseline_run_readiness.json` at the current PTO commit so generated
  review artifacts remain byte-for-byte reproducible.

## Architecture Quality

The run keeps baseline mutation isolated: the pinned upstream checkout under
`tmp/baselines/vllm` remains unmodified, while the benchmark uses the copied
source overlay under `tmp/cuda-backend/paper-baselines/source-overlays/`.

The viewer data records this as an execution attempt rather than a result row.
That keeps the full `vllm_serving_and_throughput` contract strict: serving
latency metrics and H200 target-model evidence must be present before
`results.json` claims an imported vLLM serving baseline.

## Evaluation Run

The first attempt failed before model load because `HF_HOME` pointed one level
above the actual cache layout:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
HF_HOME=$PWD/tmp/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=$PWD/tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython:$PWD/tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython/python:$PWD/python:$PWD \
timeout 900 \
tmp/cuda-backend/paper-baselines/envs/vllm-27fa5aa3/bin/python -m vllm.entrypoints.cli.main bench throughput \
  --model Qwen/Qwen3-1.7B \
  --dataset-name random \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 1 \
  --num-warmups 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 256 \
  --output-json tmp/cuda-backend/paper-baselines/serving-runs/vllm/vdcores_offline_decode/vllm-throughput-qwen3-1p7b-batch1-bringup.json
```

The successful rerun set `HUGGINGFACE_HUB_CACHE` directly to the cache root:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
HF_HOME=$PWD/tmp/hf-home-vllm HUGGINGFACE_HUB_CACHE=$PWD/tmp/huggingface_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=$PWD/tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython:$PWD/tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython/python:$PWD/python:$PWD \
timeout 900 \
tmp/cuda-backend/paper-baselines/envs/vllm-27fa5aa3/bin/python -m vllm.entrypoints.cli.main bench throughput \
  --model Qwen/Qwen3-1.7B \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 1 \
  --num-warmups 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 256 \
  --output-json tmp/cuda-backend/paper-baselines/serving-runs/vllm/vdcores_offline_decode/vllm-throughput-qwen3-1p7b-batch1-bringup.json
```

## Result

- Model: `Qwen/Qwen3-1.7B`.
- Hardware: local A100, one visible GPU.
- Prompt/decode shape: `128` prompt tokens and `64` output tokens.
- Requests: `1` measured request after `1` warmup request.
- Raw throughput JSON:
  `tmp/cuda-backend/paper-baselines/serving-runs/vllm/vdcores_offline_decode/vllm-throughput-qwen3-1p7b-batch1-bringup.json`.
- Measured elapsed time: `0.2419096989906393s`.
- Throughput: `4.1337739006433765` requests/s and
  `793.6845889235282` total tokens/s.

## Review Status

This remains a partial baseline attempt. It proves that the isolated vLLM
environment can load the cached bring-up model and execute the offline
throughput path, but it does not satisfy the full `vllm_serving_and_throughput`
contract because it lacks H200 evidence, Qwen3-8B target-model coverage, and
serving latency metrics such as TTFT and ITL.

## Remaining Gaps

- Run `vllm serve` plus `vllm bench serve` so TTFT, ITL, end-to-end latency,
  and throughput are captured together.
- Repeat on H200 and promote from `Qwen/Qwen3-1.7B` bring-up to the agreed
  Qwen3-8B paper target before importing the baseline into `results.json`.
- Refresh paired run-readiness after the isolated environment exists on H200;
  the existing readiness row still reflects older source-entrypoint gaps.
