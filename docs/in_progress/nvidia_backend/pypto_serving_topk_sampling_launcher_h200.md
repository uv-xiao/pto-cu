# pypto-serving Top-K Sampling Launcher H200 Evidence

This note records a narrow pypto-serving/simpler-nv serving-route
launcher/probe for a generated Top-K sampling correctness gate. It connects
the synthetic pypto-serving source route to the existing generated Gluon
correctness harness, not to a production sampler.

## Contract

- serving-route launcher/probe for a generated Top-K sampling correctness gate.

The launcher is implemented in `examples/cuda/pypto_serving_nv_shim.py` as
`create_generated_gluon_topk_sampling_launcher(...)`. It calls
`examples/cuda/gluon_topk_sampling.py::run_topk_sampling_correctness(...)`.

The CLI selection is:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
    --pypto-serving-source --kernel-launcher gluon-topk-sampling \
    --prompt hello --max-new-tokens 1 --require-cuda
```

The selected fixture uses the existing deterministic Top-K shape:

```text
kernel_name: topk_sampling_f32
launch_kind: gluon-topk-sampling
shape: {rows: 3, vocab: 16, k: 5}
```

The serving-route result preserves review-safe metadata under
`pto_launch_results`:

- status and phase;
- `launch_kind: gluon-topk-sampling`;
- `kernel_name: topk_sampling_f32`;
- shape metadata;
- `source_sha256`;
- `artifact.source_path`;
- `artifact.manifest_path`;
- validation fields when CUDA runs, including
  `validation.values_match`, `validation.indices_match`, and
  `validation.max_abs_error`.

## H200 Source-Route Evidence

Execution details:

```text
machine class: H200
GPU: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA toolkit: nvcc Build cuda_12.8.r12.8/compiler.35404655_0
arch: compute_90
REMOTE_PTO_CU=<remote-pto-cu>
remote Git refresh: not required
python environment: <remote-pto-cu>/.venv
FastAPI: 0.138.0
HTTPX: 0.28.1
Torch: 2.12.1+cu130
Torch CUDA: 13.0
Triton: 3.7.1
```

Command:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    source .venv/bin/activate && \
    python -m pip install fastapi httpx torch triton && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source \
      --kernel-launcher gluon-topk-sampling \
      --prompt hello --max-new-tokens 1 \
      --device 0 --arch compute_90 --require-cuda'
```

Result:

```text
server: pypto-serving-source
route: /v1/completions
status: passed
status_code: 200
pto_status: passed
pto_token_ids: [1]
pto_launch_count: 1
launch_kind: gluon-topk-sampling
kernel_name: topk_sampling_f32
phase: prefill
shape: {rows: 3, vocab: 16, k: 5}
validation.values_match: true
validation.indices_match: true
validation.max_abs_error: 0.0
source_sha256: 185a2930c266d16641fe91d234020d6caf84ccc71a74c717617bb005c4aa7e66
artifact.source_path: tmp/pypto-serving-gluon-topk-sampling/prefill/topk_sampling_f32.gluon.py
artifact.manifest_path: tmp/pypto-serving-gluon-topk-sampling/prefill/topk_sampling_f32.gluon.json
```

The pass JSON also included the CPU golden and GPU result values and indices;
both matched exactly. The source route returned object `text_completion`, text
`N`, and finish reason `length`.

Two dependency-gating reruns were captured before the pass:

```text
status: skipped
reason: missing FastAPI TestClient: No module named 'fastapi'
```

After installing FastAPI and HTTPX, the route executed and generated the
artifact digest above, then the Top-K correctness gate skipped because Torch
was not installed:

```text
status: skipped
reason: torch import failed: No module named 'torch'
source_sha256: 185a2930c266d16641fe91d234020d6caf84ccc71a74c717617bb005c4aa7e66
```

## Local Verification

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
47 passed
```

## Boundary

This is a serving-route launcher/probe for a generated Top-K sampling
correctness gate. It is not FlashInfer integration, not tokenizer semantics,
not generated text correctness, not DeepSeek serving readiness, not production
serving evidence, not vLLM plugin evidence, not DeepSeek-V4-Flash correctness,
not throughput evidence, and not latency evidence.

- non-claim: not production serving evidence.
