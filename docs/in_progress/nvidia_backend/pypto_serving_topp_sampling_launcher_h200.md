# pypto-serving Top-P Sampling Launcher H200 Evidence

This note records a narrow pypto-serving/simpler-nv serving-route
launcher/probe for a generated Top-P sampling correctness gate. It connects
the synthetic pypto-serving source route to the existing generated Gluon
correctness harness, not to a production sampler.

## Contract

- serving-route launcher/probe for a generated Top-P sampling correctness gate.

The launcher is implemented in `examples/cuda/pypto_serving_nv_shim.py` as
`create_generated_gluon_topp_sampling_launcher(...)`. It calls
`examples/cuda/gluon_topp_sampling.py::run_topp_sampling_correctness(...)`.

The CLI selection is:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
    --pypto-serving-source --kernel-launcher gluon-topp-sampling \
    --prompt hello --max-new-tokens 1 --require-cuda
```

The selected fixture uses the existing deterministic Top-P shape:

```text
kernel_name: topp_sampling_f32
launch_kind: gluon-topp-sampling
shape: {rows: 3, vocab: 16, max_k: 6}
p: 0.80
```

The serving-route result preserves review-safe metadata under
`pto_launch_results`:

- status and phase;
- `launch_kind: gluon-topp-sampling`;
- `kernel_name: topp_sampling_f32`;
- shape metadata;
- sampling request metadata including `p`;
- `source_sha256`;
- `artifact.source_path`;
- `artifact.manifest_path`;
- validation fields when CUDA runs, including
  `validation.values_match`, `validation.indices_match`,
  `validation.selected_counts_match`,
  `validation.cumulative_probabilities_match`, `validation.max_abs_error`,
  and `validation.max_cumulative_probability_error`.

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
source sync: --sync plus explicit pypto-serving source clone sync
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
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source \
      --kernel-launcher gluon-topp-sampling \
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
launch_kind: gluon-topp-sampling
kernel_name: topp_sampling_f32
phase: prefill
shape: {rows: 3, vocab: 16, max_k: 6}
p: 0.80
validation.values_match: true
validation.indices_match: true
validation.selected_counts_match: true
validation.cumulative_probabilities_match: true
validation.max_abs_error: 0.0
validation.max_cumulative_probability_error: 9.999999994736442e-08
source_sha256: 9f9919d0b209af36a7546714500c78893d11a60a630f70c69b7464d3055d3311
artifact.source_path: tmp/pypto-serving-gluon-topp-sampling/prefill/topp_sampling_f32.gluon.py
artifact.manifest_path: tmp/pypto-serving-gluon-topp-sampling/prefill/topp_sampling_f32.gluon.json
```

The pass JSON also included CPU golden and GPU result values, indices,
selected counts, and cumulative probabilities. All validation checks passed.
The GPU cumulative probabilities were `[0.8000001, 0.8, 0.8]`, matching the
CPU golden within the harness tolerance. The source route returned object
`text_completion`, text `N`, and finish reason `length`.

Two dependency-gating reruns were captured before the pass:

```text
status: skipped
reason: missing FastAPI TestClient: No module named 'fastapi'
```

After installing FastAPI and HTTPX, the route executed and generated the
artifact digest above, then the Top-P correctness gate skipped because Torch
was not installed:

```text
status: skipped
reason: torch import failed: No module named 'torch'
source_sha256: 9f9919d0b209af36a7546714500c78893d11a60a630f70c69b7464d3055d3311
```

## Local Verification

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
29 passed, 22 skipped
```

## Boundary

This is a serving-route launcher/probe for a generated Top-P sampling
correctness gate. It is not FlashInfer integration, not tokenizer semantics,
not generated text correctness, not DeepSeek serving readiness, not production
serving evidence, not vLLM plugin evidence, not DeepSeek-V4-Flash correctness,
not throughput evidence, and not latency evidence.

- non-claim: not production serving evidence.
