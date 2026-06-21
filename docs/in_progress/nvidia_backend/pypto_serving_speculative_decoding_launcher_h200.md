# pypto-serving Speculative Decoding Launcher H200 Evidence

This note records a narrow pypto-serving/simpler-nv
serving-route launcher/probe for a generated speculative decoding
accept/reject correctness gate. It connects the synthetic pypto-serving
source route to the generated Gluon correctness harness, not to a production
draft/verify serving path.

## Contract

- serving-route launcher/probe for a generated speculative decoding accept/reject correctness gate.

The launcher is implemented in `examples/cuda/pypto_serving_nv_shim.py` as
`create_generated_gluon_speculative_decoding_launcher(...)`. It calls
`examples/cuda/gluon_speculative_decoding.py::run_speculative_decoding_correctness(...)`.

The CLI selection is:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
    --pypto-serving-source --kernel-launcher gluon-speculative-decoding \
    --prompt hello --max-new-tokens 1 --require-cuda
```

The selected fixture uses the broader deterministic speculative accept/reject
shape:

```text
kernel_name: speculative_accept_f32
launch_kind: gluon-speculative-decoding
shape: {rows: 3, max_draft: 6}
sampling_operator: speculative-decoding-accept-reject
```

The serving-route result preserves review-safe metadata under
`pto_launch_results`:

- status and phase;
- `launch_kind: gluon-speculative-decoding`;
- `kernel_name: speculative_accept_f32`;
- shape metadata;
- sampling request metadata;
- `source_sha256`;
- `artifact.source_path`;
- `artifact.manifest_path`;
- validation fields when CUDA runs, including
  `validation.accepted_token_ids_match`, `validation.accept_mask_match`, and
  `validation.accepted_counts_match`.

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
source sync: --sync
python environment: <remote-pto-cu>/.venv
package source: reused Min-P/Top-P H200 venv package set
pypto-serving source clone: 0b0d8a0
FastAPI: 0.138.0
HTTPX: 0.28.1
Torch: 2.12.1+cu130
Torch CUDA: 13.0
Triton: 3.7.1
server: pypto-serving-source
route: /v1/completions
```

Command:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source \
      --kernel-launcher gluon-speculative-decoding \
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
launch_kind: gluon-speculative-decoding
kernel_name: speculative_accept_f32
phase: prefill
shape: {rows: 3, max_draft: 6}
validation.accepted_token_ids_match: true
validation.accept_mask_match: true
validation.accepted_counts_match: true
source_sha256: d5f9da0a38a56d5194b8c06bf4698acbac14285c6cc038046f6622d2eb1f8db1
artifact.source_path: tmp/pypto-serving-gluon-speculative-decoding/prefill/speculative_accept_f32.gluon.py
artifact.manifest_path: tmp/pypto-serving-gluon-speculative-decoding/prefill/speculative_accept_f32.gluon.json
```

The pass JSON also includes CPU golden and GPU result accepted token ids,
accept mask, and accepted counts. The source route returns object
`text_completion`, text `N`, and finish reason `length`.

## Local Verification

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
31 passed, 28 skipped
```

## Boundary

This is a serving-route launcher/probe for a generated speculative decoding
accept/reject correctness gate. It is not FlashInfer integration, not
tokenizer semantics, not generated text correctness, not DeepSeek serving
readiness, not production serving evidence, not vLLM plugin evidence, not
DeepSeek-V4-Flash correctness, not throughput/latency evidence, and not
production draft/verify scheduler evidence.

- non-claim: not FlashInfer integration.
- non-claim: not tokenizer semantics.
- non-claim: not generated text correctness.
- non-claim: not DeepSeek serving readiness.
- non-claim: not production serving evidence.
- non-claim: not throughput/latency evidence.
