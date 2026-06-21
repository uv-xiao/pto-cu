# Gluon Top-P Sampling H200 Correctness

This note records correctness evidence for the generated Gluon
`topp_sampling_f32` primitive. The harness validates deterministic Top-P
selection over a small FP32 probability matrix whose rows are normalized
before the operator runs. The fixture avoids softmax scope in this PR.

## Harness

The harness is `examples/cuda/gluon_topp_sampling.py`. It emits structured
JSON for pass, skip, and fail cases. The current review gate is intentionally
small: `rows=2, vocab=8, max_k=5, p=0.75`. The probabilities already sum to one.
Candidates are sorted by probability descending, with equal probabilities
ordered by lower token id first.

The stdout JSON includes:

- `schema_version: 1`;
- `kernel_name: topp_sampling_f32`;
- repo-relative generated source and manifest paths;
- shape and dtype metadata;
- request metadata for the top-p sampling operator;
- CPU golden `values`, `indices`, `selected_counts`, and
  `cumulative_probabilities`;
- GPU result `values`, `indices`, `selected_counts`, and
  `cumulative_probabilities` when CUDA runs;
- validation flags for values, indices, selected counts, and the cumulative
  probability boundary;
- explicit non-claims.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_topp_sampling.py \
  --output-dir tmp/gluon-topp-sampling-local \
  --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, vocab=8, max_k=5, p=0.75`;
- dtype: `float32`;
- CPU golden values:
  `[[0.3, 0.2, 0.15, 0.1, 0.0], [0.25, 0.25, 0.15, 0.1, 0.0]]`;
- CPU golden indices: `[[1, 3, 5, 2, -1], [0, 2, 6, 4, -1]]`;
- selected counts: `[4, 4]`;
- cumulative probabilities: `[0.75, 0.75]`;
- validation: values, indices, selected counts, and cumulative probability
  boundary match;
- max absolute error: `0.0`;
- max cumulative probability error: `0.0`;
- source digest:
  `9f9919d0b209af36a7546714500c78893d11a60a630f70c69b7464d3055d3311`.

Local GPU metadata:

- GPU: NVIDIA A100 family;
- compute capability: `8.0`;
- driver: `595.71.05`;
- CUDA toolkit: `nvcc` from CUDA 12.8;
- Torch: `2.1.0+cu121`;
- Torch CUDA: `12.1`;
- Triton: `3.5.1`.

## H200 Command

Execution details:

- synced checkout: `<remote-pto-cu>`;
- python environment: `<remote-pto-cu>/.venv`;
- machine class: H200;
- GPU: NVIDIA H200 NVL;
- compute capability: `9.0`;
- memory: `143771 MiB`;
- driver: `580.126.20`;
- CUDA toolkit: `nvcc` from CUDA 12.8;
- Torch: `2.12.1+cu130`;
- Torch CUDA: `13.0`;
- Triton: `3.7.1`;
- Gluon import: available.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    source .venv/bin/activate && \
    pip install scikit-build-core nanobind cmake ninja torch triton \
      ><remote-deps-log> && \
    pip install --no-build-isolation -e . ><remote-install-log> && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/gluon_topp_sampling.py \
      --output-dir tmp/gluon-topp-sampling-h200 \
      --arch compute_90 --require-cuda'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, vocab=8, max_k=5, p=0.75`;
- dtype: `float32`;
- CPU golden values:
  `[[0.3, 0.2, 0.15, 0.1, 0.0], [0.25, 0.25, 0.15, 0.1, 0.0]]`;
- CPU golden indices: `[[1, 3, 5, 2, -1], [0, 2, 6, 4, -1]]`;
- CPU golden selected counts: `[4, 4]`;
- CPU golden cumulative probabilities: `[0.75, 0.75]`;
- GPU result values:
  `[[0.3, 0.2, 0.15, 0.1, 0.0], [0.25, 0.25, 0.15, 0.1, 0.0]]`;
- GPU result indices: `[[1, 3, 5, 2, -1], [0, 2, 6, 4, -1]]`;
- GPU result selected counts: `[4, 4]`;
- GPU result cumulative probabilities: `[0.75, 0.75]`;
- validation: values, indices, selected counts, and cumulative probabilities
  match;
- max absolute error: `0.0`;
- max cumulative probability error: `0.0`;
- source digest:
  `9f9919d0b209af36a7546714500c78893d11a60a630f70c69b7464d3055d3311`.

## Limitations

- This is a tiny static-shape top-p correctness gate, not broad vocabulary
  coverage.
- The input probabilities are already normalized; this is not a softmax gate.
- The implementation is correctness-focused, not performance-focused.
- Min-p, speculative decoding, tokenizer behavior, generated text, and
  serving integration remain separate gates.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not vLLM or simpler-nv kernel integration evidence.
- This is not DeepSeek serving correctness evidence.
- This is not generated-text or tokenizer-semantics evidence.
- This is not throughput or latency evidence.
