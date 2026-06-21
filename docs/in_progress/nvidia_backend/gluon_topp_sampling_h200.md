# Gluon Top-P Sampling H200 Correctness

This note records correctness evidence for the generated Gluon
`topp_sampling_f32` primitive. The harness validates deterministic Top-P
selection over a small FP32 probability matrix whose rows are normalized
before the operator runs. The fixture avoids softmax scope in this PR.

## Harness

The harness is `examples/cuda/gluon_topp_sampling.py`. It emits structured
JSON for pass, skip, and fail cases. The default review gate remains
`rows=2, vocab=8, max_k=5, p=0.75`. The broader H200 fixture is
`rows=3, vocab=16, max_k=6, p=0.80`. The probabilities already sum to one.
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
- strict validation flags for result payload shape, values, indices, selected
  counts, and the cumulative probability boundary;
- explicit non-claims.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_topp_sampling.py \
  --output-dir tmp/gluon-topp-sampling-local \
  --arch compute_90 --rows 3 --vocab 16 --max-k 6 --p 0.80
```

Without CUDA tooling or a visible NVIDIA GPU it reports a skip. With
`--require-cuda`, skipped cases return a non-zero exit status.

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
      --output-dir tmp/gluon-topp-shape-coverage-h200 \
      --arch compute_90 --rows 3 --vocab 16 --max-k 6 --p 0.80 \
      --require-cuda'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=3, vocab=16, max_k=6, p=0.80`;
- dtype: `float32`;
- CPU golden values:
  `[[0.3, 0.25, 0.15, 0.1, 0.0, 0.0],`
  `[0.25, 0.2, 0.15, 0.15, 0.05, 0.0],`
  `[0.22, 0.18, 0.14, 0.1, 0.09, 0.07]]`;
- CPU golden indices:
  `[[0, 2, 3, 1, -1, -1], [5, 1, 2, 3, 0, -1],`
  `[8, 0, 4, 2, 5, 1]]`;
- CPU golden selected counts: `[4, 5, 6]`;
- CPU golden cumulative probabilities: `[0.8, 0.8, 0.8]`;
- GPU result values:
  `[[0.3, 0.25, 0.15, 0.1, 0.0, 0.0],`
  `[0.25, 0.2, 0.15, 0.15, 0.05, 0.0],`
  `[0.22, 0.18, 0.14, 0.1, 0.09, 0.07]]`;
- GPU result indices:
  `[[0, 2, 3, 1, -1, -1], [5, 1, 2, 3, 0, -1],`
  `[8, 0, 4, 2, 5, 1]]`;
- GPU result selected counts: `[4, 5, 6]`;
- GPU result cumulative probabilities: `[0.8000001, 0.8, 0.8]`;
- validation: values shape, indices shape, selected counts shape, cumulative
  probabilities shape, values, indices, selected counts, and cumulative
  probabilities match;
- max absolute error: `0.0`;
- max cumulative probability error: `9.999999994736442e-08`;
- source digest:
  `9f9919d0b209af36a7546714500c78893d11a60a630f70c69b7464d3055d3311`.

## Limitations

- This is a bounded two-fixture top-p correctness gate, not broad vocabulary
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
