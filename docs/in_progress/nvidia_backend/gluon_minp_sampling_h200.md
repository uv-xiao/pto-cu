# Gluon Min-P Sampling H200 Correctness

This note records correctness evidence for the generated Gluon
`minp_sampling_f32` primitive. The harness validates deterministic Min-P
selection over a small FP32 probability matrix whose rows are normalized
before the operator runs. The fixture avoids softmax scope in this PR.

## Harness

The harness is `examples/cuda/gluon_minp_sampling.py`. It emits structured
JSON for pass, skip, and fail cases. The default review fixture remains
`rows=2, vocab=8, max_k=5, min_p=0.5`. The broader H200 fixture uses
`rows=3, vocab=16, max_k=6, min_p=0.5` and covers varied selected counts
plus probability ties. The probabilities already sum to one. Candidates whose
probability is at least
`probability >= min_p * row_max_probability` are sorted by probability
descending, with equal probabilities ordered by lower token id first. Unused
output slots are filled with `0.0` values and `-1` indices.

The stdout JSON includes:

- `schema_version: 1`;
- `kernel_name: minp_sampling_f32`;
- repo-relative generated source and manifest paths;
- shape and dtype metadata;
- request metadata for the min-p sampling operator;
- CPU golden `values`, `indices`, and `selected_counts`;
- GPU result `values`, `indices`, and `selected_counts` when CUDA runs;
- validation flags for values, indices, selected counts, and explicit payload
  shape checks: `values_shape_match`, `indices_shape_match`, and
  `selected_counts_shape_match`;
- explicit non-claims.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_minp_sampling.py \
  --output-dir tmp/gluon-minp-sampling-local \
  --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, vocab=8, max_k=5, min_p=0.5`;
- dtype: `float32`;
- CPU golden values:
  `[[0.3, 0.2, 0.15, 0.0, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]]`;
- CPU golden indices: `[[1, 3, 5, -1, -1], [0, 2, 6, -1, -1]]`;
- selected counts: `[3, 3]`;
- GPU result values:
  `[[0.3, 0.2, 0.15, 0.0, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]]`;
- GPU result indices: `[[1, 3, 5, -1, -1], [0, 2, 6, -1, -1]]`;
- GPU result selected counts: `[3, 3]`;
- validation: values, indices, and selected counts match;
- max absolute error: `0.0`;
- source digest:
  `12ea20654b6d1298f0900684ffea875f5ad84ad35fc5f292e85d6f50f552b03b`.

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
    .venv/bin/python examples/cuda/gluon_minp_sampling.py \
      --output-dir tmp/gluon-minp-shape-coverage-h200 \
      --arch compute_90 --rows 3 --vocab 16 --max-k 6 --min-p 0.50 \
      --require-cuda'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=3, vocab=16, max_k=6, min_p=0.5`;
- dtype: `float32`;
- CPU golden values:
  `[[0.3, 0.25, 0.15, 0.15, 0.0, 0.0], [0.25, 0.2, 0.15,`
  `0.15, 0.13, 0.0], [0.22, 0.18, 0.14, 0.12, 0.11, 0.11]]`;
- CPU golden indices:
  `[[0, 2, 1, 3, -1, -1], [5, 1, 2, 3, 0, -1], [8, 0, 4, 2, 1, 5]]`;
- CPU golden selected counts: `[4, 5, 6]`;
- GPU result values:
  `[[0.3, 0.25, 0.15, 0.15, 0.0, 0.0], [0.25, 0.2, 0.15,`
  `0.15, 0.13, 0.0], [0.22, 0.18, 0.14, 0.12, 0.11, 0.11]]`;
- GPU result indices:
  `[[0, 2, 1, 3, -1, -1], [5, 1, 2, 3, 0, -1], [8, 0, 4, 2, 1, 5]]`;
- GPU result selected counts: `[4, 5, 6]`;
- validation: `values_shape_match`, `indices_shape_match`,
  `selected_counts_shape_match`, values, indices, and selected counts match;
- max absolute error: `0.0`;
- source digest:
  `12ea20654b6d1298f0900684ffea875f5ad84ad35fc5f292e85d6f50f552b03b`.

## Limitations

- This is a small deterministic min-p correctness gate, not exhaustive
  vocabulary coverage.
- The input probabilities are already normalized; this is not a softmax gate.
- The implementation is correctness-focused, not performance-focused.
- Speculative decoding, tokenizer behavior, generated text, and serving
  integration remain separate gates.
- This is not FlashInfer integration evidence.
- This is not vLLM or simpler-nv kernel integration evidence.
- This is not DeepSeek serving correctness evidence.
- This is not generated-text or tokenizer-semantics evidence.
- This is not throughput or latency evidence.
