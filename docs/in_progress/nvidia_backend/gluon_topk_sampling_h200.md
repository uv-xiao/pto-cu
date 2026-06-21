# Gluon Top-K Sampling H200 Correctness

This note records correctness evidence for the generated Gluon
`topk_sampling_f32` primitive. The harness validates deterministic top-k
selection over a small FP32 logits matrix and checks both selected `values`
and token `indices`.

## Harness

The harness is `examples/cuda/gluon_topk_sampling.py`. It emits structured
JSON for pass, skip, and fail cases. The current review gate is intentionally
small: `rows=2, vocab=8, k=3`. Ties are deterministic: equal logits are
ordered by lower token id first.

The stdout JSON includes:

- `schema_version: 1`;
- `kernel_name: topk_sampling_f32`;
- repo-relative generated source and manifest paths;
- shape and dtype metadata;
- request metadata for the top-k sampling operator;
- CPU golden `values` and `indices`;
- GPU result `values` and `indices` when CUDA runs;
- validation flags for `values_match` and `indices_match`;
- explicit non-claims.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_topk_sampling.py \
  --output-dir tmp/gluon-topk-sampling-local \
  --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, vocab=8, k=3`;
- dtype: `float32`;
- CPU golden values: `[[0.9, 0.9, 0.8], [0.9, 0.25, 0.25]]`;
- CPU golden indices: `[[1, 3, 7], [7, 3, 4]]`;
- validation: values match and indices match;
- max absolute error: `0.0`;
- source digest:
  `185a2930c266d16641fe91d234020d6caf84ccc71a74c717617bb005c4aa7e66`.

Local GPU metadata:

- GPU: NVIDIA A100 family;
- compute capability: `8.0`;
- driver: `595.71.05`;
- CUDA toolkit: `nvcc` from CUDA 12.8.

## H200 Command

The first fresh remote venv command followed the standard dependency shape
with `scikit-build-core`, `nanobind`, `cmake`, and `ninja`; it reached the
harness and returned a structured skip because that H200 environment did not
provide `torch` or `triton` through system site packages. The passing run used
the same synced checkout and installed `torch` and `triton` into the remote
venv before invoking the harness.

Execution details:

- synced checkout: `<remote-pto-cu>`;
- python environment: `<remote-pto-cu>/.venv`;
- machine class: H200;
- GPU: NVIDIA H200 NVL;
- compute capability: `9.0`;
- memory: `143771 MiB`;
- driver: `580.126.20`;
- Torch: `2.12.1+cu130`;
- Torch CUDA: `13.0`;
- Triton: `3.7.1`;
- Gluon import: available.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/gluon_topk_sampling.py \
      --output-dir tmp/gluon-topk-sampling-h200 \
      --arch compute_90 --require-cuda'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, vocab=8, k=3`;
- dtype: `float32`;
- CPU golden values: `[[0.9, 0.9, 0.8], [0.9, 0.25, 0.25]]`;
- CPU golden indices: `[[1, 3, 7], [7, 3, 4]]`;
- GPU result values: `[[0.9, 0.9, 0.8], [0.9, 0.25, 0.25]]`;
- GPU result indices: `[[1, 3, 7], [7, 3, 4]]`;
- validation: values match and indices match;
- max absolute error: `0.0`;
- source digest:
  `185a2930c266d16641fe91d234020d6caf84ccc71a74c717617bb005c4aa7e66`.

## Limitations

- This is a tiny static-shape top-k correctness gate, not broad vocabulary
  coverage.
- The implementation is correctness-focused, not performance-focused.
- Top-p, min-p, speculative decoding, tokenizer behavior, generated text, and
  serving integration remain separate gates.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not vLLM or simpler-nv kernel integration evidence.
- This is not DeepSeek serving correctness evidence.
- This is not generated-text or tokenizer-semantics evidence.
- This is not throughput or latency evidence.
