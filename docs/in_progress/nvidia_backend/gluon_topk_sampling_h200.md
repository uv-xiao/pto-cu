# Gluon Top-K Sampling H200 Correctness

This note records correctness evidence for the generated Gluon
`topk_sampling_f32` primitive. The harness validates deterministic Top-K
selection over FP32 logits matrices and checks both selected `values` and
token `indices`.

## Harness

The harness is `examples/cuda/gluon_topk_sampling.py`. It emits structured
JSON for pass, skip, and fail cases. It supports deterministic fixtures
selected by shape arguments:

- default fixture: `rows=2, vocab=8, k=3`;
- broader fixture: `rows=3, vocab=16, k=5`.

The broader fixture includes tied and negative logits. Ties are
deterministic: equal logits are ordered by lower token id first.

The stdout JSON includes:

- `schema_version: 1`;
- `kernel_name: topk_sampling_f32`;
- repo-relative generated source and manifest paths;
- shape and dtype metadata;
- request metadata for the Top-K sampling operator;
- CPU golden `values` and `indices`;
- GPU result `values` and `indices` when CUDA runs;
- strict validation flags for result payload shape, `values`, and `indices`;
- explicit non-claims.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_topk_sampling.py \
  --output-dir tmp/gluon-topk-shape-coverage-local \
  --arch compute_90 --rows 3 --vocab 16 --k 5
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `rows=3, vocab=16, k=5`;
- dtype: `float32`;
- CPU golden values:
  `[[2.0, 2.0, 1.5, 1.5, 1.2], [3.0, 3.0, 2.0, 2.0, 1.5], [4.0, 4.0, 3.5, 3.5, 3.5]]`;
- CPU golden indices:
  `[[7, 11, 2, 3, 9], [5, 7, 6, 9, 11], [3, 4, 6, 7, 15]]`;
- GPU result values:
  `[[2.0, 2.0, 1.5, 1.5, 1.2], [3.0, 3.0, 2.0, 2.0, 1.5], [4.0, 4.0, 3.5, 3.5, 3.5]]`;
- GPU result indices:
  `[[7, 11, 2, 3, 9], [5, 7, 6, 9, 11], [3, 4, 6, 7, 15]]`;
- validation: values shape, indices shape, values, and indices match;
- max absolute error: `0.0`;
- source digest:
  `185a2930c266d16641fe91d234020d6caf84ccc71a74c717617bb005c4aa7e66`.

Local GPU metadata:

- GPU: NVIDIA A100 family;
- compute capability: `8.0`;
- driver: `595.71.05`;
- CUDA toolkit: `nvcc` from CUDA 12.8.

## H200 Command

The fresh remote run synced this checkout to `<remote-pto-cu>`, installed the
Python build and CUDA dependencies in `<remote-pto-cu>/.venv`, then invoked
the broadened fixture with `--require-cuda`.

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
      >"<remote-temp>/topk-shapes-deps.log" && \
    pip install --no-build-isolation -e . \
      >"<remote-temp>/topk-shapes-install.log" && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/gluon_topk_sampling.py \
      --output-dir tmp/gluon-topk-shape-coverage-h200 \
      --arch compute_90 --rows 3 --vocab 16 --k 5 --require-cuda'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=3, vocab=16, k=5`;
- dtype: `float32`;
- CPU golden values:
  `[[2.0, 2.0, 1.5, 1.5, 1.2], [3.0, 3.0, 2.0, 2.0, 1.5], [4.0, 4.0, 3.5, 3.5, 3.5]]`;
- CPU golden indices:
  `[[7, 11, 2, 3, 9], [5, 7, 6, 9, 11], [3, 4, 6, 7, 15]]`;
- GPU result values:
  `[[2.0, 2.0, 1.5, 1.5, 1.2], [3.0, 3.0, 2.0, 2.0, 1.5], [4.0, 4.0, 3.5, 3.5, 3.5]]`;
- GPU result indices:
  `[[7, 11, 2, 3, 9], [5, 7, 6, 9, 11], [3, 4, 6, 7, 15]]`;
- validation: values shape, indices shape, values, and indices match;
- max absolute error: `0.0`;
- source digest:
  `185a2930c266d16641fe91d234020d6caf84ccc71a74c717617bb005c4aa7e66`.

## Limitations

- This is a small two-fixture Top-K correctness gate, not broad sampling
  coverage.
- The implementation is correctness-focused, not performance-focused.
- Top-P, Min-P, speculative decoding, tokenizer behavior, generated text, and
  serving integration remain separate gates.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not vLLM or simpler-nv kernel integration evidence.
- This is not DeepSeek serving correctness evidence.
- This is not generated-text or tokenizer-semantics evidence.
- This is not throughput or latency evidence.
