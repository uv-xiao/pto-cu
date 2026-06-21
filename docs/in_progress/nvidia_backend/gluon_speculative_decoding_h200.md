# Gluon Speculative Decoding H200 Correctness

This note records correctness evidence for the generated Gluon
`speculative_accept_f32` primitive. The harness validates a deterministic
speculative-decoding accept/reject boundary over a small static fixture. It
does not use a model, tokenizer, serving stack, or generated text.

## Harness

The harness is `examples/cuda/gluon_speculative_decoding.py`. It emits
structured JSON for pass, skip, and fail cases. The current review gate is
intentionally small: `rows=2, max_draft=4`.

For each row, inputs are draft token ids, draft probabilities, target
probabilities for those same draft tokens, and deterministic thresholds. The
acceptance rule is:

`threshold <= min(1.0, target_probability / draft_probability)`.

The row accepts while that rule holds and stops at first reject per row; in
other words, stop at first reject per row and mask the tail.
Rejected and tail positions write `-1` token ids and `0` accept-mask entries.

The stdout JSON includes:

- `schema_version: 1`;
- `kernel_name: speculative_accept_f32`;
- repo-relative generated source and manifest paths;
- shape and dtype metadata;
- request metadata for the speculative-decoding accept/reject operator;
- CPU golden `accepted_token_ids`, `accept_mask`, and `accepted_counts`;
- GPU result `accepted_token_ids`, `accept_mask`, and `accepted_counts` when
  CUDA runs;
- validation flags for accepted token ids, accept mask, and accepted counts;
- explicit non-claims.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_speculative_decoding.py \
  --output-dir tmp/gluon-speculative-decoding-local \
  --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, max_draft=4`;
- CPU golden accepted token ids:
  `[[10, 11, 12, 13], [20, -1, -1, -1]]`;
- CPU golden accept mask: `[[1, 1, 1, 1], [1, 0, 0, 0]]`;
- CPU golden accepted counts: `[4, 1]`;
- GPU result accepted token ids:
  `[[10, 11, 12, 13], [20, -1, -1, -1]]`;
- GPU result accept mask: `[[1, 1, 1, 1], [1, 0, 0, 0]]`;
- GPU result accepted counts: `[4, 1]`;
- validation: accepted token ids, accept mask, and accepted counts match;
- source digest:
  `d5f9da0a38a56d5194b8c06bf4698acbac14285c6cc038046f6622d2eb1f8db1`.

Local GPU metadata:

- GPU: NVIDIA A100 80GB PCIe;
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
    .venv/bin/python examples/cuda/gluon_speculative_decoding.py \
      --output-dir tmp/gluon-speculative-decoding-h200 \
      --arch compute_90 --require-cuda'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=2, max_draft=4`;
- CPU golden accepted token ids:
  `[[10, 11, 12, 13], [20, -1, -1, -1]]`;
- CPU golden accept mask: `[[1, 1, 1, 1], [1, 0, 0, 0]]`;
- CPU golden accepted counts: `[4, 1]`;
- GPU result accepted token ids:
  `[[10, 11, 12, 13], [20, -1, -1, -1]]`;
- GPU result accept mask: `[[1, 1, 1, 1], [1, 0, 0, 0]]`;
- GPU result accepted counts: `[4, 1]`;
- validation: accepted token ids, accept mask, and accepted counts match;
- source digest:
  `d5f9da0a38a56d5194b8c06bf4698acbac14285c6cc038046f6622d2eb1f8db1`.

## Limitations

- This is a tiny static-shape speculative accept/reject correctness gate, not
  broad draft-length or batch coverage.
- The input probabilities are provided directly; this is not a softmax gate.
- The implementation is correctness-focused, not performance-focused.
- This does not implement proposer/verifier model scheduling.
- This is not FlashInfer integration evidence.
- This is not vLLM or simpler-nv kernel integration evidence.
- This is not DeepSeek serving correctness evidence.
- This is not generated-text or tokenizer-semantics evidence.
- This is not throughput or latency evidence.
