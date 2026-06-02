# CUDA Examples: Qwen Persistent Task Bodies

## Qwen Persistent Task Bodies

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies.cu
```

Expected output: command exits 0; output JSON records generated Qwen task
bodies with token, mutable KV-cache, weight, shape-linear, QK RMSNorm/RoPE,
GQA decode-attention grouping, full-vocab argmax, proxy numeric oracle, and
unit-math/decode-attention oracles.

The artifact renders through the existing persistent DAG source generator.
It is source-level integration evidence, not full Qwen serving correctness.
The generated CUDA source now uses descriptor `rows`, `cols`, `inner`, `lda`,
and `ldb` fields for QKV, attention-output, MLP, and logits linear
projections when full tensor metadata is present. The QK norm task computes
RMS scale from descriptor `cols`, `inner`, and `lda` fields before applying
Q/K norm weight slots and pairwise RoPE rotation when cos/sin table slots are
bound. The attention-output task now has a shape-gated path that reads mutable
`c`/`d` KV-cache fields and computes a max-stabilized softmax reduction over
the bounded `inner` decode window with GQA query-head to KV-head grouping from
descriptor `rows`, `lda`, and `ldb`. It also accepts runtime
`tensor_args[1]` as a `kv_page_table` and maps logical decode steps to
physical pages before reading key/value cache. This is diagnostic source
evidence; tiled softmax and full decode-loop import remain open.
The logits shape path scans the
descriptor vocab width for device-side argmax feedback. The persistent DAG
ABI also exposes mutable `c` and `d` fields, so the artifact records KV-cache
writeback field access before `cuda_live` decode-loop execution. The numeric
oracle keeps the controlled proxy formulas for fallback review only; it must
not be promoted as full Qwen correctness. The Qwen unit math oracle records
RMSNorm, projection, single-token attention cache writeback, SiLU/SwiGLU, and
logits equations for a hidden-size-4 reference. The generated CUDA source now
contains that unit-math path, and the live example below executes it. The
manifest also
records `qwen_kernel_source_map`, which points reviewers to local FlashInfer,
vLLM, and SGLang source snapshots under `tmp/sources/kernel-references/`.
