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
bodies with shape-aware token embedding lookup, mutable KV-cache, weight,
shape-linear, QK RMSNorm/RoPE, GQA decode-attention grouping, paged tiled
attention, full-vocab argmax, proxy numeric oracle, and
unit-math/decode-attention oracles.

The artifact renders through the existing persistent DAG source generator.
It is source-level integration evidence, not full Qwen serving correctness.
The embedding task uses descriptor `cols`/`ldb` fields to turn runtime
`input_ids` into hidden-vector columns through `embed_tokens.weight` instead
of the old four-value proxy index when the weight slot is bound.
The generated CUDA source now uses descriptor `rows`, `cols`, `inner`, `lda`,
and `ldb` fields for QKV, attention-output, MLP, and logits linear
projections when full tensor metadata is present. The QK norm task computes
RMS scale from descriptor `cols`, `inner`, and `lda` fields before applying
separate Q/K norm weight slots and pairwise RoPE rotation when cos/sin table
slots are bound; descriptors expose Q-width plus KV-width rather than blending
Q/K weights into one vector. The normalized K region is also written back
through mutable `task->c` key-cache storage using the decode position and
descriptor page size; QK-norm accepts runtime `tensor_args[4]` as a
`kv_page_table` and maps logical decode pages before writing normalized K
cache. The QK-norm source derives the batch row from `j / task->cols` before
reading Q/K regions and before forming the batch-local K-cache write index.
The attention-output task now has a shape-gated path that reads mutable
`c`/`d` KV-cache fields and computes a max-stabilized softmax reduction over
the bounded `inner` decode window with GQA query-head to KV-head grouping from
descriptor `rows`, `lda`, and `ldb`. It also accepts runtime `tensor_args[1]`
as a `kv_page_table` and maps logical decode steps to physical pages before
reading key/value cache. The softmax max and weighted sum passes apply the
Qwen head-dim attention scale and are bounded by descriptor-controlled
attention tiles. When
`o_proj_weight` is bound as `tensor_args[0]`, the task recomputes bounded
attention columns, multiplies them by `o_proj_weight`, and writes projected
hidden columns; `scalar_args[1]` limits that diagnostic projection width for
resource-backed unit runs. Post-attention RMSNorm now consumes `task->b` as
the layer residual source and reduces over `attention_output + residual`
before applying `post_attention_layernorm.weight`. The MLP down-projection
task also consumes `task->b` and adds the launch-packet residual source after
the down projection. This is diagnostic source evidence; full decode-loop
import remains open.
The logits shape path now computes descriptor-bounded hidden-by-vocab tiles
before device-side argmax feedback. It reads the hidden vector through
`task->a`, reads vocab projection weights through `tensor_args[0]`, bounds the
hidden loop with `inner`, `lda`, and `ldb`, and uses `scalar0` as the logits
tile size from the descriptor. The QKV task now binds runtime `kv_page_table` as
`tensor_args[3]`, uses descriptor `scalar0` as page size, consumes the current
decode position from `scalar_args[2]`, and writes key/value projections to
batch-local physical KV-cache slots with logical-to-physical fallback. The
numeric oracle keeps the controlled proxy formulas for fallback review only;
it must not be promoted as full Qwen correctness. The Qwen unit math oracle
records RMSNorm, projection, single-token attention cache writeback,
SiLU/SwiGLU, and logits equations for a hidden-size-4 reference. The generated
CUDA source now contains that unit-math path, and the live example below
executes it. The manifest also records `qwen_kernel_source_map`, which points
reviewers to local FlashInfer, vLLM, and SGLang source snapshots under
`tmp/sources/kernel-references/`.
