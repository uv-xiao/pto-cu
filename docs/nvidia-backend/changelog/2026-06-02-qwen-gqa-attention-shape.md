# 2026-06-02 Qwen GQA Attention Shape

## Code And Data Changed

- Changed the `qwen_attention_o` descriptor shape contract to expose query
  heads, KV heads, and head dimension through existing task fields.
- Updated generated decode-attention source to map query heads to KV heads and
  index `c`/`d` KV-cache rows by `step`, `kv_head`, and `head_col`.
- Expanded the decode-attention oracle from a per-column hidden-size-4 example
  to a hidden-size-8 GQA reference with four query heads and two KV heads.
- Updated the source map, CUDA example manifest/docs, and paper-readiness
  matrix text to keep remaining gates focused on paged/tiled attention,
  full-loop execution, and row import.

## Architecture Quality

This keeps the persistent DAG ABI unchanged. The task body uses fields already
carried by launch packets: `lda` for head dimension, `ldb` for KV heads, and
`inner` for the bounded diagnostic window. A later
`2026-06-03-qwen-attention-batch-rows.md` changelog refines the contract so
`rows` stays reserved for workload batch rows and query heads are derived from
descriptor width and head dimension. The implementation clamps uneven
query-to-KV mapping rather than relying on undefined indexing when descriptor
fields are malformed.

## Evaluation Run

- Passed `tests/ut/py/test_nvidia_qwen_task_body_math.py`,
  `tests/ut/py/test_nvidia_qwen_graph_materialization.py`, and
  `tests/ut/py/test_nvidia_review_artifacts.py` with 86 tests:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest ... -q`.
- Generated
  `tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-gqa-attention.cu`
  and compiled it to PTX with `nvcc -ptx -arch=compute_80`.

## Remaining Gaps

PTO full-serving rows still require paged KV addressing, tiled softmax
promotion, complete decode-loop execution, and viewer import for both
MPK-policy and VDCores-policy workloads.
