# 2026-06-02 Qwen Decode Attention Reduction

## Code And Data Changed

- Added a bounded diagnostic decode-attention path to
  `qwen_attention_o` in the generated persistent-device Qwen task bodies.
- The path reads shape fields plus mutable `c`/`d` KV-cache fields, computes
  max-stabilized softmax over the bounded `inner` decode window, and writes
  the reduced context to `out`.
- Added `qwen_decode_attention_oracle` and
  `qwen_bounded_decode_attention_reduction_source` so reviewers can connect
  the generated CUDA source to a deterministic hidden-size-8 GQA reference.
- Updated the Qwen source map, CUDA example manifest/docs, and paper-readiness
  matrix text to describe this as diagnostic evidence.

## Architecture Quality

The old projection fallback remains available when KV-cache pointers are not
bound. The new source path is explicitly diagnostic: it proves device-side
KV-cache consumption, reduction wiring, and GQA head grouping, while keeping
paged KV addressing, tiled softmax, full decode-loop execution, and viewer row
import as remaining promotion gates.

## Evaluation Run

- Passed `tests/ut/py/test_nvidia_qwen_task_body_math.py` and
  `tests/ut/py/test_nvidia_review_artifacts.py` with 70 tests:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest ... -q`.
- Generated
  `tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-decode-attention.cu`
  and compiled it to PTX with `nvcc -ptx -arch=compute_80`.

## Remaining Gaps

PTO full-serving rows still require paged KV addressing, tiled softmax
promotion, complete decode-loop execution, and viewer import for both
MPK-policy and VDCores-policy workloads.
