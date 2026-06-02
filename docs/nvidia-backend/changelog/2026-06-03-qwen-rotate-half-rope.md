# Qwen Rotate-Half RoPE

## Code And Data Changed

- Updated generated `qwen_attention_qk_norm` RoPE source from adjacent
  even/odd pairing to Qwen-style split-half `rotate_half` pairing.
- Kept the existing half-head RoPE table allocation and mapped
  `rope_index = head_col % (head_dim / 2)`.
- Added the manifest contract `qwen_qk_norm_rotate_half_rope_source`.

## Architecture Quality

This aligns the generated QK RMSNorm plus RoPE body with the Qwen3 reference
layout used by Transformers, Mirage-MPK Qwen3, and SGLang-style
`rotate_half`. It removes a model-fidelity mismatch in the persistent-device
task body without changing the runtime ABI or RoPE buffer lifetime.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `63 passed`.

A100 first-layer live-session smoke passed:

```text
tmp/cuda-backend/qwen-rotate-half-rope-first-layer-2026-06-03/
```

A100 full-descriptor one-step MPK-policy smoke passed:

```text
tmp/cuda-backend/qwen-rotate-half-rope-full-descriptor-1step-mpk-2026-06-03/
```

The full-descriptor artifact reports 255 completed tasks, zero scheduler
errors, full-vocab diagnostic logits reference pass, and sampled token
`116324`.

## Remaining Gaps

Full Qwen correctness remains open. The first-step token still does not match
the stored Hugging Face reference token `151667`, so the next work remains
kernel and launch-state fidelity rather than result import.
