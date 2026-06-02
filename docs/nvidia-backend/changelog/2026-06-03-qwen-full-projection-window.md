# Qwen Full Projection Window

## Code And Data Changed

- Added `--resource-backed-projection-active-cols` to the Qwen decode-loop
  runner.
- Added projection-window descriptor rewriting for `qwen_attention_qkv`,
  `qwen_mlp_gate_up`, and `qwen_mlp_down` without changing the generated
  device task bodies.
- Recorded `projection_active_cols_policy` in resource-backed execution
  artifacts.

## Architecture Quality

This separates two correctness knobs that were previously easy to confuse:
logits active vocabulary columns and projection active columns. Full-Qwen
correctness attempts now need both `--resource-backed-logits-active-cols full`
and `--resource-backed-projection-active-cols full`; otherwise the run remains
a bounded diagnostic even when full logits are checked.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q
```

Result: `37 passed`.

A100 first-layer smoke with full projection windows passed:

```text
tmp/cuda-backend/qwen-projection-full-first-layer-2026-06-03-default-cache/
```

The artifact records `projection_active_cols_policy.mode=full_descriptor_cols`
with `scalar1=6144` for QKV, `scalar1=12288` for MLP gate/up, and
`scalar1=4096` for MLP down. The run completed 10 tasks with zero scheduler
errors and passed the diagnostic logits projection reference.

## Remaining Gaps

This is not a full-serving correctness result. The smoke used the first-layer
descriptor selection and the default bounded logits window. A full promotion
still requires full projection columns, full logits columns, all selected
layers, policy-length MPK and VDCores runs, and Hugging Face token/logit
agreement.
