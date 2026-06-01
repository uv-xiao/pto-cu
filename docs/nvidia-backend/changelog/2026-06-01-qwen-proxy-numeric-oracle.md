# 2026-06-01 Qwen Proxy Numeric Oracle

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_task_bodies_impl/oracle.py`.
- Extended the Qwen persistent task-body manifest with a
  `numeric_oracle` block and `controlled_proxy_numeric_oracle` contract.
- Updated the CUDA example manifest, README, serving scaffold, and in-progress
  evaluation notes to describe the oracle as proxy evidence only.

## Architecture Quality

The generated Qwen task-body artifact now separates two claims. It proves that
the current scaffold formulas have deterministic host-side expected values,
while the remaining gap still states that the generated bodies are not
numerically correct Qwen kernels. This prevents proxy arithmetic from being
mistaken for full-serving model correctness.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::\
test_qwen_persistent_task_bodies_render_generated_source -q
```

Result: `1 passed`.

## Remaining Gaps

- Replace proxy task bodies with numerically correct Qwen kernels.
- Execute the `cuda_live` decode loop and import full-serving viewer rows.
