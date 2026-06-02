# Qwen Final Norm Live

## Code And Data Changed

- Added a block-threaded full RMSNorm branch to the generated
  `qwen_final_norm` persistent-device task body.
- Extended the Qwen unit-math live DAG from four tasks to five tasks:
  `RMSNorm -> QKV -> SwiGLU -> FinalRMSNorm -> logits`.
- Updated the compact benchmark-viewer unit-math row to point at
  `tmp/cuda-backend/qwen-final-norm-rmsnorm/qwen-unit-math-live.json` and
  removed the older four-task diagnostic row.
- Added explicit evidence symbols for
  `qwen_final_norm_full_rmsnorm_source` and
  `qwen_unit_math_final_norm_live_execution`.

## Architecture Quality

The unit-math live path now exercises the final RMSNorm kernel on device
instead of only proving it through generated source. The logits task receives
the unit-math projection scalar contract, so the five-task live DAG checks the
same final-normalized hidden state that feeds the logits projection.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_unit_math_live.py --device 0 --arch compute_80 \
  --repeat-runs 3 \
  --cache-root tmp/cuda-backend/qwen-final-norm-rmsnorm/unit-live-cache \
  --output-json tmp/cuda-backend/qwen-final-norm-rmsnorm/qwen-unit-math-live.json
```

Result: pass. The live run completed 15 persistent-device task executions,
reported zero scheduler errors, and observed final RMSNorm and logits outputs
with maximum absolute error below `1e-6`.

Additional verification:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_unit_math_live.py \
  -q
```

Result: passed.

## Remaining Gaps

This closes one unit-math kernel gap only. PTO still needs full-serving
Qwen/Qwen3-8B rows for MPK and VDCores with end-to-end generated-token
correctness before the LLM-serving paper claim can be marked ready.
