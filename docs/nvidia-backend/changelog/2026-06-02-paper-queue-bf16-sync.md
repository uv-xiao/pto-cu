# 2026-06-02 Paper Queue BF16 Sync

## Code And Data Changed

- Updated the PTO Qwen3-8B full-serving missing-evidence record to include the
  BF16/FP32 `tensor_arg_dtypes` persistent task ABI and dtype-aware CUDA tensor
  reads.
- Regenerated the paper-readiness audit and work queue from the matrix data.

## Architecture Quality

The viewer now reflects the current blocker accurately. The dtype path for
real Qwen resident weights is implemented; the remaining PTO full-serving
blocker is full model-shape kernel math plus result import.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`.

## Remaining Gaps

PTO still needs full attention, sampling, full decode-loop execution, and
full-serving row import before MPK-policy and VDCores-policy rows can be used
as paper-ready evidence.
