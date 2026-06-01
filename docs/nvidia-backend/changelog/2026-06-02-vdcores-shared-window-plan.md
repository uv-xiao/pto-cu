# 2026-06-02 VDCores Shared-Instruction Window Plan

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/vdcores_instruction_window_plan.py`
  to derive a segmented shared-instruction runtime plan from the existing
  VDCores Qwen3-8B H200 capacity artifact.
- Generated the raw analysis artifact under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-shared-window-plan-0a0392d2/
```

- Added the analysis attempt to the benchmark viewer execution-attempt data
  and kept the VDCores Qwen3-8B paper-serving row blocked.

## Architecture Quality

The analysis keeps the preferred VDCores recovery path explicit. The
global-instruction variant can run `-N 64 -b 5`, but it fails correctness.
The shared-runtime path therefore needs segmented or token-windowed execution
that reloads or advances instruction windows while preserving resident tensor
state, KV-cache state, dependencies, correctness checks, and per-window timing.

## Evaluation Run

The script consumed:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-instruction-capacity-0a0392d2/instruction-capacity-n64.json
```

It derived these lower bounds for the default 512-instruction shared table:

- 5 compute-instruction windows per SM.
- 30 memory-instruction windows per SM.
- 30 worst-case instruction windows per SM for the Qwen3-8B decode64 path.

## Remaining Gaps

- Implement the segmented shared-instruction builder/runtime path or fix the
  global-instruction correctness regression.
- Re-run Qwen3-8B `-N 64 -b 5` with correctness passing before importing the
  VDCores paper-serving row.
