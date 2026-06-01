# 2026-06-02 Qwen Resource-Backed Logits Sampling

## Code And Data Changed

- Added bounded logits-prefix readback to the resource-backed Qwen diagnostic
  execution path.
- Imported the new logits coverage fields into benchmark-viewer result rows.
- Kept the PTO full-serving work item blocked because the sampled prefix is
  not full-vocabulary Qwen logits.

Raw artifact:

```text
tmp/cuda-backend/pto-serving-resource-backed-logits-sampling-2026-06-02/qwen-decode-loop-runner.json
```

## Architecture Quality

The execution artifact now reports whether logits readback covers the whole
buffer, how many elements were written, how many were sampled, and a stable
top-k token view over the written prefix. This makes output evidence
reviewable and prevents diagnostic rows from looking like full-serving
correctness rows.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-repeat-runs 3 \
  --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-logits-sampling-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 3 submissions, 765 completed tasks, 0 errors,
  65,536 written logits-prefix elements sampled from a 2,430,976-element
  logits buffer, stable top token id 0 across repeats.
- `vdcores_offline_decode`: 3 submissions, 765 completed tasks, 0 errors,
  65,536 written logits-prefix elements sampled from a 2,430,976-element
  logits buffer, stable top token id 0 across repeats.

## Remaining Gaps

- Generate and validate full-vocabulary Qwen logits instead of the current
  diagnostic prefix.
- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Execute the full token-by-token decode loop and import full-serving PTO rows
  for `mpk_offline_decode` and `vdcores_offline_decode`.
