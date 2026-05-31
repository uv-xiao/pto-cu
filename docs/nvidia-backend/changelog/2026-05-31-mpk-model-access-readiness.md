# 2026-05-31 MPK Model-Access Readiness

## Code And Data Changed

- Added per-run `model_access` metadata to the MPK and VDCores paper-baseline
  run records.
- Updated the paper-baseline run-readiness probe so public-model runs can pass
  model-access readiness without `HF_TOKEN`, while explicitly gated runs still
  require it.
- Updated the paper-readiness audit so passed run-readiness records are not
  emitted as actionable work-queue items.

## Architecture Quality

Model access is now a run-level contract instead of a baseline-wide shortcut.
That matters because MPK uses public Qwen3 model artifacts, while the current
VDCores `llama3` path directly reads `HF_TOKEN` for
`meta-llama/Llama-3.1-8B-Instruct`. The readiness data now reflects that
difference instead of blocking MPK on a token it does not need.

## Evaluation Run

Unauthenticated Hugging Face API snapshots for the MPK Qwen models were saved
under `tmp/sources/huggingface/`:

```text
tmp/sources/huggingface/qwen-qwen3-8b-api.json
tmp/sources/huggingface/qwen-qwen3-1p7b-api.json
```

Both snapshots report `private=false`, `gated=false`, and `disabled=false`.
After regenerating readiness data, these MPK runs now report `pass` readiness:

- `mpk_qwen3_native_vs_persistent`
- `mpk_persistent_scheduler_trace`

## Remaining Gaps

The MPK runs are ready to execute, but they are still `planned_not_run` until
their raw artifacts are captured and imported. VDCores Llama runs remain
blocked on `HF_TOKEN`, and vLLM/SGLang remain blocked on package-entrypoint
readiness.
