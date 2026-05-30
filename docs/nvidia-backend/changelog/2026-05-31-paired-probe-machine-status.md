# 2026-05-31 Paired Probe Machine Status

## Code And Data Changed

Added `latest_machine_status` to the committed paper-baseline probe data. Each
probe now records A100 and H200 status separately, links the machine-specific
raw probe JSON file under `tmp/`, and carries the blocking gaps reported by the
paired probe artifact.

Updated the benchmark viewer to render those machine statuses under each paper
baseline, and tightened the viewer-data validator plus focused tests so every
paper-baseline probe must expose exactly A100 and H200 readiness status.

## Architecture Quality

The aggregate `latest_status` remains useful for a quick baseline summary, but
it hides why a baseline is partial. The per-machine status makes the paper
readiness state reviewable without opening raw JSON by hand, while still
linking every rendered status back to the raw A100 or H200 probe artifact.

This is especially important for SGLang: A100 and H200 are both partial, but
for different reasons. The committed viewer data now preserves that difference.

## Evaluation Run

The machine statuses are derived from
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`:

- MPK: A100 pass, H200 pass.
- VDCores: A100 pass, H200 pass.
- vLLM: A100 partial and H200 partial due missing `vllm` module.
- SGLang: A100 partial due source benchmark import failures, H200 partial due
  missing `sglang`/`orjson` import path.
- ThunderKittens: A100 pass, H200 pass.

The validation command is:

```bash
.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

Machine-level readiness is still setup evidence, not benchmark completion.
MPK, VDCores, vLLM, SGLang, and the full ThunderKittens sweeps still need
captured raw runs before their paper rows can be promoted to current result
evidence.
