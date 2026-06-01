# 2026-06-01 vLLM H200 MPK Serving Capture

## Code And Data Changed

- Imported an H200 vLLM `Qwen/Qwen3-8B` serving row for the MPK-comparable
  batch-1 policy: prompt tokens `64`, decode tokens `1024`, and concurrency
  `1`.
- Added a vLLM execution-attempt record that points at the preserved raw
  server and benchmark artifacts under `tmp/`.
- Kept the vLLM paper-baseline run partial because the MPK-comparable sweep,
  repeated samples, and cross-method comparisons are still incomplete.

## Architecture Quality

The benchmark viewer now has vLLM evidence for both serving workload shapes:
the VDCores-comparable `128 -> 64` sweep and the MPK-comparable
`64 -> 1024` bring-up. The imported result row uses the same serving-result
schema as the VDCores-comparable rows, so reviewers can compare TTFT, ITL,
throughput, request shape, and raw artifact location without special casing.

## Evaluation Run

The H200 checkout was refreshed through the documented tree-sync fallback.
Repository Actions stayed disabled, and no upstream repository was edited or
pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-mpk-qwen3-8b-batch1-7e939170/`

Serving result:

- model: `Qwen/Qwen3-8B`
- GPU: H200 on `bizhaoh200`
- request shape: input tokens `64`, output tokens `1024`, batch/concurrency
  `1`
- completed requests: `1`
- failed requests: `0`
- mean TTFT: `63.15663317218423 ms`
- mean ITL: `5.947963752941331 ms`
- output throughput: `166.5269980770844 tokens/s`
- request throughput: `0.16262402155965272 req/s`

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result:

```text
benchmark viewer data validation passed
```

## Remaining Gaps

- Capture MPK-comparable vLLM batch `2`, `4`, `8`, and `16`.
- Repeat vLLM serving samples for variance and paper confidence intervals.
- Capture the matching MPK serving path and import it into the viewer.
- Compare the same serving workload against PTO persistent-device once that
  runner is available.
