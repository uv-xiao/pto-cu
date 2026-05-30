# 2026-05-31 Serving Policy

## Code And Data Changed

Added `docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json` as
the review-facing contract for LLM serving workloads. The benchmark viewer now
loads that file, renders a Serving Policies tab, and shows serving policy IDs
beside paper-baseline reproduction runs.

The paper-baseline run data now links MPK, VDCores, vLLM, SGLang, and
ThunderKittens-family runs to serving policy IDs. The `llm_serving_decode`
benchmark and paper-evaluation matrix now refer to the selected policy file
instead of leaving model, prompt, decode, and batch settings undefined.

## Architecture Quality

The policy separates MPK-comparable and VDCores-comparable serving workloads
because the source papers use different decode lengths and context policies.
This avoids forcing incompatible paper baselines into one vague row while
still making every future raw result point to an explicit policy ID.

The validator now checks that serving policies have stable IDs, source
evidence, primary and bring-up models, prompt/decode policies, batch sizes,
hardware targets, required metrics, blockers, and valid links to baseline run
records. This keeps the viewer data human-readable and machine-checked.

## Evaluation Run

No new performance run was made in this slice. The evaluation work was source
and contract extraction from existing local artifacts:

- MPK paper text under
  `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt`;
- VDCores paper text under `tmp/sources/arxiv-2605.03190-vdcores.txt`;
- MPK Qwen3 demo source under `tmp/baselines/mirage-mpk/demo/qwen3/demo.py`;
- VDCores Llama and Qwen scheduling notes under `tmp/baselines/vdcores/`.

The selected policies are:

- `mpk_offline_decode`: Qwen3-8B primary, Qwen3-1.7B bring-up, 64 prompt
  tokens, 1024 decode tokens, offline batch sizes 1, 2, 4, 8, and 16.
- `vdcores_offline_decode`: Qwen3-8B cross-paper target, Llama-3.1-8B current
  VDCores demo path, 128 context tokens, 64 decode tokens, offline batch sizes
  1, 2, 4, 8, and 16.

The contract is verified by the benchmark-viewer validator:

```bash
.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

The policies are selected, but no MPK, VDCores, vLLM, SGLang, or PTO serving
result has been promoted from them yet. The next evaluation slices must install
or build those runtimes on H200, run the selected policy commands, and import
raw JSON into the viewer result schema.
