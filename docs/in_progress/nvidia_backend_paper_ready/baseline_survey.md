# Baseline Source Survey

This page is the stable entrypoint for paper-baseline source readiness. The
full survey is split into focused files so MPK, VDCores, vLLM, SGLang,
ThunderKittens, and PTO comparison state can be reviewed independently while
preserving the original `baseline_survey.md` path used by review guards.

## Local Source State

| System | Local source | Review notes |
| --- | --- | --- |
| MPK | `tmp/baselines/mirage-mpk` | upstream `mirage-project/mirage`, serving entrypoints include `demo/qwen3/demo.py` and `benchmark/benchmark_serving.py` |
| VDCores | `tmp/baselines/vdcores` | upstream `vdcores/vdcores`, Qwen schedules under `app/python/qwen3/` and `app/python/qwen3_1p7b/` |
| vLLM | `tmp/baselines/vllm` | source includes `vllm bench serve` and `vllm bench throughput` paths |
| SGLang | `tmp/baselines/sglang` | source includes `bench_serving`, offline throughput, and one-batch benchmark paths |
| ThunderKittens | `tmp/baselines/thunderkittens` | source includes kernel Makefiles, correctness tests, and benchmark scripts |

Committed viewer data mirrors this source state in
`docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json`,
`docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json`, and
`docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json`. The current
serving policy IDs are `mpk_offline_decode` and `vdcores_offline_decode`.

## Survey Map

- [Source state and serving policy](baseline_survey/source-state-and-policy.md)
- [MPK and VDCores notes](baseline_survey/mpk-and-vdcores.md)
- [vLLM and SGLang notes](baseline_survey/serving-frameworks.md)
- [ThunderKittens notes](baseline_survey/thunderkittens.md)
- [PTO comparison mapping and actions](baseline_survey/pto-comparison-and-actions.md)

## Review Contract

Use this survey to prove that external baseline sources are pinned under
`tmp/`, that source-specific run commands are discoverable, and that viewer
contracts expose the same paper-baseline readiness state without relying on
private shell history.
