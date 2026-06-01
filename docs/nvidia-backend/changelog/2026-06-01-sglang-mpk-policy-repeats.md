# 2026-06-01 SGLang MPK-Policy H200 Repeats

## Code And Data Changed

- Added `sglang_serving_repeats_to_paper_results.py` to aggregate repeated
  SGLang `bench_serving` and `bench_offline_throughput` captures into the
  paper-baseline importer format.
- Imported ten H200 SGLang rows for `Qwen/Qwen3-8B` under the
  `mpk_offline_decode` serving policy: online serving and offline engine modes
  for batch sizes 1, 2, 4, 8, and 16.
- Updated the paper evaluation matrix so SGLang MPK-policy evidence is no
  longer treated as missing for the shared Qwen3-8B serving policy.

## Architecture Quality

The importer keeps raw SGLang command artifacts under `tmp/` and commits only
the normalized viewer records, run contract, execution-attempt metadata, and
review-facing changelog. This preserves the rule that paper claims must point
to explicit raw evidence while keeping the committed dataset compact.

The run used the tree-sync remote-evaluation fallback for the standalone
pto-cu checkout. No upstream repository was edited, pushed, or reconfigured.

## Evaluation Run

Raw artifact root:

```text
tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-mpk-qwen3-8b-repeats-91ddaf86/
```

The H200 run used one SGLang server for `Qwen/Qwen3-8B`, prompt length 64,
decode length 1024, bfloat16, `--disable-piecewise-cuda-graph`, offline model
cache, and three samples per batch.

| Batch | Online output tok/s | Offline output tok/s |
| ----- | ------------------- | -------------------- |
| 1 | 174.146 | 171.575 |
| 2 | 339.453 | 340.278 |
| 4 | 673.744 | 670.666 |
| 8 | 1321.610 | 1316.960 |
| 16 | 2647.308 | 2649.133 |

Initial offline-throughput commands with `--dataset-name random` attempted a
network-backed ShareGPT seed fetch under Hugging Face offline mode. The rerun
used a local ShareGPT-shaped seed file in the artifact directory and completed
all offline samples.

## Remaining Gaps

- SGLang `bench_one_batch` is still not represented in the imported rows.
- Final paper readiness still depends on matching PTO full-serving, MPK
  persistent-kernel, VDCores full-serving, and ThunderKittens-family rows.
