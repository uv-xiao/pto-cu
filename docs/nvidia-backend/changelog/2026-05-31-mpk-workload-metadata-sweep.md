# 2026-05-31 MPK Workload Metadata Sweep

## Code And Data Changed

- Added `mpk_qwen3_0p6b_workload_metadata_sweep_h200` to the benchmark
  viewer execution-attempt data.
- Regenerated paper-readiness audit, work-queue, and goal-progress artifacts
  from the viewer data.
- Updated review-artifact tests so the latest MPK persistent-scheduler blocker
  is matching workload metadata, not only the earlier sanitizer token state.

Raw H200 artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/workload-metadata-901ec9c1/
```

## Architecture Quality

The persistent-device scheduler comparison now has a stricter MPK baseline
gate. The patched MPK run can exit successfully, but a successful run is not
paper-grade if its decode shape does not match the requested workload.

The sweep ran the same patched persistent demo with `--max-new-tokens 1` and
`--max-new-tokens 2`. Both runs generated 89 tokens, reached predecode
`step=127`, and built the same 7261-task, 1870-event graph with a final
`output_token` argmax-reduce task. The viewer now records that
`max_new_tokens` is not honored by this persistent demo path, so MPK rows must
not be imported as matching PTO/VDCores workload evidence yet.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 sweep used the carried local baseline patches:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python demo/qwen3/demo.py \
  --model Qwen/Qwen3-0.6B \
  --max-new-tokens <1|2> \
  --max-seq-length 128 \
  --max-num-batched-requests 1 \
  --max-num-batched-tokens 1 \
  --ignore-eos \
  --use-mirage \
  --output-dir <artifact-root>/max_new_tokens_<N>/build \
  --save-tokens <artifact-root>/max_new_tokens_<N>/tokens.json
```

Observed result:

- requested max-new-tokens: `[1, 2]`;
- observed generated lengths: `[89, 89]`;
- observed predecode steps: `[127, 127]`;
- task counts: `[7261, 7261]`;
- event counts: `[1870, 1870]`;
- exit status: both runs exited 0.

The focused test was run before the viewer record existed and failed because
`mpk_qwen3_0p6b_workload_metadata_sweep_h200` was absent from
`paper_baseline_execution_attempts.json`. After adding the record and
regenerating derived artifacts, the focused test passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  -q
```

Result: `1 passed`.

## Remaining Gaps

- MPK still has no paper-grade matching-workload persistent result for the
  scheduler-overhead comparison.
- The MPK run command or persistent demo loop must enforce the intended
  decode length and report scheduler/resource metadata for the same workload
  used by PTO and VDCores.
