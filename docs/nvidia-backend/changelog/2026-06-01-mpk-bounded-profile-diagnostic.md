# 2026-06-01 MPK Bounded Profile Diagnostic

## Code And Data Changed

- Added `mpk_qwen3_0p6b_bounded_profile_diagnostic_h200` to the benchmark
  viewer execution-attempt data.
- Added the carried MPK profiler diagnostic patch under
  `docs/nvidia-backend/baseline-patches/`.
- Regenerated paper-readiness audit, work-queue, and goal-progress artifacts.
- Updated review-artifact tests so the latest MPK persistent-scheduler
  blocker is the profiling/correctness conflict.

Raw H200 artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/bounded-profile-9668bcef/
```

## Architecture Quality

This slice moves the MPK scheduler trace gap from a vague
`scheduler/resource/latency` placeholder to a concrete baseline-runtime
failure mode. The bounded non-profiled MPK run honors the requested decode
length, but enabling MPK profiling changes the observed token state.

The diagnostic patch has three baseline-side changes:

- create missing Perfetto block/group tracks instead of crashing on
  `KeyError: (16, 0)`;
- synchronize CUDA before exporting the profiler buffer;
- increase the demo profiler tensor to `30000 * 256` `uint64` entries.

Those changes let MPK export a Perfetto trace, but they do not make the
profiled run correct. The viewer therefore keeps the attempt `partial` and
does not import a paper-grade result row.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 run used the carried snapshot-pointer, predecode, and profiler
diagnostic patches:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  <repo-venv>/python demo/qwen3/demo.py \
  --model Qwen/Qwen3-0.6B \
  --max-new-tokens 2 \
  --max-seq-length 41 \
  --max-num-batched-requests 1 \
  --max-num-batched-tokens 1 \
  --ignore-eos \
  --use-mirage \
  --profiling \
  --trace-name <artifact-root>/<variant>/mpk_bounded_decode \
  --output-dir <artifact-root>/<variant>/build \
  --save-tokens <artifact-root>/<variant>/tokens.json
```

Observed result:

- unpatched profile export status: `1`;
- unpatched profile export blocker: `KeyError: (16, 0)`;
- profiler-export patch status: `0`;
- synchronized-export patch status: `0`;
- large-buffer patch status: `0`;
- exported Perfetto trace size in the large-buffer run: `454969` bytes;
- profiled saved `generate_length`: `0`;
- profiled predecode `step`: `1`;
- profiled predecode generated length: `-37`.

The focused tests were run before the viewer record existed and failed because
`mpk_qwen3_0p6b_bounded_profile_diagnostic_h200` was absent. After adding the
record and regenerating derived artifacts, the focused tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_matches_current_viewer_data \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  -q
```

Result: `3 passed`.

## Remaining Gaps

- MPK profiling and correctness do not yet coexist for the bounded-decode
  workload.
- The exported trace proves that scheduler trace export can be unblocked with
  carried patches, but it is not valid paper-grade evidence because token
  progress is wrong.
- The next MPK slice should identify why `--profiling` changes persistent
  token/step state before importing scheduler, resource-policy, or latency
  rows.
