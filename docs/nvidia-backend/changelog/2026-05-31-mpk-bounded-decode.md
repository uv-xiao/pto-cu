# 2026-05-31 MPK Bounded Decode

## Code And Data Changed

- Added `mpk_qwen3_0p6b_bounded_decode_h200` to the benchmark viewer
  execution-attempt data.
- Updated the persistent-device scheduler matrix gap from generic matching
  workload metadata to the remaining scheduler/resource/latency import.
- Regenerated paper-readiness audit, work-queue, and goal-progress artifacts
  from the viewer data.
- Updated review-artifact tests so the latest MPK persistent-scheduler
  attempt is the bounded-decode run.

Raw H200 artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/bounded-decode-901ec9c1/
```

## Architecture Quality

This slice resolves the immediate MPK decode-length mismatch without changing
upstream MPK or pto-cu runtime code. MPK's persistent device loop terminates on
`max_seq_length` or EOS, not on `max_new_tokens`, so the matching-workload
smoke bounds `max_seq_length` to the observed prompt length plus the requested
decode token count.

The result keeps the comparison conservative: the viewer records that bounded
decode now honors one-token and two-token requests, while still marking the
attempt `partial` because paper-grade scheduler/resource/latency distributions
are not imported and the run uses carried local baseline patches.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 run used the carried local MPK snapshot-pointer and predecode patches:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python demo/qwen3/demo.py \
  --model Qwen/Qwen3-0.6B \
  --max-new-tokens <1|2> \
  --max-seq-length <40|41> \
  --max-num-batched-requests 1 \
  --max-num-batched-tokens 1 \
  --ignore-eos \
  --use-mirage \
  --output-dir <artifact-root>/max_new_tokens_<N>_max_seq_<S>/build \
  --save-tokens <artifact-root>/max_new_tokens_<N>_max_seq_<S>/tokens.json
```

Observed result:

- requested max-new-tokens: `[1, 2]`;
- bounded max sequence lengths: `[40, 41]`;
- observed prompt lengths: `[39, 39]`;
- observed generated lengths: `[1, 2]`;
- observed predecode steps: `[39, 40]`;
- task counts: `[7261, 7261]`;
- event counts: `[1870, 1870]`;
- exit status: both runs exited 0.

The focused test was run before the viewer record existed and failed because
`mpk_qwen3_0p6b_bounded_decode_h200` was absent from
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

- MPK still needs scheduler/resource/latency distributions imported for the
  bounded-decode workload before it can be a paper-grade persistent-device
  scheduler baseline.
- The bounded-decode path depends on carried MPK baseline patches recorded
  under `docs/nvidia-backend/baseline-patches/`.
- The next MPK slice should capture the same bounded workload with
  scheduler-overhead timing, dispatch/resource metadata, and a complete raw
  result import path.
