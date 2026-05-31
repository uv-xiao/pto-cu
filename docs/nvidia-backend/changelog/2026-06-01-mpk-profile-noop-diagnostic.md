# 2026-06-01 MPK Profile No-Op Diagnostic

## Code And Data Changed

- Added `mpk_qwen3_0p6b_profile_noop_diagnostic_h200` to the benchmark
  viewer execution-attempt data.
- Added the carried MPK profiler all-no-op diagnostic patch under
  `docs/nvidia-backend/baseline-patches/`.
- Regenerated paper-readiness audit, work-queue, and goal-progress artifacts.
- Updated review-artifact tests so the latest MPK persistent-scheduler
  blocker is the profile compile path changing token progress.

Raw H200 artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/profile-write-diagnostic-034bada3/
```

## Architecture Quality

This slice narrows the MPK scheduler-trace blocker. The previous diagnostic
proved that Perfetto export can be made to complete, but the profiled run
still lost decode progress. This run removed profiler event writes and then
removed all profiler macros while keeping the same `--profiling` compile path.

Both variants still produced the bad token state. That means profiler buffer
writes and Perfetto export are not sufficient explanations for the corruption.
The review-facing blocker now points at the profile compile path or profile
mode itself.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 run used the carried snapshot-pointer, predecode, profiler-export,
and profiler-no-op diagnostic patches:

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

- profiler-event no-op status: `0`;
- profiler-event no-op trace size: `1141` bytes;
- profiler-event no-op saved `generate_length`: `0`;
- profiler-event no-op predecode `step`: `1`;
- all-profiler-macros no-op status: `0`;
- all-profiler-macros no-op trace size: `89` bytes;
- all-profiler-macros no-op saved `generate_length`: `0`;
- all-profiler-macros no-op predecode `step`: `1`.

The focused tests were run before the viewer record existed and failed because
`mpk_qwen3_0p6b_profile_noop_diagnostic_h200` was absent and derived
paper-readiness artifacts still pointed at the earlier bounded-profile
attempt.

## Remaining Gaps

- The MPK profile compile path still corrupts token progress even with
  profiler event writes and init macros disabled.
- Paper-grade MPK scheduler, resource-policy, and latency rows remain blocked
  until `--profiling` preserves the same bounded-decode token state as the
  non-profiled run.
- The next MPK diagnostic should isolate what `-DMPK_ENABLE_PROFILING` or
  profile mode changes in persistent scheduler state.
