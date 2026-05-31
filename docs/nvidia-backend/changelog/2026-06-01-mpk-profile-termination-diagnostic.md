# 2026-06-01 MPK Profile Termination Diagnostic

## Code And Data Changed

- Added `mpk_qwen3_0p6b_profile_termination_diagnostic_h200` to the
  benchmark viewer execution-attempt data.
- Added the carried MPK profile-termination diagnostic patch under
  `docs/nvidia-backend/baseline-patches/`.
- Regenerated paper-readiness audit, work-queue, and goal-progress artifacts.
- Updated review-artifact tests so the latest MPK persistent-scheduler
  execution attempt is the passing profile-termination diagnostic.

Raw H200 artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/profile-termination-diagnostic-fa357d52/
```

## Architecture Quality

This slice turns the MPK `--profiling` token-corruption symptom into a concrete
source-level cause. In offline `prepare_next_batch`, `MPK_ENABLE_PROFILING`
forced the request-done branch to `true`, so the profiled run finalized after
the first batch even though the bounded decode workload expected two generated
tokens.

The diagnostic patch removes only that profile-only early termination and
keeps `-DMPK_ENABLE_PROFILING` enabled. With that change, the real-profiler
variant preserves bounded-decode correctness and exports a non-empty Perfetto
trace. The viewer marks this execution diagnostic `pass`, while the paper
claim remains blocked until scheduler/resource/latency rows are imported into
paper-grade result records.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 run used carried snapshot-pointer, predecode, profiler-export, and
profile-termination diagnostic patches:

```bash
CUDA_HOME=/usr/local/cuda-12.8 PATH=$CUDA_HOME/bin:$PATH \
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

- all-profiler-macros no-op status: `0`;
- all-profiler-macros no-op saved `generate_length`: `2`;
- all-profiler-macros no-op predecode `step`: `40`;
- all-profiler-macros no-op trace size: `89` bytes;
- real-profiler status: `0`;
- real-profiler saved `generate_length`: `2`;
- real-profiler predecode `step`: `40`;
- real-profiler trace size: `16872606` bytes.

An initial SSH run failed before compilation because `nvcc` was not on the
non-interactive shell `PATH`. The rerun set `CUDA_HOME` and `PATH` explicitly,
matching the remote-evaluation policy.

## Remaining Gaps

- MPK scheduler/resource/latency rows are still not imported into paper-grade
  viewer result records.
- The profile-termination patch is diagnostic evidence, not an upstreamed
  baseline fix.
- The next MPK slice should parse the Perfetto trace and kernel metadata into
  comparable scheduler/resource/latency result rows for the bounded workload.
