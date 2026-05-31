# 2026-05-31 MPK Execution Attempts

## Code And Data Changed

- Added `paper_baseline_execution_attempts.json` to the benchmark viewer data.
- Updated the HTML viewer to show execution attempts beside each paper
  baseline's planned runs, readiness checks, and probes.
- Validated the new execution-attempt data in the benchmark viewer guard and
  focused review-artifact tests.
- Captured H200 MPK Qwen3-0.6B bring-up artifacts under
  `tmp/cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/`.
- Updated the MPK setup notes with the execution-time dependencies:
  `accelerate==1.8.0`, `transformers==4.57.1`, `protobuf`, `fastapi`,
  `uvicorn`, and `tg4perfetto`.

## Architecture Quality

Execution attempts are intentionally separate from imported paper results.
This lets reviewers see concrete H200 progress and blockers without treating
partial bring-up as paper-grade latency or scheduler evidence. The viewer now
distinguishes:

- readiness probes: source paths and imports are available;
- run readiness: planned commands and expected artifacts are well formed;
- execution attempts: commands were actually run, with logs and generated
  artifacts;
- imported results: measured records are normalized into `results.json`.

## Evaluation Run

H200 native Qwen3-0.6B bring-up passed with a bounded two-token run:

```text
Prompt length 39, generate length 1, per-token latency 476.39776611328125 ms
Saved tokens to ../../cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/native-token2.json
```

The first live fetch attempt failed because H200 could not reach
`huggingface.co`. Re-running with a shared H200 cache and offline Hugging Face
environment variables resolved model access.

MPK persistent-kernel bring-up reached megakernel generation, compilation, and
asynchronous launch on H200:

```text
[MPK INIT] Total tasks: 7261, Total events: 1870
Saved compiled kernel to: .../mpk_launcher_rank0.cpython-312-x86_64-linux-gnu.so
Finished Launching Persistent Kernel (Async)
```

The profiled path then failed in MPK profiler export with `KeyError: (16, 0)`.
The non-profiled path failed at `torch.cuda.synchronize()` with
`CUDA error: an illegal memory access was encountered`. Retrying with
`max_num_batched_tokens=8` reproduced the illegal-memory-access failure after
kernel launch.

## Remaining Gaps

- MPK persistent-kernel execution is not yet paper-grade evidence because it
  fails after launch.
- The MPK profiler exporter needs investigation for missing `(block_idx,
  group_idx)` mappings in H200 Qwen3-0.6B traces.
- The persistent megakernel needs CUDA-side debugging for the illegal memory
  access seen after launch.
- Imported MPK result rows should wait until persistent execution produces
  correctness and scheduler-overhead artifacts.
