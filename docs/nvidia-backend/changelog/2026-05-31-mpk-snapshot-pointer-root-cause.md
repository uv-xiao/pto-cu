# 2026-05-31 MPK Snapshot Pointer Root Cause

## Code And Data Changed

- Added MPK execution-attempt records for a local baseline patch that assigns
  `meta_tensors[10]` to
  `RuntimeConfig::paged_kv_indices_snapshot`.
- Regenerated `paper_readiness_audit.json`,
  `paper_readiness_work_queue.json`, and `goal_progress.json` so the latest
  MPK blocker is the patch reproducibility and sanitizer decode gap, not the
  already-explained null write.
- Updated the review-artifact tests to require the patched MPK diagnostic as
  the latest `mpk_persistent_scheduler_trace` execution attempt.

## Architecture Quality

The MPK one-token failure is now traced to a concrete baseline runtime-config
handoff bug. The local MPK clone allocates and passes
`paged_kv_indices_snapshot`, and the C++ header documents
`meta_tensors[10]`, but `init_persistent_kernel` did not assign that pointer
into `global_runtime_config`. A one-line local patch clears the prior invalid
global write through address `0x0`.

This does not make MPK paper-ready yet. It turns an opaque illegal-address
failure into an actionable baseline patch and exposes the next reviewer-visible
gate: make the patch reproducible and make the sanitized run reach token
export instead of failing in `tokenizer.decode` with `OverflowError`.

## Evaluation Run

The unpatched memcheck source evidence is:

```bash
compute-sanitizer --tool memcheck --target-processes all \
  python demo/qwen3/demo.py --model Qwen/Qwen3-0.6B \
    --max-new-tokens 1 --max-seq-length 128 \
    --max-num-batched-requests 1 --max-num-batched-tokens 1 \
    --ignore-eos --use-mirage
```

The patched H200 smoke passed with exit status `0` and saved token output:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python demo/qwen3/demo.py --model Qwen/Qwen3-0.6B \
    --max-new-tokens 1 --max-seq-length 128 \
    --max-num-batched-requests 1 --max-num-batched-tokens 1 \
    --ignore-eos --use-mirage
```

The patched H200 memcheck reached kernel launch without invalid global access
records, but the application still exited nonzero while decoding token IDs:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  compute-sanitizer --tool memcheck --target-processes all \
  python demo/qwen3/demo.py --model Qwen/Qwen3-0.6B \
    --max-new-tokens 1 --max-seq-length 128 \
    --max-num-batched-requests 1 --max-num-batched-tokens 1 \
    --ignore-eos --use-mirage
```

Focused review tests passed after the data refresh:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_readiness_audit_matches_current_viewer_data or paper_readiness_work_queue_matches_current_audit or benchmark_viewer_data_contracts_are_complete'
```

## Remaining Gaps

- The MPK patch is local under `tmp/baselines/mirage-mpk`; it has not been
  upstreamed and must not be pushed to upstream from this project.
- The sanitized patched run still reports one sanitizer error summary because
  `cudaDeviceSetLimit` returns `cudaErrorInvalidValue`, and the demo then
  raises `OverflowError` during tokenizer decode.
- MPK cannot be imported as paper-grade persistent-device evidence until the
  patch is reproducible, sanitized execution reaches token export, and
  scheduler trace/resource-policy/latency metrics are captured.
