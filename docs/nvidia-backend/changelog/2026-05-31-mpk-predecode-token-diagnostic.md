# 2026-05-31 MPK Predecode Token Diagnostic

## Code And Data Changed

- Added the MPK predecode memcheck execution attempt
  `mpk_qwen3_0p6b_snapshot_pointer_patch_predecode_memcheck_h200`.
- Added the raw summary artifact under
  `tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/token1_memcheck_predecode/`
  for local inspection.
- Regenerated `paper_readiness_audit.json`,
  `paper_readiness_work_queue.json`, and `goal_progress.json` so the latest
  MPK blocker reflects the generated token state.
- Updated the review-artifact tests to require the predecode attempt and its
  `first_generated_token_id == -1` evidence.

## Architecture Quality

The previous patched memcheck run proved that assigning
`paged_kv_indices_snapshot` clears the invalid global write, but the remaining
`OverflowError` did not identify whether token export or token production was
bad. The temporary predecode instrumentation in the ignored MPK baseline clone
dumps `generated_ids` before `tokenizer.decode`.

That dump shows the persistent run reaches step `39`, prompt length `39`, and
one generated token with id `-1`. The next MPK architecture question is
therefore the persistent argmax/output-token path under sanitizer, not the
already-explained snapshot pointer handoff. No upstream repository was edited
or pushed; the diagnostic patch remains only under `tmp/`.

## Evaluation Run

The H200 command was:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  compute-sanitizer --tool memcheck --target-processes all \
  python demo/qwen3/demo.py --model Qwen/Qwen3-0.6B \
    --max-new-tokens 1 --max-seq-length 128 \
    --max-num-batched-requests 1 --max-num-batched-tokens 1 \
    --ignore-eos --use-mirage
```

The run produced `tokens.json.predecode.json` before failing in
`tokenizer.decode`. The important fields are:

```json
{
  "step": 39,
  "prompt_length": 39,
  "generated_length": 1,
  "generated_token_ids": [-1]
}
```

The focused review tests passed after the data refresh:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_readiness_audit_matches_current_viewer_data or paper_readiness_work_queue_matches_current_audit or benchmark_viewer_data_contracts_are_complete'
```

## Remaining Gaps

- The snapshot-pointer fix and predecode dump are local tmp baseline patches.
- The sanitized MPK run still exits nonzero because generated token id `-1`
  is invalid for tokenizer decode.
- Paper-grade MPK evidence still needs a reproducible patch path, valid token
  output under sanitizer, scheduler/resource-policy metrics, and latency rows.
