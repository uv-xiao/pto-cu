# 2026-06-01 MPK Native Token Viewer Import

## Code And Data Changed

- Added `mpk_native_token_capture.py` to normalize an MPK native
  `demo/qwen3/demo.py --save-tokens` artifact into paper-baseline raw JSON.
- Added the `mpk_qwen3_native_token_bringup` run contract with explicit
  native-only wording and expected artifacts.
- Added a separate `mpk_native_qwen3_0p6b_token2_bringup` serving workload and
  command-plan filter so the native row does not inherit the full MPK paper
  batch ladder.
- Imported the H200 native Qwen3-0.6B two-token artifact into
  `results.json` as an `llm_serving_decode` MPK viewer row.
- Updated the LLM-serving matrix from `planned_no_results` to
  `partial_current_capture`, and narrowed its missing evidence to the
  remaining same-workload paper gaps.
- Refreshed run readiness, the paper-readiness audit, the work queue, and goal
  progress from the updated viewer data.

## Architecture Quality

The imported row is deliberately scoped as native torch bring-up evidence. It
does not replace the existing `mpk_qwen3_native_vs_persistent` paper run, and
it does not claim MPK persistent-kernel correctness or scheduler evidence. The
viewer now exposes the successful MPK native H200 artifact while the audit
continues to block paper readiness on the persistent-kernel and matching
same-workload baseline rows.

## Evaluation Run

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/
```

The source artifact is the prior H200 native run:

```text
native-token2.json
native-token2.log
```

It was normalized with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/mpk_native_token_capture.py \
    --input tmp/cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/native-token2.json \
    --output tmp/cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/native-token2-paper-results.json \
    --machine bizhaoh200 --pto-commit 61af3f01
```

Then imported with `paper_baseline_results_update.py`, producing
`native-token2-viewer-result-records.json` and a viewer result containing
prompt tokens, decode tokens, end-to-end latency, per-output-token latency,
and output-token throughput.

## Remaining Gaps

- The MPK persistent-kernel path remains the paper-critical blocker.
- The native bring-up row uses Qwen3-0.6B and a two-token decode, so it is not
  comparable to the Qwen3-8B repeated serving rows.
- LLM-serving paper readiness still requires same-workload MPK persistent,
  VDCores gated Llama or approved public-model replacement, vLLM, SGLang,
  ThunderKittens-family, and PTO rows.
