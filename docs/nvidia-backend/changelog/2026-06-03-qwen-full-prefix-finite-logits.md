# 2026-06-03 Qwen Full-Prefix Finite Logits

## Code And Data Changed

- Added no runtime code in this report.
- Captured fresh A100 resource-backed MPK evidence after the MLP residual
  binding fix, extending the verified prefix boundary from eight layers to the
  full 36-layer Qwen/Qwen3-8B DAG with full projection and full logits
  windows.
- Captured a local comparison record against the current Hugging Face
  reference artifact. The comparison still fails token/logit agreement, so the
  new evidence is not eligible for benchmark-viewer import as full-serving
  correctness.
- Updated the Qwen full-serving remaining-gap status to separate the closed
  finite-logits blocker from the still-open Hugging Face agreement blocker.

## Architecture Quality

The full-prefix run exercises the same generated persistent-device task graph
shape as the paper-readiness path: token pointers, resident safetensors, live
activation workspace, KV cache, device token feedback, final norm, and full
vocabulary logits. The run proves that the previous row-0 NaN and empty top-k
failure no longer blocks full-prefix MPK execution.

The strict promotion gate remains unchanged. Internal diagnostic projection
agreement proves launch and readout consistency for the generated task bodies,
but paper-readiness still requires full-model token/logit agreement against
the Hugging Face Qwen/Qwen3-8B reference for both MPK and VDCores
policy-length rows.

## Evaluation Run

Full 36-layer MPK prefix:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --workspace-cuda-live \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 --resource-backed-repeat-runs 1 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols full \
  --resource-backed-logits-active-cols full --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-full-prefix-after-mlp-residual-fix-420s.json
```

Result:

- 255 graph tasks scheduled, 255 completed, zero scheduler errors.
- No row-0 non-finite activations were reported.
- Full logits buffer checked: 2,430,976 finite logits out of 2,430,976
  sampled logits, with all sampled logits nonzero.
- Row-0 top-k was populated. The top token was `220` with logit `6.285215`.
- The internal diagnostic logits reference passed over 3,904 checked
  elements, 16 checked rows, and `max_abs_error=6.04e-06`.
- Device feedback observed the sampled token: `sampled_token_id=220`,
  `output_ids_value=220`, and `next_input_value=220`.

The local comparison record is:

```text
tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-full-prefix-hf-token-comparison.json
```

It compares the full-prefix artifact above with:

```text
tmp/cuda-backend/qwen-full-model-reference-mpk-1step-2026-06-03/reference.json
```

The comparison status is `fail`: PTO selected token `220`, while the Hugging
Face reference selected token `151667` with top logit `36.0` at decode
position 17.

## Remaining Gaps

The old full-prefix finite-logits blocker is closed for MPK on the local A100
evidence path. Full-serving correctness remains open because PTO token/logit
output still diverges from the Hugging Face reference. Next work should focus
on model-correct Qwen math fidelity across prefill, KV state, attention, and
decode semantics, then re-run MPK and VDCores policy-length rows only after
the Hugging Face comparison passes.
