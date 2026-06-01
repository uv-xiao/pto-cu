# 2026-06-01 MPK Qwen3 8B Persistent Import

## Code And Data Changed

- Added `mpk_qwen3_persistent_capture.py` to normalize MPK Qwen3-8B
  persistent-kernel token artifacts into the paper-baseline viewer schema.
- Imported the H200 MPK persistent decode-1024 row for
  `mpk_qwen3_native_vs_persistent` into `results.json`.
- Marked `mpk_qwen3_native_vs_persistent` as `imported_to_viewer`, replaced
  its expected artifacts with the actual Qwen3-8B run outputs, and added a
  matching execution-attempt record.
- Updated the LLM-serving matrix and work queue so MPK persistent Qwen3-8B is
  no longer listed as missing evidence. PTO full serving, VDCores full
  serving, and ThunderKittens-family full serving remain open.
- Updated serving-workload notes to distinguish native Qwen3-0.6B bring-up
  from the imported Qwen3-8B persistent-kernel row.

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/
```

## Architecture Quality

The importer keeps MPK evidence reviewable without changing or pushing
upstream MPK. It validates that both native and persistent runs exited with
status 0, checks the expected `torch` and `mpk` modes, requires positive
prompt/decode lengths, and preserves the raw artifact root in the viewer row.

The row is intentionally labeled with an asynchronous timing caveat. The MPK
demo reports one combined prefill+decode per-token value around persistent
kernel launch, not a final repeated latency distribution with separate TTFT
and ITL measurements.

## Evaluation Run

The H200 run used cached `Qwen/Qwen3-8B` artifacts with:

- `CUDA_VISIBLE_DEVICES=0`;
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- batch size 1;
- actual prompt length 39 tokens;
- `max_seq_length=1063`, producing 1024 generated tokens.

The persistent log shows MPK initialization, `compute_90a` compilation through
`nvcc`, and successful token save:

```text
tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/persistent-batch1-decode1024.log
```

The normalized raw import is:

```text
tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/paper-baseline-results.json
```

Imported with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py \
    tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/paper-baseline-results.json \
    --artifact-root tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/ \
    --viewer-output tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/viewer-result-records.json
```

## Remaining Gaps

- Repeat MPK Qwen3-8B across the full batch ladder before making final MPK
  latency-distribution claims.
- Import matching PTO persistent-device full-serving, VDCores full-serving,
  and ThunderKittens-family full-serving rows for the shared Qwen3-8B serving
  policies.
