# 2026-05-31 ThunderKittens Serving Capture

## Code And Data Changed

- Extended `thunderkittens_mha_capture.py` so the bounded H100 MHA wrapper can
  emit normalized `llm_serving_decode` raw rows with serving-policy metadata.
- Updated `paper_serving_command_plan.py` so ThunderKittens serving commands
  pass the paper-baseline run ID, benchmark ID, serving workload ID, prompt
  tokens, decode tokens, and padded kernel shape.
- Preserved `batch_size`, `prompt_tokens`, and `decode_tokens` in
  `paper_baseline_viewer_export.py` so the HTML viewer shows the serving
  policy used by imported baseline rows.
- Extended the benchmark-viewer data guard to allow the
  `partial_controlled_results` serving-policy status.
- Imported five H200 ThunderKittens rows into `results.json` and promoted
  `thunderkittens_decode_attention_tile` to `imported_to_viewer`.
- Regenerated `serving_command_plan.json`, `paper_readiness_audit.json`,
  `paper_readiness_work_queue.json`, and `goal_progress.json`.

## Architecture Quality

The capture remains a repo-owned wrapper around the cloned ThunderKittens
source under `tmp/baselines/`; no upstream source was modified. The H100 MHA
kernel requires at least `n=256` for a nonzero launch grid, so the VDCores
64-token decode policy is represented as a padded `B,1,256,64` kernel shape.
The raw and viewer records keep the serving contract explicit with
`prompt_tokens=128`, `decode_tokens=64`, batch size, repeat policy, correctness,
and raw artifact root.

## Evaluation Run

H200 command family:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py \
  --baseline-dir tmp/baselines/thunderkittens/kernels/attention/mha_h100 \
  --output tmp/cuda-backend/paper-baselines/serving-runs/thunderkittens/vdcores_offline_decode/thunderkittens-mha-batch<B>.json \
  --machine <h200-host> --pto-commit f7662782 --cuda-toolkit 12.8 \
  --paper-baseline-run-id thunderkittens_decode_attention_tile \
  --benchmark-id llm_serving_decode \
  --serving-workload-id vdcores_offline_decode \
  --prompt-tokens 128 --decode-tokens 64 \
  --shape <B>,1,256,64 --warmup 5 --repeats 20 --causal
```

Captured batch sizes were `1,2,4,8,16`. All rows passed PyTorch
scaled-dot-product attention correctness with 20 timed CUDA-event repeats. The
imported p50 end-to-end latencies were `32127 ns`, `28767 ns`, `32384 ns`,
`32320 ns`, and `28799 ns`; imported throughput was `1992093`, `4449542`,
`7905138`, `15841584`, and `35556790` tokens/s respectively.

## Remaining Gaps

This is a controlled serving-family tile baseline, not a full Llama/Qwen
serving benchmark. The LLM-serving paper claim still needs MPK, VDCores, vLLM,
SGLang, and full PTO serving captures before it can be marked paper-ready.
