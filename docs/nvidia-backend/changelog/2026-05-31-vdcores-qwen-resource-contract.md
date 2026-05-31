# 2026-05-31 VDCores Qwen Resource Contract

## Code And Data Changed

- Switched `vdcores_resource_policy_trace` from the gated Llama demo path to
  the public `app/python/qwen3_1p7b` VDCores path.
- Updated the VDCores source-entrypoint probe to check the Qwen scheduler and
  runtime-context files on A100 and H200.
- Added a viewer execution-attempt row for the H200 Qwen3-1.7B synthetic
  schedule dry-build.
- Regenerated paper-baseline run readiness, paper-readiness audit, work queue,
  and goal-progress data. The work queue dropped from nine items to eight.

## Architecture Quality

The resource-policy run contract now separates public bring-up from final
paper-grade measurement. The dry-build proves that the Qwen schedule can be
constructed and that the selected compute-operator set is explicit, but it
does not claim correctness or latency. Those final measurements still require
real Qwen execution and import through the paper-baseline result path.

## Evaluation Run

The H200 dry-build command completed for both a bounded one-layer probe and the
full synthetic Qwen3-1.7B schedule:

```text
[dry-build] built qwen3-1.7b schedule with hidden=2048, intermediate=6144, head_dim=128, layers=28, max_seq_len=512
[dry-build] logits_epoch=3, logits_slice=50688, vocab_size=151936
[compute-ops] wrote 9 operators to tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-contract/compute_ops_full.txt
```

The paired A100/H200 source probe was refreshed under
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-61af3f01/`.

## Remaining Gaps

- `vdcores_resource_policy_trace` is ready to run, but is still
  `planned_not_run` and not imported into viewer results.
- The dry-build attempt is not paper-grade evidence because it does not launch
  the model, check correctness, or record device timing.
- The separate VDCores serving run still uses the existing Llama serving-demo
  contract and remains blocked on gated model access.
