# 2026-05-31 VDCores Qwen Runtime Rebuild

## Code And Data Changed

- Added three VDCores H200 execution attempts:
  - selected `dae.runtime` rebuild for the Qwen3-1.7B compute-op set;
  - rebuilt-runtime Qwen correctness launch;
  - one-layer launched stage sweep.
- Added raw `tmp/` summary JSON files for the rebuild, correctness launch, and
  stage sweep artifacts.
- Extended the review artifact tests to require the rebuilt-runtime attempts
  and to prove they remain launch-failure evidence, not paper-grade
  correctness or resource-policy results.

## Architecture Quality

The new evidence separates the VDCores blockers more precisely:

- H200 model access is now resolved by syncing `Qwen/Qwen3-1.7B` into the
  shared Hugging Face cache and running with offline cache settings.
- The selected-op `dae.runtime` rebuild is resolved by using the captured
  nine-op Qwen compute list, pinned CUTLASS headers, and the existing
  `nvcc -include cfloat` workaround.
- The remaining blocker is now a runtime device launch failure: even the
  one-layer `final_rms` launch cut fails with CUDA illegal memory access.

No VDCores source files were edited or pushed upstream. The rebuild used
generated build state in the H200 evaluation worktree and raw logs under
`tmp/`.

## Evaluation Run

The local host downloaded `Qwen/Qwen3-1.7B` into
`tmp/huggingface_cache/models--Qwen--Qwen3-1.7B`, then synced that model cache
to the H200 shared Hugging Face cache. H200 offline config loading reported
`qwen3`, hidden size `2048`, and `28` layers.

The selected runtime rebuild used:

```bash
CPATH=<cutlass-include> \
DAE_COMPUTE_OPS_FILE=<qwen-compute-ops> \
make clean pyext NVCC="nvcc -include cfloat"
```

The successful rebuild selected nine compute ops, including one dynamic GEMV
family op, installed `dae.runtime`, and verified that all required Qwen compute
ops were exported by `runtime.supported_compute_ops`.

The rebuilt correctness run used:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --correctness
```

It loaded both checkpoint shards and reached `launch_dae`, then failed with
CUDA illegal memory access.

A one-layer launch sweep ran `final_rms`, `logits`, `argmax`, `restore`, and
`full` stage cuts with `--launch`. Every stage failed at `launch_dae`; the
earliest failure is `final_rms`.

## Remaining Gaps

- Diagnose the VDCores `launch_dae` illegal memory access on the rebuilt
  Qwen3-1.7B runtime.
- Capture a passing VDCores correctness run before importing resource-policy
  or timing records into the benchmark viewer.
- Keep the paper-readiness matrix blocked until VDCores has comparable
  correctness, queue/resource-policy metadata, and latency artifacts.
