# 2026-05-31 VDCores Qwen Correctness Fetch Blocker

## Code And Data Changed

- Added a VDCores H200 execution attempt for the real
  `app/python/qwen3_1p7b/sched.py --correctness` path.
- Added a raw `tmp/` summary JSON beside the synced H200 metadata, status, and
  log files so reviewers can inspect the exact failure mode.
- Extended the benchmark-viewer data validator with an explicit
  `blocked_before_model_load` execution-attempt status for pre-launch model
  fetch blockers.
- Extended the review artifact tests to require this execution attempt and to
  verify that it is not counted as a measured resource-policy result.

## Architecture Quality

The VDCores row now distinguishes three evidence states:

- source and extension readiness are available;
- synthetic schedule construction is available;
- real Qwen correctness and resource-policy measurement are still blocked
  before model load.

That distinction keeps the persistent-device scheduler claim conservative. The
paper-readiness matrix still requires a measured VDCores run with queue and
resource-policy metadata.

## Evaluation Run

On the H200 host, the VDCores baseline clone had `origin` push disabled, the
`dae.runtime` extension was present, and CUDA reported an H200 GPU. The bounded
attempt ran:

```bash
cd tmp/baselines/vdcores
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --correctness
```

The run repeatedly timed out fetching
`Qwen/Qwen3-1.7B/resolve/main/config.json` from Hugging Face before model load.
It was terminated after the repeated timeout pattern was captured. It did not
reach schedule execution, correctness checking, or resource-policy measurement.

## Remaining Gaps

- Put `Qwen/Qwen3-1.7B` in the shared H200 Hugging Face cache, or provide a
  reliable model-fetch path for the H200 evaluation host.
- Re-run VDCores correctness with offline cache settings after the model is
  available.
- Run the VDCores timing/resource-policy path and import the measured raw JSON
  into viewer result records.
