# ThunderKittens Serving Tile Refresh

## Code And Data Changed

- Refreshed the five H200 ThunderKittens `vdcores_offline_decode` serving-tile
  viewer rows from raw artifacts under
  `tmp/cuda-backend/paper-baselines/serving-runs/thunderkittens/`.
- Used the SSH fallback path after remote `git fetch origin
  goal/nvidia-paper-ready` failed on the H200 host with the known GitHub SSH
  key issue.
- Kept the result kind as `paper_baseline_serving_tile_capture`, preserving
  `serving_coverage=controlled_attention_tile_proxy`.

## Architecture Quality

The committed viewer rows now point at a current-branch H200 capture for the
same planned batch ladder, while the paper-readiness work queue still keeps
ThunderKittens full-serving evidence open instead of treating the proxy row as
paper-ready serving coverage.

## Evaluation Run

Focused verification passed:

- H200 `thunderkittens_mha_capture.py` for batches `1,2,4,8,16`, each with
  `5` warmups and `20` timed CUDA-event repeats.
- `validate_benchmark_viewer_data.py`
- `validate_nvidia_changelog.py`
- `check_nvidia_review_ready.py`
- `pytest -q -k 'thunderkittens_capture_builds_serving_decode_result or
  benchmark_viewer_has_json_backed_review_data'`
- `git diff --check`

## Remaining Gaps

ThunderKittens still needs a full-serving Qwen/Qwen3-8B row beyond the
controlled MHA tile proxy before the LLM-serving paper-baseline claim can be
promoted.
