# 2026-05-31 ThunderKittens Dependency Probe

## Code And Data Changed

Extended the ThunderKittens paper-baseline probe with the Python modules
required by the selected `kernels/attention/mha_h100` PyTorch-extension path:
`torch`, `pybind11`, `numpy`, `pandas`, `matplotlib`, and `tqdm`. The
benchmark-viewer validator now rejects ThunderKittens probe data that omits
those dependencies.

## Architecture Quality

The probe now reflects the actual build and run surface rather than only
source-file existence. This keeps the viewer from showing a paper baseline as
ready when H200 can parse the scripts but cannot build or execute the selected
kernel path.

## Evaluation Run

The focused review-data test first failed because the ThunderKittens probe had
no `python_module` checks for the selected kernel dependencies:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k benchmark_viewer_has_json_backed_review_data
```

The initial H200 environment probe showed `torch`, `pybind11`, and `tqdm`
were missing for the selected ThunderKittens run path, while `numpy`,
`pandas`, and `matplotlib` were present. Those missing modules were then
installed into the remote pto-cu project venv and the paired probe artifact was
refreshed under
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-67c5c655/`.

## Remaining Gaps

ThunderKittens is still not a full captured performance baseline. The next
step is to run the full correctness plus benchmark commands and export repeat
statistics into the viewer result schema.
