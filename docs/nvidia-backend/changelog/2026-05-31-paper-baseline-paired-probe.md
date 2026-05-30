# 2026-05-31 Paper Baseline Paired Probe

## Code And Data Changed

Added `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py`
to run the committed paper-baseline readiness probe locally and on the remote
H200 host. The paired runner builds local, remote, tree-sync, and copy-back
commands with the same remote fallback contract used by the CUDA paired
benchmark scripts. It also syncs `tmp/baselines/` separately so remote probes
can see the paper-baseline source checkouts without copying generated
`tmp/cuda-backend/` outputs.

## Architecture Quality

The paired probe keeps paper-baseline readiness evidence separate from full
benchmark execution. It records local A100 and remote H200 probe artifacts
under one `tmp/` directory and writes command examples so reviewers can
reproduce the path without relying on private shell history.

## Evaluation Run

The focused TDD red check first failed because
`paper_baseline_pair_probe.py` and this changelog report were missing:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_pair_probe or review_policy_changelog'
```

After implementation, the focused review artifact test, remote-evaluation
validator, and NVIDIA review guard passed. The paired fallback probe was also
run with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py \
    --sync-remote-tree
```

It wrote `a100-probe.json`, `h200-probe.json`, and
`paired-probe-summary.json` under
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-bdec348b/`. A100
passed MPK, VDCores, SGLang, and ThunderKittens and remained partial for vLLM.
H200 remained partial for MPK, VDCores, vLLM, SGLang, and ThunderKittens
because the required runtime Python packages are not installed there.

## Remaining Gaps

This report adds the paired probe harness. It does not claim that MPK,
VDCores, vLLM, SGLang, or ThunderKittens have been built or benchmarked on
H200; those raw performance captures remain future paper-evaluation slices.
