# CUDA Backend Status: Review Gate Policy

## Review Gate Policy

Repository GitHub Actions are intentionally closed while the standalone
NVIDIA backend ultimate goal is active. This prevents inherited Ascend a2a3/a5
jobs or stale repository checks from blocking CUDA-only progress. The checked
in branch keeps no runnable workflow YAML under `.github/workflows/`.

The review path is local and explicit:

- `docs/ci.md` defines the closed-CI policy and reopening procedure.
- `docs/ci/nvidia-manual-review.workflow.yml` preserves a future manual review
  recipe outside GitHub's runnable workflow directory.
- `.agents/checks/check_nvidia_review_ready.py` fails if runnable workflow
  YAML reappears under `.github/workflows/` while the closed-CI policy is
  active.
- `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`, and
  `check_nvidia_review_ready.py` are the branch-level document, data, and
  policy gates for reviewable CUDA progress.

Verification evidence:

```bash
test ! -d .github/workflows || find .github/workflows -maxdepth 1 \
  -type f \( -name '*.yml' -o -name '*.yaml' \)
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/check_nvidia_review_ready.py
```

Result: `.github/workflows/` has no runnable workflow YAML, and the NVIDIA
review guard passed. The guard also checks that the archived manual review
recipe remains present under `docs/ci/`.

Future CUDA hardware CI is useful infrastructure, but it is not a backend
implementation blocker for this standalone branch. Reopening automatic CI
requires updating `docs/ci.md`, the review guard, and a changelog report in
the same change.
