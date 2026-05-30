# 2026-05-31 Remote Evaluation Contract

## Code And Data Changed

- Added `.agents/checks/validate_remote_evaluation.py`.
- Wired the remote-evaluation validator into the NVIDIA review guard and
  focused artifact tests.
- Updated the work-preparation note to require recording the remote commit or
  tree-sync source commit for remote runs.
- Extended the shared contracts with a machine-checkable remote evaluation
  contract.

## Architecture Quality

Remote CUDA evaluation now has an enforceable fallback policy instead of a
prose-only rule. The validator imports the paired CUDA smoke, persistent smoke,
full benchmark, stream benchmark, and lifecycle matrix scripts and checks that
they expose:

- default remote Git refresh;
- explicit `rsync` tree sync;
- no remote `git fetch` or `git checkout` after tree sync;
- explicit remote CUDA and `PYTHONPATH` environment setup;
- stable sync excludes for `.venv`, build outputs, `tmp`, Python caches, and
  pytest caches.

## Evaluation Run

Expected verification for this report:

```bash
.venv/bin/python .agents/checks/validate_remote_evaluation.py

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

git diff --check
```

## Remaining Gaps

- The validator proves command construction and policy coverage; it does not
  run remote H200 jobs.
- Future remote captures should include the selected refresh path and copied
  artifact path in their raw JSON or history entries.
