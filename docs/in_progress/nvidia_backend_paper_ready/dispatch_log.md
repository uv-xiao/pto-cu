# NVIDIA Backend Paper-Ready Dispatch Log

This log records dispatcher-worker activity for the standalone pto-cu NVIDIA
backend goal. It is required review evidence; do not rely on private terminal
scrollback or unstated session memory.

## Logging Schema

Each entry must include:

- timestamp;
- dispatcher session or PR;
- worker id and objective;
- exact Codex command or script invocation;
- parent goal and child slice;
- branch name and PR URL or planned PR slot;
- allowed scope and files;
- dependencies and blocked assumptions;
- verification commands and results;
- merge decision and merge commit, when applicable;
- handoff summary and remaining gaps.

## Entries

### 2026-05-31 - Goal Activation And Standalone Scope

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; planned PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned preparation.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, preparation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`; PR pending.
- Allowed scope and files: `docs/in_progress/`,
  `docs/nvidia-backend/changelog/`, `.agents/checks/`,
  `tests/ut/py/test_nvidia_review_artifacts.py`, and local `tmp/` notes.
- Dependencies and blocked assumptions: user clarified that this is a
  standalone project; upstream repositories must not be modified.
- Verification commands and results: initial focused test was intentionally red
  before adding goal artifacts:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  failed because the ultimate-goal docs and changelog did not exist.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next dispatcher should finish this
  preparation PR, then dispatch child slices for viewer data expansion,
  evidence guard hardening, CUDA runtime maturity, examples, remote evaluation,
  and paper-ready baseline reproduction.

### 2026-05-31 - Baseline Source Survey

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned baseline
  source survey and viewer data expansion.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, baseline readiness slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `docs/in_progress/nvidia_backend_paper_ready/`,
  `docs/nvidia-backend/benchmark-viewer/`, `.agents/checks/`,
  `tests/ut/py/test_nvidia_review_artifacts.py`, and local `tmp/` notes.
- Dependencies and blocked assumptions: MPK and VDCores source clones are
  survey inputs only; full builds and benchmark runs remain child slices.
- Verification commands and results: the focused artifact test was run before
  implementation and failed because `paper_baselines.json` and
  `baseline_survey.md` did not exist.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next baseline worker should capture vLLM,
  SGLang, and ThunderKittens sources, then attempt MPK and VDCores build/run
  reproduction on compatible GPU hosts.

### 2026-05-31 - Paper Baseline Source Capture Completion

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned baseline
  source capture.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, baseline readiness slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `docs/in_progress/nvidia_backend_paper_ready/`,
  `docs/nvidia-backend/benchmark-viewer/`,
  `tests/ut/py/test_nvidia_review_artifacts.py`, `.agents/checks/`, and
  local `tmp/` notes.
- Dependencies and blocked assumptions: vLLM, SGLang, and ThunderKittens were
  cloned for source survey only; no framework builds or GPU evaluations were
  attempted in this slice.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  focused review artifact pytest, Python compile checks, JSON syntax checks,
  `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: all required paper baseline families now
  have source commits and inspected entry points. Next child slices should
  build and run each baseline, then write importers for raw benchmark JSON.

### 2026-05-31 - Automatic CI Disabled During Ultimate Goal

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned CI policy
  update.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, review-flow slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `.github/workflows/ci.yml`,
  `.agents/checks/check_nvidia_review_ready.py`,
  `tests/ut/py/test_nvidia_review_artifacts.py`, and goal/changelog docs.
- Dependencies and blocked assumptions: local verification remains mandatory;
  GitHub Actions are manual-only so automatic checks do not block exploratory
  ultimate-goal progress.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, focused review artifact pytest, Python
  compile checks, JSON syntax checks, `node --check` on the viewer, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future slices should rely on the local
  guard/test commands recorded in work preparation and may manually dispatch
  the workflow when a branch is ready for external review.

### 2026-05-31 - Benchmark Viewer Schema Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned viewer
  contract hardening.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, review-data slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `.agents/checks/`,
  `tests/ut/py/test_nvidia_review_artifacts.py`,
  `docs/nvidia-backend/benchmark-viewer/data/`, goal docs, and changelog
  docs.
- Dependencies and blocked assumptions: the validator checks schema and
  evidence references; it does not prove raw benchmark artifacts are complete
  or paper-ready.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, focused review artifact pytest, Python
  compile checks, JSON syntax checks, `node --check` on the viewer, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future result-importer work should emit
  `result_records` that satisfy this schema before changing viewer tables.
