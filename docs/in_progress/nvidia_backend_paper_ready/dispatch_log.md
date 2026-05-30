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

### 2026-05-31 - Benchmark Viewer Contract Rendering

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned viewer
  rendering hardening.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, viewer expansion slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `docs/nvidia-backend/benchmark-viewer/`,
  `.agents/checks/`, `tests/ut/py/test_nvidia_review_artifacts.py`, shared
  contracts, and changelog docs.
- Dependencies and blocked assumptions: viewer rendering now exposes the
  schema fields required for human review, but it still depends on future
  result importers for full paper-grade statistics.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  focused review artifact pytest, `validate_benchmark_viewer_data.py`,
  `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future viewer work should add filters,
  richer statistic breakdowns, and imported MPK/VDCores result rows once raw
  artifacts exist.

### 2026-05-31 - Viewer Result Export Path

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned evaluation
  importer slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, evaluation structure
  slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files:
  `.agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py`,
  `docs/nvidia-backend/benchmark-viewer/data/`, review guards, focused tests,
  shared contracts, evaluation plan, and changelog docs.
- Dependencies and blocked assumptions: the exporter covers current PTO compact
  capture rows only; future paper baseline importers must extend the mapping
  for MPK, VDCores, vLLM, SGLang, and ThunderKittens.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  focused review artifact pytest, `validate_benchmark_viewer_data.py`, Python
  compile checks, JSON syntax checks, `node --check`, `git diff --check`, and
  local export regeneration against the current compact capture.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future evaluation slices should generate
  viewer result records from raw captures instead of hand-editing result rows.

### 2026-05-31 - Changelog Contract Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned reporting
  guard slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, changelog/reporting slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `.agents/checks/`,
  `tests/ut/py/test_nvidia_review_artifacts.py`,
  `docs/nvidia-backend/changelog/`, and shared contracts.
- Dependencies and blocked assumptions: the validator checks report structure
  and index coverage, not whether every described runtime behavior is complete.
- Verification commands and results: passed `validate_nvidia_changelog.py`,
  `check_nvidia_review_ready.py`, focused review artifact pytest, Python
  compile checks, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future child PRs that add changelog
  reports must keep the index and four-section report contract passing.

### 2026-05-31 - CUDA Example Contract Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned example
  guard slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, examples slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `.agents/checks/`, `examples/cuda/`,
  `tests/ut/py/test_nvidia_review_artifacts.py`, shared contracts, and
  changelog docs.
- Dependencies and blocked assumptions: the validator checks that examples map
  to committed benchmark and method IDs; it does not run GPU examples.
- Verification commands and results: passed `validate_cuda_examples.py`,
  `check_nvidia_review_ready.py`, focused review artifact pytest, Python
  compile checks, JSON syntax checks, `validate_nvidia_changelog.py`, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future CUDA examples should update
  `examples/cuda/manifest.json` and README in the same slice.

### 2026-05-31 - Paper Evaluation Matrix

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned evaluation
  matrix slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready evaluation
  slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data and UI,
  `.agents/checks/`, focused review artifact tests, shared contracts,
  evaluation plan, and changelog docs.
- Dependencies and blocked assumptions: the matrix records paper-readiness
  gaps but does not itself reproduce MPK, VDCores, vLLM, SGLang, or
  ThunderKittens results.
- Verification commands and results: passed `validate_benchmark_viewer_data.py`,
  `check_nvidia_review_ready.py`, focused review artifact pytest,
  `validate_nvidia_changelog.py`, `node --check`, JSON syntax checks, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future evaluation importers should turn
  missing matrix evidence into viewer result records backed by raw artifacts.

### 2026-05-31 - Remote Evaluation Contract

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned remote
  evaluation guard slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, remote evaluation fallback
  slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: `.agents/checks/`, CUDA evaluation scripts as
  imported evidence, focused review artifact tests, shared contracts,
  work-preparation docs, and changelog docs.
- Dependencies and blocked assumptions: the validator checks command
  construction and policy coverage; it does not run H200 jobs.
- Verification commands and results: passed
  `validate_remote_evaluation.py`, `check_nvidia_review_ready.py`, focused
  review artifact pytest, `validate_nvidia_changelog.py`, Python compile
  checks, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future remote captures should record
  whether Git refresh or tree-sync fallback produced each raw artifact.

### 2026-05-31 - Paper Baseline Runs

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned paper
  baseline reproduction contract slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data and UI,
  `.agents/checks/`, focused review artifact tests, shared contracts,
  evaluation plan, baseline survey, and changelog docs.
- Dependencies and blocked assumptions: the run records define reproduction
  commands and expected artifacts; they do not execute MPK, VDCores, vLLM,
  SGLang, or ThunderKittens.
- Verification commands and results: passed `validate_benchmark_viewer_data.py`,
  `check_nvidia_review_ready.py`, focused review artifact pytest,
  `validate_nvidia_changelog.py`, `node --check`, JSON syntax checks, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future baseline importer slices should
  turn these expected artifacts into viewer result rows.

### 2026-05-31 - Paper Baseline Result Importer

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-baseline importer slice.
- Exact Codex command or script invocation: not applicable because no worker
  was launched in this entry.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data,
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py`,
  `.agents/checks/`, focused review artifact tests, shared contracts,
  evaluation plan, evaluation skill docs, and changelog docs.
- Dependencies and blocked assumptions: the importer defines a stable path
  from raw paper-baseline JSON to viewer rows; it does not run MPK, VDCores,
  vLLM, SGLang, or ThunderKittens.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  focused review artifact pytest, `validate_benchmark_viewer_data.py`,
  `validate_nvidia_changelog.py`, Python compile checks, JSON syntax checks,
  `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future evaluation slices should run the
  baseline commands, write raw artifacts under
  `tmp/cuda-backend/paper-baselines/`, import viewer result rows, and update
  the paper-evaluation matrix only with measured evidence.
