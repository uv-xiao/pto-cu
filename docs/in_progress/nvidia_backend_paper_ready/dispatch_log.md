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
  repository GitHub Actions are disabled and checked-in workflows are
  manual-only so automatic checks do not block exploratory ultimate-goal
  progress.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, focused review artifact pytest, Python
  compile checks, JSON syntax checks, `node --check` on the viewer, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future slices should rely on the local
  guard/test commands recorded in work preparation. A reviewer who needs the
  manual workflow must deliberately re-enable repository Actions, dispatch it,
  disable Actions again, and record that run here.

### 2026-05-31 - VDCores MInst Provenance Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  provenance diagnostic.
- Exact Codex command or script invocation: H200 instruction dump with
  `QWEN1P7B_NO_PREFETCH=all python app/python/qwen3_1p7b/sched.py
  --hf-cache-dir <shared-hf-cache> --debug-num-layers 1 --debug-stop-after
  final_rms -N 1 -i 68`; structured `runpy` provenance dump without launch;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, VDCores baseline
  diagnostic slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files:
  `docs/nvidia-backend/benchmark-viewer/data/`,
  `docs/nvidia-backend/changelog/`,
  `docs/in_progress/nvidia_backend_paper_ready/dispatch_log.md`,
  `tests/ut/py/test_nvidia_review_artifacts.py`, and raw `tmp/` artifacts.
- Dependencies and blocked assumptions: upstream VDCores was used as a source
  input only; no upstream repository was edited or pushed.
- Verification commands and results: the focused viewer test first failed
  because the new execution attempt was absent, then passed after adding the
  viewer record and regenerating derived artifacts:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data
  -q` -> `1 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the diagnostic proves generated
  Python-side direct 1D `MInst` effective addresses map to live tensor ranges
  before launch. The next VDCores slice should inspect runtime/device-side
  mutation, instruction upload, or CUDA memcheck allocator visibility for
  direct 1D `cp_async_bulk` loads.

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

### 2026-05-31 - Paper Baseline Readiness Probes

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-baseline probe slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data and UI,
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`,
  `.agents/checks/`, focused review artifact tests, shared contracts,
  evaluation plan, evaluation skill docs, and changelog docs.
- Dependencies and blocked assumptions: probes are safe readiness checks, not
  full baseline builds or benchmark runs. The local probe found A100 CUDA
  readiness and source-entrypoint readiness; it also found vLLM is not
  installed in the current environment.
- Verification commands and results: passed the focused paper-baseline probe
  pytest selector, `validate_benchmark_viewer_data.py`,
  `check_nvidia_review_ready.py`, `node --check`, JSON syntax check, and
  Python compile check for the probe script.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future slices should run the H200 probe
  through the remote fallback path, install/build vLLM, and turn readiness
  probes into real raw benchmark captures.

### 2026-05-31 - Paired Paper Baseline Probe Harness

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned paired
  paper-baseline probe harness.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py
  --sync-remote-tree`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready remote
  baseline evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: paired probe script, remote-evaluation validator,
  focused review artifact tests, CUDA evaluation skill docs, shared contracts,
  evaluation plan, dispatch log, and changelog docs.
- Dependencies and blocked assumptions: the harness proves command
  construction and artifact structure for paired readiness probes. Full
  baseline builds and benchmark runs still depend on per-baseline setup.
- Verification commands and results: TDD red checks failed before
  implementation because the paired probe script, changelog report, and
  baseline-source sync command were missing. The paired `--sync-remote-tree`
  probe wrote `a100-probe.json`, `h200-probe.json`, and
  `paired-probe-summary.json` under
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-bdec348b/`.
  Local A100 reported MPK, VDCores, SGLang, and ThunderKittens as pass and
  vLLM as partial. Remote H200 reported MPK, VDCores, vLLM, SGLang, and
  ThunderKittens as partial because runtime Python dependencies are not
  installed there yet.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: run the paired probe with
  `--sync-remote-tree` to capture A100/H200 readiness under `tmp/`, then use
  the results to decide the first H200 paper-baseline build slice.

### 2026-05-31 - ThunderKittens Dependency Probe Tightening

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  ThunderKittens readiness correction.
- Exact Codex command or script invocation:
  `ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 ... importlib.util`
  module check and
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py
  --sync-remote-tree`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer probe data, viewer validator,
  focused review artifact tests, evaluation plan, dispatch log, and changelog
  docs.
- Dependencies and blocked assumptions: the selected ThunderKittens
  `mha_h100` path builds as a PyTorch extension. Source entrypoints alone are
  insufficient readiness evidence.
- Verification commands and results: TDD red check failed because
  ThunderKittens had no dependency module probes. H200 dependency inspection
  found `torch=False`, `pybind11=False`, `numpy=True`, `pandas=True`,
  `matplotlib=True`, and `tqdm=False` for `/usr/bin/python3`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: install the missing H200 dependencies
  before attempting the ThunderKittens `make`, correctness, and benchmark
  capture.

### 2026-05-31 - ThunderKittens H200 Setup And Quick Smoke

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  ThunderKittens H200 setup and first raw capture.
- Exact Codex command or script invocation:
  `ssh bizhaoh200 ... .venv/bin/python -m pip install pybind11 tqdm`,
  `ssh bizhaoh200 ... .venv/bin/python -m pip install torch==2.8.0+cu128
  --index-url https://download.pytorch.org/whl/cu128`,
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py
  --sync-remote-tree`, and an H200 Python here-doc that imports
  `tmp/baselines/thunderkittens/kernels/attention/mha_h100/_C`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: remote project venv under `/data/.../pto-cu/.venv`,
  local and remote `tmp/` artifacts, benchmark viewer data, focused review
  artifact tests, evaluation plan, dispatch log, and changelog docs.
- Dependencies and blocked assumptions: this installs dependencies only in the
  remote pto-cu venv and does not modify upstream repositories. The quick smoke
  is a controlled setup artifact, not the full ThunderKittens paper benchmark.
- Verification commands and results: H200 import check reported
  `torch 2.8.0+cu128`, CUDA available, H200 device visible, and required
  modules present. The paired probe under
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-67c5c655/`
  reports MPK, VDCores, and ThunderKittens as pass on H200 and vLLM/SGLang as
  partial. The selected ThunderKittens `mha_h100` extension built on H200, and
  the quick smoke wrote
  `tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-67c5c655/quick-smoke.json`
  with correctness pass against PyTorch SDPA.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: run the full ThunderKittens
  `test_correctness.py` and `benchmark.py` or a scripted equivalent that
  captures repeat statistics before promoting this baseline beyond
  setup-ready.

### 2026-05-31 - ThunderKittens Bounded H200 MHA Capture

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned bounded
  ThunderKittens MHA capture for repeat statistics and viewer import.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py
  --baseline-dir tmp/baselines/thunderkittens/kernels/attention/mha_h100
  --output
  tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/capture.json
  --machine <h200-host> --pto-commit 5915346e --cuda-toolkit 12.8 --shape
  1,1,768,64 --shape 1,4,1536,64 --warmup 5 --repeats 20 --causal`, followed
  by `paper_baseline_viewer_export.py` for viewer-result import.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: local and remote `tmp/` artifacts, CUDA evaluation
  scripts, benchmark viewer data, focused review artifact tests, evaluation
  plan, dispatch log, and changelog docs.
- Dependencies and blocked assumptions: this capture uses the existing
  ThunderKittens H200 build and remote project venv dependencies. It is a
  bounded capture because the upstream default scripts sweep larger shapes
  and remain future paper-evaluation work.
- Verification commands and results: the capture wrote
  `tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/capture.json`
  and `viewer-result-records.json`. Both shapes passed correctness against
  PyTorch scaled-dot-product attention. The imported viewer rows have twenty
  CUDA-event samples each, with p50 device time `36864 ns` for
  `b=1,h=1,n=768,d=64` and `49279 ns` for `b=1,h=4,n=1536,d=64`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: ThunderKittens now has imported H200
  repeat evidence, but paper-ready promotion still requires full upstream
  correctness and benchmark sweeps plus shape alignment with PTO tensor-core
  workloads.

### 2026-05-31 - Shared Serving Policy Contract

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned paper
  serving policy extraction and viewer contract update.
- Exact Codex command or script invocation: local `rg`/`sed` inspection of
  `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt`,
  `tmp/sources/arxiv-2605.03190-vdcores.txt`,
  `tmp/baselines/mirage-mpk/demo/qwen3/demo.py`,
  `tmp/baselines/vdcores/app/python/llama3/sched.py`, and VDCores scheduling
  notes.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data and renderer, review
  validators/tests, evaluation plan, baseline survey, dispatch log, and
  changelog docs.
- Dependencies and blocked assumptions: no upstream repositories were edited.
  The policy uses existing paper text and cloned source as evidence. It
  records two serving policies because MPK and VDCores paper workloads are not
  identical.
- Verification commands and results: `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  `test_nvidia_review_artifacts.py`, Python compile checks, JSON syntax
  checks, `node --check`, and `git diff --check` passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: `mpk_offline_decode` and
  `vdcores_offline_decode` now define comparable model, prompt, decode, batch,
  hardware, baseline-run, and metric policy. The next slices must run MPK,
  VDCores, vLLM, SGLang, and PTO serving-equivalent commands against those
  policies and import raw results.

### 2026-05-31 - Serving Command Plan Materialization

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned command
  materialization for the MPK/VDCores serving policies.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py
  --output
  tmp/cuda-backend/paper-baselines/serving-runs/plan-7cad653c.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation helper script, review tests and
  guards, evaluation docs, changelog docs, and generated `tmp/` command-plan
  artifact. No upstream repositories were edited.
- Dependencies and blocked assumptions: the generated plan is launch evidence,
  not performance evidence. Long H200 runs still need installed MPK, VDCores,
  vLLM, and SGLang runtime dependencies and model access.
- Verification commands and results: the plan was generated successfully and
  JSON syntax-checked. It contains 30 rows across MPK, VDCores, vLLM, and
  SGLang: five batch sizes for each single-policy baseline and ten rows each
  for vLLM/SGLang across both serving policies.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: baseline owners should execute the
  generated commands, preserve the raw artifacts named in each command row,
  then import normalized raw JSON through `paper_baseline_viewer_export.py`.

### 2026-05-31 - Paired Probe Dependency Tightening

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned readiness
  correction for MPK/VDCores model-stack dependencies.
- Exact Codex command or script invocation: installed `transformers` in the
  H200 project venv, then ran
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py
  --sync-remote-tree`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer probe data, review validators and
  tests, evaluation plan, dispatch log, changelog docs, and generated `tmp/`
  paired probe artifacts. No upstream repositories were edited.
- Dependencies and blocked assumptions: MPK and VDCores import Transformers
  in their selected model entrypoints, so readiness must check that module
  explicitly. vLLM and SGLang remain partial on H200 because their packages
  are not installed in the project venv.
- Verification commands and results: the paired probe wrote
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-57de1a6b/`.
  A100 reports MPK, VDCores, SGLang, and ThunderKittens pass with vLLM
  partial. H200 reports MPK, VDCores, and ThunderKittens pass with vLLM and
  SGLang partial.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next setup work should install or build
  vLLM and SGLang in the H200 project venv, rerun the paired probe, and then
  execute the generated serving command plan.

### 2026-05-31 - SGLang Source Import Probe

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned readiness
  correction for SGLang source execution.
- Exact Codex command or script invocation: regenerated the serving command
  plan with `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py
  --output
  tmp/cuda-backend/paper-baselines/serving-runs/plan-43b927ed.json`,
  then reran `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py
  --sync-remote-tree`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer run/probe data, probe and command
  scripts, review validators, focused tests, evaluation docs, changelog docs,
  and generated `tmp/` artifacts. No upstream repositories were edited.
- Dependencies and blocked assumptions: SGLang source-file existence is not
  enough; selected benchmark modules must import from the pinned checkout
  before long H200 serving runs are meaningful.
- Verification commands and results: the paired probe wrote
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`.
  A100 reports MPK, VDCores, and ThunderKittens pass, vLLM partial, and
  SGLang partial because SGLang benchmark imports hit a torch/torchvision
  operator-registration mismatch. H200 reports MPK, VDCores, and
  ThunderKittens pass, vLLM partial, and SGLang partial because SGLang source
  imports require the missing `orjson` dependency.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: install or build SGLang's runtime
  dependency stack in the H200 project venv, resolve the local
  torch/torchvision mismatch before A100 SGLang serving capture, then run the
  source-path command plan and import raw serving results into the viewer.

### 2026-05-31 - Current Artifact Evidence Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned evidence
  guard tightening.
- Exact Codex command or script invocation: no worker invocation. Updated
  `.agents/checks/validate_benchmark_viewer_data.py` and focused tests so
  current result artifacts, current paper-matrix raw artifacts, and latest
  paired probe roots must resolve to local `tmp/` JSON evidence.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, code-document evidence
  guard slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer validator, focused review tests,
  shared contracts, dispatch log, and changelog docs. No upstream
  repositories were edited.
- Dependencies and blocked assumptions: planned future artifacts remain
  allowed in `expected_artifacts`; only current evidence paths must exist.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest, Python compile for the benchmark-viewer validator,
  JSON syntax checks, `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next evaluation slices should promote
  MPK, VDCores, vLLM, SGLang, or full ThunderKittens outputs to current
  evidence only after their raw JSON artifacts exist under `tmp/`.

### 2026-05-31 - Paired Probe Machine Status

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned benchmark
  viewer readiness-status slice.
- Exact Codex command or script invocation: no worker invocation. Read
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/` and
  added A100/H200 `latest_machine_status` entries to
  `paper_baseline_probes.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, human-reviewable
  benchmark status slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer probe data and rendering,
  viewer-data validators, focused tests, shared contracts, dispatch log, and
  changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: the machine statuses are setup
  readiness only; benchmark results still require raw run captures.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest, Python compile for touched guard scripts, JSON
  syntax checks, `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future paired probes should refresh both
  the aggregate `latest_status` and the per-machine A100/H200 status entries
  before claiming a baseline is setup-ready on either GPU class.

### 2026-05-31 - Probe Status Artifact Consistency

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  evidence-consistency guard slice.
- Exact Codex command or script invocation: no worker invocation. Updated
  `validate_benchmark_viewer_data.py` to load the raw A100/H200 probe JSON
  referenced by `latest_machine_status` and compare status plus blocking gaps.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, code-document evidence
  guard slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer validator, focused tests, shared
  contracts, dispatch log, and changelog docs. No upstream repositories were
  edited.
- Dependencies and blocked assumptions: the guard assumes the raw probe JSON
  artifacts named in committed viewer data remain under local `tmp/` for human
  review.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest with 17 tests, Python compile for the viewer-data
  validator, JSON syntax checks, `node --check` on the viewer, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future probe refreshes must update
  committed status summaries and raw artifacts together, or the validator will
  reject the branch before review.

### 2026-05-31 - Probe Status Updater

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned probe status
  refresh workflow slice.
- Exact Codex command or script invocation: no worker invocation. Added
  `.agents/skills/cuda-backend-eval/scripts/paper_probe_status_update.py` to
  materialize committed probe status from paired A100/H200 probe artifacts.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer
  readiness-status workflow slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation skill script, focused tests,
  shared contracts, dispatch log, changelog docs, and skill instructions. No
  upstream repositories were edited.
- Dependencies and blocked assumptions: the updater requires both
  `a100-probe.json` and `h200-probe.json` in the paired artifact directory.
- Verification commands and results: `paper_probe_status_update.py` reproduced
  the committed `paper_baseline_probes.json` from
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/` with no
  diff. Passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest with 18 tests, Python compile for the updater, JSON
  syntax checks, `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future paired probe reruns should call
  the updater before committing viewer data, then run the viewer-data
  validator to prove the committed summary matches raw artifacts.

### 2026-05-31 - Paper Readiness Audit

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  human-reviewable paper-readiness status slice.
- Exact Codex command or script invocation: no worker invocation. Added
  `.agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py` and
  generated `docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation guard slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation skill script, benchmark viewer
  data/rendering, viewer-data validator, focused tests, shared contracts,
  evaluation plan, dispatch log, changelog docs, and skill instructions. No
  upstream repositories were edited.
- Dependencies and blocked assumptions: the audit is derived from committed
  viewer JSON; it does not replace raw benchmark captures.
- Verification commands and results: `paper_readiness_audit.py` regenerated
  the committed `paper_readiness_audit.json` with no diff. Passed
  `check_nvidia_review_ready.py`, `validate_benchmark_viewer_data.py`,
  `validate_nvidia_changelog.py`, `validate_cuda_examples.py`,
  `validate_remote_evaluation.py`, focused review artifact pytest with 19
  tests, Python compile for touched guard scripts, JSON syntax checks,
  `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the audit currently reports
  `not_paper_ready`, zero ready claims, and four blocked claims. Future
  matrix, run, probe, or result changes must regenerate the audit before
  review.

### 2026-05-31 - Persistent Baseline Run Contracts

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-baseline run contract slice.
- Exact Codex command or script invocation: no worker invocation. Added
  `mpk_persistent_scheduler_trace` and `vdcores_resource_policy_trace` to
  `paper_baseline_runs.json`, then regenerated `paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready evaluation
  planning and benchmark viewer guard slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data, viewer-data validator,
  focused tests, baseline survey, evaluation plan, dispatch log, and
  changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: these records are planned H200-class
  runs; no MPK or VDCores raw scheduler traces were captured in this slice.
- Verification commands and results: `paper_readiness_audit.py` regenerated
  the committed `paper_readiness_audit.json` with no diff. Passed
  `check_nvidia_review_ready.py`, `validate_benchmark_viewer_data.py`,
  `validate_nvidia_changelog.py`, focused review artifact pytest with 19
  tests, Python compile for touched guard scripts, JSON syntax checks,
  `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the persistent-device paper claim now
  has explicit MPK and VDCores run records. The next evaluation slice should
  execute them, normalize raw JSON, and import viewer result rows.

### 2026-05-31 - A100 Driver Graph Viewer Evidence

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  host-schedule baseline evidence slice.
- Exact Codex command or script invocation: no worker invocation. Extended
  `capture_imports.json` for `direct_driver_graph`, exported
  `tmp/cuda-backend/host-launch-a100-8b6cdaee/cuda-benchmark.json` with
  `cuda_viewer_export.py`, imported the A100 graph row into `results.json`,
  and regenerated `paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data, CUDA evaluation raw
  artifacts under `tmp/`, focused tests, evaluation plan, dispatch log, and
  changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this is a local A100 host-launch
  microbenchmark capture from commit `8b6cdaee`; it does not cover H200,
  tensor-shape graph launches, or full launch-statistic sweeps.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest with 19 tests, paper-readiness audit regeneration
  plus diff check, Python compile checks for touched guard/import scripts,
  JSON syntax checks for viewer/example/import artifacts, `node --check` on
  the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the host-schedule launch-overhead claim
  now has viewer evidence for A100 `direct_driver_graph`. The remaining gaps
  are direct runtime rows, H200 graph rows, selected tensor launch shapes, and
  p50/p90/p99 distribution-ready captures.

### 2026-05-31 - A100 Driver Launch Viewer Evidence

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned raw Driver
  launch evidence slice.
- Exact Codex command or script invocation: no worker invocation. Added the
  `direct_driver` viewer method and capture-import mapping, re-exported
  `tmp/cuda-backend/host-launch-a100-8b6cdaee/cuda-benchmark.json` with
  `cuda_viewer_export.py`, imported the A100 direct Driver launch row into
  `results.json`, and regenerated `paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data, CUDA evaluation raw
  artifacts under `tmp/`, focused tests, evaluation plan, dispatch log, and
  changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: `direct_driver` is a CUDA Driver API
  `cuLaunchKernel` baseline, not a CUDA Runtime API baseline, so the Runtime
  API row remains a paper-readiness gap.
- Verification commands and results: passed `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest with 19 tests, paper-readiness audit regeneration
  plus diff check, Python compile checks for touched guard/import scripts,
  JSON syntax checks for viewer/example/import artifacts, `node --check` on
  the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the host-schedule launch-overhead claim
  now has A100 raw Driver launch and Driver graph replay rows. Remaining gaps
  are direct CUDA Runtime API rows, H200 Driver launch and graph rows,
  selected tensor launch shapes, and p50/p90/p99 distribution-ready captures.

### 2026-05-31 - A100 Runtime Launch Viewer Evidence

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned Runtime API
  launch evidence slice.
- Exact Codex command or script invocation: no worker invocation. Added a
  `direct_runtime` benchmark path backed by an nvcc-built shared library that
  calls `cudaLaunchKernel`, captured
  `tmp/cuda-backend/host-launch-runtime-a100-e429c07b/cuda-benchmark.json`,
  imported the A100 Runtime API row into `results.json`, and regenerated
  `paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA benchmark helper, benchmark viewer data, CUDA
  evaluation raw artifacts under `tmp/`, focused tests, evaluation plan,
  dispatch log, and changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this is a local A100 vector-shape
  Runtime API microbenchmark; H200 Runtime API evidence and tensor-shape launch
  evidence remain open.
- Verification commands and results: passed focused
  `test_cuda_benchmark_report.py` selectors for the Runtime API path, a real
  A100 `--single-baseline direct_runtime` sample, the 10-repeat A100 Runtime
  API capture, `check_nvidia_review_ready.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `validate_cuda_examples.py`, `validate_remote_evaluation.py`, focused
  review artifact pytest with 19 tests, paper-readiness audit regeneration
  plus diff check, Python compile checks for touched benchmark/import/guard
  scripts, JSON syntax checks for viewer/example/import artifacts,
  `node --check` on the viewer, and `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the host-schedule launch-overhead claim
  now has A100 PTO, Runtime API, raw Driver API, and Driver graph rows. A
  later H200 host-launch entry closes the H200 vector Runtime/Driver row gap;
  selected tensor launch shapes and p50/p90/p99 distribution-ready captures
  remain open.

### 2026-05-31 - H200 Host Launch Viewer Evidence

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned H200
  host-launch evidence slice.
- Exact Codex command or script invocation: no worker invocation. Used the
  documented remote fallback by syncing the local tree to `bizhaoh200` with
  `rsync`, then ran:
  `CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py --device 0
  --sizes 1024 --repeats 10 --block-dim 256 --arch compute_90 --label
  host-launch-h200-ec8f272e --output-dir
  tmp/cuda-backend/host-launch-h200-ec8f272e`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark viewer data, CUDA evaluation raw
  artifacts under `tmp/`, focused tests, evaluation plan, dispatch log, and
  changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: remote Git refresh was avoided by the
  tree-sync fallback. The capture covers the H200 `n=1024` vector launch
  shape only; tensor launch shapes and distribution sweeps remain open.
- Verification commands and results: captured ten passing H200 repeats for
  `pto_host_schedule`, `direct_runtime`, `direct_driver`, and
  `direct_driver_graph`, validated the raw capture with
  `cuda_validate_capture.py`, exported viewer records with
  `cuda_viewer_export.py`, and regenerated `paper_readiness_audit.json`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the host-schedule launch-overhead claim
  now has A100 and H200 vector evidence for PTO host-schedule, CUDA Runtime
  API, CUDA Driver API, and CUDA Driver graph paths. Remaining gaps are
  selected tensor launch shapes and p50/p90/p99 distribution-ready sweeps.

### 2026-05-31 - Viewer Latency Distribution Fields

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned benchmark
  viewer evidence-quality slice.
- Exact Codex command or script invocation: no worker invocation. Extended
  `cuda_viewer_export.py` to emit p50, p90, p99, mean, standard deviation,
  minimum, and maximum host/device latency fields for each imported capture
  group, then regenerated viewer records from the current raw CUDA artifacts.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA viewer exporter, benchmark viewer data and
  rendering, viewer validators, focused tests, evaluation plan, dispatch log,
  and changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this improves distribution visibility
  for already-imported repeated captures. It does not add selected tensor
  launch shapes or new stream-count/graph-replay sweeps.
- Verification commands and results: focused TDD test first failed because
  exported records lacked `host_wall_p50_ns`; after implementation it passed.
  Full verification is recorded in the matching changelog report and PR body.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: repeated A100/H200 host-launch viewer
  rows now expose p50/p90/p99/mean/stdev/min/max fields. Remaining
  host-launch gaps are selected tensor launch shapes and actual stream-count
  or graph-replay sweep captures.

### 2026-05-31 - Tensor Launch Viewer Evidence

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned selected
  tensor launch evidence slice.
- Exact Codex command or script invocation: no worker invocation. Added
  direct CUDA Driver, Runtime, and Driver Graph naive SGEMM baselines, then
  captured A100 and H200 artifacts with:
  `CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
  PYTHONPATH=$PWD:$PWD/python PTO_SOURCE_COMMIT=09462d04 .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py --device 0
  --sizes 1024 --repeats 10 --block-dim 256 --arch compute_80 --label
  tensor-launch-a100-09462d04 --output-dir
  tmp/cuda-backend/tensor-launch-a100-09462d04` locally, and the matching
  `compute_90` command on `bizhaoh200` after `rsync` tree sync.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark viewer and
  paper-ready evaluation evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, viewer import data,
  focused tests, raw artifacts under `tmp/`, evaluation plan, dispatch log,
  and changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: remote Git refresh was avoided by the
  tree-sync fallback. `PTO_SOURCE_COMMIT` records the synced source commit in
  H200 raw metadata.
- Verification commands and results: focused TDD dispatch tests passed, A100
  single-baseline smokes passed for the three new direct SGEMM baselines, both
  10-repeat raw captures validated with 110 rows and `16x16x16` tensor-tile
  requirements, `cuda_viewer_export.py` imported six tensor viewer records,
  and `paper_readiness_audit.py` regenerated the committed audit.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the host-schedule launch-overhead claim
  now has selected A100/H200 tensor-launch evidence for CUDA Runtime, Driver,
  and Driver Graph rows. Remaining launch-overhead gaps are stream-count and
  graph-replay sweeps across selected vector and tensor shapes.

### 2026-05-31 - Paper Baseline Results Updater

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-baseline import automation slice.
- Exact Codex command or script invocation: no worker invocation. Added
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py`
  to wrap paper-baseline raw JSON import, update viewer `results.json`, mark
  referenced paper-baseline runs as `imported_to_viewer`, and regenerate a
  paper-readiness audit from the same inputs.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-baseline evaluation
  and strict code/document evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, focused review tests,
  goal docs, dispatch log, and changelog docs. No upstream repositories were
  edited.
- Dependencies and blocked assumptions: this adds the committed-data update
  bridge for future measured MPK/VDCores imports. It does not create new MPK
  or VDCores measured results and must not be used to promote unmeasured
  artifacts.
- Verification commands and results: focused TDD fixture first failed because
  `paper_baseline_results_update.py` was missing; after implementation it
  passed. Full verification is recorded in the matching changelog report and
  PR body.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future paper-baseline runs can now move
  from normalized raw artifacts to viewer results, run status, and readiness
  audit with one script. Remaining blockers are the actual MPK/VDCores
  matching scheduler runs, broader ThunderKittens sweeps, and serving
  baseline captures.

### 2026-05-31 - Paper Baseline Run Readiness

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  MPK/VDCores scheduler-run readiness slice.
- Exact Codex command or script invocation: no worker invocation. Added
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`
  and generated:
  `tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-3157ea68/run-readiness.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-baseline evaluation
  visibility and strict evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validators, focused tests, goal docs, dispatch log, and changelog
  docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: readiness evidence is not benchmark
  evidence. It intentionally leaves MPK/VDCores scheduler runs blocked until
  model access and extension build prerequisites are resolved and measured
  raw artifacts are imported.
- Verification commands and results: focused TDD fixture first failed because
  the run-readiness script was missing; after implementation it passed. The
  viewer data validator and `node --check` passed after wiring the JSON into
  the HTML viewer.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the viewer now shows MPK scheduler-run
  readiness as partial due to missing `HF_TOKEN`, and VDCores readiness as
  partial due to missing `HF_TOKEN` plus missing `dae.runtime` build output.
  Remaining blockers are still the measured MPK/VDCores scheduler imports,
  broader ThunderKittens sweeps, and serving baseline captures.

### 2026-05-31 - Planned Run Readiness Coverage

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-baseline visibility slice.
- Exact Codex command or script invocation: no worker invocation. Generalized
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`
  and regenerated:
  `tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-1ace72fb/run-readiness.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready evaluation
  visibility and strict evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data,
  validators, focused tests, goal docs, dispatch log, and changelog docs. No
  upstream repositories were edited.
- Dependencies and blocked assumptions: readiness records remain pre-run
  evidence only. They expose blockers for planned runs, but measured raw JSON
  still has to be imported before any paper claim can be promoted.
- Verification commands and results: focused TDD fixture first failed because
  `--probes` was unsupported and only scheduler-run readiness was emitted.
  After implementation, the focused test, benchmark-viewer data validator, and
  review guard passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the viewer now has run-readiness rows
  for pending MPK, VDCores, vLLM, and SGLang runs. MPK/VDCores are blocked by
  model access and VDCores extension build state; vLLM/SGLang are blocked by
  paired-probe dependency/import gaps. These are still not measured results.

### 2026-05-31 - Readiness Audit Run Readiness

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-readiness audit integration slice.
- Exact Codex command or script invocation: no worker invocation. Extended
  `.agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py` and
  regenerated `docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready evaluation
  visibility and strict evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validators, focused tests, goal docs, dispatch log, and changelog
  docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: run-readiness blockers are pre-run
  evidence. They make claim blockers more explicit, but they do not replace
  measured raw benchmark artifacts.
- Verification commands and results: focused TDD fixture first failed because
  the audit lacked `paper_baseline_run_readiness_statuses`. After
  implementation, the focused audit test and benchmark-viewer data validator
  passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: paper-readiness audit claims now include
  run-readiness status and blockers for pending MPK, VDCores, vLLM, and SGLang
  runs. The next readiness movement still requires actual imported baseline
  measurements and broader ThunderKittens sweeps.

### 2026-05-31 - Paper Baseline Required Metric Gate

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned measured
  evidence acceptance-guard slice.
- Exact Codex command or script invocation: no worker invocation. Extended
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py`
  to validate raw rows against each run's `required_metrics`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, strict code/document
  evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, focused review tests, goal
  docs, dispatch log, and changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this is an import gate. It does not
  create measured baseline results; it prevents incomplete raw artifacts from
  becoming committed viewer evidence.
- Verification commands and results: focused TDD fixture first failed because
  an MPK scheduler trace without `scheduler_overhead` was imported. After
  implementation, both the rejecting fixture and the successful update fixture
  passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: future MPK, VDCores, vLLM, SGLang, and
  ThunderKittens raw captures must now satisfy the committed `required_metrics`
  contract before `results.json`, run status, or audit data can change.

### 2026-05-31 - ThunderKittens Serving Run Contract

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned paper
  baseline planning and audit-coverage slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`
  regenerated run-readiness data with output root
  `tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-cf273f6d/`,
  then `paper_readiness_audit.py` regenerated the committed audit.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready evaluation
  and strict evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation planning scripts, benchmark-viewer
  JSON contracts, focused review tests, goal docs, dispatch log, and changelog
  docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this adds a planned
  ThunderKittens decode-attention serving-equivalent contract. It does not
  claim measured LLM-serving performance.
- Verification commands and results: focused tests for the readiness audit and
  serving command planner passed after initially failing for the missing
  ThunderKittens run/readiness record and missing planner rows.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the LLM-serving paper claim now has an
  attached ThunderKittens planned run and readiness record. Actual H200
  captures for MPK, VDCores, vLLM, SGLang, and ThunderKittens serving-family
  rows remain required before the claim can be promoted.

### 2026-05-31 - Imported Baseline Artifact Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned strict
  evidence guardrail slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`
  regenerated run-readiness data with output root
  `tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-e8f3288f/`,
  then `paper_readiness_audit.py` regenerated the committed audit.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, strict code/document
  evidence guardrail and paper-ready evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation validators, benchmark-viewer JSON
  contracts, focused review tests, goal docs, dispatch log, and changelog
  docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this separates imported bounded
  ThunderKittens MHA evidence from the planned full upstream correctness and
  benchmark sweeps. It does not create new measured performance data.
- Verification commands and results: focused TDD test first failed because
  missing expected artifacts on `imported_to_viewer` runs were accepted. After
  implementation and data split, focused tests and viewer-data validation
  passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: imported paper-baseline runs now need
  existing `expected_artifacts`. The tensor-core claim still needs the planned
  `thunderkittens_full_sweep` raw captures before paper promotion.

### 2026-05-31 - Serving Baseline Metadata Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned strict
  evidence guardrail slice.
- Exact Codex command or script invocation:
  `gh api -X PUT repos/uv-xiao/pto-cu/actions/permissions --input -` with
  `{"enabled":false}` reconfirmed repository Actions are disabled, then
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`
  regenerated run-readiness data with output root
  `tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-112a881d/`,
  followed by `paper_readiness_audit.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, strict code/document
  evidence guardrail and paper-ready evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation validators and skill docs,
  benchmark-viewer JSON contracts, focused review tests, goal docs, dispatch
  log, and changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: repository GitHub Actions remain
  disabled during the ultimate goal. This slice only tightens serving-run
  metadata; it does not create measured serving performance data.
- Verification commands and results: focused TDD test first failed because an
  LLM-serving run without `batch_or_concurrency_policy` was accepted. After
  implementation, the focused test passed; broader verification is recorded in
  the final commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: every LLM-serving paper-baseline run now
  requires model/prompt shape and batch/concurrency policy before import.
  Actual MPK, VDCores, vLLM, SGLang, and ThunderKittens serving-family H200
  captures remain required before paper promotion.

### 2026-05-31 - Serving Command Viewer Data

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned viewer and
  paper-evaluation reviewability slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`
  generated
  `docs/nvidia-backend/benchmark-viewer/data/serving_command_plan.json`,
  then `paper_readiness_audit.py` regenerated the committed audit after the
  LLM-serving matrix gained the command-plan evidence reference.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark-viewer expansion
  and paper-ready evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer JSON and rendering code, CUDA
  evaluation validator and skill docs, focused review tests, goal docs,
  dispatch log, and changelog docs. No upstream repositories were edited.
- Dependencies and blocked assumptions: this commits a reproducible launch
  contract for future H200 serving runs. It does not create measured baseline
  results or remove imported-run blockers.
- Verification commands and results: focused viewer-data test first failed
  because `serving_command_plan.json` did not exist. After implementation, the
  focused test, viewer-data validator, and JavaScript syntax check passed;
  broader verification is recorded in the final commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: reviewers can now inspect 35 serving
  command rows in the HTML viewer. The LLM-serving claim still needs PTO,
  MPK, VDCores, vLLM, SGLang, and ThunderKittens raw H200 artifacts imported
  into viewer results.

### 2026-05-31 - Paper Readiness Next Actions

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-readiness audit reviewability slice.
- Exact Codex command or script invocation:
  `paper_readiness_audit.py` regenerated the committed audit after the audit
  builder started copying next actions from matrix gaps, run-readiness records,
  and readiness probes.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark-viewer expansion
  and strict code/document evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer rendering code, generated viewer
  JSON, CUDA evaluation validator and skill docs, focused review tests, goal
  docs, dispatch log, and changelog docs. No upstream repositories were
  edited.
- Dependencies and blocked assumptions: this is an audit and reviewability
  improvement only; it does not create new measured performance data.
- Verification commands and results: focused TDD tests first failed because
  `next_actions` were missing from the committed audit and viewer. After
  implementation, focused tests and broader verification are recorded in the
  final commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the HTML viewer can now show per-claim
  next actions. The overall audit remains `not_paper_ready` until the planned
  raw baseline captures are imported.

### 2026-05-31 - Paper Readiness Work Queue

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-readiness work-queue reviewability slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py`
  generated
  `docs/nvidia-backend/benchmark-viewer/data/paper_readiness_work_queue.json`
  from `paper_readiness_audit.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark-viewer expansion
  and strict code/document evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer rendering code, generated viewer
  JSON, CUDA evaluation validator and skill docs, focused review tests, goal
  docs, dispatch log, and changelog docs. No upstream repositories were
  edited.
- Dependencies and blocked assumptions: this is a generated work-planning
  artifact only; it does not create new measured performance data.
- Verification commands and results: focused TDD tests first failed because
  `paper_readiness_work_queue.py` and the committed work-queue JSON did not
  exist. After implementation, focused tests and broader verification are
  recorded in the final commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: reviewers can now inspect the 13
  remaining paper-readiness actions as one table. The queue still points at
  raw captures that must be run and imported before paper promotion.

### 2026-05-31 - Goal Progress Audit

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  ultimate-goal progress audit slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/nvidia_goal_progress.py`
  generated `docs/nvidia-backend/benchmark-viewer/data/goal_progress.json`
  from the current paper audit, work queue, matrix, baseline records, and
  goal docs.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, durable project planning
  and strict evidence guardrail slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer rendering code, generated viewer
  JSON, CUDA evaluation validators and skill docs, focused review tests, goal
  docs, dispatch log, and changelog docs. No upstream repositories were
  edited.
- Dependencies and blocked assumptions: this is an acceptance-criteria audit.
  It does not create measured paper-grade performance data.
- Verification commands and results: focused TDD tests first failed because
  `nvidia_goal_progress.py` and `goal_progress.json` did not exist. After
  implementation, focused tests and broader verification are recorded in the
  final commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the viewer now exposes goal-level
  progress. It reports seven met criteria and one in-progress criterion:
  final paper-grade results remain blocked by the 13 queued raw-capture
  actions.

### 2026-05-31 - Review Artifact Refresh

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned generated
  artifact maintenance slice.
- Exact Codex command or script invocation:
  `.agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`
  regenerates `paper_readiness_audit.json`,
  `paper_readiness_work_queue.json`, and `goal_progress.json` in dependency
  order.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, strict evidence guardrail
  and benchmark-viewer maintenance slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts and skill docs, focused
  review tests, goal docs, dispatch log, and changelog docs. No upstream
  repositories were edited.
- Dependencies and blocked assumptions: this is generated-artifact
  maintenance only; it does not create measured paper-grade performance data.
- Verification commands and results: focused TDD first failed because the
  unified refresh script did not exist. After implementation, the focused
  refresh regression and broader verification are recorded in the final commit
  summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: generated review artifacts now have one
  refresh command. The paper-readiness audit still has 13 queued actions.

### 2026-05-31 - PTO Serving-Equivalent Evidence

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  LLM-serving paper-readiness evidence slice.
- Exact Codex command or script invocation:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data
  -q` first failed after adding the expected PTO serving-equivalent contract.
  `.agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`
  then regenerated the derived audit, work queue, and goal progress JSON after
  the data update.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, benchmark-viewer evidence
  and paper-ready evaluation planning slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, focused review tests, goal
  docs, dispatch log, and changelog docs. No upstream repositories were
  edited.
- Dependencies and blocked assumptions: the PTO result is a controlled H200
  attention-tile proxy for `vdcores_offline_decode`, not an end-to-end LLM
  serving run. Full paper-grade serving still needs MPK, VDCores, vLLM,
  SGLang, ThunderKittens-family, and PTO serving artifacts imported from raw
  H200 runs.
- Verification commands and results: the focused TDD test failed before the
  data update and passed afterward. Broader guard results are recorded in the
  commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the LLM serving matrix now has PTO
  `persistent_device` H200 controlled serving-equivalent evidence. The
  work queue drops the LLM serving blocker count from 13 to 12 and keeps the
  remaining raw-baseline import action visible.

### 2026-05-31 - MPK Snapshot Pointer Root Cause

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  baseline diagnostic slice.
- Exact Codex command or script invocation: traced the MPK memcheck null write
  from `prepare_next_batch` to the unassigned
  `RuntimeConfig::paged_kv_indices_snapshot` pointer, applied a one-line
  local patch under `tmp/baselines/mirage-mpk`, copied that ignored baseline
  file to the H200 checkout, and ran the one-token Qwen3-0.6B MPK smoke plus
  memcheck with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, MPK tmp artifacts, focused
  review tests, dispatch log, and changelog docs. No upstream repositories
  were edited or pushed.
- Dependencies and blocked assumptions: the local MPK patch is not upstreamed
  and is not part of pto-cu source. It proves the root cause of the first
  sanitizer invalid write, but paper-grade MPK rows still need a reproducible
  patch path and successful sanitized token export.
- Verification commands and results: the focused TDD test first failed because
  the latest MPK execution attempt was still
  `mpk_qwen3_0p6b_token1_memcheck_h200`; after adding the patched attempts and
  refreshing derived artifacts, the focused review tests passed. Broader guard
  results are recorded in the commit summary for this slice.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the MPK persistent scheduler blocker now
  points to a concrete snapshot-pointer runtime-config assignment and a
  follow-up sanitizer decode/export issue, rather than an unexplained
  scheduler null write.

### 2026-05-31 - MPK Predecode Token Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  baseline diagnostic slice.
- Exact Codex command or script invocation: added temporary predecode
  instrumentation under the ignored `tmp/baselines/mirage-mpk` clone, synced
  it to the H200 checkout, and reran the patched Qwen3-0.6B one-token MPK
  memcheck with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, MPK tmp artifacts, focused
  review tests, dispatch log, and changelog docs. No upstream repositories
  were edited or pushed.
- Dependencies and blocked assumptions: the latest evidence depends on a
  local tmp baseline patch and diagnostic print. It proves the next MPK
  blocker is invalid generated token state under sanitizer, not token export
  alone.
- Verification commands and results: the focused TDD tests first failed
  because the latest MPK execution attempt still pointed to the prior patched
  memcheck run; after adding the predecode attempt and refreshing derived
  artifacts, the focused tests passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the MPK persistent scheduler blocker now
  points to generated token id `-1` after the snapshot pointer patch. Next
  work should inspect MPK's persistent argmax/output-token path and make the
  local baseline patch reproducible without touching upstream.

### 2026-05-31 - MPK Reproducibility Patches

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  baseline reproducibility slice.
- Exact Codex command or script invocation: converted the ignored
  `tmp/baselines/mirage-mpk` diffs into committed patch files under
  `docs/nvidia-backend/baseline-patches/`, then wired those paths into the
  patched MPK execution-attempt records.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, committed baseline patch
  files, focused review tests, validator, dispatch log, and changelog docs.
  No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the snapshot-pointer patch is still a
  pto-cu-carried baseline delta, not an upstream MPK change. It can now be
  reviewed and reapplied from this repo.
- Verification commands and results: the focused TDD test and viewer-data
  validator first failed because patched MPK attempts had no committed patch
  references; after adding `reproducibility_patches`, both checks passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: MPK patch reproducibility is now an
  explicit review artifact. The remaining MPK blocker is generated token id
  `-1` under sanitizer plus missing paper-grade scheduler/resource/latency
  import.

### 2026-05-31 - MPK Workload Metadata Sweep

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  matching-workload diagnostic.
- Exact Codex command or script invocation: ran the H200 patched MPK Qwen3
  0.6B persistent demo twice with `--max-new-tokens 1` and
  `--max-new-tokens 2`, both with offline Hugging Face cache, max sequence
  length 128, one request, one batched token, `--ignore-eos`, and token JSON
  export under
  `tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/workload-metadata-901ec9c1/`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts. No upstream
  repositories were edited or pushed.
- Dependencies and blocked assumptions: the run uses carried local MPK baseline
  patches as input artifacts. The sweep proves successful patched execution is
  still not a matching workload because `max_new_tokens` is not honored.
- Verification commands and results: the focused TDD test first failed because
  `mpk_qwen3_0p6b_workload_metadata_sweep_h200` was absent from viewer data.
  After adding the execution attempt and refreshing derived artifacts, the
  focused test passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: MPK persistent rows must not be imported
  as paper-grade scheduler evidence until the run command or demo loop
  enforces the intended decode length and records scheduler/resource metadata
  for the same workload used by PTO and VDCores.

### 2026-05-31 - VDCores No-Prefetch Sweep

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  baseline diagnostic slice.
- Exact Codex command or script invocation: ran the H200 Qwen3-1.7B
  `final_rms` launch with `QWEN1P7B_NO_PREFETCH` set to `all`, `q_proj`,
  `k_proj`, `v_proj`, `out_proj`, `gate_low`, `gate_high`, `up_low`,
  `up_high`, and `down_proj`, then reran the all-stage no-prefetch variant
  under `compute-sanitizer --tool memcheck`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  baseline evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts. No upstream
  repositories were edited or pushed.
- Dependencies and blocked assumptions: all no-prefetch variants still fail
  after model load and `launch_dae`; disabling per-stage async prefetch
  routing is not sufficient to import a VDCores paper-grade row.
- Verification commands and results: the focused TDD test first failed because
  `vdcores_qwen3_1p7b_no_prefetch_sweep_h200` was absent from viewer data.
  After adding the execution attempt and refreshing derived artifacts, the
  focused test passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the VDCores work-queue blocker now
  points to `MInst` load-address, coordinate, and tensor-descriptor provenance
  for the earliest `final_rms` schedule. Paper-grade VDCores correctness,
  scheduler/resource-policy, and latency evidence remain missing.

### 2026-05-31 - MPK Bounded Decode Smoke

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  matching-workload diagnostic.
- Exact Codex command or script invocation: ran the H200 patched MPK Qwen3
  0.6B persistent demo twice with `--max-new-tokens 1 --max-seq-length 40`
  and `--max-new-tokens 2 --max-seq-length 41`, both with offline Hugging
  Face cache, one request, one batched token, `--ignore-eos`, and token JSON
  export under
  `tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/bounded-decode-901ec9c1/`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts. No upstream
  repositories were edited or pushed.
- Dependencies and blocked assumptions: the run uses carried local MPK
  baseline patches as input artifacts. Bounded decode solves the immediate
  `max_new_tokens` mismatch by using the persistent loop's existing
  `max_seq_length` termination policy.
- Verification commands and results: the focused TDD test first failed because
  `mpk_qwen3_0p6b_bounded_decode_h200` was absent from viewer data. After
  adding the execution attempt and refreshing derived artifacts, the focused
  test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data
  -q` -> `1 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the MPK persistent scheduler blocker now
  moves from decode-length matching to scheduler/resource/latency import for
  the bounded-decode workload. Paper-grade MPK scheduler evidence remains
  partial until those metrics are captured and imported.

### 2026-06-01 - MPK Bounded Profile Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  scheduler-trace diagnostic.
- Exact Codex command or script invocation: ran the H200 patched MPK Qwen3
  0.6B persistent demo with `--max-new-tokens 2 --max-seq-length 41`,
  offline Hugging Face cache, one request, one batched token, `--ignore-eos`,
  `--use-mirage`, `--profiling`, and Perfetto trace export under
  `tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/bounded-profile-9668bcef/`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, committed baseline patch files, and local
  `tmp/` raw artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the diagnostic uses carried local MPK
  baseline patches. The unpatched profiler exporter reaches kernel launch but
  fails with `KeyError: (16, 0)`. The profiler diagnostic patch exports a
  Perfetto trace, but the profiled run reports predecode `step=1` and saved
  `generate_length=0`, so it is not paper-grade evidence.
- Verification commands and results: the focused TDD test first failed because
  `mpk_qwen3_0p6b_bounded_profile_diagnostic_h200` was absent from viewer
  data. After adding the execution attempt and refreshing derived artifacts,
  the focused tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_matches_current_viewer_data
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit
  -q` -> `3 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: MPK profiling trace export can now be
  diagnosed with a reproducible patch, but profiling and correctness do not
  yet coexist. The next MPK slice should explain why `--profiling` changes
  persistent token/step state before importing scheduler, resource-policy, or
  latency rows.

### 2026-06-01 - MPK Profile No-Op Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  profile-mode diagnostic.
- Exact Codex command or script invocation: ran the H200 patched MPK Qwen3
  0.6B persistent demo with `--max-new-tokens 2 --max-seq-length 41`,
  offline Hugging Face cache, one request, one batched token, `--ignore-eos`,
  `--use-mirage`, `--profiling`, and Perfetto trace export under
  `tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/profile-write-diagnostic-034bada3/`.
  The first variant no-oped profiler event writes while keeping profiler init;
  the second variant no-oped `PROFILER_CLOSURE_PARAMS_DECL`, `PROFILER_INIT`,
  and all event macros.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, committed baseline patch files, and local
  `tmp/` raw artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the diagnostic uses carried local MPK
  baseline patches. Both no-op variants exit 0 but report predecode `step=1`
  and saved `generate_length=0`, so they are not paper-grade evidence.
- Verification commands and results: the focused TDD test first failed because
  `mpk_qwen3_0p6b_profile_noop_diagnostic_h200` was absent from viewer data
  and the derived readiness files still referenced the bounded-profile
  attempt.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: profiler event writes and Perfetto
  export are no longer sufficient root-cause explanations. The next MPK slice
  should identify why `-DMPK_ENABLE_PROFILING` or profile mode corrupts
  token/step state before importing scheduler, resource-policy, or latency
  rows.

### 2026-06-01 - MPK Profile Termination Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  profile-mode root-cause diagnostic.
- Exact Codex command or script invocation: ran the H200 patched MPK Qwen3
  0.6B persistent demo with `--max-new-tokens 2 --max-seq-length 41`,
  offline Hugging Face cache, one request, one batched token, `--ignore-eos`,
  `--use-mirage`, `--profiling`, and Perfetto trace export under
  `tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/profile-termination-diagnostic-fa357d52/`.
  The first run failed before compilation because `nvcc` was not on the
  non-interactive SSH `PATH`; the rerun set `CUDA_HOME=/usr/local/cuda-12.8`
  and `PATH=$CUDA_HOME/bin:$PATH`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready MPK baseline
  evaluation slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, committed baseline patch files, and local
  `tmp/` raw artifacts copied back from H200. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: the diagnostic uses carried local MPK
  baseline patches. It removes only the `MPK_ENABLE_PROFILING` early
  request-done override in offline `prepare_next_batch`; the real-profiler
  variant keeps profiler event writes enabled.
- Verification commands and results: the focused TDD test first failed because
  the derived paper-readiness artifacts still referenced the previous
  profile-no-op attempt. After adding the execution attempt and refreshing
  derived artifacts, the focused tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_matches_current_viewer_data
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit
  -q` -> `3 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: MPK profiling can now preserve
  bounded-decode correctness and export a real Perfetto trace with carried
  patches. The remaining MPK paper-readiness gap is importing comparable
  scheduler/resource/latency rows into viewer results for the bounded workload.

### 2026-06-01 - MPK Scheduler Trace Import

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK
  scheduler-trace import.
- Exact Codex command or script invocation: copied the successful H200 MPK
  bounded-decode profiler run into
  `tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json` and
  imported it with `.agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready persistent
  scheduler baseline evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer results, paper-baseline run
  status, paper evaluation matrix, generated paper-readiness audit/work-queue
  and goal-progress data, focused tests, dispatch log, changelog docs, and
  local `tmp/` raw artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the imported MPK row uses carried
  local MPK patches from the profile-termination diagnostic. Device elapsed
  time comes from the MPK demo CUDA events; scheduler overhead comes from
  observed Perfetto scheduler slice begin/end pairs.
- Verification commands and results: the first focused pytest run failed after
  import because the audit no longer reports execution/readiness actions for
  `mpk_persistent_scheduler_trace` once the run is `imported_to_viewer`. After
  updating the test expectation, the focused review tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_matches_current_viewer_data
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit
  tests/ut/py/test_nvidia_review_artifacts.py::test_nvidia_goal_progress_matches_current_artifacts
  -q` -> `4 passed`. `validate_benchmark_viewer_data.py` also passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the persistent scheduler-overhead matrix
  now cites an MPK H200 viewer result and no longer lists MPK scheduler-trace
  import as missing evidence. The remaining blocker for that claim is VDCores
  queue/resource-policy evidence.

### 2026-06-01 - VDCores Runtime LD Warp Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  runtime load-warp diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: temporarily synced an ignored
  `tmp/baselines/vdcores/include/dae/pipeline/ldwarp.cuh` debug print to H200,
  rebuilt VDCores with `debug=64`, then ran the Qwen3-1.7B one-layer
  `final_rms` cut under compute-sanitizer with `QWEN1P7B_NO_PREFETCH=all`,
  offline Hugging Face cache, `--debug-num-layers 1`,
  `--debug-stop-after final_rms`, `-N 1`, and `--launch`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: remote PTO checkout was stale and had
  unrelated dirty committed-data files, so the raw artifact records the remote
  PTO commit separately. VDCores source remained at
  `5247328cf3f893ed9df95f9f38e7e9a97f0cbfb1`; the temporary debug header was
  restored locally and remotely after capture.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `32 passed`;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: SM64 reported a runtime direct 1D load
  address of `0x761ba2034000`; compute-sanitizer reported the same address as
  the first invalid 4096-byte `cp_async_bulk` read. The VDCores blocker is now
  why the runtime-consumed address is outside sanitizer allocation tracking or
  otherwise unmapped during launch, not missing Python-side MInst provenance.

### 2026-06-01 - VDCores Pointer Attribute Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  pointer-attribute diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: synced a temporary wrapper under
  `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-pointer-attrs-sm64-c6e25d3e/`
  to H200, monkey-patched `dae_app` for one process, built the Qwen3-1.7B
  one-layer `final_rms` schedule without launch, simulated sampled direct 1D
  effective addresses, then called `cuMemGetAddressRange` and 4096-byte
  device-to-host `cudaMemcpy` from those addresses.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the probe classifies prelaunch direct
  addresses and does not launch the failing kernel. It therefore narrows the
  pointer-validity question but does not produce VDCores correctness or
  queue/resource-policy timing.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `32 passed`;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: 80 sampled direct 1D load/store
  effective addresses were not classified by `cuMemGetAddressRange`, but all
  80 copied successfully with device-to-host `cudaMemcpy`. The next VDCores
  diagnostic should combine same-process pointer classification with the
  failing launch, or inspect why `cp_async_bulk` and compute-sanitizer treat
  `cudaMemcpy`-readable direct 1D addresses as out of bounds.

### 2026-06-01 - VDCores Device Colocation Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  device-colocation diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: ran the launch pointer-attribute
  wrapper under
  `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-launch-pointer-attrs-ac8dab26/`;
  then ran the Qwen3-1.7B one-layer `final_rms` and one-layer `full` launch
  commands with `CUDA_VISIBLE_DEVICES=7`, `QWEN1P7B_NO_PREFETCH=all`, offline
  Hugging Face cache, `--debug-num-layers 1`, `-N 1`, and `--launch`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the latest remote VDCores extension
  still had debug-print instrumentation from the prior diagnostic build. The
  result is diagnostic evidence, not paper-grade timing.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `32 passed`;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the launch pointer probe found direct
  source-load pointers on devices 0 and 7 while VDCores launched on current
  device 0. Restricting visibility to physical GPU 7 made the one-layer
  `final_rms` launch return status 0, so cross-device weight placement was the
  original direct 1D load failure. The one-layer full schedule still fails
  later with illegal instruction, so VDCores still lacks correctness and
  queue/resource-policy timing.

### 2026-06-01 - VDCores Logits Stage Bisect

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  logits-stage bisect for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: ran Qwen3-1.7B one-layer
  co-located cuts on H200 with `CUDA_VISIBLE_DEVICES=7`,
  `QWEN1P7B_NO_PREFETCH=all`, offline Hugging Face cache,
  `--debug-num-layers 1`, `-N 1`, and `--launch` for
  `--debug-stop-after logits`, `argmax`, `restore`, and `full`; then reran
  the `logits` cut with `QWEN1P7B_LOGITS_SPLIT_M=1,2,3,6`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the remote VDCores extension still had
  debug-print instrumentation from prior diagnostics, so this is diagnostic
  evidence, not timing evidence.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `32 passed`;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the first failing co-located stage is
  `logits`; `argmax`, `restore`, and `full` inherit the failure. Default
  `QWEN1P7B_LOGITS_SPLIT_M=6` reaches the kernel and fails with illegal
  instruction after invalid slot-allocation or TMA-coordinate signals. Split
  values `1`, `2`, and `3` fail before launch on VDCores auto-folding
  placement assertions, so they are not simple launchable workarounds. VDCores
  still lacks correctness and queue/resource-policy timing.

### 2026-06-01 - Repository CI Closed For Ultimate Goal

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned repository
  gate policy update after user direction to keep repo CI closed during the
  NVIDIA backend ultimate goal.
- Exact Codex command or script invocation: queried
  `gh api repos/uv-xiao/pto-cu/actions/permissions`, disabled workflow
  `286106490` with
  `gh api --method PUT repos/uv-xiao/pto-cu/actions/workflows/286106490/disable`,
  attempted to disable GitHub's synthetic dependency-graph workflow
  `286106506`, then re-queried workflow states and PR checks.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, dispatcher policy and
  review-gate hygiene.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: own repository settings for `uv-xiao/pto-cu`,
  goal work-preparation policy, changelog index, and changelog report. No
  upstream repositories were edited, pushed, or reconfigured.
- Dependencies and blocked assumptions: GitHub rejects disabling the synthetic
  `Dependency Graph` workflow with HTTP 422. Repository Actions remain disabled
  at the settings level, so no Actions workflow should run as a PR gate.
- Verification commands and results:
  `gh api repos/uv-xiao/pto-cu/actions/permissions` ->
  `{"enabled":false,"sha_pinning_required":false}`;
  `gh api repos/uv-xiao/pto-cu/actions/workflows` ->
  `NVIDIA Manual Review: disabled_manually`, `Dependency Graph: active`;
  `gh pr checks 1 --repo uv-xiao/pto-cu` ->
  `no checks reported on the 'main' branch`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: repo CI is closed for the active
  ultimate goal. Local review guards, benchmark artifacts, changelog reports,
  and dispatch-log evidence remain the required progress gates.

Follow-up tracked-state closure after user confirmation:

- Removed the runnable `.github/workflows/ci.yml` file from the branch.
- Archived the future manual review recipe at
  `docs/ci/nvidia-manual-review.workflow.yml`.
- Updated `docs/ci.md`, the review guard, and focused tests so closed repo CI
  means no workflow YAML exists under `.github/workflows/` during the ultimate
  goal.

### 2026-06-01 - VDCores Logits Schedule Introspection

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  logits scheduling diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: copied the tmp-only probe
  `vdcores_logits_schedule_probe.py` to the H200 checkout, then ran it from
  `tmp/baselines/vdcores` with `CUDA_VISIBLE_DEVICES=7`,
  `QWEN1P7B_NO_PREFETCH=all`, `QWEN1P7B_LOGITS_SPLIT_M=6`, offline Hugging
  Face cache, `--debug-num-layers 1`, `--debug-stop-after logits`, and
  `-N 1`. The probe monkey-patched `dae_app`, called `dae.build_instructions()`,
  and did not call `launch_dae`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue data, focused review tests, dispatch log,
  changelog docs, and local `tmp/` raw artifacts copied back from H200. No
  upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the remote pto-cu checkout had local
  viewer-data modifications, so the run used the allowed tmp probe-copy path
  instead of a remote Git refresh. The remote VDCores checkout remains the
  same diagnostic baseline used by the earlier logits-stage failure.
- Verification commands and results:
  remote `logits-schedule-probe-status.txt` -> `0`; raw summary ->
  `logits_epoch=3`, `logits_slice=50688`, `vocab_size=151936`,
  `SM64 first logits PC=38`, desc32/33/34 counts `12/3/2`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the illegal-instruction blocker is now
  localized to the direct logits GEMV instruction window after `final_rms`.
  The next diagnostic should inspect VDCores memory-slot allocation and
  `RepeatM` loop handling for the desc32/33/34 direct TMA sequence before any
  queue/resource-policy timing import.

### 2026-06-01 - VDCores Slot/Repeat Source Analysis

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned source/log
  diagnostic for the latest VDCores resource-policy blocker.
- Exact Codex command or script invocation: analyzed the prior H200
  co-located logits split-6 failure log and the build-only logits schedule
  summary with the tmp-only script
  `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-slot-repeat-source-analysis-858717e4/vdcores_slot_repeat_analysis.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue data, focused review tests, dispatch log,
  changelog docs, and local `tmp/` raw/source-analysis artifacts. No upstream
  repositories were edited or pushed.
- Dependencies and blocked assumptions: this was source/log analysis, not a
  new launch. It uses the previous H200 co-located logits split-6 failure and
  the previous build-only schedule dump as evidence.
- Verification commands and results:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/checks/check_nvidia_review_ready.py` -> passed;
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `32 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/checks/validate_benchmark_viewer_data.py` -> passed;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/checks/validate_nvidia_changelog.py` -> passed;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the blocker now points at two concrete
  VDCores runtime hazards in the direct logits GEMV window: slot `0` can be
  reused for a `desc33` load before STU consumes the earlier `desc34` store
  metadata, and `RepeatM` feeds invalid `desc33` coordinates `(65535,127,0)`
  where the build-only schedule expects `[0,0,0,0]`. VDCores still lacks
  correctness and queue/resource-policy timing.

### 2026-06-01 - VDCores Slot Lifetime PC44-PC52 Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  slot-lifetime diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: copied the tmp-only
  `vdcores-slot-lifetime-diagnostic-pc44-pc52.patch` into the H200 VDCores
  checkout, rebuilt with `DAE_DIAG_SLOT_LIFETIME` and
  `DAE_DIAG_WAIT_AFTER_WB_ALLOC`, then ran `CUDA_VISIBLE_DEVICES=7`,
  `QWEN1P7B_NO_PREFETCH=all`, `QWEN1P7B_LOGITS_SPLIT_M=6`, offline Hugging
  Face cache, `--debug-num-layers 1`, `--debug-stop-after logits`, `-N 1`,
  and `--launch`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the remote pto-cu checkout still had
  unrelated dirty viewer-data files, so this used the allowed tmp patch-copy
  path instead of remote Git refresh. The remote VDCores checkout was clean
  before applying the diagnostic patch.
- Verification commands and results: remote rebuild status `0`; valid launch
  status `1`; PC48 wait changed flags from `0x00fffffe` to `0x00ffffff`;
  STU copied slot 0 opcode `0443`; `Unknown mem wb opcode` did not recur;
  PC50 still applied `addr_accum=0x7fffff` and produced desc33 coordinates
  `(65535,127,0,0)`, ending in illegal instruction.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: writeback metadata lifetime is a real
  hazard but not sufficient to fix the logits-stage failure. The next VDCores
  diagnostic should instrument the PC49/PC50 `RepeatM` shuffle lane source and
  packed coordinate delta encoding before attempting queue/resource-policy
  timing import.

### 2026-06-01 - VDCores RepeatM Lane-Source Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  lane-source diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: copied the tmp-only
  `vdcores-repeat-lane-source-diagnostic.patch` into the H200 VDCores
  checkout, rebuilt with `DAE_DIAG_REPEAT_LANES`,
  `DAE_DIAG_WAIT_AFTER_WB_ALLOC`, and `DAE_DIAG_USE_SPLIT_U64_SHUFFLE`, then
  ran `CUDA_VISIBLE_DEVICES=7`, `QWEN1P7B_NO_PREFETCH=all`,
  `QWEN1P7B_LOGITS_SPLIT_M=6`, offline Hugging Face cache,
  `--debug-num-layers 1`, `--debug-stop-after logits`, `-N 1`, and
  `--launch`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the remote pto-cu checkout still had
  unrelated dirty viewer-data files, so this used the allowed tmp patch-copy
  path instead of remote Git refresh. The remote and local VDCores checkouts
  were restored to clean state after artifact capture.
- Verification commands and results before local review gates: remote rebuild
  status `0`; launch status `1`; no `Unknown mem wb opcode` appeared after
  the PC48 wait; forcing split 32-bit shuffles made PC50 update address `0x0`
  and desc33 coordinates `(0,0,0)`, but PC52 updated address
  `0x20000331000000`, produced desc32 coordinates `(0,12544,3)`, and ended
  with illegal memory access.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the split-shuffle variant is not a
  valid fix and rules out a simple native `uint64_t` shuffle lowering bug as
  the sole root cause. The next VDCores diagnostic should leave `addr_accum`
  unchanged and instrument actual `RepeatM` active masks, producer-lane
  ownership, source-lane validity, and packed coordinate-delta lifetime before
  attempting queue/resource-policy timing import.

### 2026-06-01 - VDCores RepeatM State Lite Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  lite RepeatM state diagnostic for the persistent scheduler baseline gap.
- Exact Codex command or script invocation: copied the tmp-only
  `vdcores-repeat-state-lite.patch` into the H200 VDCores checkout, rebuilt
  with `DAE_DIAG_REPEAT_STATE_LITE` and `DAE_DIAG_WAIT_AFTER_WB_ALLOC`, then
  ran `CUDA_VISIBLE_DEVICES=7`, `QWEN1P7B_NO_PREFETCH=all`,
  `QWEN1P7B_LOGITS_SPLIT_M=6`, offline Hugging Face cache,
  `--debug-num-layers 1`, `--debug-stop-after logits`, `-N 1`, and
  `--launch`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, generated
  paper-readiness audit/work-queue/goal-progress data, focused review tests,
  dispatch log, changelog docs, and local `tmp/` raw artifacts copied back
  from H200. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the remote pto-cu checkout still had
  unrelated dirty viewer-data files, so this used the allowed tmp patch-copy
  path instead of remote Git refresh. A broader per-lane diagnostic perturbed
  the schedule, so the committed review data uses the narrower lane-0 debug
  macro run. The remote and local VDCores checkouts were restored to clean
  state after artifact capture.
- Verification commands and results before local review gates: remote rebuild
  status `0`; launch status `1`; no `Unknown mem wb opcode` appeared after
  the PC48 wait; PC49 decoded `RepeatM` as `reg_start=0`, `reg_end=5`,
  `size=2`, `arg=0x0000`, and `address=0x1000000000`; after PC49, lane-0
  `gpr0=0x1000000000` and `gpr1=0x0`; PC50 consumed source lane `0` but the
  runtime `addr_accum` was `0x7fffff`, producing desc33 coordinates
  `(65535,127,0)` and ending with illegal instruction.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the latest evidence rules out bad PC49
  `RepeatM` field encoding and localizes the bad coordinate to allocwarp
  RepeatM accumulator transport between decode and consumers. The next
  VDCores diagnostic should isolate native 64-bit shuffle/register transport
  and then test a lane-0-only split transport or explicit 32-bit
  packed-coordinate transport before importing queue/resource-policy timing.

### 2026-06-01 - VDCores Guarded RepeatM Benchmark

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  guarded RepeatM full-launch and benchmark capture for the persistent
  scheduler baseline gap.
- Exact Codex command or script invocation: copied the tmp-only
  `vdcores-repeat-guard-native.patch` into the H200 VDCores checkout, rebuilt
  with `DAE_DIAG_GUARD_REPEAT_SHUFFLE` and
  `DAE_DIAG_WAIT_AFTER_WB_ALLOC`, then ran the offline full launch with
  `CUDA_VISIBLE_DEVICES=7`, `QWEN1P7B_NO_PREFETCH=all`,
  `QWEN1P7B_LOGITS_SPLIT_M=6`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `-N 1`, and `--launch`. The benchmark reused the
  guarded runtime without debug printing and ran `DAE_BENCH_WARMUP=1` with
  `-N 1` and `-b 5`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, imported
  viewer result data, generated paper-readiness audit/work-queue/goal-progress
  data, focused review tests, dispatch log, changelog docs, and local `tmp/`
  raw artifacts copied back from H200. No upstream repositories were edited or
  pushed.
- Dependencies and blocked assumptions: the first full attempt without offline
  Hugging Face flags stalled in model HEAD retries and was terminated with
  status `143`; the offline rerun completed. The local and remote VDCores
  checkouts were restored to clean state after artifact capture.
- Verification commands and results before local review gates: patch apply
  status `0`; rebuild status `0`; full launch status `0`; benchmark status
  `0`; patch restore status `0`; guarded PC49 skipped the invalid pre-repeat
  shuffle from source lane 48; benchmark on 132 H200 SMs reported min
  `1774240 ns`, median `1778528 ns`, average `1779008 ns`, and max
  `1785504 ns`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the latest evidence supports the
  root cause that unguarded `addr_accum` shuffles execute before `RepeatM` is
  active and can read invalid source lanes. The next VDCores slice should run
  guarded correctness, then add queue-pressure and scheduler-overhead metadata
  before marking the VDCores resource-policy trace imported-to-viewer.

### 2026-06-01 - VDCores Guarded RepeatM Correctness

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  guarded RepeatM correctness capture for the persistent scheduler baseline
  gap.
- Exact Codex command or script invocation: copied the tmp-only
  `vdcores-repeat-guard-native.patch` into the H200 VDCores checkout, rebuilt
  through the remote project venv with the selected Qwen3-1.7B compute-op
  list, pinned CUTLASS include path, `-include cfloat`,
  `DAE_DIAG_GUARD_REPEAT_SHUFFLE`, and
  `DAE_DIAG_WAIT_AFTER_WB_ALLOC`, then ran the offline correctness command
  with `CUDA_VISIBLE_DEVICES=7`, `QWEN1P7B_NO_PREFETCH=all`,
  `QWEN1P7B_LOGITS_SPLIT_M=6`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, and `--correctness`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data, imported
  viewer result correctness metadata, generated paper-readiness
  audit/work-queue/goal-progress data, focused review tests, dispatch log,
  changelog docs, and local `tmp/` raw artifacts copied back from H200. No
  upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the first rebuild attempt failed
  before CUDA because non-interactive `python` was absent; the second failed
  because CUTLASS headers were not on the include path; the third failed
  because `FLT_MAX` needed the existing `-include cfloat` workaround and
  system `pip` was blocked by PEP 668. The successful run used the project
  venv first on `PATH`.
- Verification commands and results before local review gates: patch apply
  status `0`; rebuild status `0`; correctness status `0`; patch restore
  status `0`; all 17 logged correctness checks passed; final token agreement
  was `ref=25, dae=25`; local and remote VDCores checkouts were clean after
  capture.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: guarded RepeatM now has full-layer
  Qwen3-1.7B single-token correctness, so correctness is no longer part of the
  persistent-device scheduler blocker. The next VDCores slice should export
  queue-pressure and scheduler-overhead metadata comparable with PTO
  persistent-device and MPK.

### 2026-06-01 - VDCores Queue/Scheduler Trace Import

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  queue-pressure and scheduler-overhead import for the persistent scheduler
  baseline gap.
- Exact Codex command or script invocation: applied the tmp-only guarded
  RepeatM plus profile-slot diagnostics patch to the H200 VDCores checkout,
  generated the Qwen3-1.7B compute-op list with `--dry-build -w`, rebuilt
  through the remote project venv with pinned CUTLASS headers, `-include
  cfloat`, `DAE_DIAG_GUARD_REPEAT_SHUFFLE`, and
  `DAE_DIAG_WAIT_AFTER_WB_ALLOC`, then ran `-b 5` for the timing trace and
  `--correctness` in a separate fresh process.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence slice.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer result data, paper-baseline run
  status, execution-attempt data, generated paper-readiness
  audit/work-queue/goal-progress data, focused review tests, dispatch log,
  changelog docs, and local `tmp/` raw artifacts copied back from H200. No
  upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: a combined `-b 5 --correctness`
  command emitted valid timing diagnostics but failed the subsequent
  correctness check after repeated launches. The imported evidence uses the
  clean benchmark-only status `0` plus a separate fresh correctness status
  `0`.
- Verification commands and results before local review gates: rebuild status
  `0`; benchmark status `0`; correctness status `0`; five H200 benchmark
  iterations reported median `1787488 ns` and average `1786304 ns`; mean
  allocwarp scheduler resident time was `1763558 ns`; max slot pressure was
  `24 / 24`; all 17 correctness checks passed with final token
  `ref=25, dae=25`; local and remote VDCores checkouts were clean after
  capture.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: `vdcores_resource_policy_trace` is now
  imported to the viewer, and the persistent-device scheduler claim no longer
  has a VDCores queue/scheduler metadata blocker. The remaining paper-readiness
  work is now in serving baseline runs and official tensor-core baseline gaps.

### 2026-06-01 - VDCores Diagnostic Scope Contract

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned review
  contract correction for the VDCores persistent scheduler evidence row.
- Exact Codex command or script invocation: edited the paper evaluation matrix,
  VDCores paper-baseline run metadata, imported viewer result metadata,
  focused review tests, and changelog docs; then refreshed generated
  paper-readiness artifacts with
  `.agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready VDCores
  resource-policy evidence scope.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, generated
  audit/work-queue/goal-progress data, focused review tests, dispatch log, and
  changelog docs. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: the VDCores queue/scheduler fields are
  diagnostic-scope fields exported by a tmp-only profile-slot patch. They
  cannot honestly be described as a non-diagnostic baseline trace unless
  VDCores grows stable instrumentation with the same semantics.
- Verification commands and results:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`
  refreshed audit/work-queue/goal-progress JSON;
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `33 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`
  -> `benchmark viewer data validation passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py`
  -> `nvidia changelog validation passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/check_nvidia_review_ready.py`
  -> `nvidia review guard passed`;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json && git diff --check`
  -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the top work-queue item now asks for a
  stable VDCores instrumentation mode or an explicit diagnostic-only paper
  treatment, keeping final latency/correctness rows separate from diagnostic
  scheduler counters.

### 2026-06-01 - Serving Baseline Probe Scope

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM/SGLang
  source-entrypoint and dependency probe refinement for the LLM serving
  paper-baseline blocker.
- Exact Codex command or script invocation: updated
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py` and
  `paper_baseline_probes.json`, then ran
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py --sync-remote-tree --local-python .venv/bin/python`.
  The paired runner used tree sync for H200, then copied back
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-86ea3913/h200-probe.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline evaluation readiness.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: probe tooling, benchmark-viewer probe and
  run-readiness data, generated audit/work-queue/goal-progress data, focused
  review tests, dispatch log, changelog docs, and local `tmp/` probe
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: vLLM source imports now pass from the
  pinned checkout on A100 and H200, but installed `vllm`, server import, and
  engine argument imports remain blocked on runtime dependencies. SGLang
  benchmark module imports are now checked with `PYTHONNOUSERSITE=1`; both
  A100 and H200 report missing isolated `orjson` and `torchvision`, and H200
  also reports no installed `sglang` module.
- Verification commands and results:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py --sync-remote-tree --local-python .venv/bin/python`
  -> wrote paired A100/H200 probes under
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-86ea3913`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py --commit 86ea3913 --output-root tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-86ea3913`
  -> wrote run-readiness JSON;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`
  -> refreshed audit/work-queue/goal-progress JSON;
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q`
  -> `33 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`
  -> `benchmark viewer data validation passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py`
  -> `nvidia changelog validation passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/check_nvidia_review_ready.py`
  -> `nvidia review guard passed`;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json && git diff --check`
  -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the serving-baseline blocker is narrower
  and reviewable. The next execution slice should build isolated vLLM and
  SGLang evaluation environments instead of installing their large dependency
  stacks into the shared project venv.

### 2026-06-01 - Serving Baseline Environment Plans

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned isolated
  environment planning for vLLM and SGLang serving-baseline execution.
- Exact Codex command or script invocation: added
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py`,
  updated the run-readiness generator, viewer, validator, tests, and
  changelog docs, then ran
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline evaluation environment readiness.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validator/tests, generated audit/work-queue/goal-progress data,
  dispatch log, changelog docs, and local `tmp/` environment-plan artifacts.
  No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: vLLM and SGLang dependency stacks are
  intentionally not installed into the project `.venv`. The environment plans
  require dedicated `tmp/` venvs and `PYTHONNOUSERSITE=1` validation before
  actual serving benchmark runs.
- Verification commands and results:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`
  -> wrote `paper_baseline_environment_plans.json`, refreshed
  run-readiness, audit, work-queue, and goal-progress JSON;
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q -k 'environment_plan or run_readiness_probe_exports_run_blockers or benchmark_viewer_has_json_backed_review_data'`
  -> `3 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`
  -> `benchmark viewer data validation passed`;
  `node --check docs/nvidia-backend/benchmark-viewer/viewer.js` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM and SGLang now have reviewable
  isolated-environment recipes in the benchmark viewer. The next execution
  slice should materialize those venvs on the evaluation host, run the
  validation commands, then capture serving benchmark raw JSON.

### 2026-06-01 - Serving Baseline Environment Attempt

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned bounded
  execution evidence for the vLLM isolated environment plan.
- Exact Codex command or script invocation: added
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py`,
  updated the benchmark viewer, validator, tests, changelog docs, and ran
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --max-steps 3 --timeout-seconds 300`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline environment materialization evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validator/tests, dispatch log, changelog docs, and local `tmp/`
  environment-attempt artifacts. No upstream repositories were edited or
  pushed.
- Dependencies and blocked assumptions: the vLLM attempt intentionally stopped
  after three of nine plan steps to avoid pulling the full torch/vLLM stack in
  this slice. It created the dedicated `tmp/` venv, upgraded build tools, and
  installed explicit `uvloop`; remaining requirement and validation steps are
  still pending.
- Verification commands and results:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --max-steps 3 --timeout-seconds 300`
  -> wrote
  `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-ef065acd/environment-attempt.json`;
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q -k 'environment_attempt_captures_bounded_steps or benchmark_viewer_has_json_backed_review_data'`
  -> `2 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`
  -> `benchmark viewer data validation passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: continue vLLM from step 4
  (`requirements/common.txt` and `requirements/cuda.txt`), then editable
  install and validation imports. Start a matching SGLang environment attempt
  after vLLM setup is either complete or blocked with concrete install logs.

### 2026-06-01 - vLLM Environment Materialization

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM
  environment materialization and blocker capture.
- Exact Codex command or script invocation: added resumable/appendable
  environment attempts, fixed vLLM build-requirement planning, tightened pip
  setup to `PYTHONNOUSERSITE=1` plus env-local `PATH`, regenerated viewer data,
  recreated `tmp/cuda-backend/paper-baselines/envs/vllm-27fa5aa3`, and ran
  bounded vLLM setup windows for steps 1-6.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline environment materialization evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validator/tests, dispatch log, changelog docs, and local `tmp/`
  environment-attempt artifacts. No upstream repositories were edited or
  pushed.
- Dependencies and blocked assumptions: vLLM runtime/CUDA requirements and
  CUDA build requirements now install in the isolated env with user-site
  disabled. Editable install reaches the source build and fails at
  `csrc/spinloop.cpp` under `Py_LIMITED_API=0x030b0000` because `Py_buffer` and
  `PyBuffer_Release` are not visible.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q -k 'environment_plan_exports_isolated_serving_envs or environment_attempt_appends_resume_window'`
  -> `2 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --max-steps 3 --timeout-seconds 300 --commit 1d3242de`
  -> wrote
  `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de/environment-attempt.json`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --start-step 4 --max-steps 1 --attempt-id-suffix step04 --append-viewer --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step04 --timeout-seconds 900 --commit 1d3242de`
  -> step 4 passed;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --start-step 5 --max-steps 1 --attempt-id-suffix step05 --append-viewer --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step05 --timeout-seconds 600 --commit 1d3242de`
  -> step 5 passed;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --start-step 6 --max-steps 1 --attempt-id-suffix step06 --append-viewer --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step06 --timeout-seconds 900 --commit 1d3242de`
  -> step 6 failed with the `spinloop.cpp` limited-API compile blocker.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: decide whether to patch the pinned vLLM
  source locally, adjust build flags for `spinloop`, or use an
  upstream-supported prebuilt/skip-extension route before vLLM serving runs.
  SGLang environment materialization is still pending.

### 2026-06-01 - vLLM Spinloop Preflight

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM
  spinloop source/ABI preflight gate.
- Exact Codex command or script invocation: added
  `.agents/skills/cuda-backend-eval/scripts/vllm_spinloop_preflight.py`,
  added `preflight_commands` to environment plans, updated the environment
  attempt runner to execute preflight steps before remaining install steps, and
  replayed vLLM setup windows at commit `94ba2e06`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline environment materialization evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validator/tests, dispatch log, changelog docs, and local `tmp/`
  environment-attempt artifacts. No upstream repositories were edited or
  pushed.
- Dependencies and blocked assumptions: vLLM setup now fails before the long
  editable build when the pinned source uses `USE_SABI 3.11` for `spinloop`,
  `spinloop.cpp` uses `Py_buffer`/`PyBuffer_Release`, and the isolated env uses
  Python 3.10 headers.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q -k 'vllm_spinloop_preflight or environment_plan_exports_isolated_serving_envs or environment_attempt_appends_resume_window'`
  -> `3 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline vllm --start-step 6 --max-steps 1 --attempt-id-suffix step06 --append-viewer --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-94ba2e06-step06 --timeout-seconds 60 --commit 94ba2e06`
  -> step 6 preflight failed in `0.108s` with a JSON blocker under
  `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-94ba2e06-step06/`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: resolve the preflight blocker by using a
  Python 3.11+ baseline environment or by adding a reviewed local
  reproducibility patch/build flag that removes `Py_LIMITED_API` from the
  spinloop CXX compile. SGLang environment materialization is still pending.

### 2026-06-01 - vLLM Spinloop Source Overlay

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM
  source-overlay path for Python 3.10 baseline environment setup.
- Exact Codex command or script invocation: added
  `.agents/skills/cuda-backend-eval/scripts/vllm_spinloop_source_overlay.py`,
  updated the vLLM environment plan to install from a copied source overlay,
  regenerated the environment plan at commit `460a34ba`, and ran
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py
  --baseline vllm --start-step 6 --max-steps 2 --attempt-id-suffix
  step06_overlay_preflight --append-viewer --output-root
  tmp/cuda-backend/paper-baselines/environment-attempts/vllm-460a34ba-step06-overlay-preflight
  --timeout-seconds 300 --commit 460a34ba`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline environment materialization evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data and
  rendering, validator/tests, dispatch log, changelog docs, and local `tmp/`
  source-overlay/environment-attempt artifacts. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: the source overlay copies the pinned
  vLLM checkout into `tmp/cuda-backend/paper-baselines/source-overlays/` and
  applies the spinloop-only `-UPy_LIMITED_API` CXX compile option there. The
  pinned upstream checkout under `tmp/baselines/vllm` remains unmodified.
- Verification commands and results:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
  -k 'environment_plan_exports_isolated_serving_envs or vllm_spinloop'`
  -> `3 passed`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py
  --commit 460a34ba --output-root
  tmp/cuda-backend/paper-baselines/environment-plans/environment-plans-460a34ba`
  -> wrote the current environment plan;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py
  --baseline vllm --start-step 6 --max-steps 2 --attempt-id-suffix
  step06_overlay_preflight --append-viewer --output-root
  tmp/cuda-backend/paper-baselines/environment-attempts/vllm-460a34ba-step06-overlay-preflight
  --timeout-seconds 300 --commit 460a34ba`
  -> step 6 overlay creation passed and step 7 overlay preflight passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the Python 3.10 spinloop preflight
  blocker is now resolved through a local overlay without upstream mutation.
  The next vLLM slice should run editable install from step 8 and capture the
  first remaining install or validation blocker. SGLang environment
  materialization is still pending.

### 2026-06-01 - vLLM Environment Validation

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM
  isolated environment materialization and validation.
- Exact Codex command or script invocation: updated the vLLM environment plan
  to pass `VLLM_VERSION_OVERRIDE` from the pinned upstream checkout, excluded
  `.deps/` from source-overlay copies, added env-local SciPy before validation,
  then ran `paper_baseline_environment_attempt.py` for the clean overlay
  install window and validation window at commit `e1c48975`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline environment materialization evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data,
  validator/tests, dispatch log, changelog docs, and local `tmp/`
  environment-attempt artifacts. No upstream repositories were edited or
  pushed.
- Dependencies and blocked assumptions: the successful environment is local
  A100 evidence. It does not yet prove H200 vLLM readiness or serving
  benchmark performance.
- Verification commands and results:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py
  --baseline vllm --start-step 6 --max-steps 3 --attempt-id-suffix
  step06_overlay_preflight_install_clean --append-viewer --output-root
  tmp/cuda-backend/paper-baselines/environment-attempts/vllm-e1c48975-step06-overlay-preflight-install-clean
  --timeout-seconds 1800 --commit e1c48975`
  -> steps 6, 7, and 8 passed; editable install completed in `760.226s`;
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py
  --baseline vllm --start-step 9 --max-steps 5 --attempt-id-suffix
  step09_scipy_validation --append-viewer --output-root
  tmp/cuda-backend/paper-baselines/environment-attempts/vllm-e1c48975-step09-scipy-validation
  --timeout-seconds 300 --commit e1c48975`
  -> env-local SciPy install and all four validation imports passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM local environment readiness is now
  captured. The next serving slice should run vLLM benchmark commands and
  import raw JSON results, or separately materialize the same environment on
  H200. SGLang environment materialization is still pending.

### 2026-06-01 - vLLM Local Offline Throughput Bring-Up

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned first vLLM
  benchmark-command execution from the validated isolated environment.
- Exact Codex command or script invocation: ran
  `vllm bench throughput` from
  `tmp/cuda-backend/paper-baselines/envs/vllm-27fa5aa3` against
  `Qwen/Qwen3-1.7B` with the source overlay, one local A100, random prompt
  length `128`, output length `64`, one warmup request, and one measured
  request.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline capture evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, changelog docs, dispatch
  log, and local `tmp/` vLLM run artifacts. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: the first run failed before model load
  because `HF_HOME` pointed above the actual Hugging Face cache. The rerun set
  `HUGGINGFACE_HUB_CACHE=tmp/huggingface_cache` and completed.
- Verification commands and results:
  `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
  HF_HOME=$PWD/tmp/hf-home-vllm
  HUGGINGFACE_HUB_CACHE=$PWD/tmp/huggingface_cache HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
  PYTHONPATH=$PWD/tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython:$PWD/tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython/python:$PWD/python:$PWD
  timeout 900
  tmp/cuda-backend/paper-baselines/envs/vllm-27fa5aa3/bin/python -m
  vllm.entrypoints.cli.main bench throughput --model Qwen/Qwen3-1.7B
  --dataset-name random --random-input-len 128 --random-output-len 64
  --num-prompts 1 --num-warmups 1 --dtype bfloat16
  --gpu-memory-utilization 0.50 --max-model-len 256 --output-json
  tmp/cuda-backend/paper-baselines/serving-runs/vllm/vdcores_offline_decode/vllm-throughput-qwen3-1p7b-batch1-bringup.json`
  -> passed and wrote raw throughput JSON with elapsed time
  `0.2419096989906393s`, `4.1337739006433765` requests/s, and
  `793.6845889235282` total tokens/s.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM can now execute an offline
  throughput bring-up from the isolated local environment. The attempt remains
  partial and is intentionally not imported into `results.json` because it is
  local A100 evidence, uses the Qwen3-1.7B bring-up model, and lacks serving
  TTFT/ITL metrics. The next serving slice should run `vllm serve` plus
  `vllm bench serve` on H200 for Qwen3-8B or the agreed bring-up model.

### 2026-06-01 - vLLM H200 Serving Capture

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned H200 vLLM
  serving capture for the paper baseline evaluation track.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the standalone pto-cu checkout on `bizhaoh200`, ran the
  vLLM isolated environment setup through bounded H200 attempts, installed
  env-local `pandas`, `numexpr`, and `bottleneck` after the first server
  launch exposed system package binary incompatibility, then ran `vllm serve`
  and `vllm bench serve` for `Qwen/Qwen3-8B` with input length `128`, output
  length `64`, batch/concurrency `1`, `request_rate=inf`, and
  `CUDA_VISIBLE_DEVICES=7`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline capture evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data,
  changelog docs, dispatch log, and local `tmp/` vLLM run artifacts. No
  upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The first H200 server attempt failed before readiness because
  `vllm._aiter_ops` imported a system `pandas` package that was incompatible
  with the environment NumPy. Installing env-local `pandas`, `numexpr`, and
  `bottleneck` resolved that serving-path blocker.
- Verification commands and results:
  H200 environment attempt
  `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-840a847f-h200-full/`
  -> steps 1 through 8 passed and stopped at the bounded attempt limit;
  H200 validation attempt
  `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-840a847f-h200-validation/`
  -> steps 9 through 13 passed before the additional Qwen3 validation guard was
  added;
  H200 serving artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-vdcores-qwen3-8b-batch1-840a847f-pandas/`
  -> server readiness passed, `bench-serve-status.txt` is `0`, completed
  requests `1`, failed requests `0`, mean TTFT `64.89939196035266 ms`, mean
  ITL `5.8076567134805135 ms`, output throughput
  `148.25853760123334 tokens/s`, and request throughput
  `2.316539650019271 req/s`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM now has one successful H200
  Qwen3-8B serving point imported into the benchmark viewer. The attempt is
  still partial for paper use until batch `2`, `4`, `8`, and `16`, repeated
  samples, MPK serving comparison, and PTO persistent-device comparison are
  captured under the same workload policy.

### 2026-06-01 - vLLM H200 VDCores Serving Sweep

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM H200
  VDCores-comparable serving sweep for the paper baseline evaluation track.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the standalone pto-cu checkout on `bizhaoh200`, started
  one isolated vLLM server for `Qwen/Qwen3-8B` on `CUDA_VISIBLE_DEVICES=7`
  with `max_model_len=192`, `bfloat16`, and `gpu_memory_utilization=0.80`,
  then ran `vllm bench serve` for batch/concurrency `2`, `4`, `8`, and `16`
  with input length `128`, output length `64`, `request_rate=inf`,
  `ignore_eos`, and `temperature=0`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline sweep evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, changelog docs, dispatch
  log, and local `tmp/` vLLM run artifacts. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The sweep reuses the previously materialized isolated vLLM environment and
  source overlay; it does not exercise the MPK-comparable 1024-token policy.
- Verification commands and results:
  H200 serving artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-vdcores-qwen3-8b-sweep-89fe1705/`
  -> server readiness passed, status code `0`, batch `2`, `4`, `8`, and `16`
  each returned `bench-serve-status.txt=0`, and all rows completed with zero
  failed requests.
  Batch `2`: mean TTFT `80.22279106080532 ms`, mean ITL
  `5.732923454146773 ms`, output throughput `287.79827100668354 tokens/s`.
  Batch `4`: mean TTFT `34.042275743559 ms`, mean ITL
  `5.801673921283394 ms`, output throughput `635.7423030915141 tokens/s`.
  Batch `8`: mean TTFT `82.2809441597201 ms`, mean ITL
  `5.931017967495357 ms`, output throughput `1108.0923559038586 tokens/s`.
  Batch `16`: mean TTFT `116.38721390045248 ms`, mean ITL
  `5.672932683309127 ms`, output throughput `2119.4372078699585 tokens/s`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM now has the full
  VDCores-comparable H200 Qwen3-8B batch sweep imported into the benchmark
  viewer. The vLLM paper-baseline run is still partial until the
  MPK-comparable 1024-token policy, repeated samples, MPK serving comparison,
  and PTO persistent-device comparison are captured under the same workload
  policy.

### 2026-06-01 - vLLM H200 MPK Serving Capture

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM H200
  MPK-comparable serving capture for the paper baseline evaluation track.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the standalone pto-cu checkout on `bizhaoh200`, started
  one isolated vLLM server for `Qwen/Qwen3-8B` on `CUDA_VISIBLE_DEVICES=7`
  with `max_model_len=1088`, `bfloat16`, and `gpu_memory_utilization=0.80`,
  then ran `vllm bench serve` for input length `64`, output length `1024`,
  batch/concurrency `1`, `request_rate=inf`, `ignore_eos`, and
  `temperature=0`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline MPK-policy evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, changelog docs, dispatch
  log, and local `tmp/` vLLM run artifacts. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The capture reuses the previously materialized isolated vLLM environment and
  source overlay. It covers only MPK-comparable batch `1`, not the full
  batch/concurrency sweep.
- Verification commands and results:
  H200 serving artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-mpk-qwen3-8b-batch1-7e939170/`
  -> server readiness passed, status code `0`, `bench-serve-status.txt=0`,
  completed requests `1`, failed requests `0`, mean TTFT
  `63.15663317218423 ms`, mean ITL `5.947963752941331 ms`, output throughput
  `166.5269980770844 tokens/s`, and request throughput
  `0.16262402155965272 req/s`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM now has one MPK-comparable H200
  Qwen3-8B serving point imported into the benchmark viewer. The vLLM
  paper-baseline run is still partial until MPK-comparable batch `2`, `4`,
  `8`, and `16`, repeated samples, MPK serving comparison, and PTO
  persistent-device comparison are captured under the same workload policy.

### 2026-06-01 - vLLM H200 MPK Serving Sweep

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM H200
  MPK-comparable serving sweep for the paper baseline evaluation track.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the standalone pto-cu checkout on `bizhaoh200`, started
  one isolated vLLM server for `Qwen/Qwen3-8B` on `CUDA_VISIBLE_DEVICES=7`
  with `max_model_len=1088`, `bfloat16`, and `gpu_memory_utilization=0.80`,
  then ran `vllm bench serve` for input length `64`, output length `1024`,
  batch/concurrency `2`, `4`, `8`, and `16`, `request_rate=inf`,
  `ignore_eos`, and `temperature=0`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready serving
  baseline MPK-policy evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, changelog docs, dispatch
  log, tests, and local `tmp/` vLLM run artifacts. No upstream repositories
  were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The capture reuses the isolated vLLM environment and source overlay. It
  covers only one sample per batch, not paper confidence intervals.
- Verification commands and results:
  H200 serving artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-mpk-qwen3-8b-sweep-908438de/`
  -> server readiness passed, status code `0`, and all four benchmark
  commands exited `0`.
- Captured metrics: batch `2` completed `2`, failed `0`, mean TTFT
  `72.44588294997811 ms`, mean ITL `5.906464019926849 ms`, output throughput
  `334.7536743804337 tokens/s`; batch `4` completed `4`, failed `0`, mean
  TTFT `39.24347530119121 ms`, mean ITL `5.97836579881674 ms`, output
  throughput `665.049038750595 tokens/s`; batch `8` completed `8`, failed
  `0`, mean TTFT `57.89269332308322 ms`, mean ITL `6.042956362181649 ms`,
  output throughput `1311.7874753053661 tokens/s`; batch `16` completed
  `16`, failed `0`, mean TTFT `98.91026074183173 ms`, mean ITL
  `5.985480906558565 ms`, output throughput `2629.5086844579146 tokens/s`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: vLLM now has single-sample H200
  Qwen3-8B serving rows for the full MPK-comparable and VDCores-comparable
  batch matrix. The vLLM paper-baseline run is still partial until repeated
  samples, MPK serving comparison, and PTO persistent-device comparison are
  captured under the same workload policy.

### 2026-06-01 - SGLang H200 Environment Attempt

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned SGLang H200
  isolated environment setup evidence for the paper baseline evaluation track.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the standalone pto-cu checkout on `bizhaoh200`, then ran
  `PYTHONPATH=$PWD:$PWD/python python3 .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline sglang --start-step 1 --max-steps 2 --attempt-id-suffix h200_step01_02 --output-root tmp/cuda-backend/paper-baselines/environment-attempts/sglang-1cbb7b83-h200-step01-02 --timeout-seconds 300 --commit 1cbb7b83`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready SGLang
  serving-baseline environment setup.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer environment-attempt data,
  changelog docs, dispatch log, tests, and local `tmp/` SGLang attempt
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The attempt is intentionally bounded to steps `1` and `2`; the editable
  SGLang install and validation imports remain unrun.
- Verification commands and results:
  H200 environment artifact
  `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-1cbb7b83-h200-step01-02/`
  -> step `1` venv creation passed in `2.681 s`; step `2` pip/setuptools/wheel
  upgrade passed in `2.923 s`; attempt status is `partial` with blocker
  `bounded attempt stopped at step 2 of 9 environment steps`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: SGLang now has first H200 environment
  setup evidence in the same viewer path as vLLM. Continue from step `3` to
  install `tmp/baselines/sglang/python[all]`, then validate imports and run the
  serving/offline/one-batch benchmarks.

### 2026-06-01 - SGLang H200 Environment Validation

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned continuation
  of the SGLang H200 isolated environment setup and validation slice.
- Exact Codex command or script invocation: ran
  `PYTHONPATH=$PWD:$PWD/python python3 .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline sglang --start-step 3 --max-steps 1 --attempt-id-suffix h200_step03 --output-root tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step03 --timeout-seconds 1800 --commit df219d33`,
  then ran
  `PYTHONPATH=$PWD:$PWD/python python3 .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py --baseline sglang --start-step 4 --max-steps 6 --attempt-id-suffix h200_step04_09 --output-root tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step04-09 --timeout-seconds 300 --commit df219d33`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready SGLang
  serving-baseline environment setup.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer environment-attempt data,
  changelog docs, dispatch log, tests, and local `tmp/` SGLang attempt
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  This validates the SGLang environment only; server launch and benchmark raw
  JSON are still separate work.
- Verification commands and results:
  H200 environment artifact
  `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step03/`
  -> step `3` editable install passed in `147.943 s`;
  `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step04-09/`
  -> steps `4` through `9` all passed, validating imports for `sglang`,
  `orjson`, `torchvision`, `sglang.bench_serving`,
  `sglang.bench_offline_throughput`, and `sglang.bench_one_batch`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: SGLang now has a passing isolated H200
  environment and benchmark-module import evidence. The next slice should run
  SGLang server launch, serving benchmark, offline throughput, and one-batch
  captures under the shared Qwen3-8B workload policy.

### 2026-06-01 - SGLang H200 Serving Bring-Up

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned SGLang H200
  serving bring-up for the paper baseline evaluation track.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the standalone pto-cu checkout on `bizhaoh200`, then ran
  bounded SGLang server and benchmark commands from the isolated
  `tmp/cuda-backend/paper-baselines/envs/sglang-7ed53d15` environment on
  `CUDA_VISIBLE_DEVICES=7`. The final online attempt started
  `sglang.launch_server` with `--disable-piecewise-cuda-graph`, then ran
  `sglang.bench_serving` with `--dataset-name random-ids`,
  `--tokenize-prompt`, requested input length `128`, requested output length
  `64`, `num_prompts=1`, `max_concurrency=1`, `request_rate=inf`, and
  `warmup_requests=0`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready SGLang
  serving-baseline execution.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer execution-attempt data,
  generated paper-readiness data, changelog docs, dispatch log, tests, and
  local `tmp/` SGLang serving artifacts. No upstream repositories were edited
  or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  SGLang online serving now reaches H200 readiness, but the result is not a
  paper row because the measured token shape does not match the planned
  VDCores policy and offline/one-batch remain unresolved.
- Verification commands and results:
  initial artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-8a61669c/`
  -> server failed during piecewise CUDA graph warmup with an illegal memory
  access; retry artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-disablepwcg-8a61669c/`
  -> server reached readiness but `bench_serving` tried to fetch an uncached
  ShareGPT helper file in offline mode; local-data artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-localdata-8a61669c/`
  -> online serving passed but measured `38/44` tokens, offline failed with a
  context-length/tokenization error, and one-batch failed with `input_ids`
  `None`; final random-ids artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-randomids-8a61669c/`
  -> online serving passed with one completed request, zero failed requests,
  request throughput `0.9697372581994511 req/s`, output throughput
  `42.66843936077585 tok/s`, mean TTFT `771.7542587779462 ms`, and mean ITL
  `5.566300629356572 ms`, but still measured `38/44` tokens instead of the
  requested `128/64`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the next SGLang slice should fix the
  token-shape contract before importing viewer result rows, then resolve
  offline throughput and one-batch command failures before running the full
  batch ladder.

### 2026-06-01 - SGLang Fixed-Range H200 Capture

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned SGLang
  fixed-range serving and offline evidence import.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the remote standalone checkout on `bizhaoh200`. The H200 run
  used `CUDA_VISIBLE_DEVICES=0`, the isolated
  `tmp/cuda-backend/paper-baselines/envs/sglang-7ed53d15` environment,
  `Qwen/Qwen3-8B`, `bfloat16`, `--disable-piecewise-cuda-graph`, and
  `--random-range-ratio 1.0` after source inspection showed
  `range_ratio=0` samples variable lengths. The online command used
  `sglang.bench_serving --dataset-name random-ids --tokenize-prompt` with
  `128` input tokens, `64` output tokens, `num_prompts=1`,
  `max_concurrency=1`, and `request_rate=inf`. The offline command used
  `sglang.bench_offline_throughput --context-length 384 --skip-warmup` with a
  local ShareGPT-shaped seed file.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready SGLang
  serving-baseline execution.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: SGLang command-plan generation, benchmark-viewer
  results, benchmark-viewer execution-attempt data, generated paper-readiness
  data, changelog docs, dispatch log, tests, and local `tmp/` SGLang serving
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  SGLang online serving and offline engine throughput now have exact `128/64`
  H200 result rows, but `bench_one_batch` still fails before producing a row,
  so the SGLang run remains partial.
- Verification commands and results:
  source diagnosis reproduced `range_ratio=0.0 -> [38]/[44]` and
  `range_ratio=1.0 -> [128]/[64]`; artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-fixedrange-bfc1c581/`
  -> `server_ready_status=0`, `bench_serving_status=0`,
  `offline_status=0`, `one_batch_status=1`. Online serving captured one
  completed request, zero failed requests, `128` input tokens, `64` output
  tokens, request throughput `2.48693394676766 req/s`, output throughput
  `159.16377259313023 tok/s`, mean TTFT `36.877373699098825 ms`, and mean ITL
  `5.6076819948371375 ms`. Offline engine throughput captured one successful
  request, `128` input tokens, `64` output tokens, latency
  `0.4909931207075715 s`, and output throughput
  `130.34805845705012 tok/s`. One-batch still fails with
  `AttributeError: 'NoneType' object has no attribute 'long'` in
  `vocab_parallel_embedding.py`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next SGLang work should fix or bypass
  the one-batch synthetic-input path, then run batch `2`, `4`, `8`, and `16`
  with repeated samples before upgrading SGLang from partial evidence to a
  paper-ready baseline.

### 2026-06-01 - SGLang Fixed-Range H200 Batch Sweep

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned SGLang
  fixed-range batch-ladder evidence import.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the remote standalone checkout on `bizhaoh200`, started
  one SGLang `Qwen/Qwen3-8B` server on `CUDA_VISIBLE_DEVICES=0` with
  `bfloat16`, `--context-length 256`, and
  `--disable-piecewise-cuda-graph`, then ran
  `sglang.bench_serving` for batches `2`, `4`, `8`, and `16` with
  `--dataset-name random-ids`, `--tokenize-prompt`,
  `--random-range-ratio 1.0`, prompt length `128`, output length `64`,
  `max_concurrency=batch`, and `request_rate=inf`. After stopping the server,
  ran `sglang.bench_offline_throughput` for the same batch ladder with
  `--context-length 384`, `--skip-warmup`, and a local ShareGPT-shaped seed
  file.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready SGLang
  serving-baseline execution.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer result data,
  benchmark-viewer execution-attempt data, generated paper-readiness data,
  changelog docs, dispatch log, tests, and local `tmp/` SGLang serving sweep
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The SGLang online/offline batch ladder now has exact `128/64` H200 result
  rows for batches `2`, `4`, `8`, and `16`, but `bench_one_batch` remains
  unresolved and the sweep has only one sample per batch.
- Verification commands and results: artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-sweep-cfbdcf0c/`
  -> `server_ready_status=0`, online batch statuses all `0`, offline batch
  statuses all `0`. Online serving captured zero failed requests for every
  batch; output throughput was `173.9169706505443`, `484.47103616685513`,
  `1177.5193163104066`, and `1793.5576766141346` tok/s for batches `2`,
  `4`, `8`, and `16`. Offline output throughput was
  `288.98312805396284`, `562.3668373531619`, `1141.1108121634202`, and
  `2170.7243728274034` tok/s.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next SGLang work should fix or bypass
  the one-batch synthetic-input path and then add repeated samples for
  variance before upgrading SGLang from partial evidence to a paper-ready
  baseline.

### 2026-06-01 - SGLang Fixed-Range H200 Repeats

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned SGLang
  repeated sample capture and import.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the remote standalone checkout on `bizhaoh200`, started
  one SGLang `Qwen/Qwen3-8B` server on `CUDA_VISIBLE_DEVICES=0` with
  `bfloat16`, `--context-length 256`, and
  `--disable-piecewise-cuda-graph`, then ran three
  `sglang.bench_serving` repeats for batches `1`, `2`, `4`, `8`, and `16`
  with `--dataset-name random-ids`, `--tokenize-prompt`,
  `--random-range-ratio 1.0`, prompt length `128`, output length `64`,
  `max_concurrency=batch`, and `request_rate=inf`. After stopping the server,
  ran three `sglang.bench_offline_throughput` repeats for the same batch
  ladder with `--context-length 384`, `--skip-warmup`, and a local
  ShareGPT-shaped seed file. Imported the aggregate raw JSON with
  `paper_baseline_results_update.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready SGLang
  serving-baseline execution.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: paper-baseline importer metric preservation,
  benchmark-viewer result data, benchmark-viewer execution-attempt data,
  generated paper-readiness data, changelog docs, dispatch log, tests, and
  local `tmp/` SGLang serving repeat artifacts. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The SGLang online/offline batch ladder now has three samples for each batch,
  but `bench_one_batch` remains unresolved and final paper claims still need
  matching PTO persistent-device, MPK, VDCores, and vLLM serving-policy
  evidence.
- Verification commands and results: artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-repeats-eb75a235/`
  -> `server_ready_status=0`, all 15 online statuses `0`, and all 15 offline
  statuses `0`. Online output throughput means were `161.02159972046658`,
  `308.03817224015523`, `597.9727093342293`, `1194.254514947911`, and
  `2171.7566025420783` tok/s for batches `1`, `2`, `4`, `8`, and `16`.
  Offline output throughput means were `156.19079316906286`,
  `288.1244764888034`, `554.4822850296894`, `1136.0633983460384`, and
  `1784.4900809591322` tok/s; the batch-16 offline row preserves raw samples
  `2167.286434107662`, `2237.551058756805`, and `948.6327500129296` tok/s.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next SGLang work should fix or bypass
  the one-batch synthetic-input path, then align SGLang repeated rows with
  matching PTO persistent-device, MPK, VDCores, and vLLM serving-policy
  evidence before treating the serving baseline as paper-ready.

### 2026-06-01 - vLLM H200 Serving Repeats

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned vLLM
  repeated sample capture and import.
- Exact Codex command or script invocation: used the documented tree-sync
  fallback to refresh the remote standalone checkout on `bizhaoh200`, then
  started one vLLM `Qwen/Qwen3-8B` server on `CUDA_VISIBLE_DEVICES=0` for the
  VDCores-shaped `128/64` policy and one server for the MPK-shaped `64/1024`
  policy. For each policy, ran `vllm bench serve` for batches `1`, `2`, `4`,
  `8`, and `16` with three repeats, `max_concurrency=batch`,
  `request_rate=inf`, `--ignore-eos`, and `--temperature 0`. Imported the
  aggregate raw JSON with `paper_baseline_results_update.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-ready vLLM
  serving-baseline execution.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer result data,
  benchmark-viewer execution-attempt data, generated paper-readiness data,
  changelog docs, dispatch log, tests, and local `tmp/` vLLM serving repeat
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The H200 vLLM environment entrypoints pass in the isolated evaluation env.
  The vLLM probe remains partial because A100 runtime validation was not
  rerun for this H200 paper-baseline capture.
- Verification commands and results: artifact
  `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-qwen3-8b-repeats-eb75a235/`
  -> all 15 VDCores-shaped serving statuses `0` and all 15 MPK-shaped serving
  statuses `0`. VDCores-shaped output throughput means were
  `155.0161459156807`, `312.86978656376226`, `606.447657421981`,
  `1153.2822637361564`, and `2159.7779254046995` tok/s for batches `1`,
  `2`, `4`, `8`, and `16`. MPK-shaped output throughput means were
  `153.16926887618368`, `316.10883967853846`, `629.2727994297298`,
  `1256.784287803509`, and `2411.4020405041215` tok/s.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: next LLM-serving work should capture
  matching MPK and VDCores baseline rows under the same Qwen3-8B policies, and
  either rerun or explicitly waive A100 vLLM runtime validation for this
  H200-only baseline path.

### 2026-06-01 - MPK Native Token Viewer Import

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK native
  H200 bring-up normalization, viewer import, and command-plan contract fix.
- Exact Codex command or script invocation: normalized
  `tmp/cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/native-token2.json`
  with `mpk_native_token_capture.py`, imported the generated raw result through
  `paper_baseline_results_update.py`, regenerated `serving_command_plan.json`
  with `paper_serving_command_plan.py`, and refreshed derived review artifacts
  with `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, MPK paper-baseline
  bring-up evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation skill script/docs,
  benchmark-viewer result data, serving workload/run contracts, generated
  review data, changelog docs, dispatch log, tests, and local `tmp/` MPK native
  artifacts. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The imported row is native torch bring-up evidence only. It is not MPK
  persistent-kernel evidence and does not satisfy the full Qwen3-8B MPK paper
  serving workload.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `40 passed`;
  `jq empty` over viewer data and MPK generated artifacts -> passed;
  `py_compile` for `mpk_native_token_capture.py` and
  `paper_serving_command_plan.py` -> passed; `git diff --check` -> passed.
  The imported viewer row records `prompt_tokens=39`, `decode_tokens=2`,
  `end_to_end_latency_ns=476397766`, `time_per_output_token_ns=238198883`,
  and `throughput_tokens_per_s=4.198172499406724`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the viewer now exposes the MPK native
  Qwen3-0.6B H200 bring-up artifact without attaching it to the full MPK paper
  ladder. The MPK persistent-kernel path remains the paper-critical blocker,
  and LLM-serving paper readiness still needs same-workload MPK persistent,
  VDCores, vLLM, SGLang, ThunderKittens-family, and PTO rows.

### 2026-06-01 - Serving Gap Audit Narrowing

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  paper-readiness audit cleanup after vLLM and SGLang H200 serving rows were
  imported.
- Exact Codex command or script invocation: added vLLM and SGLang
  `llm_serving_decode` H200 evidence refs to `paper_evaluation_matrix.json`,
  added shape-aware `viewer_result` matching in the audit and validator,
  narrowed the LLM-serving missing-evidence text, updated
  `nvidia_goal_progress.py`, then regenerated review artifacts with
  `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, paper-readiness queue
  accuracy.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer matrix/audit/work-queue data,
  generated goal progress and run-readiness data, changelog docs, dispatch
  log, tests, and CUDA evaluation helper wording. No upstream repositories
  were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  This slice changed review data only and did not run new GPU benchmarks.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `40 passed`;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `py_compile nvidia_goal_progress.py` -> passed; `git diff --check` ->
  passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the LLM-serving work queue no longer
  asks reviewers to import vLLM rows that already exist, and SGLang is now
  scoped to the VDCores-shaped `128/64` policy it actually covers. Remaining
  full-serving gaps are MPK persistent-kernel, VDCores full serving, SGLang
  MPK-policy, ThunderKittens-family full serving, and PTO full serving under
  the shared Qwen3-8B policies.

### 2026-06-01 - MPK Qwen3 8B Persistent Import

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned MPK H200
  persistent-kernel capture import.
- Exact Codex command or script invocation: tree-synced the standalone repo to
  `bizhaoh200`, ran native torch and MPK persistent `Qwen/Qwen3-8B` token
  captures in offline Hugging Face mode, copied artifacts back under
  `tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/`, added
  `mpk_qwen3_persistent_capture.py`, imported the normalized raw JSON through
  `paper_baseline_results_update.py`, then regenerated review artifacts with
  `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, MPK serving evidence for
  the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, CUDA evaluation helper
  script, changelog docs, dispatch log, and tests. No upstream repositories
  were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The MPK demo reports one combined prefill+decode per-token timing around
  asynchronous persistent-kernel launch, so the imported row is execution
  coverage with a latency caveat, not a final latency-distribution claim.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `41 passed`;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `py_compile mpk_qwen3_persistent_capture.py` -> passed; `git diff --check`
  -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: MPK persistent Qwen3-8B is no longer a
  matrix missing-evidence item. Remaining work items are PTO persistent-device
  full serving, VDCores full serving plus its `HF_TOKEN` readiness blocker,
  ThunderKittens-family full serving, and the official ThunderKittens upstream
  tensor-core sweep gaps.

### 2026-06-01 - VDCores Qwen3 8B Preflight

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  serving-readiness correction for the paper-target Qwen3-8B path.
- Exact Codex command or script invocation: ran a bounded H200 preflight with
  `HF_TOKEN= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
  python app/python/qwen3/sched.py --hf-cache-dir <shared-hf-cache>/hub
  --correctness`, copied artifacts back under
  `tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-preflight-b6df049f/`,
  added the execution-attempt record, updated the serving command generator,
  regenerated `serving_command_plan.json`, and refreshed review artifacts with
  `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, VDCores full-serving
  evidence for the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, CUDA evaluation helper
  script, in-progress evaluation docs, changelog docs, dispatch log, and
  tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  Qwen3-8B model load passed from the offline H200 cache, but the run stopped
  before correctness because the compiled VDCores `dae.runtime` lacks the
  Qwen3-8B compute-operator set reported by the launcher.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `41 passed`;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the VDCores serving work item now points
  at `app/python/qwen3/sched.py` and carries a specific missing-operators
  execution blocker. Remaining work is to rebuild `dae.runtime` with that
  `DAE_COMPUTE_OPS` superset, rerun correctness, then build a paper-serving
  harness that imports Qwen3-8B latency/throughput rows.

### 2026-06-01 - VDCores Qwen3 8B Rebuild And Correctness

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  Qwen3-8B runtime rebuild and bounded execution validation.
- Exact Codex command or script invocation: on `bizhaoh200`, built
  `dae.runtime` with `DAE_COMPUTE_OPS_FILE=<artifact>/qwen3-8b-compute-ops.vdcore.build`,
  the H200 CUTLASS include path, and `-include cfloat`; then ran
  `app/python/qwen3/sched.py --hf-cache-dir <shared-hf-cache>/hub --correctness`,
  `-N 1 -b 5`, and the paper-policy `-N 64 -b 5` command.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, VDCores full-serving
  evidence for the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, in-progress evaluation docs,
  changelog docs, dispatch log, and tests. No upstream repositories were
  edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The runtime rebuild and bounded correctness passed, but the 64-token
  VDCores serving policy still fails before launch with
  `assert len(self.cinsts) <= ctensor.shape[0]` in the instruction builder.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `41 passed`;
  `jq empty docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: VDCores Qwen3-8B no longer has a
  missing-operator blocker. It now needs instruction-capacity work before the
  full `vdcores_offline_decode` 64-token row can be imported with final
  latency/throughput metrics.

### 2026-06-01 - VDCores Qwen3 8B Global Instruction Diagnostic

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned VDCores
  Qwen3-8B instruction-capacity and correctness diagnostic.
- Exact Codex command or script invocation: measured the `-N 64` Qwen3-8B
  schedule on `bizhaoh200` by monkeypatching `Launcher.build_instructions` to
  emit per-SM instruction counts before allocation. Then applied
  `docs/nvidia-backend/baseline-patches/vdcores-qwen3-8b-global-insts-16384.patch`
  to the remote VDCores checkout, rebuilt `dae.runtime` with the Qwen3-8B
  compute-op file, ran `-N 64 -b 1`, ran `-N 64 -b 5`, ran `--correctness`,
  restored the source patch, and rebuilt the shared-instruction runtime.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, VDCores full-serving
  evidence for the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, baseline reproducibility
  patch, in-progress evaluation docs, changelog docs, dispatch log, and tests.
  No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The 64-token schedule requires up to `2177` compute instructions and
  `15042` memory instructions per SM, beyond the default shared-load
  `512`-instruction limit. The global-instruction runtime with
  `numInsts=16384` runs timing but fails correctness thresholds, so it cannot
  be imported as a paper result.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed; `jq empty
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `41 passed`.
  Raw H200 logs show `-N 64 -b 5` exit status `0`, median execution time
  `303995744 ns`, and `--correctness` exit status `1`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the VDCores Qwen3-8B full-serving
  blocker is now sharper: fix correctness for the global-instruction path or
  implement a segmented/token-windowed schedule under the shared-instruction
  runtime before importing the VDCores paper-serving row.

### 2026-06-01 - ThunderKittens FA3 Comparator Capture

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  ThunderKittens official-comparator gap narrowing.
- Exact Codex command or script invocation: cloned FlashAttention under
  `tmp/baselines/flash-attention`, built FlashAttention-3 from
  `tmp/baselines/flash-attention/hopper` on `bizhaoh200` with
  `MAX_JOBS=4`, `FLASH_ATTENTION_FORCE_BUILD=TRUE`, SM80 disabled, FP16/FP8
  disabled, non-128 head dimensions disabled, and optional split/paged/local
  variants disabled. Then ran the unmodified ThunderKittens
  `kernels/attention/mha_h100/benchmark.py` and `test_correctness.py` with a
  `PYTHONPATH` shim that requests `return_attn_probs=True` from
  `flash_attn_interface.flash_attn_func`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, tensor-core baseline
  evidence for the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, in-progress evaluation docs,
  changelog docs, dispatch log, tests, and raw `tmp/` artifacts. No upstream
  repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The broad default FA3 build was stopped because it compiled many unused
  SM80 and head-dimension variants; the narrowed build completed and installed
  `flash-attn-3` into the remote project venv. The compatibility shim is
  required because the current FA3 API returns only `out` by default while the
  ThunderKittens benchmark expects `(out, lse)`.
- Verification commands and results: remote FA3 build status `0`; FA3 API
  probe showed `return_attn_probs=True` returns `(out, lse)`; official
  ThunderKittens benchmark status `0`; official correctness status `0`. FA3
  rows completed for forward/backward, causal/non-causal, and sequence
  lengths 768, 1536, 3072, 6144, and 12288.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the tensor-core work queue no longer
  treats FA3 bindings as missing. The remaining official ThunderKittens sweep
  blocker was PyTorch reference OOM in selected 6144- and 12288-token cells
  before the isolated-reference capture below.

### 2026-06-01 - ThunderKittens Isolated PyTorch Reference Cells

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  ThunderKittens official-reference OOM isolation.
- Exact Codex command or script invocation: copied
  `tmp/cuda-backend/paper-baselines/thunderkittens/pt-reference-isolated-60d797cd/run_pt_reference_cell.py`
  to `bizhaoh200`, then launched each selected PyTorch reference cell as a
  fresh process with `CUDA_VISIBLE_DEVICES=0` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The matrix covered
  forward/backward, causal/non-causal, and sequence lengths 6144 and 12288.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, tensor-core baseline
  evidence for the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data, in-progress evaluation docs,
  changelog docs, dispatch log, tests, and raw `tmp/` artifacts. No upstream
  repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  Isolating the PyTorch references separates allocator fragmentation or
  monolithic benchmark sequencing from true dense-reference memory capacity.
- Verification commands and results: isolated 6144-token cells passed for
  forward/backward and causal/non-causal modes. Isolated 12288-token cells
  still OOM for all four selected modes. `validate_benchmark_viewer_data.py`
  -> passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed; `jq empty
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `41 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: the tensor-core work queue now excludes
  6144-token PyTorch references from the blocker. The remaining official
  ThunderKittens sweep gap is 12288-token dense PyTorch reference capacity.

### 2026-06-01 - ThunderKittens Dense Reference Policy

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned tensor-core
  paper-readiness policy update.
- Exact Codex command or script invocation: promoted
  `tensor_core_tile_baselines` to `ready_for_paper_claim`, added the accepted
  `thunderkittens_dense_pytorch_12288_oom_policy` exception to
  `paper_evaluation_matrix.json`, added exception rendering to the HTML
  viewer, tightened the benchmark-viewer validator, and regenerated
  `paper_readiness_audit.json`, `paper_readiness_work_queue.json`, run
  readiness, environment plans, and goal progress with
  `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, tensor-core baseline
  evidence for the paper-readiness matrix.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data and viewer code,
  in-progress evaluation docs, changelog docs, dispatch log, validators, and
  tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The policy depends on the isolated-reference capture proving that selected
  6144-token dense PyTorch reference cells pass while selected 12288-token
  dense PyTorch reference cells still OOM, and on FA3 comparator artifacts
  covering sequence length 12288.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed; `jq empty
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `41 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: tensor-core tile baselines are now
  paper-ready with an explicit OOM/not-applicable footnote policy for
  infeasible 12288-token dense PyTorch reference cells. The active work queue
  now contains only LLM-serving paper-baseline gaps.

### 2026-06-01 - PTO Full-Serving Preflight

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned PTO
  full-serving gap preflight for the LLM-serving paper claim.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-26c38df3/pto-serving-preflight.json`,
  then `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, LLM-serving PTO
  full-serving evidence.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA evaluation scripts, benchmark-viewer data,
  in-progress evaluation docs, changelog docs, dispatch log, and tests. No
  upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled,
  `.github/workflows/` stayed empty, and PR #1 had no status-check rollup. The
  preflight is not performance evidence; it records the missing PTO Qwen
  full-serving path.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed; `jq empty
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed; `py_compile pto_serving_preflight.py` ->
  passed; `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` ->
  `43 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: implement or import PTO
  persistent-device Qwen/Qwen3-8B model loading, tokenization, KV-cache, and
  decode-loop execution before importing `mpk_offline_decode` or
  `vdcores_offline_decode` PTO full-serving rows.

### 2026-06-01 - LLM Serving Coverage Guard

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned evidence
  guard for LLM-serving result coverage.
- Exact Codex command or script invocation: updated serving result rows,
  matrix evidence refs, the benchmark-viewer validator, result-producing
  scripts, and the HTML viewer so `llm_serving_decode` rows declare
  `serving_coverage`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, strict
  code-document-evidence guardrails for paper-ready LLM-serving evaluation.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: benchmark-viewer data and code, CUDA evaluation
  scripts, in-progress evaluation docs, changelog docs, dispatch log,
  validators, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled,
  `.github/workflows/` stayed empty, and PR #1 remained a standalone
  `uv-xiao/pto-cu:main` review. The guard does not produce new performance
  rows; it prevents proxy rows from satisfying full-serving evidence.
- Verification commands and results: `validate_benchmark_viewer_data.py` ->
  passed; `validate_nvidia_changelog.py` -> passed;
  `check_nvidia_review_ready.py` -> passed; `jq empty
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed; `pytest
  tests/ut/py/test_nvidia_review_artifacts.py -q` -> `43 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: full-serving rows must now use
  `full_serving` or `full_serving_latency_caveat`; controlled attention-tile
  proxies, native bring-up rows, and one-token diagnostics remain visible but
  cannot close the PTO, VDCores, or ThunderKittens full-serving gaps.

### 2026-06-01 - PTO Qwen Serving Scaffold

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned lifecycle
  scaffold for the PTO persistent-device Qwen/Qwen3-8B full-serving blocker.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/persistent_qwen_serving_scaffold.py --output-json
  tmp/cuda-backend/pto-serving-scaffold-76d4fca4/qwen-serving-scaffold.json`,
  then `pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-76d4fca4/pto-serving-preflight.json`,
  then `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, PTO full-serving
  implementation readiness for the LLM-serving paper claim.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA examples, CUDA evaluation scripts,
  benchmark-viewer data, in-progress evaluation docs, changelog docs,
  dispatch log, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The scaffold is not a performance row; it makes missing PTO Qwen lifecycle
  stages executable and reviewable.
- Verification commands and results: `validate_cuda_examples.py` -> passed;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed; `check_nvidia_review_ready.py` ->
  passed; `jq empty examples/cuda/manifest.json
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed; `py_compile
  persistent_qwen_serving_scaffold.py pto_serving_preflight.py` -> passed;
  `pytest tests/ut/py/test_nvidia_review_artifacts.py -q` -> `44 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: implement `qwen_tokenizer`,
  `qwen_weight_loader`, `kv_cache_lifecycle`, `decode_loop_runner`, and
  `viewer_result_import` before importing PTO Qwen/Qwen3-8B full-serving rows.

### 2026-06-01 - Qwen Serving Lifecycle Plan

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned lifecycle
  planning slice for the PTO persistent-device Qwen/Qwen3-8B full-serving
  blocker.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/qwen_serving_lifecycle_plan.py --output-json
  tmp/cuda-backend/pto-serving-lifecycle-e3c977f8/qwen-serving-lifecycle-plan.json`,
  then `persistent_qwen_serving_scaffold.py --output-json
  tmp/cuda-backend/pto-serving-scaffold-e3c977f8/qwen-serving-scaffold.json`,
  then `pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-e3c977f8/pto-serving-preflight.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, PTO full-serving
  implementation readiness for the LLM-serving paper claim.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA examples, CUDA evaluation scripts,
  benchmark-viewer data, in-progress evaluation docs, changelog docs,
  dispatch log, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The lifecycle plan uses the Qwen/Qwen3-8B config snapshot saved at
  `tmp/sources/qwen3-8b-config-d117af2f.json` for local inspection. This
  slice does not load model weights or produce a full-serving timing row.
- Verification commands and results: `validate_cuda_examples.py` -> passed;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed; `check_nvidia_review_ready.py` ->
  passed; `jq empty examples/cuda/manifest.json
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed; `py_compile
  qwen_serving_lifecycle_plan.py persistent_qwen_serving_scaffold.py
  pto_serving_preflight.py` -> passed; `pytest
  tests/ut/py/test_nvidia_review_artifacts.py -q` -> `45 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: tokenizer integration, safetensors
  loading, CUDA allocation/binding, generated Qwen kernel bodies,
  decode-loop execution, and viewer-result import remain required before
  importing PTO Qwen/Qwen3-8B full-serving rows.

### 2026-06-01 - Qwen Prompt Accounting

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned tokenizer
  and prompt-accounting slice for the PTO persistent-device Qwen/Qwen3-8B
  full-serving blocker.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/qwen_prompt_accounting.py --mode offline --output-json
  tmp/cuda-backend/pto-serving-tokenizer-b95ff321/qwen-prompt-accounting.json`,
  then `persistent_qwen_serving_scaffold.py --output-json
  tmp/cuda-backend/pto-serving-scaffold-b95ff321/qwen-serving-scaffold.json`,
  then `pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-b95ff321/pto-serving-preflight.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, PTO full-serving
  implementation readiness for the LLM-serving paper claim.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA examples, CUDA evaluation scripts,
  benchmark-viewer data, in-progress evaluation docs, changelog docs,
  dispatch log, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The tokenizer files were cached under `tmp/hf-tokenizers/` for local
  inspection; this slice records prompt accounting only and does not bind token
  IDs into CUDA runtime buffers.
- Verification commands and results: `validate_cuda_examples.py` -> passed;
  `validate_benchmark_viewer_data.py` -> passed;
  `validate_nvidia_changelog.py` -> passed; `check_nvidia_review_ready.py` ->
  passed; `jq empty examples/cuda/manifest.json
  docs/nvidia-backend/benchmark-viewer/data/*.json` -> passed;
  `git diff --check` -> passed; `py_compile qwen_prompt_accounting.py
  qwen_serving_lifecycle_plan.py persistent_qwen_serving_scaffold.py
  pto_serving_preflight.py` -> passed; `pytest
  tests/ut/py/test_nvidia_review_artifacts.py -q` -> `46 passed`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: runtime token-ID binding, safetensors
  loading, CUDA allocation/binding, generated Qwen kernel bodies,
  decode-loop execution, and viewer-result import remain required before
  importing PTO Qwen/Qwen3-8B full-serving rows.

### 2026-06-01 - Qwen Weight Inventory

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned safetensors
  index inventory slice for the PTO persistent-device Qwen/Qwen3-8B
  full-serving blocker.
- Exact Codex command or script invocation:
  `curl -L --fail --silent --show-error
  https://huggingface.co/Qwen/Qwen3-8B/raw/d117af2f304f02a8647f88fe05b61cfb405a1d9e/model.safetensors.index.json
  -o tmp/sources/qwen3-8b-model-safetensors-index-d117af2f.json`,
  then `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/qwen_weight_inventory.py --output-json
  tmp/cuda-backend/pto-serving-weights-edbd4390/qwen-weight-inventory.json`,
  then `persistent_qwen_serving_scaffold.py --output-json
  tmp/cuda-backend/pto-serving-scaffold-edbd4390/qwen-serving-scaffold.json`,
  then `pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-edbd4390/pto-serving-preflight.json`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, PTO full-serving
  implementation readiness for the LLM-serving paper claim.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA examples, CUDA evaluation scripts,
  benchmark-viewer data, in-progress evaluation docs, changelog docs,
  dispatch log, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The Qwen safetensors index was saved under `tmp/sources/` for local
  inspection; this slice records shard/tensor inventory only and does not open
  safetensors shards or bind weights into CUDA memory.
- Verification commands and results: passed
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py
  -q` with `47 passed`; passed `validate_cuda_examples.py`,
  `validate_benchmark_viewer_data.py`, `validate_nvidia_changelog.py`,
  `check_nvidia_review_ready.py`, `git diff --check`, `jq empty
  examples/cuda/manifest.json docs/nvidia-backend/benchmark-viewer/data/*.json`,
  and `py_compile` for the Qwen serving/inventory scripts and PTO serving
  preflight.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: safetensors tensor open, tensor
  shape/dtype validation, CUDA weight binding, runtime token-ID binding, CUDA
  KV-cache allocation/binding, generated Qwen kernel bodies, decode-loop
  execution, and viewer-result import remain required before importing PTO
  Qwen/Qwen3-8B full-serving rows.

### 2026-06-01 - Qwen Weight Shape Contract

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned expected
  shape/dtype contract slice for the PTO persistent-device Qwen/Qwen3-8B
  full-serving blocker.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/qwen_weight_inventory.py --output-json
  tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json`,
  then `persistent_qwen_serving_scaffold.py --output-json
  tmp/cuda-backend/pto-serving-scaffold-e06636e9/qwen-serving-scaffold.json`,
  then `pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-e06636e9/pto-serving-preflight.json`,
  then `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, PTO full-serving
  implementation readiness for the LLM-serving paper claim.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA examples, CUDA evaluation scripts,
  benchmark-viewer data, in-progress evaluation docs, changelog docs,
  dispatch log, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The Qwen config and safetensors index were saved under `tmp/sources/` for
  local inspection; this slice derives expected tensor shapes from config and
  still does not open safetensors shards or bind weights into CUDA memory.
- Verification commands and results: focused TDD test first failed because
  `qwen_weight_inventory.py` did not accept `--config-json`; after
  implementation, passed `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py -q` with `47 passed`; passed
  `validate_cuda_examples.py`, `validate_benchmark_viewer_data.py`,
  `validate_nvidia_changelog.py`, `check_nvidia_review_ready.py`,
  `jq empty examples/cuda/manifest.json
  docs/nvidia-backend/benchmark-viewer/data/*.json`, `py_compile` for the
  Qwen serving/inventory scripts and PTO serving preflight, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: safetensors tensor open, actual
  safetensors shape/dtype validation, CUDA weight binding, runtime token-ID
  binding, CUDA KV-cache allocation/binding, generated Qwen kernel bodies,
  decode-loop execution, and viewer-result import remain required before
  importing PTO Qwen/Qwen3-8B full-serving rows.

### 2026-06-01 - Qwen Safetensors Metadata Probe

- Dispatcher session or PR: local Codex session on
  `goal/nvidia-paper-ready`; PR targets `uv-xiao/pto-cu:main`.
- Worker id and objective: no worker dispatched; dispatcher-owned
  safetensors-header probe slice for the PTO persistent-device Qwen/Qwen3-8B
  full-serving blocker.
- Exact Codex command or script invocation:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/qwen_safetensors_metadata.py --weight-inventory-json
  tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json
  --output-json
  tmp/cuda-backend/pto-serving-safetensors-ff252c1f/qwen-safetensors-metadata.json`,
  then `persistent_qwen_serving_scaffold.py --output-json
  tmp/cuda-backend/pto-serving-scaffold-ff252c1f/qwen-serving-scaffold.json`,
  then `pto_serving_preflight.py --output
  tmp/cuda-backend/pto-serving-preflight-ff252c1f/pto-serving-preflight.json`,
  then `refresh_nvidia_review_artifacts.py`.
- Parent goal and child slice:
  `docs/in_progress/nvidia_backend_paper_ready.md`, PTO full-serving
  implementation readiness for the LLM-serving paper claim.
- Branch name and PR URL: `goal/nvidia-paper-ready`,
  `https://github.com/uv-xiao/pto-cu/pull/1`.
- Allowed scope and files: CUDA examples, CUDA evaluation scripts,
  benchmark-viewer data, in-progress evaluation docs, changelog docs,
  dispatch log, and tests. No upstream repositories were edited or pushed.
- Dependencies and blocked assumptions: repository Actions stayed disabled.
  The Qwen config and safetensors index were saved under `tmp/sources/`; this
  slice proves the header-parse/metadata-compare path on synthetic
  safetensors data and reports the five real Qwen shards missing locally.
- Verification commands and results: focused TDD test first failed because
  `qwen_safetensors_metadata.py` did not exist; scaffold/preflight tests then
  failed until the probe was wired into the lifecycle evidence. After
  implementation, passed `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_review_artifacts.py -q` with `48 passed`; passed
  `validate_cuda_examples.py`, `validate_benchmark_viewer_data.py`,
  `validate_nvidia_changelog.py`, `check_nvidia_review_ready.py`,
  `jq empty examples/cuda/manifest.json
  docs/nvidia-backend/benchmark-viewer/data/*.json`, `py_compile` for the
  Qwen serving/inventory/metadata scripts and PTO serving preflight, and
  `git diff --check`.
- Merge decision and merge commit: pending.
- Handoff summary and remaining gaps: place or download the real Qwen
  safetensors shards, validate actual safetensors metadata, bind CUDA weights,
  bind runtime token IDs, allocate/bind CUDA KV-cache storage, generate Qwen
  kernel bodies, execute the decode loop, and import viewer results before PTO
  Qwen/Qwen3-8B full-serving rows can be added.
