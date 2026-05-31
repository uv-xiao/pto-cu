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
