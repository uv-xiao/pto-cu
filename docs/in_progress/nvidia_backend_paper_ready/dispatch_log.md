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
