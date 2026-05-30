# NVIDIA Backend Shared Contracts

## Benchmark Viewer Contract

Benchmark data lives under
`docs/nvidia-backend/benchmark-viewer/data/`. Viewer records must be readable
without private context and pass
`.agents/checks/validate_benchmark_viewer_data.py`.

Benchmark, method, and paper-baseline records store their stable identifier in
the JSON `id` field. Result rows reference those identifiers as
`benchmark_id`, `method_id`, and `paper_baseline_id` when applicable.

Each benchmark record must include:

- `benchmark_id`: stable identifier used by results and reports;
- `description`: workload intent in human language;
- `math`: mathematical operation or dependency graph explanation;
- `code`: kernel or pseudo-code explanation;
- `run.command`: exact command or script entry point;
- `run.inputs`: workload shape, graph shape, dtype, and repeat policy;
- `evidence_refs`: repo-relative paths and required symbols.

Each method record must include:

- `method_id`: stable runtime or baseline identifier;
- `name`: human-readable method name;
- `category`: PTO runtime, vendor baseline, framework baseline, paper
  baseline, or diagnostic baseline;
- `launch_model`: host API, CUDA Graph, persistent device scheduler, library
  call, framework executor, or paper baseline executor;
- `evidence_refs`: code, docs, or source notes proving the method definition.

Each result record must include:

- `benchmark_id`;
- `method_id`;
- `hardware`: GPU model, host class, driver, CUDA toolkit, and clock policy
  when available;
- `commit`: local pto-cu commit;
- `inputs`: shape, dtype, graph, and repeat count;
- `statistic`: median, p50, p90, p99, mean, standard deviation, and sample
  count as applicable;
- `raw_artifact`: tmp path for JSON, CSV, log, or report input;
- `correctness`: pass, fail, skipped, or not applicable with reason.

The viewer must render these review-critical fields, not only validate them in
JSON. At minimum it shows `run.inputs`, method `category`, method
`launch_model`, result hardware, result statistic sample count, correctness,
and raw artifact path.

Raw CUDA benchmark captures should flow into viewer result records through
`.agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py`, using
`docs/nvidia-backend/benchmark-viewer/data/capture_imports.json` as the
committed mapping between raw capture baselines and viewer benchmark/method
IDs. Hand-edited result rows must match this schema and identify the raw
artifact directory under `tmp/`.

Raw paper-baseline captures should flow into viewer result records through
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py`.
Each raw row must include `paper_baseline_run_id`, `benchmark_id`, `hardware`,
`inputs`, `metrics`, and `correctness`. The importer uses
`paper_baseline_runs.json` to map the run to the matching paper-baseline
method ID, then writes viewer-compatible records with the raw artifact path
kept under `tmp/`.

Each paper-baseline record must include:

- `paper_baseline_id`: stable identifier for MPK, VDCores, or a baseline used
  by their papers;
- `paper_role`: why the baseline is required for paper readiness;
- `status`: source cloned, planned, build-ready, evaluated, or blocked;
- `source.upstream_url`: canonical repository or paper URL;
- `source.local_tmp_path`: local inspection path under `tmp/`;
- `source.commit`: commit, tag, release, or pending marker;
- `paper_baselines_to_reproduce`: named baselines or configurations from the
  source paper;
- `next_action`: concrete next command or setup step.

Each paper-baseline run record must include:

- `paper_baseline_id`: the MPK, VDCores, or paper-baseline system;
- `paper_evaluation_id`: the matrix claim this run would satisfy;
- `hardware_targets`: required GPU class for the run;
- `workload`: model, input, output, and batch or concurrency policy;
- `setup_commands` and `run_commands`: exact commands to execute;
- `expected_artifacts`: raw outputs under `tmp/`;
- `required_metrics`: metrics needed before import;
- `import_target`: viewer result file and importer notes.

Each paper-evaluation matrix claim must include:

- `id`: stable claim identifier;
- `claim`: the paper claim or comparison being prepared;
- `status`: planned, partially captured, or ready for paper claim;
- `workload_ids`: benchmark records that define the workload;
- `method_ids`: PTO, vendor, framework, or diagnostic methods in scope;
- `paper_baseline_ids`: MPK, VDCores, or paper-baseline systems in scope;
- `hardware_targets`: required GPU classes;
- `required_metrics`: correctness, timing, throughput, scheduler, resource,
  and artifact fields needed before promotion;
- `current_evidence_refs`: viewer rows, stable docs, or tmp artifact roots
  that support the current status;
- `missing_evidence`: explicit gaps before the claim is paper-ready;
- `promotion_gate`: condition for moving from planned or partial to
  paper-ready.

## Code Evidence Contract

Docs may describe implemented behavior only when one of these is true:

- a guard or test checks the referenced file or symbol;
- `evidence_refs` points to code or viewer data containing the symbol;
- the same PR adds a changelog report that names the implementation path and
  verification command.

Planned behavior must use planned language and must not appear in benchmark
result tables as implemented capability.

## Changelog Report Contract

Every child PR that changes NVIDIA backend code, examples, viewer data,
evaluation results, or stable docs adds a changelog report under
`docs/nvidia-backend/changelog/`.

Each changelog report must state:

- what code, data, or docs changed;
- how the architecture became clearer or more maintainable;
- what evaluation or verification ran;
- which artifacts prove the result;
- which gaps remain.

The index must link every report.
The report set must pass `.agents/checks/validate_nvidia_changelog.py`.

## Example Contract

`examples/cuda/` contains runnable examples that match evaluated workloads.
An example is reviewable only when its README entry names the matching
benchmark, command, expected output, and runtime it exercises.
The example manifest must pass `.agents/checks/validate_cuda_examples.py`.

## Source Notes Contract

Used source notes stay under `tmp/`. The source note for a paper or baseline
must include the upstream URL, local path, access date, why it was used, and
which evaluation contract it informs.

Raw source clones are not committed. Stable docs can name the source and the
local tmp note that captured the details.

## Remote Evaluation Contract

Paired CUDA evaluation scripts must support two refresh paths:

- remote Git refresh by default, using low-speed and timeout guards;
- explicit SSH tree-sync fallback when remote Git credentials or network
  transport fail.

Tree sync must exclude local virtualenvs, build outputs, raw tmp artifacts,
Python caches, and pytest caches. Remote commands must set CUDA and
`PYTHONPATH` explicitly and must not run `git fetch` or `git checkout` after a
tree sync. This contract is checked by
`.agents/checks/validate_remote_evaluation.py`.
