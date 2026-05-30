# NVIDIA Backend Shared Contracts

## Benchmark Viewer Contract

Benchmark data lives under
`docs/nvidia-backend/benchmark-viewer/data/`. Viewer records must be readable
without private context.

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

## Example Contract

`examples/cuda/` contains runnable examples that match evaluated workloads.
An example is reviewable only when its README entry names the matching
benchmark, command, expected output, and runtime it exercises.

## Source Notes Contract

Used source notes stay under `tmp/`. The source note for a paper or baseline
must include the upstream URL, local path, access date, why it was used, and
which evaluation contract it informs.

Raw source clones are not committed. Stable docs can name the source and the
local tmp note that captured the details.
