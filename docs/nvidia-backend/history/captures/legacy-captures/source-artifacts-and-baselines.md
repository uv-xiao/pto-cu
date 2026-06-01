# Legacy Capture Sources And Baselines

This archive preserves the earlier CUDA backend evaluation narrative that
preceded the focused landing page and current-capture summary. See
[evaluation.md](../../../evaluation.md) for the active index and
[evaluation-current.md](../../../evaluation-current.md) for the latest paired
A100/H200 capture.

The archived content below summarizes earlier CUDA backend evaluation
evidence. The measurements are early runtime microbenchmarks, not end-to-end
LLM serving results. They are shaped by the VDCores and MPK papers only at the
evaluation structure level: fixed GPU work, repeated problem sizes, selected
launch baselines, local A100 runs, and remote H200 runs.

The archived raw reports are under `tmp/`:

- `tmp/cuda-backend/index.md`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/a100.json`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/h200.json`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/cuda-smoke-report.md`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/cuda-smoke-report.svg`
- `tmp/cuda-backend/a100-rangewide-cc6869f7/cuda-benchmark.md`
- `tmp/cuda-backend/h200-rangewide-cc6869f7/cuda-benchmark.md`
- `tmp/cuda-backend/combined-rangewide-cc6869f7/cuda-benchmark.md`
- `tmp/cuda-backend/combined-rangewide-cc6869f7/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-rangewide-cc6869f7/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-taskcount-7194bfc9/cuda-benchmark.md`
- `tmp/cuda-backend/h200-taskcount-7194bfc9/cuda-benchmark.md`
- `tmp/cuda-backend/combined-taskcount-7194bfc9/cuda-benchmark.md`
- `tmp/cuda-backend/combined-taskcount-7194bfc9/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-taskcount-7194bfc9/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-gridext-3eeb399a/cuda-benchmark.md`
- `tmp/cuda-backend/h200-gridext-3eeb399a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-gridext-3eeb399a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-gridext-3eeb399a/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-gridext-3eeb399a/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/h200-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/h200-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/h200-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/combined-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/combined-dag-323f4587/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/h200-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/combined-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/combined-reuse-bcf54a88/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-tensor-8950e029/cuda-benchmark.md`
- `tmp/cuda-backend/h200-tensor-8950e029/cuda-benchmark.md`
- `tmp/cuda-backend/combined-tensor-8950e029/cuda-benchmark.md`
- `tmp/cuda-backend/combined-tensor-8950e029/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-tensor-8950e029/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-graph-ba2cdd0e/cuda-benchmark.md`
- `tmp/cuda-backend/h200-graph-ba2cdd0e/cuda-benchmark.md`
- `tmp/cuda-backend/combined-graph-ba2cdd0e/cuda-benchmark.md`
- `tmp/cuda-backend/combined-graph-ba2cdd0e/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-graph-ba2cdd0e/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-current-6c49c5cf/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-6c49c5cf/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-current-db0acd4c/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-db0acd4c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-current-b060039c/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-b060039c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark-ratios.svg`

The tensor descriptor smoke data was captured from commit `38db010e`. The
wider vector/task range data was captured from commit `cc6869f7`. The
task-count sweep data was captured from commit `7194bfc9`. The extended
worker-grid data was captured from commit `3eeb399a`. The earlier worker-grid
data was captured from commit `e430bc1b`. The stream concurrency data was
captured from commit `37bebf44`. The DAG-chain data was captured from commit
`323f4587`. The scratch-reuse DAG data was captured from commit `bcf54a88`.
The tensor-tile DAG data was captured from commit `8950e029`. The CUDA Graph
launch-baseline data was captured from commit `ba2cdd0e`. The previous
current-capture data with the `8x4x12` tensor descriptor, before adding the
compiler-backed host-schedule row, was captured from commit `6c49c5cf`.
The previous current-capture data with the compiler-backed host-schedule row,
unary square row, and scalar AXPY row was captured from commit `db0acd4c`.
The previous current-capture data with scalar affine and triad DAG rows was
captured from commit `b060039c`.

`tmp/cuda-backend/index.md` is a generated local index that includes both
benchmark artifacts and compact smoke-report artifacts. It records tensor-tile
descriptor shapes when a benchmark or smoke payload carries that metadata.

## Current Baselines

- `direct_driver`: thin CUDA Driver API launch path for the same vector-add
  PTX kernel.
- `direct_driver_graph`: same Driver API vector-add kernel replayed through a
  CUDA Graph, with graph instantiation outside the measured interval. This is
  a host-launch amortization baseline, not a device-side scheduler.
- `pto_host_schedule`: PTO CUDA host runtime C API and manifest dispatch.
- `pto_persistent_device`: descriptor-array persistent executor.
- `pto_persistent_queue`: scheduler block plus bounded device ring queue.
- `pto_persistent_dag`: generated-dispatch-like task selection with fan-in
  counters.
- `pto_persistent_dag_chain`: five-task generated-dispatch DAG with a
  post-fan-in dependency chain. It reuses the same compiled device binary as
  `pto_persistent_dag`; the difference is only the runtime graph descriptors.
- `pto_persistent_dag_reuse`: six-task generated-dispatch DAG that reuses a
  scratch buffer after the buffer's last dependent completes. It is a
  lifecycle validation row rather than a throughput row.
- `pto_persistent_dag_tensor`: four-task generated-dispatch DAG with a tiled
  GEMM task followed by residual, gate, and fan-in elementwise tasks. The
  benchmark row uses the default 16x16x16 descriptor unless the benchmark is
  run with `--tensor-rows`, `--tensor-cols`, and `--tensor-inner`.
- `*_batch`: same-work rows with six vector-add task descriptors. These rows
  compare repeated host launches with one persistent launch over the same
  descriptor count.
- `pto_persistent_device_grid_batch`: direct persistent-device batch row with
  a swept number of CUDA worker blocks assigned to each task descriptor.

Ratios are relative to the matched host-schedule row for the same GPU, vector
length, and task count. For batch rows, the reference is
`pto_host_schedule_batch`, not the one-task `pto_host_schedule` row.
Generated reports also include a `DAG Shape Rows` table that compares
`pto_persistent_dag_*` rows against `pto_persistent_dag` for the same GPU and
vector length. Use that table for graph-shape interpretation because the
chain, reuse, and tensor DAGs intentionally have different task counts.
The generated `cuda-benchmark-ratios.svg` file visualizes the same matched
reference ratios used by the main Markdown table; use it for launch-overhead
and stream-concurrency comparisons where the rows share a reference task
count.
