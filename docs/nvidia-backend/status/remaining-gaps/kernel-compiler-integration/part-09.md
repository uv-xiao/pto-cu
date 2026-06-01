# CUDA Backend Status: Kernel Compiler Integration Part 9

After promoting the graph tagged-inout, graph triad, and graph quad baselines
to the selected benchmark matrix, the paired-current validator then expected
`954` full paired samples or `72` compact paired samples. The focused
benchmark/report TDD selector passed locally after adding the graph-triad and
graph-quad rows:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_triad_dag or graph_quad_dag or \
      paired_current_requires_generic_args_baseline or \
      compact_current_preset_matches_docs_gate or \
      current_a100_h200_workflow or \
      validate_command_matches_configured_capture or \
      include_persistent_device_modes or same_work_batch_modes or \
      omits_empty_batch_sweeps'
```

Result: `9 passed, 206 deselected`.

The graph tensor-arity rows were then captured through the compact paired
A100/H200 benchmark gate:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-benchmark-working
```

The paired runner wrote local A100, remote H200, and merged reports under
`tmp/cuda-backend/graph-tensor-arity-benchmark-working/`, then validated the
combined `combined-current-943620bf/cuda-benchmark.json` with the
`compact-current` preset. It required 72 samples, source-paper provenance,
sanitized command examples, generated Markdown/SVG reports, zero scheduler
errors, triad dispatch `[6,2,1]`, quad dispatch `[8,2,1]`, graph fan-in
`[0,0,2]`, and graph dependents `[2,2]`.

| GPU | Base DAG ns | Fixed triad ns | Graph triad ns | Fixed quad ns | Graph quad ns |
| --- | ----------- | -------------- | -------------- | ------------- | ------------- |
| A100 | 47104 | 33792 | 29696 | 39936 | 33792 |
| H200 | 41472 | 35296 | 30880 | 33376 | 28704 |

The combined report directory contains `cuda-benchmark.json`,
`cuda-benchmark.md`, `cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

The graph scalar-scale DAG is now also part of the selected paired benchmark
matrix as `pto_persistent_dag_graph_scalar_scale`. The row keeps the same
scalar-scale generated-dispatch body as the fixed
`pto_persistent_dag_scalar_scale` row, but represents it through explicit
runtime graph metadata. The paired gate first failed under TDD because the
validator preset, paired benchmark row list, and `run_single_sample(...)`
dispatcher did not know the graph scalar-scale baseline. After wiring the
row, the focused tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_scalar_scale or paired_current_requires_generic_args_baseline or \
      compact_current_preset_matches_docs_gate or \
      pair_benchmark_builds_current_a100_h200_workflow or \
      pair_benchmark_validate_command_matches_configured_capture'
```

Result: `5 passed, 258 deselected`.

The local A100 single-row check validated dispatch `[11,2,1]`, graph fan-in
`[0,0,2]`, dependents `[2,2]`, scalar metadata `scalar0=2.0`, and zero device
scheduler errors:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scalar_scale \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --output-dir tmp/cuda-backend/graph-scalar-scale-benchmark-working/a100-single
```

The paired A100/H200 compact selected-baseline gate then validated
`80` samples with source-paper provenance, command examples, Markdown/SVG
reports, visible graph topology, tensor throughput rows, zero scheduler
errors, and the new graph scalar-scale topology requirements:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-scale-benchmark-working
```

Artifacts:

- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/a100-current-993254e8/`
- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/h200-current-993254e8/`
- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/combined-current-993254e8/`
- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/index.md`

| GPU | N | Dispatch | Fan-in | Dependents | Scalar args | Device ns | Status |
| --- | - | -------- | ------ | ---------- | ----------- | --------- | ------ |
| A100 | 1024 | `11,2,1` | `0,0,2` | `2,2` | `scalar0=2.0` | 28672 | pass |
| H200 | 1024 | `11,2,1` | `0,0,2` | `2,2` | `scalar0=2.0` | 27712 | pass |

The generated benchmark Markdown and SVG graph-metadata sections now render a
`Scalar args` column, so the scalar descriptor value is visible outside the
raw JSON artifact.

The graph scalar AXPY and affine DAGs are now selected benchmark rows as
`pto_persistent_dag_graph_scalar_axpy` and
`pto_persistent_dag_graph_scalar_affine`. They use the same generated task
bodies as the fixed scalar DAG rows, but run through explicit graph descriptor
metadata. The first focused TDD run failed because the paired benchmark row
list, capture preset, and `run_single_sample(...)` dispatcher did not include
the two graph scalar rows. After wiring them, the focused regression subset
passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'paired_current_requires_generic_args_baseline or \
      compact_current_preset_matches_docs_gate or \
      pair_benchmark_builds_current_a100_h200_workflow or \
      pair_benchmark_validate_command_matches_configured_capture or \
      omits_empty_batch_sweeps or include_persistent_device_modes or \
      same_work_batch_modes or graph_scalar_axpy_dag or \
      graph_scalar_affine_dag'
```

Result: `9 passed, 257 deselected`.

A100 single-row checks validated the two promoted rows before the paired gate:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scalar_axpy \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --output-dir tmp/cuda-backend/graph-scalar-variants-benchmark-working/a100-axpy-single

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scalar_affine \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --output-dir tmp/cuda-backend/graph-scalar-variants-benchmark-working/a100-affine-single
```

The compact selected-baseline gate then validated `84` A100/H200 samples with
source-paper provenance, command examples, generated Markdown/SVG reports,
visible graph topology, scalar metadata, tensor throughput rows, and zero
scheduler errors:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-benchmark-working
```

Artifacts:

- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/a100-current-93fc927d/`
- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/h200-current-93fc927d/`
- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/combined-current-93fc927d/`
- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/index.md`

| GPU | N | Baseline | Dispatch | Fan-in | Dependents | Scalar args | Device ns |
| --- | - | -------- | -------- | ------ | ---------- | ----------- | --------- |
| A100 | 1024 | `pto_persistent_dag_graph_scalar_axpy` | `4,2,1` | `0,0,2` | `2,2` | `scalar0=1.5` | 28672 |
| A100 | 1024 | `pto_persistent_dag_graph_scalar_affine` | `5,2,1` | `0,0,2` | `2,2` | `scalar0=1.5,scalar1=0.5` | 34816 |
| H200 | 1024 | `pto_persistent_dag_graph_scalar_axpy` | `4,2,1` | `0,0,2` | `2,2` | `scalar0=1.5` | 25280 |
| H200 | 1024 | `pto_persistent_dag_graph_scalar_affine` | `5,2,1` | `0,0,2` | `2,2` | `scalar0=1.5,scalar1=0.5` | 25600 |

The selected paired benchmark gate now includes
`pto_persistent_dag_graph_tensor_core`. The paired runner was first updated
under TDD because it still omitted the row from its selected baseline list
even though the benchmark and validator accepted it. The focused tests passed:

The paired-current validator now also rejects stale tensor reports with
`--require-report-tensor-throughput`. That gate checks that
`cuda-benchmark.md` contains the `Tensor Throughput Rows` table and that
`cuda-benchmark-throughput.svg` visibly includes each required tensor/core and
cuBLAS baseline with the requested tensor descriptor shape.
The gate was validated on the current compact paired A100/H200 capture at
artifact label `a9d028de`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/tensor-throughput-gate-current-working
```

The paired runner wrote and validated
`tmp/cuda-backend/tensor-throughput-gate-current-working/combined-current-a9d028de/cuda-benchmark.json`
with the `compact-current` preset. It required `84` samples, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports,
visible tensor throughput rows, zero scheduler errors, graph descriptor
topology, graph task-argument metadata, and selected tensor/cuBLAS baseline
rows. Selected rows:

| GPU | Host ns | Base DAG ns | Graph tensor-core ns | cuBLAS graph ns | Grid batch ns |
| --- | ------- | ----------- | -------------------- | --------------- | ------------- |
| A100 | 19456 | 46080 | 37888 | 11264 | 35840 |
| H200 | 13984 | 39904 | 32288 | 9472 | 28128 |

The graph tensor-core row validates dispatch `10,1,2,1`, fan-in
`0,1,1,2`, dependents `1,2,3,3`, tensor tile `16x16x16`, and zero scheduler
errors on both GPUs. The tensor-throughput table reports A100
`0.86 GF/s` for graph tensor-core and `2.91 GF/s` for cuBLAS Graph; H200
reports `1.01 GF/s` for graph tensor-core and `3.46 GF/s` for cuBLAS Graph.

The previous graph-unary compact paired A100/H200 capture is:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-unary-benchmark-working
```

The paired runner wrote and validated
`tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark.json`
with the `compact-current` preset. It required 78 samples, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports, zero
scheduler errors, graph tensor-core metadata, tagged scalar graph metadata,
and the graph unary-square row. The graph unary-square row validates dispatch
`[7,1,1]`, graph fan-in `[0,1,1]`, graph dependents `[1,2]`, and a
three-task explicit graph descriptor for the same one-input square task body
used by the fixed unary DAG row.

| GPU | Fixed unary ns | Graph unary ns | Tagged scalar ns | Graph tensor-core ns |
| --- | -------------- | -------------- | ---------------- | -------------------- |
| A100 | 41984 | 36864 | 34816 | 39936 |
| H200 | 32416 | 31968 | 31552 | 40864 |

