# CUDA Backend Status: Kernel Compiler Integration Part 6

The graph tensor-core smoke then added the same explicit graph descriptor
coverage for the WMMA first task. The focused red test first failed because
`graph_tensor_core_tile` was not accepted as a DAG shape and the paired
runner did not classify it as a tensor-tile shape. After adding the shape, the
focused unit selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k graph_tensor_core_tile
```

Result: `2 passed, 218 deselected`.

Paired A100/H200 evidence used the graph tensor-core shape with two launches:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_tensor_core_tile --task-count 4 --queue-capacity 2 \
    --repeat-runs 2 --n 256 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-core-working
```

The artifact directory is:

```text
tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/
```

Both GPUs validated `runtime=persistent_device`, `mode=dag`, dispatch
`[10,1,2,1]`, graph fan-in `[0,1,1,2]`, dependents `[1,2,3,3]`, tensor tile
`16x16x16`, tensor-core metadata `wmma:m16n16k8:tf32->f32`,
`launch_completed_counts=[4,4]`, zero scheduler errors, and resource policy
`scheduler_blocks=1`, `worker_blocks=4`, `block_dim=256`, `grid_dim=5`.

| GPU | Device ns | Host ns | Per-launch device ns | Status |
| --- | --------- | ------- | -------------------- | ------ |
| A100 | 76800 | 106982 | `49152,27648` | pass |
| H200 | 57472 | 75314 | `31712,25760` | pass |

The same explicit graph tensor-core descriptor is now covered by the normal
L2 `SceneTestCase` path. The descriptor-only red test first failed because
`persistent_dag_graph_f32` accepted a graph task with `func_id=10` and an
incompatible `rows=8` WMMA tile. The graph adapter now applies the same
tensor-core compatibility guard as the fixed tensor-core adapter:
`rows` and `cols` in multiples of `16`, and `inner` divisible by `8`.

Focused local A100 coverage:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'graph_tensor_core_tile_args or graph_tensor_core_tile_with_ctypes_data' \
  --platform cuda
```

Result: `3 passed, 75 deselected`, covering graph descriptor construction,
the incompatible tensor-core descriptor rejection, and a no-torch real-data
ctypes scene through `Worker` / `ChipWorker`.

The same no-torch real-data scene passed on remote H200 after syncing the
working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda-12.8 \
   PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k graph_tensor_core_tile_with_ctypes_data --platform cuda'
```

Result: `1 passed, 77 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The explicit graph tensor-tile path is also exposed as the
`pto_persistent_dag_graph_tensor` benchmark baseline. It uses
`dag_shape=graph_tensor_tile`, receives the same `--tensor-rows`,
`--tensor-cols`, and `--tensor-inner` descriptor flags as the scalar tensor
DAG, and records both graph dependency metadata and tensor descriptor metadata
in the benchmark JSON and Markdown report. A one-repeat A100/H200 sample at
`N=512`, `16x16x16` is under
`tmp/cuda-backend/combined-graph-tensor-current-working/`. The capture
validated source-paper metadata, sanitized command examples, generated report
files, zero scheduler errors, and expected dispatch `[3,1,2,1]`. The sample
device times were `51200 ns` on A100 and `38080 ns` on H200.

The explicit graph tensor-core path is now exposed as the
`pto_persistent_dag_graph_tensor_core` benchmark and tensor-shape sweep
baseline. It uses `dag_shape=graph_tensor_core_tile`, preserves the same
graph descriptor metadata as the smoke path, and records the WMMA first task
as dispatch `[10,1,2,1]`.

The focused benchmark tests first failed because the tensor-shape sweep parser
rejected the new baseline, `run_single_sample` treated it as unknown, and the
publish validators still expected only the fixed tensor-core row. After adding
the baseline to the benchmark, sweep, and validator flows, the focused selector
passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_tensor_core or tensor_sweep_validator_compact_preset_keeps_dispatch_commas'
```

Paired A100/H200 evidence used the tensor-shape sweep with one repeat and a
`16x16x16` descriptor:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_graph_tensor_core \
    --shapes 16x16x16 --n 256 --repeats 1 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-core-benchmark-working
```

The validated working-tree artifact is:

```text
tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/
```

Validation used:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-sweep.json \
    --require-artifact a100 --require-artifact h200 \
    --require-baseline pto_persistent_dag_graph_tensor_core \
    --require-size 256 --require-shape 16x16x16 \
    --expected-repeats 1 --expected-result-count 2 \
    --require-dispatch pto_persistent_dag_graph_tensor_core=10,1,2,1 \
    --require-report-files --require-report-throughput \
    --require-command-examples \
    --require-source-papers
```

| GPU | Device ns | Host ns | PTX | Graph fan-in | Status |
| --- | --------- | ------- | --- | ------------ | ------ |
| A100 | 52224 | 73631 | `compute_80` | `0,1,1,2` | pass |
| H200 | 50144 | 64644 | `compute_90` | `0,1,1,2` | pass |

Both rows record graph dependents `[1,2,3,3]`, tensor tile `16x16x16`,
tensor-core metadata `wmma:m16n16k8:tf32->f32`, source-paper metadata, and
zero scheduler errors. The artifact label uses the then-current `HEAD`
because the sweep was captured from an uncommitted working tree.

The persistent scalar-scale scene-test adapter was then added to cover the
single-tensor plus scalar descriptor shape on the persistent-device runtime.
It compiles a generated-dispatch `func_id=11` task body, runs it before the
existing multiply/add fan-in branch, and uses ctypes-backed CPU tensors so the
same selector can run on H200 without `torch`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k scalar_scale --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k scalar_scale --platform cuda'
```

Result: local A100 reported `2 passed, 48 deselected`; remote H200 reported
`2 passed, 48 deselected` after the known PTO-ISA SSH refresh warning. The
full local CUDA scene-test file was also rerun after this adapter and reported
`50 passed`.

The same scalar-scale task body was promoted to the no-torch standalone
persistent DAG smoke so it can be captured without the full scene-test
framework. The smoke uses generated-dispatch `func_id` sequence `[11,2,1]`,
with the first task computing `tmp0 = scalar0 * a`, an independent multiply
branch computing `tmp1 = a * b`, and the final task adding both branches.
Focused local coverage:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'scalar_scale' --platform cuda

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_backend.py::test_cuda_persistent_device_smoke_runs_dispatch_dag_scalar_scale \
    -q --platform cuda
```

Results: `2 passed, 146 deselected` for the shape/paired-runner unit tests,
and `1 passed` for the local A100 real-data CUDA smoke.

The paired A100/H200 smoke was then captured with a tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape scalar_scale --task-count 3 --queue-capacity 2 \
    --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired runner validated both artifacts with
`runtime=persistent_device`, `mode=dag`, `dag_shape=scalar_scale`,
`completed_count=3`, `dispatch_func_ids=[11,2,1]`, `scalar0=2.0`, zero
scheduler errors, and generated report files. The A100 row reported
`device_wall_ns=40960` and `host_wall_ns=61301`; H200 reported
`device_wall_ns=25856` and `host_wall_ns=34808`.

The same DAG shape was then promoted to the selected benchmark path as
`pto_persistent_dag_scalar_scale`. Focused benchmark/report tests and a local
A100 single-baseline sample were run:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'scalar_scale or worker_and_dag_tables or old_captures_without_scalar_affine or include_all_default_persistent or include_batch_mode' \
    --platform cuda

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_scale \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --label scalar-scale-baseline-a100
```

Results: the focused report tests passed with `6 passed, 144 deselected`; the
local A100 single-baseline sample reported `status=pass`,
`dispatch_func_ids=[11,2,1]`, `scalar0=2.0`, and
`device_wall_ns=47104`.

