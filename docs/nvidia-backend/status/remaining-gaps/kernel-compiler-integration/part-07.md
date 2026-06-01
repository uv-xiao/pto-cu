# CUDA Backend Status: Kernel Compiler Integration Part 7

The same real-data ctypes graph test was run on the remote H200 after syncing
the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k inferred_graph_with_ctypes --platform cuda'
```

Result: `1 passed, 43 deselected`. The remote command also printed the known
PTO-ISA SSH refresh warning before the selected CUDA test passed.

The same real-data graph path was then run without an explicit `temporaries`
map, so `tmp0` and `tmp1` were allocated from task outputs:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k auto_temp_graph_with_ctypes --platform cuda'
```

Result: `1 passed, 45 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The explicit graph descriptor scalar fields now also resolve scalar argument
names from `TaskArgsBuilder`, not just numeric literals in the descriptor. The
focused TDD selector first failed because `_make_graph_task` called
`float("alpha")` for a graph task `scalar0` field. After adding scalar-name
resolution, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'graph_scalar_scale or scalar_field_names' --platform cuda
```

Result: `2 passed, 64 deselected`. The same real-data graph scalar-scale
ctypes selector passed on remote H200 after syncing the tree:
`1 passed, 65 deselected`, with the known PTO-ISA SSH refresh warning printed
before pytest.

The same scalar-scale graph shape is now covered by the no-torch paired
persistent-smoke workflow as `graph_descriptor_scalar_scale`, so it can be
validated outside `SceneTestCase` while still recording explicit runtime graph
metadata:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_scale --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-scale-working
```

Result:
`tmp/cuda-backend/graph-scalar-scale-working/persistent-graph_descriptor_scalar_scale-repeat2-smoke-15e9038f/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_scalar_scale`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, dispatch `[11,2,1]`,
`graph_descriptor.fanin=[0,0,2]`, `graph_descriptor.dependents=[2,2]`,
`scalar0=2.0`, resource policy `scheduler_blocks=1`, `worker_blocks=3`,
`block_dim=256`, `grid_dim=4`, and zero scheduler errors on both GPUs. A100
reported per-launch device times `[33792,19456]`, total
`device_wall_ns=53248`, and `host_wall_ns=82872`. H200 reported per-launch
device times `[39616,20096]`, total `device_wall_ns=59712`, and
`host_wall_ns=84805`.

The remaining fixed scalar variants now have the same no-torch graph
descriptor coverage. The paired persistent-smoke runner captured
`graph_descriptor_scalar_axpy` and `graph_descriptor_scalar_affine` under
`tmp/cuda-backend/graph-scalar-variants-working/`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_axpy --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-working
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_affine --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-working
```

Both captures validated `runtime=persistent_device`, `mode=dag`,
`repeat_runs=2`, `launch_completed_counts=[3,3]`,
`graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, `grid_dim=4`, and
zero scheduler errors. The AXPY descriptor reported dispatch `[4,2,1]`,
`scalar0=1.5`, A100 `device_wall_ns=62464`, and H200
`device_wall_ns=59072`. The affine descriptor reported dispatch `[5,2,1]`,
`scalar0=1.5`, `scalar1=0.5`, A100 `device_wall_ns=67584`, and H200
`device_wall_ns=59616`.

The tensor-core tile descriptor was then added to the same normal L2
`SceneTestCase` path as `persistent_dag_tensor_core_tile_f32`. Its first task
uses block-wide generated dispatch with `func_id=10`, while the remaining
residual, gate, and fan-in tasks reuse the scalar tensor DAG shape. Focused
local A100 coverage:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'tensor_core_tile_args or tensor_core_tile_with_ctypes_data' \
    --platform cuda
```

Result: `2 passed, 39 deselected`. The full local CUDA scene-test file was
then rerun and reported `41 passed`.

The no-torch ctypes tensor-core scene test was also run on H200 after syncing
the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k "tensor_core_tile_with_ctypes_data" --platform cuda'
```

Result: `1 passed, 40 deselected`. The H200 venv lacks `torch`, so the
torch-backed tensor-core scene test is local-only there; the ctypes version
validates the same L2 `Worker` and `TaskArgsBuilder` path with real CUDA data.
The command also printed the known PTO-ISA SSH refresh warning before passing.

The persistent-device scene-test compiler path now also forwards
`CALLABLE["cuda"]["stream_id"]` into the prepared callable manifest. A focused
local A100 check used `stream_id=1` in the compile/plumbing test and the
no-torch tensor-core ctypes scene test:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'compiles_cuda_persistent_device_callable or tensor_core_tile_with_ctypes_data' \
    --platform cuda
```

Result: `2 passed, 39 deselected`.

The same non-default-stream tensor-core ctypes scene test was then run on H200
after syncing the working tree. Result: `1 passed, 40 deselected`; the command
printed the known PTO-ISA SSH refresh warning before passing.

The same normal L2 persistent-device scene-test path now also checks
device-side scheduler diagnostics after `Worker.run`. A bad explicit graph
with unsupported `func_id=99` raises
`CUDA persistent DAG scheduler error code=1 task_id=0 count=1` even with
`skip_golden=True`, while a good explicit graph still validates real copied
data. Focused local A100 coverage:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'reports_cuda_persistent_scheduler_errors or graph_with_ctypes or tensor_core_tile_with_ctypes_data' \
    --platform cuda
```

Result: `3 passed, 39 deselected`. The full local CUDA scene-test file was
then rerun and reported `42 passed`.

The diagnostic and good-graph ctypes tests were also run on H200 after
syncing the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k "reports_cuda_persistent_scheduler_errors or graph_with_ctypes" \
     --platform cuda'
```

Result: `3 passed, 41 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The explicit graph-descriptor path has now been promoted into the benchmark
scripts as `pto_persistent_dag_graph`, using `dag_shape=graph_descriptor`.
This row uses the generated-dispatch `func_id` sequence `[9,2,1]`, generic
tensor slots `tensor_args[0]=tmp0,tensor_args[1]=tmp3`, scalar slots
`scalar_args[0]=1.5,scalar_args[1]=0.25`, and the explicit runtime graph
metadata `fanin=[0,0,2]` and `dependents=[2,2]`.

Focused local test coverage for the benchmark/report wiring:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
  -q --platform cuda
```

Result: `108 passed`.

The focused single-baseline path was checked on A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph --sizes 4096 \
    --repeats 1 --arch compute_80
```

The same command was run on H200 with `--arch compute_90` and the remote venv
Python. Result:
`tmp/cuda-backend/persistent-graph-baseline-working/` contains `a100.json`,
`h200.json`, `cuda-smoke-report.md`, and `cuda-smoke-report.svg`. The A100
row reported `status=pass`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_80`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=36864`; H200 reported
`status=pass`, `ptx_source=nvcc-persistent-generated-dispatch-compute_90`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=31424`.

The paired persistent-smoke runner also supports `graph_descriptor`, so the
explicit graph path can be captured with the same A100/H200 lifecycle workflow
as the fixed DAG shapes. A repeat-run lifecycle smoke was captured at commit
`5139ba23` with automatic smoke artifact validation enabled:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

