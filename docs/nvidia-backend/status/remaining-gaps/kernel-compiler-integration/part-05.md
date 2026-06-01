# CUDA Backend Status: Kernel Compiler Integration Part 5

Result:
`tmp/cuda-backend/pair-current-compact-working/combined-current-c5094aa5/`
contains `cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The paired validator required `92` rows,
source-paper provenance, generated report files, zero scheduler errors, and
the pair row's dispatch `[1,1,1]`, graph fan-in `[0,1,1]`, dependents
`[1,2]`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=pair`. The pair row reported A100
`device_wall_ns=39936`, `host_wall_ns=54972`; H200 reported
`device_wall_ns=25120`, `host_wall_ns=33814`. The current-summary DAG table
now includes `Graph Pair Inout/DAG`; the compact selected capture reported
A100 `0.87x` and H200 `0.68x` versus the base
`pto_persistent_dag` row.

The same tagged graph shape is now also in the paired persistent-smoke report
flow as `graph_descriptor_tagged`, with scalar inputs recorded beside tensor
roles in `graph_task_args`. The current A100/H200 JSON plus Markdown/SVG
artifacts are under
`tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/`.
The paired validator accepted dispatch `9,2,1`, fan-in `[0,0,2]`,
dependents `[2,2]`, repeat completions `[3,3]`, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, scalar metadata
`scalar_args[0]=1.5`, `scalar_args[1]=0.25`, tagged task args
`input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]`,
`input:a,input:b,output:tmp2`, and
`input:tmp1,input:tmp2,output_existing:out`, and zero scheduler errors on
both GPUs. A100 reported per-launch device times `[44032,24576]`; H200
reported `[24768,18720]`. The regenerated Markdown/SVG smoke report and
artifact index also show graph fan-in/dependents and `Graph task args`, so
this artifact visibly ties the paired hardware result back to the tagged
tensor/scalar task-argument lowering form.

The host-schedule generic-args adapter was checked with a failing test first,
then local A100 and remote H200 real-data ctypes scene tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'builds_cuda_elementwise_generic_args' --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'host_schedule_elementwise_generic_args_with_ctypes_data' \
  --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k host_schedule_elementwise_generic_args_with_ctypes_data \
     --platform cuda'
```

Results: the local unit check reported `1 passed, 55 deselected`, the local
A100 real-data scene reported `1 passed, 55 deselected`, and the H200
real-data scene reported `1 passed, 55 deselected` after the known PTO-ISA SSH
refresh warning.

The same generic host-schedule ABI is now covered by the no-torch Worker
smoke runner, so it can be captured without the `SceneTestCase` framework:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op generic_args --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-generic_args-smoke-72c8186c/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The artifact validator accepted both rows with
`runtime=host_schedule` and `mode=worker/generic_args`; the A100 row reported
`ptx_source=kernel-compiler-worker-task-body-generic_args-compute_80` and
`device_wall_ns=35840`, while the H200 row reported
`ptx_source=kernel-compiler-worker-task-body-generic_args-compute_90` and
`device_wall_ns=15488`.

After widening host-schedule generic args to the four tensor/scalar slots
already present in `CudaVectorGenericArgs`, the new `generic_args4`
ctypes `SceneTestCase` and no-torch Worker smoke were run on A100 and H200.
Focused local checks:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'builds_cuda_elementwise_generic_args' --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'host_schedule_elementwise_generic_args_with_ctypes_data or \
      host_schedule_elementwise_generic_args_four_slots_with_ctypes_data' \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'generic_args4_helpers or generic_args_helpers_use_aux_tensor'
```

The paired no-torch Worker smoke was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op generic_args4 --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-generic_args4-smoke-03ed75da/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The artifact validator accepted both rows with
`runtime=host_schedule` and `mode=worker/generic_args4`; the A100 row
reported
`ptx_source=kernel-compiler-worker-task-body-generic_args4-compute_80` and
`device_wall_ns=26624`, while the H200 row reported
`ptx_source=kernel-compiler-worker-task-body-generic_args4-compute_90` and
`device_wall_ns=19552`.

The same host-schedule generic-args path is now a benchmark baseline:
`pto_host_schedule_generic_args`. It compiles a generated task-body wrapper
for the generic tensor/scalar packet and uses the same indexed tensor/scalar
values as the smoke path:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_generic_args \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --label host-generic-args-baseline-a100
```

Result: `tmp/cuda-backend/host-generic-args-baseline-working/` contains
`a100.json`, `h200.json`, `cuda-benchmark.json`, `cuda-benchmark.md`, and
SVG report files. The capture validator accepted both A100 and H200 rows with
`baseline=pto_host_schedule_generic_args`, `N=4096`, source-paper provenance,
and report files. The A100 row reported
`ptx_source=kernel-compiler-task-body-wrapper-generic-args-compute_80` and
`device_wall_ns=32768`; the H200 row reported
`ptx_source=kernel-compiler-task-body-wrapper-generic-args-compute_90` and
`device_wall_ns=17664`.

The graph-descriptor adapter was checked with focused local tests:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k 'graph_args or graph_edges or graph_temporaries or mixed_graph' \
  --platform cuda

.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k graph_with_ctypes --platform cuda
```

Result: the focused descriptor and real-data graph tests passed locally on
A100. The full CUDA scene-test file was also rerun locally and reported
`48 passed`.

The mixed explicit/inferred graph path was also run on the remote H200 after
syncing the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k mixed_graph_with_ctypes --platform cuda'
```

Result: `1 passed, 47 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The graph adapter now also forwards tensor-tile descriptor fields from each
graph task: rows, columns, inner dimension, leading dimensions, and per-tile
strides. This lets an explicit `persistent_dag_graph_f32` descriptor run the
same scalar tiled-GEMM first task as `persistent_dag_tensor_tile_f32`, then
feed residual, gate, and fan-in elementwise tasks. The focused red/green test
first failed with `rows == 0`, then passed after descriptor-field lowering.

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k graph_tensor_tile_args \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k 'graph_tensor_tile' \
  --platform cuda
```

Result: the second command reported `2 passed, 52 deselected` on the local
A100 path, covering both the struct descriptor and a real-data ctypes scene.
The full local CUDA scene-test file was also rerun after this adapter and
reported `54 passed`. The same no-torch real-data scene was run on the remote
H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k graph_tensor_tile_with_ctypes_data --platform cuda'
```

Result: `1 passed, 53 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The same explicit graph tensor-tile shape is now part of the persistent smoke
tooling as `--dag-shape graph_tensor_tile`. It records both graph dependency
metadata and tensor-tile descriptor metadata in the smoke JSON, then validates
the paired A100/H200 artifacts with expected dispatch `3,1,2,1`, completed
count `4`, repeat count `2`, tensor descriptor `16x16x16`, and generated
Markdown/SVG report files.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 512 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape graph_tensor_tile \
    --repeat-runs 2 --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --output-json tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/a100.json
```

The H200 run used the same command after syncing the working tree, with
`--arch compute_90` and output
`tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/h200.json`.
The report and validator commands then produced:

```text
tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/cuda-smoke-report.md
tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/cuda-smoke-report.svg
validated tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/a100.json,
tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/h200.json
```

Result summary: A100 reported `device_wall_ns=98304`, H200 reported
`device_wall_ns=67968`, and both reported zero scheduler errors,
`launch_completed_counts=[4,4]`, dispatch `[3,1,2,1]`, and
`graph_descriptor.dependents=[1,2,3,3]`.

