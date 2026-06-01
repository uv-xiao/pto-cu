# CUDA Backend Status: Latest Local Verification Part 5

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.json \
    --preset paired-current
```

Result: `validated tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.json`.

After adding `persistent_dag_unary_square_f32` to the normal SceneTestCase L2
persistent-device argument builders, the CUDA scene-test file was rerun
locally on A100:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py -q
```

Result: `30 passed`. The new unary persistent scene slice also uses a
ctypes-backed real-data case so it can run on the H200 checkout without
`torch`.

The same unary scene-test slice was run on the remote H200 checkout after
syncing the current local tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k unary_square'
```

Result: `2 passed, 28 deselected`.

The unary graph descriptor path was then promoted from a fixed
`persistent_dag_unary_square_f32` builder to an explicit runtime graph shape
named `graph_descriptor_unary_square`. The focused TDD check first failed
because the paired A100/H200 smoke runner rejected that shape at argument
parsing. After adding the smoke shape and graph-metadata expectations, the
focused local checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_descriptor_unary_square or unary_square_dag_shape'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'graph_unary_square or unary_square'
```

Results: `3 passed, 215 deselected` for the smoke workflow tests, and
`4 passed, 71 deselected` for the SceneTestCase tests. The paired A100/H200
repeat-run smoke was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_unary_square --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-unary-square-working
```

Result:
`tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required dispatch `[7,1,1]`,
`graph_descriptor.fanin=[0,1,1]`, `graph_descriptor.dependents=[1,2]`,
`launch_completed_counts=[3,3]`, zero scheduler errors,
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, and `grid_dim=4`.

| GPU | Device ns | Host ns | Per-launch device ns |
| --- | --------- | ------- | -------------------- |
| A100 | 75776 | 106800 | `[49152,26624]` |
| H200 | 57056 | 74823 | `[36960,20096]` |

After adding the four-input host-schedule ABI, the no-torch Worker quad smoke
was captured on local A100 and remote H200 with runtime rebuild enabled:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op quad --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-quad-smoke-4327698e/` contains `a100.json`,
`h200.json`, `cuda-smoke-report.md`, and `cuda-smoke-report.svg`. A100
reported `status=pass`, `ptx_arch=compute_80`, and `device_wall_ns=21504`;
H200 reported `status=pass`, `ptx_arch=compute_90`, and
`device_wall_ns=18752`. The local A100 quad smoke was also checked at
`N=65536` to exercise the same CUDA fused multiply-add rounding path used by
`ctx->a[i] * ctx->b[i] + ctx->c[i] * ctx->d[i]`.

After promoting the four-input host-schedule ABI to a benchmark baseline, the
focused single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_quad --sizes 4096 \
    --repeats 1 --arch compute_80
```

The same command was run on H200 with `--arch compute_90`. Result:
`tmp/cuda-backend/host-quad-baseline-working/` contains `a100.json` and
`h200.json`. The A100 row reported `status=pass`,
`ptx_source=kernel-compiler-task-body-wrapper-quad-compute_80`, and
`device_wall_ns=8192`; H200 reported `status=pass`,
`ptx_source=kernel-compiler-task-body-wrapper-quad-compute_90`, and
`device_wall_ns=24960`.

The full paired current benchmark was refreshed at commit `c0dc1372`.
Result: `tmp/cuda-backend/combined-current-c0dc1372/` contains the raw JSON,
Markdown report, median-device SVG, and ratio SVG. The combined JSON has
`684` samples, including `18` `pto_host_schedule_quad` samples and `18`
`pto_persistent_dag_quad` samples. The paired-current validator reported:
`validated tmp/cuda-backend/combined-current-c0dc1372/cuda-benchmark.json`.

After adding generic persistent DAG tensor/scalar argument slots, the
`generic_args` smoke was run locally on A100 and remotely on H200 with a tree
sync. The graph uses generated-dispatch `func_id` sequence `[9, 2, 1]`; the
first task computes from the base tensor fields plus `tensor_args[0]`,
`tensor_args[1]`, `scalar_args[0]`, and `scalar_args[1]`, and the final task
joins with an independent `a * b` branch. Result:
`tmp/cuda-backend/persistent-generic_args-smoke-7c99f607/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. Both runs reported zero scheduler errors and
argument metadata
`scalar_args[0]=1.5,scalar_args[1]=0.25,tensor_args[0]=tmp0,tensor_args[1]=tmp3`.
The same descriptor shape now also has normal `SceneTestCase` L2 coverage
through `persistent_dag_generic_args_f32`, using ctypes-backed CPU tensors so
the path remains usable on the H200 host without requiring `torch`. The
focused local command passed on A100:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k generic_args_with_ctypes --platform cuda
```

The same command was run on H200 after syncing the working tree to
`bizhaoh200`; it passed with `1 passed, 35 deselected`.

After promoting generic persistent DAG tensor/scalar argument slots to a
benchmark baseline, the focused single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_generic_args --sizes 4096 \
    --repeats 1 --arch compute_80
```

The same command was run on H200 with `--arch compute_90`. Result:
`tmp/cuda-backend/persistent-generic-args-baseline-working/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row reported `status=pass`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_80`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=30720`; H200 reported
`status=pass`, `ptx_source=nvcc-persistent-generated-dispatch-compute_90`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=33600`.

The full paired current benchmark was then refreshed at commit `61cf96cd`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-61cf96cd/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`, and
`cuda-benchmark-ratios.svg`. The combined JSON has `720` samples, including
`18` `pto_persistent_dag_generic_args` samples and `18`
`pto_persistent_dag_graph` samples. All rows reported pass
status. The paired-current validator reported:
`validated tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.json`.
The compact DAG table now includes `Graph Descriptor/DAG`; the H200 graph
descriptor ratios versus `pto_persistent_dag` are `0.95x`, `1.05x`, and
`1.00x` for `N=1024,65536,1048576`, while the A100 ratios are `1.08x`,
`1.19x`, and `1.05x`. Treat the DAG-shape rows as correctness and scheduler
shape evidence rather than tuned throughput claims.

The compact paired-current gate was refreshed again at commit `8e868bfe`
after the tensor-core scene-test and persistent `stream_id` plumbing changes.
It uses the WMMA-compatible `16x16x16` tensor descriptor:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-8e868bfe/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The combined JSON has `56` samples and the
compact-current validator reported:
`validated tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark.json`.
This capture validates all 28 selected baselines on A100 and H200, including
`pto_persistent_dag_scalar_scale`, `pto_persistent_dag_graph_diamond`, and
`pto_persistent_dag_graph_tensor`. Selected A100 device times for
host/base-DAG/tensor/tensor-core/cuBLAS/grid-batch were
`29696/48128/38912/36864/53247/49152 ns`; H200 reported
`14880/36512/48960/33632/37631/30176 ns`.

The compact paired-current gate was refreshed at commit `2aedb40f` after
adding the host-schedule generic-args benchmark row. It uses the same command
shape, `N=1024`, one repeat, `batch_tasks=2`, `worker_blocks_per_task=4`, and
the WMMA-compatible `16x16x16` tensor descriptor.

Result: `tmp/cuda-backend/combined-current-2aedb40f/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The combined JSON has `58` samples and the
compact-current validator reported:
`validated tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark.json`.
This capture validates all 29 selected baselines on A100 and H200, including
`pto_host_schedule_generic_args`. Selected A100 device times for host,
host-generic, base-DAG, persistent-generic, tensor, tensor-core, cuBLAS, and
grid-batch were `29696/43008/44032/29696/41984/41984/49152/40960 ns`; H200
reported `17920/36032/38112/31168/47008/32543/51520/32896 ns`.

The compact paired-current gate was refreshed again at commit `b2c5c8a4`
after promoting `pto_persistent_dag_graph_generic_args4` into the selected
benchmark path. It uses the same command shape, `N=1024`, one repeat,
`batch_tasks=2`, `worker_blocks_per_task=4`, and the WMMA-compatible
`16x16x16` tensor descriptor.

