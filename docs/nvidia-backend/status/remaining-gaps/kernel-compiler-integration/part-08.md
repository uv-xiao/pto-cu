# CUDA Backend Status: Kernel Compiler Integration Part 8

Result:
`tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-5139ba23/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired runner then ran
`cuda_validate_smoke.py`, which accepted both JSON payloads, required the
`a100` and `h200` artifacts, checked `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[9,2,1]`, zero scheduler
errors, and the generated report files. The A100 row reported
`device_wall_ns=51200` and `host_wall_ns=74021`; H200 reported
`device_wall_ns=51616` and `host_wall_ns=71847`. This validates that the
explicit graph descriptor path can reuse one prepared generated-dispatch
callable across two launches after resetting fan-in, ready flags, counters,
and scratch/output buffers.

The paired persistent-smoke runner now also requires expected generated
dispatch sequences for the existing DAG shapes: `chain`, `fork_join`,
`scratch_reuse`, tensor-tile and tensor-core-tile, scalar AXPY/scale/affine,
triad, quad, unary-square, `generic_args`, and `graph_descriptor`. The focused
unit selector for these paired workflow builders passed with `8 passed, 142
deselected`, after first failing because the validation command omitted those
`--expected-dispatch` checks.

The generic-argument descriptor path was then captured with repeat-run
lifecycle reuse on A100 and H200:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape generic_args --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired runner validated
`runtime=persistent_device`, `mode=dag`, `dag_shape=generic_args`,
`repeat_runs=2`, `launch_completed_counts=[3,3]`,
`dispatch_func_ids=[9,2,1]`, zero scheduler errors, and generated report
files. A100 reported per-launch device times `[44032,25600]`, total
`device_wall_ns=69632`, and `host_wall_ns=106096`. H200 reported per-launch
device times `[21088,19808]`, total `device_wall_ns=40896`, and
`host_wall_ns=4953113`. This extends lifecycle evidence beyond the
graph-descriptor repeat-run path to generic indexed tensor/scalar descriptor
slots.

The persistent generic-argument generated-dispatch body now also consumes all
four bounded descriptor slots when they are present. The `generic_args4` smoke
maps `tensor_args[0]=tmp0`, `tensor_args[1]=tmp3`, `tensor_args[2]=a`, and
`tensor_args[3]=b`, with scalar slots `[1.5, 0.25, 0.125, 0.0625]`.
Focused TDD checks first failed because the paired runner omitted the
`generic_args4` dispatch expectation and the scene-test task body ignored
slots 2 and 3. After the fix, local A100 checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'persistent_generic_args_four_slots or \
      persistent_device_generic_args_four_slots or generic_args4_workflow' \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape generic_args4 \
    --output-json \
      tmp/cuda-backend/persistent-generic_args4-smoke-working/a100.json
```

The paired A100/H200 repeat-run smoke was then captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape generic_args4 --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=generic_args4`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[9,2,1]`, zero scheduler
errors, resource policy `scheduler_blocks=1`, `worker_blocks=3`,
`block_dim=256`, `grid_dim=4`, and generated report files. A100 reported
per-launch device times `[33792,23552]`, total `device_wall_ns=57344`, and
`host_wall_ns=81101`. H200 reported per-launch device times `[24192,17888]`,
total `device_wall_ns=42080`, and `host_wall_ns=59414`. The same
`persistent_dag_generic_args_f32` four-slot scene-test path passed on H200
with `1 passed, 59 deselected` after the known PTO-ISA SSH refresh warning.

The explicit graph-descriptor adapter now has the same four-slot generic
argument coverage. The descriptor shape `graph_descriptor_generic_args4`
keeps the graph metadata path (`graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`) while mapping `tensor_args[0]=tmp0`,
`tensor_args[1]=tmp3`, `tensor_args[2]=a`, and `tensor_args[3]=b`, with
scalar slots `[1.5, 0.25, 0.125, 0.0625]`. Focused TDD checks first failed
because the smoke runner did not recognize this DAG shape and the paired
validator had no dispatch expectation. After the fix, local A100 checks
passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_generic_args_four_slots or graph_descriptor_generic_args4' \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 \
    --dag-shape graph_descriptor_generic_args4 \
    --output-json \
      tmp/cuda-backend/persistent-graph_descriptor_generic_args4-smoke-working/a100.json
```

The paired A100/H200 repeat-run smoke was then captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_generic_args4 --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor_generic_args4`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[9,2,1]`, zero scheduler
errors, `graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, and generated report files.
A100 reported per-launch device times `[33792,18432]`, total
`device_wall_ns=52224`, and `host_wall_ns=390302`. H200 reported per-launch
device times `[23936,21344]`, total `device_wall_ns=45280`, and
`host_wall_ns=62740`. The same graph-descriptor four-slot scene-test path
passed on H200 with `2 passed, 60 deselected` after the known PTO-ISA SSH
refresh warning.

The explicit graph-descriptor path also validates fixed tensor-arity task
descriptors for the generated-dispatch triad and quad shapes. Focused TDD
checks first failed because the smoke runner only accepted the fixed `triad`
and `quad` shapes, and the paired validator had no graph-descriptor dispatch
expectations. After adding `graph_descriptor_triad` and
`graph_descriptor_quad`, the focused local checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_descriptor_triad or graph_descriptor_quad'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py -q \
  -k 'graph_descriptor_triad or graph_descriptor_quad' --platform cuda
```

Results: `4 passed, 209 deselected` for the smoke workflow tests, and
`2 passed, 42 deselected` for the real CUDA A100 tests. The paired A100/H200
repeat-run smokes were then captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_triad --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-working

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_quad --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-working
```

Result:
`tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/`
and
`tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/`
contain `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validators required `runtime=persistent_device`,
`mode=dag`, `repeat_runs=2`, zero scheduler errors, graph fan-in `[0,0,2]`,
dependents `[2,2]`, `scheduler_blocks=1`, `worker_blocks=3`,
`block_dim=256`, and `grid_dim=4`. Triad validated dispatch `[6,2,1]` with
`tensor_args={"c":"tmp0"}`. Quad validated dispatch `[8,2,1]` with
`tensor_args={"c":"tmp0","d":"tmp3"}`.

| Shape | GPU | Device ns | Host ns | Per-launch device ns |
| ----- | --- | --------- | ------- | -------------------- |
| `graph_descriptor_triad` | A100 | 64512 | 95893 | `[40960,23552]` |
| `graph_descriptor_triad` | H200 | 63776 | 88670 | `[42112,21664]` |
| `graph_descriptor_quad` | A100 | 68608 | 101095 | `[44032,24576]` |
| `graph_descriptor_quad` | H200 | 62496 | 87014 | `[41120,21376]` |

The same shape is now promoted to a benchmark baseline named
`pto_persistent_dag_graph_generic_args4`. Focused TDD checks first failed
because `cuda_benchmark.py`, the paired benchmark runner, and the
paired-current capture validator did not recognize the new row or its expected
dispatch sequence. After the fix, the local benchmark/report tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_generic_args4_dag or current_a100_h200_workflow or \
      configured_capture or paired_current_requires_generic_args_baseline or \
      compact_current_preset or include_persistent_baselines or \
      same_work_batch_modes'
```

A quick A100/H200 single-baseline capture was then run through
`cuda_benchmark.py`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 4096 --repeats 1 --arch compute_80 \
    --single-baseline pto_persistent_dag_graph_generic_args4 \
    --label graph-generic-args4-baseline-a100
```

Result:
`tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor_generic_args4`,
`dispatch_func_ids=[9,2,1]`, zero scheduler errors, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, and `grid_dim=4`.
A100 reported `device_wall_ns=43008` and `host_wall_ns=58143`; H200 reported
`device_wall_ns=33664` and `host_wall_ns=43163`.

