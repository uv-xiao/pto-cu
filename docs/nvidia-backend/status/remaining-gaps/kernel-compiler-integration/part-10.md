# CUDA Backend Status: Kernel Compiler Integration Part 10

All selected PTO rows use target-specific PTX (`compute_80` on A100 and
`compute_90` on H200), report zero scheduler errors, and are visible in the
generated Markdown/SVG compact report beside the tagged scalar and tensor-core
graph rows. The previous tagged-scalar compact gate remains under
`tmp/cuda-backend/tagged-scalar-compact-current-working/`.

Graph-descriptor dependency inference now builds the producer map from the
whole descriptor before inferring omitted `dependents`, so the scene-test graph
adapter no longer requires topological task order. A focused unit test first
failed with `fanin=[0,0,0]` for a reordered graph where the final consumer is
task `0`; after the inference change it passed with `fanin=[2,0,0]`,
`dependents=[0,0]`, and dispatch sequence `[1,9,2]`. The corresponding
no-torch ctypes scene test passed locally on A100 with `2 passed, 50
deselected`, and passed on H200 with `1 passed, 51 deselected` after syncing
the working tree; the H200 command printed the known PTO-ISA SSH refresh
warning before passing.

The reordered graph descriptor was also captured through the paired persistent
smoke runner:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_reordered --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor_reordered`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[1,9,2]`, zero scheduler
errors, and generated report files. A100 reported per-launch device times
`[39936,23552]`, total `device_wall_ns=63488`, and `host_wall_ns=91185`.
H200 reported per-launch device times `[25632,20608]`, total
`device_wall_ns=46240`, and `host_wall_ns=63520`.

The DAG-chain graph descriptor was then captured through the same paired
smoke runner, proving the five-task chain dependency shape as explicit runtime
graph metadata instead of only through the fixed `chain` shape:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_chain --task-count 5 \
    --queue-capacity 3 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_chain`, `repeat_runs=2`,
`launch_completed_counts=[5,5]`, `dispatch_func_ids=[1,2,1,2,1]`,
`graph_descriptor.fanin=[0,0,2,1,1]`,
`graph_descriptor.dependents=[2,2,3,4]`, zero scheduler errors, and generated
report files. A100 reported per-launch device times `[41984,27648]`, total
`device_wall_ns=69632`, and `host_wall_ns=94042`. H200 reported per-launch
device times `[31712,25632]`, total `device_wall_ns=57344`, and
`host_wall_ns=74979`.

The diamond graph descriptor was then captured through the same paired smoke
runner:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_diamond --task-count 5 \
    --queue-capacity 3 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_diamond`, `repeat_runs=2`,
`launch_completed_counts=[5,5]`, `dispatch_func_ids=[9,2,1,2,1]`,
`graph_descriptor.fanin=[0,0,2,2,2]`,
`graph_descriptor.dependents=[2,3,2,3,4,4]`, zero scheduler errors, and
generated report files. A100 reported per-launch device times
`[49152,31744]`, total `device_wall_ns=80896`, and
`host_wall_ns=111293`. H200 reported per-launch device times
`[24096,23520]`, total `device_wall_ns=47616`, and
`host_wall_ns=4912047`.

The scratch-reuse graph descriptor was then added as
`graph_descriptor_scratch_reuse`, so the explicit runtime graph path now
covers the six-task scratch-reuse DAG shape as descriptor data instead of only
through the fixed `scratch_reuse` shape. Focused TDD checks first failed
because the paired runner rejected the new DAG shape and `_make_dag_shape`
could not build it; after the fix, the focused unit selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'scratch_reuse_graph_descriptor'
```

Result: `2 passed, 178 deselected`.

The paired A100/H200 smoke was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scratch_reuse --task-count 6 \
    --queue-capacity 3 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_scratch_reuse`, `repeat_runs=2`,
`launch_completed_counts=[6,6]`, `dispatch_func_ids=[1,2,1,2,1,1]`,
`graph_descriptor.fanin=[0,0,2,1,1,2]`,
`graph_descriptor.dependents=[2,2,3,4,5,5]`, zero scheduler errors, and
generated report files. A100 reported per-launch device times
`[55296,33792]`, total `device_wall_ns=89088`, and `host_wall_ns=121166`.
H200 reported per-launch device times `[36640,29504]`, total
`device_wall_ns=66144`, and `host_wall_ns=84089`.

The same scratch-reuse shape is now covered by the L2 `SceneTestCase`
graph-descriptor adapter with logical-output and storage-output separation.
The graph task uses `out="tmp4"` and `out_storage="tmp0"` for the reuse task,
so tensor-flow inference still sees a unique logical producer while the task
descriptor reuses the original `tmp0` buffer after its last consumer. Focused
TDD first failed because the builder allocated a distinct buffer for `tmp4`;
after adding `out_storage`, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'reused_output_storage or graph_scratch_reuse_with_ctypes' \
  --platform cuda
```

Result: `2 passed, 62 deselected`. The same real-data ctypes scene selector
passed on remote H200 after syncing the working tree:
`1 passed, 63 deselected`.

Needed:

- broader CUDA scene-test argument builders beyond the current binary
  elementwise, unary square, scalar scale, axpy, affine, triad, quad,
  host-schedule generic-args, persistent scalar/DAG tracer bullets, and
  explicit graph-descriptor scratch-storage reuse.

