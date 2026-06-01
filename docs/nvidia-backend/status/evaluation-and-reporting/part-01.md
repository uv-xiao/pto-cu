# CUDA Backend Status: Evaluation And Reporting Part 1

## Evaluation And Reporting

The current evaluation setup covers local A100 and remote H200 runs with:

- `direct_driver`;
- `direct_driver_graph`;
- `pto_host_schedule`;
- `pto_host_schedule_compiler`;
- `pto_host_schedule_unary_square`;
- `pto_host_schedule_quad`;
- `pto_host_schedule_generic_args`;
- `pto_persistent_device`;
- `pto_persistent_queue`;
- `pto_persistent_dag`;
- `pto_persistent_dag_chain`;
- `pto_persistent_dag_reuse`;
- `pto_persistent_dag_scalar_axpy`;
- `pto_persistent_dag_scalar_scale`;
- `pto_persistent_dag_scalar_affine`;
- `pto_persistent_dag_triad`;
- `pto_persistent_dag_quad`;
- `pto_persistent_dag_generic_args`;
- `pto_persistent_dag_graph`;
- `pto_persistent_dag_graph_generic_args4`;
- `pto_persistent_dag_graph_node_attrs`;
- `pto_persistent_dag_graph_node_io`;
- `pto_persistent_dag_graph_node_link`;
- `pto_persistent_dag_graph_named_callable`;
- `pto_persistent_dag_graph_node_op`;
- `pto_persistent_dag_graph_depends_on`;
- `pto_persistent_dag_graph_scalar_axpy`;
- `pto_persistent_dag_graph_scalar_scale`;
- `pto_persistent_dag_graph_scalar_affine`;
- `pto_persistent_dag_graph_reordered`;
- `pto_persistent_dag_graph_chain`;
- `pto_persistent_dag_graph_scratch_reuse`;
- `pto_persistent_dag_graph_diamond`;
- `pto_persistent_dag_graph_parallel_chains`;
- `pto_persistent_dag_graph_wide_fanout`;
- `pto_persistent_dag_graph_multi_fanin`;
- `pto_persistent_dag_graph_tagged`;
- `pto_persistent_dag_graph_tagged_inout`;
- `pto_persistent_dag_graph_role_keyed_inout`;
- `pto_persistent_dag_graph_compact_role_inout`;
- `pto_persistent_dag_graph_pair_inout`;
- `pto_persistent_dag_graph_role_map_inout`;
- `pto_persistent_dag_graph_submit_groups`;
- `pto_persistent_dag_graph_triad`;
- `pto_persistent_dag_graph_quad`;
- `pto_persistent_dag_graph_unary_square`;
- `pto_persistent_dag_unary_square`;
- `pto_persistent_dag_tensor`;
- `pto_persistent_dag_graph_tensor`;
- `pto_persistent_dag_tensor_core`;
- `pto_persistent_dag_graph_tensor_core`;
- `cublas_sgemm`;
- `cublas_sgemm_graph`;
- same-work batch rows;
- worker-grid batch rows.

The latest full paired capture at commit `743709f3` uses the `16x16x16`
tensor descriptor, sizes `1024,65536,1048576`, three repeats, task counts
`2,6,12`, and worker-grid values `32,64,128,256`. It writes artifacts under
`tmp/cuda-backend/current-head-full-layered-cross-fixed/`
`combined-current-743709f3/` and validates `1350` combined samples after the
layered-cross graph row joined the selected matrix. The paired-runner
validator checked
source-paper provenance, sanitized command examples, generated Markdown/SVG
reports, zero scheduler errors, selected tensor throughput reports, graph
topology reports, graph TaskArgs-like metadata reports, expected generated
dispatch sequences, tensor tile descriptors, graph fan-in/dependent arrays,
node attrs/ops metadata, named-callable metadata, task-argument spellings,
scratch-reuse metadata, the nine-task parallel-chains queue capacity, and the
seven-task wide-fanout, four-task multi-fan-in, and nine-task layered-cross
queue capacities. This supersedes the older `61d73b65`, `5d84690d`,
`4e81fbff`, `c183d1ad`, `f99dc6b0`,
`9ec5511e`, `cb300e82`, and `61cf96cd` full captures while keeping the same
three-size/three-repeat comparison role.

The selected benchmark preset now also includes
`pto_persistent_dag_graph_multi_fanin` and
`pto_persistent_dag_graph_layered_cross`. The compact paired A100/H200 gate
under
`tmp/cuda-backend/layered-cross-selected-current-fixed/`
`combined-current-743709f3/` validates the no-batch `N=1024` selected matrix
with `108` samples. Its layered-cross row records dispatch
`1,2,11,1,2,1,6,1,1`, fan-in `0,0,0,2,3,1,2,3,2`, dependents
`3,3,4,4,5,4,6,7,6,7,7,8,8`, scalar metadata `scalar0=2.0`, tensor metadata
`c=a`, generated-dispatch PTX, report-visible graph topology,
source-paper provenance, sanitized local/remote command examples, and zero
scheduler errors. The compact gate measured layered-cross device times of
`74752 ns` on A100 and `69664 ns` on H200. The latest full paired gate under
`tmp/cuda-backend/current-head-full-layered-cross-fixed/`
`combined-current-743709f3/` validates `1350` rows and reports layered-cross
median device times of `76800/366336/6703936 ns` on A100 and
`65408/333920/4474816 ns` on H200 for `N=1024,65536,1048576`.
The previous compact gate under
`tmp/cuda-backend/multi-fanin-selected-current-working/`
`combined-current-c1c5f765/` validates the no-batch `N=1024` selected matrix
with `106` samples. Its wide-fanout row records dispatch `1,1,2,1,1,2,1`,
fan-in `0,1,1,1,2,2,2`, dependents `1,2,3,4,4,5,5,6,6`, queue capacity `7`,
and its multi-fan-in row records dispatch `1,2,11,6`, fan-in `0,0,0,3`,
dependents `3,3,3`, scalar metadata `scalar0=2.0`, tensor metadata
`c=tmp2`, generated-dispatch PTX, report-visible graph topology,
source-paper provenance, sanitized local/remote command examples, and zero
scheduler errors. The compact gate measured multi-fan-in device times of
`51200 ns` on A100 and `46656 ns` on H200. The full `61d73b65` capture
reports multi-fan-in medians of
`44032/178240/2373216 ns` on A100 and `36608/145504/1844736 ns` on H200 for
`N=1024,65536,1048576`; its wide-fanout medians are
`61440/288064/4105472 ns` on A100 and `55008/257248/3480128 ns` on H200.

Selected current-head full-capture medians show that the compiler-backed
host-schedule row remains within `0.85x-1.51x` of the handwritten
host-schedule row across A100/H200 and vector sizes. The graph task-argument
spelling rows also validate tag, role-keyed, compact, pair-shaped, and
role-map spellings through the same in-place graph topology: dispatch
`1,1,1`, fan-in `0,1,1`, dependents `1,2`, and the same report-visible task
args. At `N=1024`, A100 reported tag/role/compact/pair/role-map medians of
`30720/30720/30720/29696/30720 ns`; H200 reported
`29056/28576/27712/27936/29056 ns`.
The selected named-callable row now appears in the full matrix at all three
sizes with graph fan-in `0,0,2`, dependents `2,2`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, graph task arg key
`named_callable`, and graph task args
`task0=callable:add,input:a,input:b,output:tmp0;`
`task1=callable:mul,input:a,input:b,output:tmp1;`
`task2=callable:add,input:a,input:b,output:out`. Median device times were
`26624/135168/2361344 ns` on A100 and `25344/132320/1907936 ns` on H200 for
sizes `1024/65536/1048576`.

A compact paired benchmark at commit `945016c3` adds
`pto_persistent_dag_graph_diamond` to the benchmark matrix and validates the
new row on A100 and H200. It uses `N=1024`, one repeat, no batch rows, the
default `16x16x16` tensor descriptor for tensor rows, and writes raw JSON,
Markdown, and SVG reports under
`tmp/cuda-backend/combined-current-945016c3/`. The validator checked `48`
combined rows, required command examples, source-paper provenance, report
files, zero scheduler errors, and the graph-diamond generated-dispatch
sequence. The graph-diamond row reported dispatch `[9,2,1,2,1]`, five
completed tasks, A100 `device_wall_ns=36864`, and H200
`device_wall_ns=31744`.

The previous graph-generic compact paired validation at commit `b2c5c8a4` uses
the default `16x16x16` tensor descriptor so the scalar tensor DAG,
`pto_persistent_dag_tensor_core`, and `cublas_sgemm` rows are all runnable in
the same paired report. It runs `N=1024`, one repeat, `batch_tasks=2`, and
`worker_blocks_per_task=4`, producing `60` combined rows under
`tmp/cuda-backend/combined-current-b2c5c8a4/`. The paired runner validated
required baselines, expected generated-dispatch sequences, command examples,
tensor descriptor metadata, source-paper provenance, and Markdown and SVG
report files. It also validates the host-schedule generic-args benchmark row.
Selected A100 device times for host, host-generic, base-DAG,
persistent-generic, graph-generic4, tensor, tensor-core, cuBLAS, and
grid-batch were `22528/35840/44032/29696/27648/37888/36864/37888/37888 ns`;
H200 reported
`16992/31264/40320/30592/27520/48992/32480/34304/31872 ns`. All PTO
persistent DAG rows reported zero device scheduler errors.

The compact paired validation at artifact label `dbb01406` adds
`pto_persistent_dag_graph_scratch_reuse` to the selected benchmark path after
the previous `06b8c0c6` graph-chain gate. It uses the same `N=1024`, one
repeat, `batch_tasks=2`, `worker_blocks_per_task=4`, and default `16x16x16`
tensor descriptor shape as the compact gate, producing `64` combined rows
under `tmp/cuda-backend/combined-current-dbb01406/`. The validator checked
source-paper provenance, sanitized command examples, report files, tensor
descriptor metadata, dispatch sequences, graph descriptor fan-in/dependent
metadata, zero scheduler errors, and the new graph-scratch-reuse row. The
compact and paired-current presets now require
`scratch_reuse=reused_buffer=tmp0,reuse_task=4` for that row. The captured row
reported
`graph_descriptor.fanin=[0,0,2,1,1,2]`,
`graph_descriptor.dependents=[2,2,3,4,5,5]`, dispatch `[1,2,1,2,1,1]`,
`scratch_reuse.reused_buffer=tmp0`, `scratch_reuse.reuse_task=4`, completed
count `6`, A100 `device_wall_ns=36864`, and H200 `device_wall_ns=38240`.
The graph-chain row remains in the same compact gate and reported A100
`device_wall_ns=35840` and H200 `device_wall_ns=34528`.

The compact paired benchmark gate at artifact label `55a144de` now includes
`pto_persistent_dag_graph_tagged_inout`, validating explicit `input`,
`output`, `inout`, and `output_existing` task-argument tags in the selected
benchmark path. It uses `N=1024`, one repeat, `batch_tasks=2`,
`worker_blocks_per_task=4`, and the default `16x16x16` tensor descriptor,
producing `68` combined rows under
`tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/`.
The validator checked required baselines, dispatch sequences, tensor
descriptor metadata, source-paper provenance, command examples, generated
Markdown/SVG reports, graph descriptor fan-in/dependent metadata, graph
task-argument metadata, and zero scheduler errors.
The tagged-inout row reported
dispatch `[1,1,1]`, `graph_descriptor.fanin=[0,1,1]`,
`graph_descriptor.dependents=[1,2]`, completed count `3`,
`graph_task_args.task1=inout:tmp1,input:b`, A100
`device_wall_ns=35840`, and H200 `device_wall_ns=30080`.

The compact paired benchmark gate at artifact label `a46db551` promotes
`pto_persistent_dag_scalar_scale` into the selected benchmark path. It uses
`N=4096`, one repeat, no batch rows, and default `16x16x16` tensor descriptor
metadata. The paired runner synced the working tree to H200, captured A100
and H200 reports, merged `44` rows, and validated required baselines,
source-paper provenance, command examples, report files, and zero scheduler
errors under `tmp/cuda-backend/combined-current-a46db551/`. The scalar-scale
row reported dispatch `[11,2,1]`, `scalar0=2.0`, `device_wall_ns=37888` on
A100, and `device_wall_ns=27744` on H200.

The supplemental tensor-shape sweep at commit `c0ada3ad` runs
`pto_persistent_dag_tensor` on local A100 and remote H200 for `8x4x12`,
`16x16x64`, and `32x16x64` descriptors with `N=4096` and two repeats. All
12 rows passed with dispatch sequence `[3,1,2,1]`; the descriptor tile counts
were `128`, `16`, and `8`, respectively. The raw JSON, Markdown, and SVG
artifacts are under `tmp/cuda-backend/tensor-shape-sweep-c0ada3ad/`. This is
still scalar tiled GEMM scheduler evidence, not tensor-core throughput.

The first tensor-core persistent DAG smoke at commit `390eda4f` runs
`tensor_core_tile` on local A100 and remote H200 with a `16x16x16` descriptor.
The generated dispatch sequence is `[10,1,2,1]`; func_id `10` is a block-wide
WMMA `m16n16k8` task body with TF32 inputs and F32 accumulation. The paired
runner validated both artifacts with zero scheduler errors, tensor descriptor
`16x16x16`, `completed_count=4`, and generated Markdown/SVG report files under
`tmp/cuda-backend/persistent-tensor_core_tile-16x16x16-smoke-390eda4f/`.
This is callable and scheduler evidence for tensor-core task bodies, not a
tuned throughput result.

The first tensor-core selected-baseline benchmark row at commit `0879aa9e`
runs `pto_persistent_dag_tensor_core` on local A100 and remote H200 with the
same `16x16x16` descriptor. The compact selected-baseline report uses
`N=256`, one repeat, no batch rows, and the usual JSON/Markdown/SVG benchmark
outputs. The tensor-core DAG row measured `37888 ns` device time on A100 and
`38656 ns` on H200, compared with `40960 ns` and `43392 ns` for the scalar
`pto_persistent_dag_tensor` row in the same report. The raw artifacts are
under `tmp/cuda-backend/combined-tensor-core-current-0879aa9e/`. Benchmark
reports now also write `cuda-benchmark-dag-deltas.svg`, which plots each
`pto_persistent_dag_*` row's signed device-time increment over the matched
`pto_persistent_dag` scheduler baseline, and
`cuda-benchmark-throughput.svg`, which plots median GF/s for tensor-DAG and
cuBLAS rows with recorded tensor tile descriptors. Benchmark Markdown reports
now include a graph descriptor metadata table, and the primary benchmark SVG
embeds graph topology and task-argument metadata for explicit graph rows.

