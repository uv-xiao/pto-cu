# CUDA Backend Status: Persistent Scheduler Generalization Part 3

- full graph construction from normal PTO task graphs;
- broader graph-lowering coverage beyond the current
  `persistent_dag_graph_f32` descriptor adapter, which already covers
  automatic default temporary allocation, logical-output/storage-output
  separation for scratch reuse, order-independent tensor-flow dependency
  inference, explicit outgoing and incoming graph edges with scalar or
  list-valued named task dependencies, top-level graph edge lists including
  string `source -> target` entries and dep-gen-style `pred`/`succ` endpoint
  dictionaries including annotated real `deps.json` rows with `source`
  metadata, dep-gen-style `task_id` graph task identities, graph task defaults
  for runnable metadata shared by dep-gen task rows, external JSON graph files
  with inline overlays, keyed task overrides for heterogeneous imported graph
  rows, list-shaped task metadata sidecars, task metadata sidecar files,
  task metadata sidecars on inline descriptors, task override sidecar files,
  adjacency dictionaries, `graph.links` aliases, `graph.nodes` aliases, node
  `id` identity aliases, node-link `data` payloads, node-style IO fields,
  dictionary-valued node IO port maps, node `op` callable aliases, callable
  metadata `callable_id` / `cid` aliases, and paired smoke including
  node-link `links` and dictionary-valued node IO port maps,
  tagged TaskArgs-like graph task lowering including `inout` producer
  chaining, role-map task-argument dictionaries with paired smoke,
  submit-shaped graph descriptors, submit-group descriptor expansion in the
  selected benchmark matrix, explicit unary square graph dispatch, tagged
  graph-descriptor paired smoke, and five-task chain, five-task fan-out/fan-in,
  and six-task scratch-reuse graph descriptor smokes;
- broader lifecycle validation beyond the current scratch-reuse,
  graph-descriptor and generic-argument repeat-run, tensor-core graph, and
  direct/queue/DAG prepared-callable repeat-run smokes. The paired lifecycle
  matrix runner now captures direct, queue, DAG-chain, incoming-edge graph,
  graph-descriptor scratch-reuse, and graph tensor-core repeat-run evidence
  across A100 and H200 in one artifact set
  (`tmp/cuda-backend/lifecycle-tensor-core-working/persistent-lifecycle-matrix-1c683c1c/`).
  The lifecycle matrix validator checks required scenarios, A100/H200
  artifacts, repeat-run completion counts, DAG-chain dispatch, graph
  depends-on dispatch/topology, graph-scratch-reuse dispatch/topology,
  graph tensor-core dispatch/topology, scratch-reuse metadata, tensor-tile
  metadata, report files, VDCores/MPK source-paper provenance, collection
  mode, sanitized reconstruction commands, and zero device scheduler errors.
  The matrix report can be regenerated from the existing per-scenario smoke
  JSON with `--collect-existing-suffix 1c683c1c` without rerunning the GPUs,
  and the validator requires that flag in the regenerated local command
  example. The artifact index reads the lifecycle metadata commit and
  collection mode so regenerated reports remain distinguishable in
  `tmp/cuda-backend/*/index.md`.
  The paired persistent-smoke validator now
  requires `scratch_reuse=reused_buffer=tmp0,reuse_task=4`, and the smoke
  Markdown/SVG report renders that physical alias beside task metadata. In
  the current capture, graph-depends-on validates
  `launch_completed_counts=[3,3]`, dispatch `1,2,1`, graph fan-in `0,0,2`,
  dependents `2,2`, and device times of `63488 ns` on A100 and `41600 ns` on
  H200. The same capture's graph-scratch-reuse scenario validates
  `launch_completed_counts=[6,6]`, dispatch `1,2,1,2,1,1`, graph fan-in
  `0,0,2,1,1,2`, dependents `2,2,3,4,5,5`, scratch reuse of `tmp0` at task
  `4`, and device times of `92160 ns` on A100 and `89056 ns` on H200. The
  graph-tensor-core scenario validates `launch_completed_counts=[4,4]`,
  dispatch `10,1,2,1`, graph fan-in `0,1,1,2`, dependents `1,2,3,3`,
  tensor tile `16x16x16`, `worker_blocks=4`, and device times of `77824 ns`
  on A100 and `56672 ns` on H200, so the remaining lifecycle gap is normal
  PTO graph breadth rather than prepared-callable reset coverage;
- broader resource policy across configurable scheduler blocks, configurable
  queue/DAG worker blocks, direct worker-blocks-per-task, callable stream id
  tracer bullet, and configurable block dimension. The
  paired A100/H200 resource-policy smoke under
  `tmp/cuda-backend/persistent-block128-working/` validates a five-task
  DAG-chain repeat run with `scheduler_blocks=1`, `worker_blocks=2`,
  `worker_blocks_per_task=1`, `stream_id=1`, `block_dim=128`, `grid_dim=3`,
  `repeat_runs=2`, dispatch `1,2,1,2,1`, generated Markdown/SVG reports, and
  zero scheduler errors. The broader resource-policy diamond capture under
  `tmp/cuda-backend/resource-policy-diamond-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-4862b62c/` validates a
  five-task graph descriptor with `worker_blocks=4`, `stream_id=2`,
  `block_dim=512`, `grid_dim=5`, repeat completions `[5,5]`, dispatch
  `9,2,1,2,1`, graph fan-in `0,0,2,2,2`, dependents `2,3,2,3,4,4`,
  scalar/tensor arg metadata, generated Markdown/SVG reports, and zero device
  scheduler errors on A100 and H200. A100 reported `device_wall_ns=72704`;
  H200 reported `device_wall_ns=53728`. The scheduler-distribution diamond
  capture under `tmp/cuda-backend/scheduler-distribution-policy-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-93e0a299/` validates the
  same graph shape with `scheduler_blocks=2`, `scheduler_init_count=2`,
  `worker_blocks=3`, `stream_id=2`, `block_dim=256`, `grid_dim=5`, repeat
  completions `[5,5]`, dispatch `9,2,1,2,1`, graph fan-in `0,0,2,2,2`,
  dependents `2,3,2,3,4,4`, scalar/tensor arg metadata, generated
  Markdown/SVG reports, and zero device scheduler errors on A100 and H200.
  A100 reported `device_wall_ns=77824`; H200 reported
  `device_wall_ns=52768`. The scheduler-loop policy capture under
  `tmp/cuda-backend/scheduler-loop-policy-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-5d7b3961/` moves
  dependent release out of worker blocks and into scheduler blocks through a
  bounded completion ring. It validates the same graph shape with
  `scheduler_loop_count=2`, `scheduler_processed_count=5`,
  `scheduler_init_count=2`, repeat completions `[5,5]`, generated
  Markdown/SVG reports, and zero device scheduler errors on A100 and H200.
  A100 reported `device_wall_ns=97280`; H200 reported
  `device_wall_ns=72928`. The scheduler by-block policy capture under
  `tmp/cuda-backend/scheduler-by-block-policy-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-01b85c21/` keeps the
  same graph and resource policy, exposes per-scheduler completion counters,
  and validates `scheduler_processed_by_block=[2,3]` on both A100 and H200.
  A100 reported `device_wall_ns=97280`; H200 reported
  `device_wall_ns=71136`. The scheduler scaling sweep under
  `tmp/cuda-backend/scheduler-scaling-working/` validates the same graph over
  `scheduler_blocks=1,2,4` with paired A100/H200 smoke JSON and a summary
  report under `scheduler-scaling-a5ca4fac/`. A100 reported
  `110592/97280/98304 ns` for `1/2/4` scheduler blocks; H200 reported
  `82240/70368/70752 ns`. The four-scheduler rows expose the current small
  graph's load-balance limit directly: A100 processed completions as
  `[0,2,3,0]` with active schedulers `2/4` and a `60.0%` busiest-scheduler
  completion share, while H200 processed `[2,1,1,1]` with active schedulers
  `4/4` and a `40.0%` busiest share. The parallel-chains capture under
  `tmp/cuda-backend/parallel-chains-working/` adds a nine-task graph with four
  roots, two joins, two parallel consumers, and one final join over
  `scheduler_blocks=4`, `worker_blocks=4`, and `queue_capacity=4`. It validates
  dispatch `1,2,1,2,1,1,2,1,1`, graph fan-in `0,0,0,0,2,2,2,2,2`, graph
  dependents `4,4,5,5,6,7,6,7,8,8`, repeat completions `[9,9]`, and zero
  scheduler errors on A100 and H200. A100 processed completions as
  `[2,1,3,3]`; H200 processed `[3,3,2,1]`. The parallel-chains scheduler
  scaling sweep under
  `tmp/cuda-backend/parallel-chains-scheduler-scaling-working/` validates the
  same graph over `scheduler_blocks=1,2,4` and writes a shape-aware summary
  under `scheduler-scaling-674ebe2e/`. A100 reported
  `155648/123904/115712 ns`; H200 reported `131104/102496/90272 ns`.
  The four-scheduler rows compare at `0.74x` and `0.69x` versus each GPU's
  matching one-scheduler row and keep all four scheduler blocks active on
  both GPUs. The current-head graph-size summary under
  `tmp/cuda-backend/scheduler-graph-size-scaling-working/`
  `scheduler-graph-size-scaling-952bdefd/` regenerates one JSON/Markdown/SVG
  report over both the five-task diamond and nine-task parallel-chain sweeps.
  The report includes task count, device ns/task, and tasks/scheduler columns;
  the four-scheduler per-task rows are `19660 ns` vs. `12856 ns` on A100 and
  `14150 ns` vs. `10030 ns` on H200 for diamond vs. parallel chains. The
  parallel-chains descriptor is now also wired into the selected benchmark
  path as `pto_persistent_dag_graph_parallel_chains`. The latest full paired
  current-head gate under
  `tmp/cuda-backend/current-head-full-layered-cross-fixed/`
  `combined-current-743709f3/` validates 1350 A100/H200 rows, including
  parallel-chain dispatch `1,2,1,2,1,1,2,1,1`, wide-fanout dispatch
  `1,1,2,1,1,2,1`, multi-fan-in dispatch `1,2,11,6`, layered-cross
  dispatch `1,2,11,1,2,1,6,1,1`,
  queue-capacity/fan-in/dependent metadata, generated-dispatch PTX,
  Markdown/SVG reports, tensor-throughput reports, source-paper provenance,
  sanitized command examples, and zero scheduler errors. The compact paired
  gate under
  `tmp/cuda-backend/parallel-chains-compact-current-working/`
  `combined-current-c3274430/` remains the 102-row quick path. The remaining
  wide-fanout compact gate under
  `tmp/cuda-backend/wide-fanout-selected-current-working/`
  `combined-current-a540a014/` validates 104 rows after adding
  `pto_persistent_dag_graph_wide_fanout` to the selected benchmark path. The
  wide-fanout row records dispatch `1,1,2,1,1,2,1`, fan-in `0,1,1,1,2,2,2`,
  dependents `1,2,3,4,4,5,5,6,6`, queue capacity `7`, and zero scheduler
  errors; the latest full gate reports wide-fanout medians
  `61440/288064/4105472 ns` on A100 and `55008/257248/3480128 ns` on H200.
  The paired smoke under
  `tmp/cuda-backend/wide-fanout-smoke-a540a014/` separately validates two
  repeat launches with scheduler completions split `[3,4]` on A100 and
  `[4,3]` on H200. The compact multi-fan-in gate under
  `tmp/cuda-backend/multi-fanin-selected-current-working/`
  `combined-current-c1c5f765/` validates 106 rows and records dispatch
  `1,2,11,6`, fan-in `0,0,0,3`, dependents `3,3,3`, and zero scheduler
  errors. Its paired smoke under
  `tmp/cuda-backend/multi-fanin-working/`
  `persistent-graph_descriptor_multi_fanin-sched2-repeat2-smoke-c1c5f765/`
  separately validates two repeat launches with scheduler completions split
  `[3,1]` on A100 and `[1,3]` on H200. The latest full gate reports
  multi-fan-in medians `44032/178240/2373216 ns` on A100 and
  `36608/145504/1844736 ns` on H200. The layered-cross compact gate under
  `tmp/cuda-backend/layered-cross-selected-current-fixed/`
  `combined-current-743709f3/` validates 108 rows and records dispatch
  `1,2,11,1,2,1,6,1,1`, fan-in `0,0,0,2,3,1,2,3,2`, dependents
  `3,3,4,4,5,4,6,7,6,7,7,8,8`, scalar metadata `scalar0=2.0`, tensor metadata
  `c=a`, and zero scheduler errors. Its paired smoke under
  `tmp/cuda-backend/layered-cross-working/`
  `persistent-graph_descriptor_layered_cross-sched3-repeat2-smoke-743709f3/`
  separately validates two repeat launches with scheduler completions split
  `[2,3,4]` on A100 and `[2,4,3]` on H200. The full layered-cross gate under
  `tmp/cuda-backend/current-head-full-layered-cross-fixed/`
  `combined-current-743709f3/` validates 1350 rows across the full selected
  A100/H200 matrix. The remaining policy gap is
  broader graph families beyond diamond, parallel-chain, wide-fanout,
  multi-fan-in, and layered-cross shapes, rather than launch
  resource partitioning, root seeding, completion-ring ownership,
  graph-size reporting, or artifact
  validation;
- broader scheduler error taxonomy beyond the current unsupported-`func_id`
  invalid-dependent-ID, dependent-range, fan-in-underflow,
  duplicate-dependent, self-dependent, initial-fan-in, and
  no-root/unreachable-task
  diagnostics. The current validators, smoke reports, lifecycle matrix
  reports, benchmark validators, and local artifact index render known
  nonzero scheduler codes with stable labels such as `7(unreachable_task)`,
  so negative A100/H200 captures are easier to triage without cross-reading
  raw runtime constants.

The role-map graph descriptor row is now part of the selected benchmark
matrix, not only the paired smoke path. The focused TDD first failed because
`cuda_benchmark.py` rejected
`pto_persistent_dag_graph_role_map_inout`, and the paired/compact validators
did not require that row or its graph metadata. After adding the benchmark
row, the focused selector reported `7 passed, 292 deselected`. The full
paired-current gate under
`tmp/cuda-backend/current-head-full-submit-groups-working/`
`combined-current-c183d1ad/` validated `1278` samples. At `N=1024`, role
spelling medians were tag/role/compact/pair/role-map
`30720/30720/30720/29696/30720 ns` on A100 and
`29056/28576/27712/27936/29056 ns` on H200. The compact paired A100/H200
gate under
`tmp/cuda-backend/role-map-selected-benchmark-working/`
`combined-current-a3c09113/` validated `98` non-batch samples with source
paper provenance, sanitized reconstruction commands, Markdown/SVG reports,
and zero scheduler errors. The role-map row validated dispatch `1,1,1`, graph
fan-in `0,1,1`, dependents `1,2`, and `graph_task_arg_key=role_map`; A100
reported `device_wall_ns=29696`, and H200 reported `device_wall_ns=25440`.

The submit-groups graph descriptor row is now also part of the selected
benchmark matrix as `pto_persistent_dag_graph_submit_groups`. The TDD red
selector first failed because `cuda_benchmark.py` rejected that baseline and
the paired/compact validators still expected `1260` full samples and `98`
compact non-batch samples. After adding the selected row and validator
metadata, the full benchmark-report test file passed with `300` tests. The
full paired A100/H200 gate under
`tmp/cuda-backend/current-head-full-submit-groups-working/`
`combined-current-c183d1ad/` validated `1278` samples with source-paper
provenance, sanitized reconstruction commands, Markdown/SVG reports, and zero
scheduler errors. The submit-groups row validated dispatch `1,1,1`,
graph fan-in `0,0,2`, dependents `2,2`, and
`graph_task_arg_key=submit_groups`; A100 reported
`device_wall_ns=25600`, and H200 reported `device_wall_ns=24160`. The compact
paired A100/H200 gate under
`tmp/cuda-backend/submit-groups-selected-benchmark-working/`
`combined-current-193ccc4d/` remains useful as the focused selector gate and
validated `100` non-batch samples before this full refresh.

