# CUDA Backend Status: Kernel Compiler Integration Part 02b

Graph nodes may now keep non-IO metadata under an `attrs` dictionary. The
adapter merges `attrs` before node IO lowering, so a node can spell
`op`, `inputs`, and `outputs` as graph structure while keeping auxiliary
runtime payloads such as `tensor_args` and `scalar_args` out of the edge
metadata. The ctypes-backed `node_attrs_graph_with_ctypes_data` scene passed
locally with `2 passed, 119 deselected`; the full local A100 CUDA scene-test
file passed with `121 passed`; and the H200 selector passed with `1 passed,
120 deselected` while still emitting the known PTO-ISA SSH refresh warning.
The same graph-node attrs shape is now capturable through the paired
persistent-smoke workflow as `graph_descriptor_node_attrs`. The capture under
`tmp/cuda-backend/persistent-graph_descriptor_node_attrs-repeat2-smoke-b1b3e28c/`
contains A100/H200 JSON plus Markdown/SVG reports with dispatch `[9,2,1]`,
graph fan-in `[0,0,2]`, graph dependents `[2,2]`, repeat completions
`[3,3]`, zero scheduler errors, tensor slots
`tensor_args[0]=tmp0,tensor_args[1]=tmp3`, scalar slots
`scalar_args[0]=1.5,scalar_args[1]=0.25`, and report metadata
`graph_node_attrs.task0=attrs:tensor_args,scalar_args`. Device times were
`72704 ns` on A100 and `45408 ns` on H200 for `N=1024`.
The paired persistent-smoke validator now also requires the same attrs
metadata in both JSON and generated Markdown/SVG reports. The working-tree
capture under
`tmp/cuda-backend/persistent-node-attrs-smoke-working/`
`persistent-graph_descriptor_node_attrs-repeat2-smoke-761fbdfe/` contains
A100/H200 JSON plus Markdown/SVG artifacts. The paired validator accepted
dispatch `[9,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
repeat completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, tensor slots
`tensor_args[0]=tmp0,tensor_args[1]=tmp3`, scalar slots
`scalar_args[0]=1.5,scalar_args[1]=0.25`, report-visible graph-node attrs
`task0=attrs:tensor_args,scalar_args`, and zero scheduler errors. A100
reported `device_wall_ns=67584`, `host_wall_ns=99134`; H200 reported
`device_wall_ns=42144`, `host_wall_ns=60174`.
The paired smoke validator now also has first-class scalar/tensor argument
checks, so the attrs path cannot silently lose the payload it is meant to
carry. The stricter working-tree capture under
`tmp/cuda-backend/persistent-node-attrs-args-smoke-working/`
`persistent-graph_descriptor_node_attrs-repeat2-smoke-46c2f848/` validated
the same dispatch/topology, report-visible graph-node attrs, scalar slots,
and tensor slots on A100 and H200. A100 reported `device_wall_ns=62464`,
`host_wall_ns=93490`; H200 reported `device_wall_ns=40672`,
`host_wall_ns=57904`.
The same shape is also promoted into the selected benchmark matrix as
`pto_persistent_dag_graph_node_attrs`. The compact paired A100/H200 capture
under
`tmp/cuda-backend/graph-node-attrs-benchmark-working/combined-current-3d129351/`
validated `78` benchmark samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, dispatch
`[9,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`, and
`graph_node_attrs.task0=attrs:tensor_args,scalar_args`. The benchmark
validator now also requires the row and generated Markdown/SVG reports to
carry `scalar_args[0]=1.5,scalar_args[1]=0.25` and
`tensor_args[0]=tmp0,tensor_args[1]=tmp3`, so the selected benchmark gate
cannot preserve the node-attrs label while losing its descriptor payload.
The compact `cuda_current_summary.py --section graph-metadata` table now
renders those scalar/tensor descriptor maps as well, so copied evaluation
tables keep the payload visible.
Device times were `40960 ns` on A100 and `33152 ns` on H200 for `N=1024`.
A current-head no-batch paired compact gate under
`tmp/cuda-backend/current-head-compact-args-summary-working/combined-current-7191db4e/`
refreshes the selected benchmark matrix after that summary update. It
validated `88` A100/H200 samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, visible graph
topology, visible tensor throughput, and report-visible scalar/tensor
descriptor payloads for `pto_persistent_dag_graph_node_attrs`. The same raw
JSON reports A100/H200 graph-node-attrs device times of `29696/31072 ns`,
graph tensor-core device times of `39936/31936 ns`, and cuBLAS Graph replay
times of `13311/9088 ns` for the default `16x16x16`, `N=1024` compact
descriptor.
The graph-node callable-alias path is now also promoted into the selected
benchmark matrix as `pto_persistent_dag_graph_node_op`. The compact paired
A100/H200 capture under
`tmp/cuda-backend/graph-node-op-benchmark-working/combined-current-7edfb7df/`
validated `88` benchmark samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, dispatch
`[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`, and
`graph_node_ops=task0=op:add=1;task1=op:mul=2;task2=op:add=1`. Device times
were `31744 ns` on A100 and `25536 ns` on H200 for `N=1024`.
The graph-node input/output spelling is now promoted into the selected
benchmark matrix as `pto_persistent_dag_graph_node_io`. The compact paired
A100/H200 capture under
`tmp/cuda-backend/graph-node-io-benchmark-working/combined-current-c0d327d2/`
validated `98` benchmark samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, dispatch
`[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
`graph_task_arg_key=node_io`, and task args
`task0=input:a,input:b,output:tmp0`,
`task1=input:a,input:b,output:tmp1`, and
`task2=input:a,input:b,output:out`.
Device times were `28672 ns` on A100 and `25632 ns` on H200 for `N=1024`.
The node-link graph descriptor spelling is now promoted into the selected
benchmark matrix as `pto_persistent_dag_graph_node_link`. It covers
`graph.nodes[*].id`, nested node `data`, and `graph.links` through the same
add/mul/add callable sequence used by the graph-node `op` row. The compact
A100/H200 capture under
`tmp/cuda-backend/graph-node-link-compact-current-preset-working/`
`combined-current-8a74e5ab/` validates the current `compact-current` preset:
`102` rows, source-paper provenance, Markdown/SVG reports, graph topology and
task-argument report metadata, tensor-throughput SVG output, sanitized command
examples, zero scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`,
graph dependents `[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, and device times of
`35840 ns` on A100 and `31808 ns` on H200 for `N=1024`.
The same node-IO graph descriptor path is now also covered by the paired
persistent-smoke report validator. The working-tree capture under
`tmp/cuda-backend/persistent-node-io-smoke-working/`
`persistent-graph_descriptor_node_io-repeat2-smoke-feddd21b/` contains A100
and H200 JSON plus Markdown/SVG artifacts. The paired validator accepted
dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
repeat completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, report-visible
`graph_task_arg_key=node_io`, task args
`task0=input:a,input:b,output:tmp0;task1=input:a,input:b,output:tmp1;task2=input:a,input:b,output:out`,
and zero scheduler errors. A100 reported `device_wall_ns=68608`,
`host_wall_ns=102364`; H200 reported `device_wall_ns=42784`,
`host_wall_ns=59753`.
The same node-op graph descriptor path is now also covered by the paired
persistent-smoke report validator. The working-tree capture under
`tmp/cuda-backend/persistent-node-op-smoke-working/`
`persistent-graph_descriptor_node_op-repeat2-smoke-32e3b1ae/` contains A100
and H200 JSON plus Markdown/SVG artifacts. The paired validator accepted
dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
repeat completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, report-visible graph-node
ops `task0=op:add=1;task1=op:mul=2;task2=op:add=1`, and zero scheduler errors.
A100 reported `device_wall_ns=65536`, `host_wall_ns=97710`; H200 reported
`device_wall_ns=55296`, `host_wall_ns=78895`.
The incoming-edge path is now covered by both a real-data L2 ctypes scene and
paired persistent-device smoke. The working-tree smoke capture under
`tmp/cuda-backend/depends-on-graph-working/persistent-graph_descriptor_depends_on-repeat2-smoke-06b988b5/`
contains A100/H200 JSON, Markdown, and SVG artifacts for
`graph_descriptor_depends_on` with dispatch `[1,2,1]`, graph fan-in
`[0,0,2]`, graph dependents `[2,2]`, `launch_completed_counts=[3,3]`,
resource policy `scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`,
`grid_dim=4`, and zero scheduler errors. This proves the CUDA runtime can
schedule edges supplied as consumer-side metadata even when the consumer's
tensor pointers stay bound to the original graph inputs.
The same real-data scene fixture now uses named `depends_on` entries. It
passed locally on A100 and remotely on H200 with selector
`depends_on_graph_with_ctypes_data`; the H200 run printed the known PTO-ISA
SSH refresh warning before pytest reported `1 passed, 101 deselected`.
The same graph notation is now promoted into the selected benchmark path as
`pto_persistent_dag_graph_depends_on`. The compact paired capture under
`tmp/cuda-backend/graph-depends-benchmark-working/combined-current-01ddf564/`
validated `84` A100/H200 samples, source-paper provenance, command examples,
Markdown/SVG report files, zero scheduler errors, and graph topology metadata.
The depends-on row recorded dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, and device times of `30720 ns` on A100 and `26112 ns` on
H200 for `N=1024`.
The graph adapter now accepts a role-keyed `task_args` task form as a first
TaskArgs-like lowering slice: `input`, `output`, `output_existing`, and
`inout` roles are lowered to the existing bounded CUDA graph descriptor fields
before temporary allocation, tensor-flow dependency inference, and task struct
construction. The adapter prefers the `role` key and still accepts the older
`tag` spelling for compatibility. This lets a scene test describe a
persistent graph in terms of task-argument roles while still using the current
statically compiled generated-dispatch callable.
The role-keyed `task_args` form also accepts scalar inputs, lowering them
through the same bounded `scalar_args` slots as explicit graph descriptors.
Scalar entries still resolve through normal `TaskArgsBuilder` scalar names
before the descriptor is launched, so a graph descriptor can now keep tensor
roles and scalar inputs in one TaskArgs-like list.
The graph adapter now also resolves named graph callables before lowering
role-keyed `task_args`: a descriptor may define `graph.callables` as either a
dictionary keyed by callable name or a list of callable specs with `name`
fields, then each graph task can use `callable: "name"` or, for list-shaped
registries, the zero-based callable index instead of embedding the raw
generated-dispatch ID. Task-local fields override callable defaults, so the
resulting descriptor still records the same generated dispatch while the
scene-test graph shape is closer to normal PTO
`submit_next_level(callable, TaskArgs, ...)` submissions.
The list-shaped callable registry slice is covered by descriptor tests and by
the real-data scene selector
`named_callable_list_graph_with_ctypes_data`. That selector passed on the
local A100 and remote H200; the H200 run also printed the known PTO-ISA SSH
refresh warning before pytest reported `1 passed, 84 deselected`.
The callable-index slice is covered by
`callable_index_graph_with_ctypes_data`, which passed on the local A100 and
remote H200 with the same known H200 PTO-ISA SSH refresh warning.
The index-only registry slice is covered by
`unnamed_callable_index_graph_with_ctypes_data`; list entries now only need
`name` when graph tasks reference callables by name. That selector passed on
the local A100 and remote H200, again with the known H200 PTO-ISA SSH refresh
warning.
The compact callable-registry slice is covered by
`compact_callable_index_graph_with_ctypes_data`; integer list entries now
lower directly to generated-dispatch `func_id` values for index-referenced
graphs. The same compact form now works for dictionary registries such as
`{"add": 1}` and is covered by
`compact_callable_dict_graph_with_ctypes_data`. The focused selector passed
locally with `2 passed, 111 deselected`, the full local A100 CUDA scene-test
file passed with `113 passed`, and the H200 selector passed with `1 passed,
112 deselected` while still emitting the known PTO-ISA SSH refresh warning.
Callable metadata in graph descriptors may now spell the generated-dispatch
ID as `func_id`, `callable_id`, or `cid`. The aliases normalize before
role-keyed `task_args` and node IO lowering, so descriptor tests can use the
same `callable_id` / `cid` terminology as the normal Worker dispatch path
while preserving the current CUDA task ABI. The focused TDD selector first
failed with `KeyError: 'func_id'`, then passed on local A100 with `2 passed,
127 deselected`; the same selector passed on remote H200 with `2 passed, 127
deselected` after syncing the working tree, with the known PTO-ISA SSH
refresh warning printed first.
The role-keyed task-argument slice is covered by descriptor and real-data
selectors
`role_keyed_task_args` and `role_keyed_inout_graph_with_ctypes_data`. These
first failed because `role` was ignored and `tmp0` was treated as an input
before it existed. After adding shared `role`/`tag` normalization, the focused
local A100 selector reported `2 passed, 91 deselected`; the remote H200
real-data selector reported `1 passed, 92 deselected` with the known PTO-ISA
SSH refresh warning.
Graph task descriptors now also accept `args` as a short alias for
`task_args`, matching the argument slot in
`submit_next_level(callable, TaskArgs, ...)` more closely while preserving
the same role-keyed lowering. The descriptor-only regression first failed
with fan-in `[0,0,0]` because `args` was ignored. After wiring the alias into
the same normalization path, the local A100 selector:

