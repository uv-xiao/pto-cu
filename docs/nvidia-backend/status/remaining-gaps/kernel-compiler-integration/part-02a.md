# CUDA Backend Status: Kernel Compiler Integration Part 02a

Host-schedule task-body compilation and persistent-device generated dispatch
now have first `KernelCompiler` entry points. Both paths can consume
`CudaTaskBody` style sources. CUDA prepared-callable artifacts can be staged
through the L2 Python `Worker` registration path. The normal scene-test flow
can compile and run host-schedule CUDA vector-add, binary elementwise, unary
square, scalar scale, axpy, two-scalar affine, three-input triad, quad, and
generic tensor/scalar callable specs and persistent-device
fork/join, chain, reuse, scalar scale, scalar AXPY, scalar affine, and
tensor-tile DAG callable specs, plus third-tensor persistent triad and
unary-square callable specs, end to end.
The fourth-tensor persistent quad callable spec also runs end to end through
ctypes-backed real data. The host-schedule quad callable spec also runs
through both the normal `SceneTestCase` L2 path and the no-torch Worker smoke
path. The host-schedule generic-args callable spec now has ctypes-backed
scene tests, so it runs on H200 without requiring `torch`; it lowers the
original two-slot `tensor_args[0:2]` and `scalar_args[0:2]` path, plus the
four-slot `tensor_args[0:4]` and `scalar_args[0:4]` path, through the host
runtime launch ABI. A generic `persistent_dag_graph_f32`
adapter can now lower explicit
runtime graph descriptors with per-task `func_id`, dependency lists, fan-in,
temporary buffers, generic tensor slots, and scalar slots through the same L2
`SceneTestCase` path. The adapter now allocates default-sized temporaries for
graph task `out` names that do not match existing input/output tensors, so the
`temporaries` map is only needed for non-default temporary sizes. It also
supports a first dependency-inference slice: when a graph task omits
`dependents`, the adapter infers its outgoing edges from tensor flow by
binding `a`/`b`/`c`/`d` and `tensor_args` reads to the nearest previous
producer for that tensor name, or to a later producer when the descriptor is
intentionally out of topological order. The inference is per task, so mixed
descriptors can keep explicit dependency lists for some tasks while inferring
omitted edges for the remaining tasks.
The graph adapter also accepts incoming dependency lists through `depends_on`
or `dependencies`, lowering consumer-side task IDs into the same flattened
dependent array. That lets scene tests express graph edges independently from
the task's tensor pointer layout, which is closer to normal task-graph
metadata than CUDA-specific outgoing edge lists.
Those incoming edges may now reference named graph tasks as well as integer
task IDs. The named form keeps descriptor tests stable when tasks are inserted
or reordered and moves the CUDA graph adapter closer to normal named PTO task
graphs.
Outgoing `dependents` may now use the same task names, so both graph edge
directions share one descriptor naming model. The edge fields may be a single
task name/id or a list of task names/ids. The real-data
`graph_with_ctypes_data` scene now uses scalar named outgoing dependents and
passed on local A100 and remote H200; the H200 run printed the known PTO-ISA
SSH refresh warning before pytest reported `1 passed, 104 deselected`.
The graph adapter also accepts a top-level `edges` list with
`{"from": <task>, "to": <task>}` dictionaries, two-item endpoint pairs, or
`"<source> -> <target>"` strings. The ctypes-backed
`edge_list_graph_with_ctypes_data` scene passed locally on A100 with pytest
reporting `5 passed, 102 deselected` for the focused selector and remotely on
H200 with pytest reporting `1 passed, 106 deselected`; the H200 run printed
the known PTO-ISA SSH refresh warning first. The new string-edge selector
first failed at `_graph_edge_endpoints`, then passed on local A100 and remote
H200 with pytest reporting `2 passed, 125 deselected` on both systems.
Top-level edge dictionaries now also accept dep-gen-style `pred` and `succ`
endpoint keys, matching the structural edge vocabulary used in `deps.json`.
The focused builder selector first failed at `_graph_edge_endpoints`, then
passed with `3 passed, 144 deselected` for the adjacent edge forms. The
ctypes-backed `dep_gen_edge_graph_with_ctypes` scene passed on local A100 and
remote H200 with `1 passed, 146 deselected`; the H200 run printed the known
PTO-ISA SSH refresh warning first.
Graph task identity now also accepts dep-gen-style `task_id` as a `name`/`id`
alias, so `deps.json`-like task rows can be referenced directly from
`pred`/`succ` edge dictionaries. The focused builder selector first failed
with `unknown dependency task name: left`, then passed with
`1 passed, 148 deselected`. The adjacent identity/edge selector passed with
`4 passed, 145 deselected`. The ctypes-backed
`dep_gen_task_id_graph_with_ctypes` scene passed on local A100 with
`1 passed, 148 deselected` and on remote H200 with
`1 passed, 148 deselected`; the H200 run printed the known PTO-ISA SSH
refresh warning first.
Real `deps.json` edge rows also include annotation fields such as
`source="creator"` or `source="tensormap"`. The adapter now prefers
`pred`/`succ` over `source`/`target` when both forms appear, so the
annotation is not misread as a task endpoint. The focused builder selector
first failed with `unknown dependency task name: creator`, then passed with
`1 passed, 150 deselected`. The adjacent edge compatibility selector passed
with `5 passed, 146 deselected`, covering dep-gen, `from`/`to`, and
node-link `source`/`target` spellings. The ctypes-backed
`annotated_dep_gen_graph_with_ctypes` scene passed on local A100 and remote
H200 with `1 passed, 150 deselected`; the H200 run printed the known PTO-ISA
SSH refresh warning first.
Dep-gen task rows that only carry identity and metadata, such as `task_id` and
`scope`, can now inherit runnable CUDA descriptor fields from graph-level
`task_defaults`, `task_template`, or `default_task`. The focused selector first
failed with `KeyError: 'func_id'`, then passed locally on A100 with
`2 passed, 151 deselected` and remotely on H200 with
`2 passed, 151 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first. This keeps real `deps.json` task lists closer to their captured
shape while still letting scene tests provide the common callable and TaskArgs
metadata needed to execute the graph.
The adapter can now load that graph descriptor from a JSON file through
`graph_path` or `graph_file`, with inline `graph` metadata applied as an
overlay. The focused selector first failed because only the inline defaults
were seen and the graph had no tasks, then passed locally on A100 with
`2 passed, 153 deselected` and remotely on H200 with
`2 passed, 153 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first. The JSON-file test uses the v2 `deps.json` shape: large string
task IDs, `tasks[]`, `tensors[]`, and annotated `pred`/`succ` edges.
Imported graph tasks can now receive keyed `task_overrides` or `task_metadata`
after defaults and raw task rows are merged. The keyed override path keeps the
raw `deps.json` structural while mapping individual task IDs to different
CUDA `func_id` values and temporary/output bindings. The focused selector
first failed because task `4294967296` still dispatched `func_id=1` and the
ctypes output was plain `a+b`, then passed locally on A100 with
`2 passed, 155 deselected` and remotely on H200 with
`2 passed, 155 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first.
List-shaped `task_metadata` sidecars are now accepted too. Each metadata entry
keys itself with `name`, `id`, or `task_id`, then merges through the same
override path. The TDD selector first failed with `graph task overrides must be
a dictionary`, then passed locally on A100 with `2 passed, 157 deselected`.
The metadata sidecar can also live in a separate JSON file named by
`task_metadata_path` or `task_metadata_file` in the inline graph overlay. The
TDD selector first failed because the sidecar file was ignored, task
`4294967296` still dispatched `func_id=1`, and the ctypes output stayed plain
`a+b`; after loading the sidecar, it passed locally on A100 with
`2 passed, 159 deselected`.
The same sidecar loader now also runs for inline graph descriptors and graph
JSON files before task extraction. The focused inline selector first failed
with task `4294967296` still dispatching `func_id=1` and then passed locally
on A100 with `2 passed, 161 deselected`.
Override sidecars are supported with `task_overrides_path` or
`task_overrides_file` using the same shared sidecar loader. The focused
selector first failed because the file was ignored and task `4294967296`
still dispatched `func_id=1`, then passed locally on A100 with
`2 passed, 163 deselected`.
Shared runnable defaults can now also live in sidecar JSON files named with
`task_defaults_path` / `task_defaults_file`, `task_template_path` /
`task_template_file`, or `default_task_path` / `default_task_file`. This lets
dep-gen-style graph files keep structural task rows while a separate CUDA
sidecar supplies the common generated-dispatch `func_id`, input tensor, and
output binding defaults. The focused selector first failed with
`KeyError: 'func_id'`, then passed locally on A100 with
`2 passed, 171 deselected` and remotely on H200 with
`2 passed, 171 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first.
Relative task metadata sidecar paths embedded in graph JSON files are now
resolved beside the graph JSON file. The focused selector first failed with
`FileNotFoundError: task_metadata.json`, then passed locally on A100 with
`2 passed, 165 deselected` and remotely on H200 with
`2 passed, 165 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first.
`SceneTestCase` now resolves CUDA persistent-device `task_sources[*].source_path`
and graph descriptor paths relative to the test class file. The focused
selector first failed with `graph_path` still equal to `deps.json` and then
with `FileNotFoundError: add.pto.cu`; after extending callable path
resolution, it passed locally on A100 with `2 passed, 167 deselected` and
remotely on H200 with `2 passed, 167 deselected`; the H200 run printed the
known PTO-ISA SSH refresh warning first.
Callable registries can now be loaded from `callables_path` or
`callables_file` sidecars before graph task extraction. The sidecar has the
same dictionary or list shape as inline `graph.callables`, letting captured
graph descriptors keep `callable`/`op` references in structural task rows
while storing generated-dispatch `func_id` metadata separately. The focused
selector first failed with `unknown graph callable: generic`, then passed
locally on A100 with `2 passed, 169 deselected` and remotely on H200 with
`2 passed, 169 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first.
Generated-dispatch task source metadata can now be loaded from
`task_sources_path` or `task_sources_file` sidecars. The sidecar accepts either
a JSON list or an object with `task_sources`/`sources`, and `source_path`
entries are resolved relative to the sidecar file before
`KernelCompiler.compile_cuda_persistent_device(...)` receives them. The
focused selector first failed with `KeyError: 'task_sources'`, then passed
locally on A100 with `2 passed, 173 deselected`.
The same task source entries now also accept `source` as an alias for
`source_path`, matching other scene-test source specs. The TDD selector first
failed with `KeyError: 'source_path'`, then passed locally on A100 with
`2 passed, 175 deselected`.
The same graph-shaped path now accepts `graph.tasks` as a dictionary keyed by
task name, so descriptor specs can keep node names in one place and reference
those names from top-level edges. The ctypes-backed
`task_dict_graph_with_ctypes_data` scene passed locally on A100 with pytest
reporting `2 passed, 107 deselected` and remotely on H200 with pytest
reporting `1 passed, 108 deselected`; the H200 run printed the known PTO-ISA
SSH refresh warning first.
Top-level `graph.edges` now also accepts adjacency dictionaries, mapping each
source task name/id to either one target or a list of targets. The
ctypes-backed `adjacency_graph_with_ctypes_data` scene passed locally on A100
with pytest reporting `2 passed, 109 deselected` and remotely on H200 with
pytest reporting `1 passed, 110 deselected`; the H200 run printed the known
PTO-ISA SSH refresh warning first.
The graph adapter now also accepts `graph.nodes` as an alias for
`graph.tasks`, so the same named-node and edge-map descriptor can use graph
terminology without changing the runtime task array. The ctypes-backed
`node_graph_with_ctypes_data` scene passed locally with `2 passed, 113
deselected`; the full local A100 CUDA scene-test file passed with
`115 passed`; and the H200 selector passed with `1 passed, 114 deselected`
while still emitting the known PTO-ISA SSH refresh warning.
List-shaped graph nodes may now spell node identity as `id`, which normalizes
to the existing `name` field used by edge lookup. The focused selector first
failed with `unknown dependency task name: left`, then passed locally on A100
with `2 passed, 129 deselected` and remotely on H200 with `2 passed, 129
deselected`; the H200 run printed the known PTO-ISA SSH refresh warning
first.
Graph nodes now also accept node-style IO fields: `inputs`, `outputs`,
`output_existing`, `inouts`, and `scalars`. These fields expand into the
existing role-keyed `task_args` lowering path, so graph-node descriptors can
avoid embedding `a`/`b`/`out` ABI field names directly. The ctypes-backed
`node_io_graph_with_ctypes_data` scene passed locally with `2 passed, 115
deselected`; the full local A100 CUDA scene-test file passed with
`117 passed`; and the H200 selector passed with `1 passed, 116 deselected`
while still emitting the known PTO-ISA SSH refresh warning.
Dictionary-valued `inputs` and `outputs` node-port maps now lower through the
same node-IO path. This covers graph schemas that carry edge ports as
`{"lhs": "a", "rhs": "b"}` or `{"value": "tmp0"}` rather than plain lists.
The focused TDD selector first failed because the adapter stringified the
port dictionaries into unknown tensor names; after flattening port values
deterministically, local A100 and synced H200 both reported
`2 passed, 135 deselected` for `-k node_port_dict --platform cuda`.
Graph nodes may now use `op` as an alias for `callable`, resolved through
`graph.callables` before node IO fields are lowered. This lets node
descriptors keep an operation-name spelling while still producing the same
generated-dispatch `func_id` task array. The ctypes-backed
`node_op_graph_with_ctypes_data` scene passed locally with `2 passed, 117
deselected`; the full local A100 CUDA scene-test file passed with
`119 passed`; and the H200 selector passed with `1 passed, 118 deselected`
while still emitting the known PTO-ISA SSH refresh warning.
