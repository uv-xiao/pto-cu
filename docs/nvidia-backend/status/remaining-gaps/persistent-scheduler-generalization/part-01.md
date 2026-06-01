# CUDA Backend Status: Persistent Scheduler Generalization Part 1

## Persistent Scheduler Generalization

The persistent-device scheduler is proven for small generated descriptors, but
it is not yet a full TensorMap/ringbuffer analogue.

The selected benchmark gate now includes the reordered explicit graph
descriptor as `pto_persistent_dag_graph_reordered`, not only the paired smoke
artifact. The compact no-batch A100/H200 capture under
`tmp/cuda-backend/graph-reordered-benchmark-working/combined-current-e038c96a/`
validated `86` rows, source-paper provenance, generated reports, zero device
scheduler errors, dispatch `1,9,2`, graph fan-in `2,0,0`, and dependents
`0,0`. At `N=1024`, A100 reported `35840 ns` device time and H200 reported
`25856 ns`, so order-independent graph dependency inference is covered by the
selected benchmark matrix as well as by the repeat-run smoke.

The explicit graph-descriptor adapter now accepts node-link style graph nodes
where list-shaped `graph.nodes` entries carry identity in `id` and task
payload under `data`. The adapter flattens `data` before callable, `op`, and
node-IO lowering, so this shape uses the same CUDA task descriptor ABI as the
existing top-level node field form. The focused TDD selector first failed with
`KeyError: 'func_id'` because `data` stayed nested; after adding early
node-data normalization, local A100 passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'node_data or node_id or node_op or callable_id' \
    --platform cuda
```

That run reported `8 passed, 125 deselected`. The synced H200 focused selector
also passed:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k node_data --platform cuda
```

That run reported `2 passed, 131 deselected` after the known PTO-ISA SSH
refresh warning.

The same node-link compatibility path now accepts `graph.links` as an alias
for top-level edge lists. A focused TDD selector first failed with graph fan-in
`[0,0,0]` because the links were ignored; after routing `links` through the
same dependency lowering as `edges`, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'node_link or node_data or graph_edges or adjacency_edges' \
    --platform cuda
```

That run reported `11 passed, 124 deselected`. The synced H200 focused
selector also passed:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k node_link --platform cuda
```

That run reported `2 passed, 133 deselected` after the known PTO-ISA SSH
refresh warning.

The node-link descriptor shape is now also available in the paired
persistent-smoke workflow as `graph_descriptor_node_link`, so the `links`
schema can produce A100/H200 JSON plus generated Markdown/SVG evidence. The
new TDD selector first failed because the smoke builder and paired runner did
not know the shape; after wiring the generated add/mul/add descriptor,
metadata expectations, and CLI choices, the focused local report tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'node_link_graph_descriptor'
```

That run reported `2 passed, 284 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_node_link --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-node-link-smoke-working
```

The capture under
`tmp/cuda-backend/persistent-node-link-smoke-working/persistent-graph_descriptor_node_link-repeat2-smoke-3e4ddb00/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, A100 device time `45056 ns`,
and H200 device time `43808 ns`.

The dictionary-valued node-port graph descriptor spelling is now promoted into
the paired persistent-smoke workflow as `graph_descriptor_node_port_dict`.
The new TDD selector first failed because both smoke runners rejected the
unknown DAG shape; after adding the direct-smoke task descriptor, paired
runner expectations, graph-node op metadata, and report-visible port-task-arg
metadata, the focused local tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k node_port_dict --platform cuda
```

That run reported `2 passed, 337 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_node_port_dict --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-node-port-dict-smoke-working
```

The capture under
`tmp/cuda-backend/persistent-node-port-dict-smoke-working/persistent-graph_descriptor_node_port_dict-repeat2-smoke-b336f9ff/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, graph task arg key
`node_port_dict`, and graph task args
`task0=input.lhs:a,input.rhs:b,output.value:tmp0;`
`task1=input.lhs:a,input.rhs:b,output.value:tmp1;`
`task2=input.lhs:tmp0,input.rhs:tmp1,output.value:out`. Device times were
`61440 ns` on A100 and `41408 ns` on H200.

The dictionary-keyed graph task descriptor spelling is now promoted into the
paired persistent-smoke workflow as `graph_descriptor_task_dict`. The TDD
selector first failed because `run_persistent_smoke` rejected the unknown DAG
shape and the paired runner rejected the CLI choice. After wiring the
add/mul/add descriptor, paired-runner expectations, graph fan-in/dependent
metadata, and report-visible task-dictionary metadata, the focused local tests
passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_task_dict or task_dict_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 343 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_task_dict --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-task-dict-smoke-working
```

The capture is under:

```text
tmp/cuda-backend/persistent-task-dict-smoke-working/persistent-graph_descriptor_task_dict-repeat2-smoke-6566536a/
```

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents
`[2,2]`, graph task arg key `task_dict`, and graph task args
`join=input:a,input:b,output:out;left=input:a,input:b,output:tmp0;`
`right=input:a,input:b,output:tmp1`. Device times were `67584 ns` on A100 and
`43456 ns` on H200.

The submit-shaped graph descriptor spelling is now also available in the
paired persistent-smoke workflow as `graph_descriptor_submits`. The TDD
selector first failed because the direct smoke rejected the unknown DAG shape;
after wiring the generated add/add/add inout descriptor, paired-runner
expectations, graph fan-in/dependent metadata, and report-visible submit
task-arg metadata, the focused local tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_submits or submits_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 339 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_submits --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-submits-smoke-working
```

The capture under
`tmp/cuda-backend/persistent-submits-smoke-working/persistent-graph_descriptor_submits-repeat2-smoke-c6130969/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,1,1]`, graph fan-in `[0,1,1]`, graph
dependents `[1,2]`, graph task arg key `submits`, and graph task args
`task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;`
`task2=input:tmp1,input:a,output_existing:out`. Device times were
`74752 ns` on A100 and `48288 ns` on H200.

The submit-group graph descriptor spelling is now available in the paired
persistent-smoke workflow as `graph_descriptor_submit_groups`. The TDD
selector first failed because `run_persistent_smoke` rejected the unknown DAG
shape and the paired runner rejected the CLI choice. After wiring the
parallel add/add/join descriptor, paired-runner expectations, graph
fan-in/dependent metadata, and report-visible submit-group task-arg metadata,
the focused local tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_submit_groups or submit_groups_graph_descriptor' \
    --platform cuda
```

