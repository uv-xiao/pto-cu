# CUDA Backend Status: Kernel Compiler Integration Part 3

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'args_alias' --platform cuda
```

reported `2 passed, 121 deselected`, covering both the descriptor and a
ctypes-backed real-data persistent-device graph scene. The same synced
working tree on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && CUDA_HOME=/usr/local/cuda \
   PATH=/usr/local/cuda/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
   tests/ut/py/test_cuda_scene_test.py -q -k args_alias --platform cuda'
```

also reported `2 passed, 121 deselected`, after the known PTO-ISA SSH refresh
warning.
Graph task descriptors now also accept role-map `task_args` dictionaries such
as `{"inputs": ["a", "b"], "output": "tmp0"}`. This keeps the descriptor
closer to a bundled `TaskArgs` object while preserving CUDA descriptor field
ordering. The focused regression first failed because the adapter iterated
dictionary keys as task-arg entries. After adding role-map normalization, the
descriptor selector reported `4 passed, 139 deselected`, the local A100 ctypes
selector reported `1 passed, 142 deselected`, and the synced H200 selector
reported `1 passed, 142 deselected` after the known PTO-ISA SSH refresh
warning.
The paired persistent-smoke/report path now also has a dedicated
`graph_descriptor_role_map_inout` shape. The focused report-helper TDD first
failed because both the persistent smoke builder and paired-smoke CLI rejected
that shape. After registering it, the focused selector reported
`2 passed, 296 deselected`. The paired A100/H200 smoke under
`tmp/cuda-backend/role-map-inout-working/`
`persistent-graph_descriptor_role_map_inout-repeat2-smoke-63e71c8a/`
validated `repeat_runs=2`, dispatch `1,1,1`, graph fan-in `0,1,1`,
dependents `1,2`, `graph_task_arg_key=role_map`, generated Markdown/SVG
reports, and zero scheduler errors. A100 reported `device_wall_ns=69632`;
H200 reported `device_wall_ns=49888`.
Graph task-list descriptors now also accept top-level `graph.submits` or
`graph.submissions` as aliases for `graph.tasks`/`graph.nodes`, so a
scene-test descriptor can look like a list of
`submit_next_level(callable, TaskArgs, ...)` calls. Submit entries still use
the same named/indexed callable resolution, `args`/`task_args` role lowering,
temporary allocation, and tensor-flow edge inference as task entries. The
focused regression first failed with
`CUDA persistent_dag_graph_f32 requires at least one graph task` because the
adapter ignored `submits`. After adding the alias, the local A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'submits_alias or args_alias' --platform cuda
```

reported `4 passed, 135 deselected`, covering descriptor construction and
ctypes-backed real-data persistent-device graph scenes. The same synced
working tree on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && CUDA_HOME=/usr/local/cuda \
   PATH=/usr/local/cuda/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
   tests/ut/py/test_cuda_scene_test.py -q \
   -k "submits_alias or args_alias" --platform cuda'
```

also reported `4 passed, 135 deselected`, after the known PTO-ISA SSH refresh
warning.
Graph descriptors now also accept `graph.submit_groups` or
`graph.submission_groups` as a bridge toward
`submit_next_level_group(callable, args_list, config)`. The current CUDA
persistent-device tracer bullet expands each `args_list` entry into one CUDA
DAG task while preserving callable resolution, TaskArgs-like role lowering,
temporary allocation, and tensor-flow dependency inference. This is not yet
the final PTO one-slot group semantics. The focused regression first failed
with `CUDA persistent_dag_graph_f32 requires at least one graph task` because
the adapter ignored `submit_groups`. After adding group expansion, the local
A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'submit_groups or submits_alias' --platform cuda
```

reported `4 passed, 137 deselected`, covering descriptor construction and
ctypes-backed real-data persistent-device graph scenes. The same synced
working tree on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && CUDA_HOME=/usr/local/cuda \
   PATH=/usr/local/cuda/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
   tests/ut/py/test_cuda_scene_test.py -q \
   -k "submit_groups or submits_alias" --platform cuda'
```

also reported `4 passed, 137 deselected`, after the known PTO-ISA SSH refresh
warning.
Graph task descriptors now also accept two-item role/name pairs in the
`task_args` list, such as `("input", "a")`, `("output", "tmp0")`,
`("inout", "tmp0")`, and `("output_existing", "out")`. These pair entries
normalize into the same role-keyed lowering path as expanded dictionaries,
so they preserve temporary allocation, existing-output aliasing, and
tensor-flow dependency inference. The focused TDD selector first failed at
descriptor construction because non-dictionary entries were rejected. After
adding pair normalization, the local A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k pair_task_args --platform cuda
```

reported `2 passed, 123 deselected`, covering both descriptor construction
and a ctypes-backed real-data persistent-device graph scene. The same synced
working tree on H200 reported `2 passed, 123 deselected` with the known
PTO-ISA SSH refresh warning. A neighboring local selector for
`pair_task_args`, `role_keyed_task_args`, `args_alias`, and
`compact_role_task_args` reported `6 passed, 119 deselected`, checking that
the new shorthand did not regress the existing graph task-argument spellings.
The role mapping now preserves the lifecycle distinction needed by CUDA
memory planning: role `output` may create a default-sized temporary, but
roles `output_existing` and `inout` must name storage that is already known at
that point in descriptor order. Descriptor construction raises before launch
if either role references an unknown tensor or temporary, avoiding a silent
scratch allocation for values that are supposed to alias existing storage.
The same lifecycle rule is now applied to explicit graph `out_storage`.
Logical `out` names still create default-sized temporaries, but an
`out_storage` alias must point at storage that has already been allocated or
declared. This keeps scratch-buffer reuse explicit and prevents typos in the
physical storage name from allocating a new buffer silently.
The negative lifecycle cases are now covered by descriptor-only regression
tests: unknown `output_existing`, unknown `inout`, and unknown `out_storage`
names fail before task struct construction. The combined tagged role selector
reported `4 passed, 68 deselected`; the scratch-storage selector reported
`2 passed, 71 deselected`; and the valid tagged-inout and scratch-reuse
real-data selectors still reported `1 passed` each on local A100 and remote
H200.

The tagged graph lowering was checked with a failing test first, then local
A100 and remote H200 real-data ctypes scene tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k tagged_task_args -m 'not requires_hardware'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k tagged_graph --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k tagged_graph --platform cuda'
```

Results: the descriptor-only test reported `1 passed, 67 deselected`; the
local A100 real-data tagged graph scene reported `1 passed, 67 deselected`;
and the H200 real-data tagged graph scene reported `1 passed, 67 deselected`
after the known PTO-ISA SSH refresh warning.
After adding scalar entries to tagged graph `task_args`, the descriptor and
local A100 real-data selector was rerun:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'scalar_task_args or tagged_graph' --platform cuda
```

Result: `2 passed, 77 deselected`.
After syncing the branch tree to H200 with `rsync`, the same selector returned
`2 passed, 77 deselected` with the known PTO-ISA SSH refresh warning.

Named-callable graph lowering was then added under TDD. The first real-data
selector failed with `KeyError: 'func_id'` because graph tasks using
`callable: "generic"` still reached task construction without resolving the
callable metadata. After adding callable-name resolution, the local A100
descriptor and real-data selectors passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'named_callable or named_callables'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'unknown_cuda_persistent_graph_callable_name'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k named_callable_graph_with_ctypes_data --platform cuda
```

Results: the first selector reported `2 passed, 80 deselected`, the unknown
callable-name guard reported `1 passed, 81 deselected`, and the local A100
real-data named-callable graph scene reported `1 passed, 81 deselected`. The
same real-data selector passed on the H200 checkout after syncing the touched
files: `1 passed, 81 deselected`, again with the known PTO-ISA SSH refresh
warning.
Tagged `inout` graph lowering was then covered with a failing descriptor test
that first produced fan-in `[0,0,1]`, leaving the in-place update as an
incorrect root. After changing tensor-flow inference to prefer the nearest
previous producer for duplicate logical tensor names, the tagged inout
descriptor and real-data ctypes scene passed locally on A100, and the same
selector passed on H200 after syncing the tree:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k tagged_inout_graph -m 'not requires_hardware'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k tagged_inout_graph \
  --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k tagged_inout_graph --platform cuda'
```

Results: descriptor-only `1 passed, 71 deselected`; local A100 real-data
`1 passed, 71 deselected`; remote H200 real-data `1 passed, 71 deselected`
with the known PTO-ISA SSH refresh warning.
The same tagged `inout` graph shape is now part of the no-torch paired
persistent-smoke workflow as `graph_descriptor_tagged_inout`:

