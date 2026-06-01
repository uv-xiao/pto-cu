# CUDA Backend Status: Kernel Compiler Integration Part 4

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_tagged_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/tagged-inout-working
```

Result:
`tmp/cuda-backend/tagged-inout-working/persistent-graph_descriptor_tagged_inout-repeat2-smoke-a8b7819c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`, tagged task args
`input:a,input:b,output:tmp1`, `inout:tmp1,input:b`, and
`input:tmp1,input:a,output_existing:out`, repeat completions `[3,3]`,
resource policy `scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`,
and zero scheduler errors on both GPUs. A100 reported per-launch device times
`[46080,25600]` and H200 reported `[28960,20512]`.
The role-keyed variant of the same in-place graph shape is now covered by
`graph_descriptor_role_keyed_inout` in the paired persistent-smoke workflow:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_role_keyed_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/role-keyed-inout-working
```

Result:
`tmp/cuda-backend/role-keyed-inout-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-5075b400/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`, `graph_task_arg_key=role`,
task args `input:a,input:b,output:tmp1`, `inout:tmp1,input:b`, and
`input:tmp1,input:a,output_existing:out`, repeat completions `[3,3]`,
resource policy `scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`,
and zero scheduler errors on both GPUs. A100 reported per-launch device times
`[43008,26624]` and H200 reported `[41792,25440]`. The generated smoke report
adds a visible `Graph task arg key` column, so the artifact distinguishes the
preferred `role` spelling from the older `tag` spelling.

The graph adapter also accepts compact role-keyed task-argument entries such
as `{"input": "a"}`, `{"output": "tmp0"}`, `{"inout": "tmp0"}`, and
`{"output_existing": "out"}`. This keeps graph descriptors closer to a
TaskArgs-style role map without repeating `tensor` plus `role` in every
entry. Mixed compact/expanded entries are rejected before task construction.
The compact role-entry slice was checked with a failing descriptor test first,
then with local A100 and remote H200 real-data ctypes scene tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'compact_role_task_args or compact_role_graph_with_ctypes_data or \
      mixed_cuda_persistent_compact_role_task_arg' --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k compact_role_graph_with_ctypes_data --platform cuda'
```

Results: the local A100 selector reported `3 passed, 93 deselected`; the H200
real-data selector reported `1 passed, 94 deselected` with the known PTO-ISA
SSH refresh warning.

The same compact role-entry graph shape is now covered by the no-torch paired
persistent-smoke workflow as `graph_descriptor_compact_role_inout`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_compact_role_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/compact-role-inout-working
```

Result:
`tmp/cuda-backend/compact-role-inout-working/persistent-graph_descriptor_compact_role_inout-repeat2-smoke-1fbef8c4/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`,
`graph_task_arg_key=compact`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, and `input:tmp1,input:a,output_existing:out`, repeat
completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, and zero scheduler errors on both GPUs.
A100 reported per-launch device times `[43008,23552]`; H200 reported
`[28896,20224]`. The generated Markdown and SVG smoke reports expose the
compact task-argument key beside the graph topology.

The same role-keyed graph smoke was rerun at current head after lifecycle
matrix indexing landed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_role_keyed_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/current-head-role-keyed-working
```

Artifact:
`tmp/cuda-backend/current-head-role-keyed-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-8030fc57/`

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator again required dispatch
`[1,1,1]`, fan-in `[0,1,1]`, dependents `[1,2]`,
`graph_task_arg_key=role`, repeat completions `[3,3]`, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, and zero scheduler
errors. A100 reported per-launch device times `[50176,26624]`; H200 reported
`[21664,20448]`. The refreshed local index for that output root records the
same role-keyed graph metadata in one row.

The same role-keyed graph shape is also part of the selected paired benchmark
matrix as `pto_persistent_dag_graph_role_keyed_inout`. A compact A100/H200
capture:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/role-keyed-benchmark-working
```

validated `72` rows under
`tmp/cuda-backend/role-keyed-benchmark-working/combined-current-a7787008/`.
The capture required dispatch `[1,1,1]`, graph fan-in `[0,1,1]`,
dependents `[1,2]`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=role`. The role-keyed row reported A100
`device_wall_ns=38912`, `host_wall_ns=52853`; H200
`device_wall_ns=20864`, `host_wall_ns=2446166`. Both rows passed with zero
device scheduler errors.

The compact role-entry spelling is now also part of the selected paired
benchmark matrix as `pto_persistent_dag_graph_compact_role_inout`, beside the
role-keyed row. A failing benchmark/report test first required the new row in
`cuda_benchmark.py`, `cuda_pair_benchmark.py`, `cuda_validate_capture.py`, and
`cuda_current_summary.py`. A no-batch A100/H200 capture:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/compact-role-benchmark-working
```

validated `74` rows under
`tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f/`.
The capture required dispatch `[1,1,1]`, graph fan-in `[0,1,1]`,
dependents `[1,2]`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=compact`. The compact role row reported A100
`device_wall_ns=50176`, `host_wall_ns=62766`; H200
`device_wall_ns=27360`, `host_wall_ns=36621`. Both rows passed with zero
device scheduler errors, and the current-summary DAG table now includes a
`Graph Compact Role Inout/DAG` column.
The combined benchmark report was regenerated under
`tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f-report-role-spelling/`
so `cuda-benchmark.md` now has a focused `Graph Role Spelling Rows` section
and `cuda-benchmark.svg` exposes the same tag/role/compact rows in
`graph role spelling:` metadata.

The graph adapter also accepts pair-shaped task-argument entries, where each
entry is written as a two-item role/name pair. This spelling is useful as a
minimal structured form when JSON-like role maps would add unnecessary keys,
but it must still lower to the same TaskArgs role flow as tag, role-keyed,
and compact spellings. A failing benchmark/report selector first required
`graph_descriptor_pair_inout` and
`pto_persistent_dag_graph_pair_inout` in the persistent smoke, paired smoke,
benchmark, validator, and current-summary scripts. The focused selector then
passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
  -q -k 'pair_inout or graph_role_spelling or \
          validate_command_matches_configured_capture or paired_current or \
          compact_current or current_summary'
```

Result: `17 passed, 267 deselected`.

The no-torch paired persistent-smoke workflow now covers the same in-place
graph shape as `graph_descriptor_pair_inout`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_pair_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-pair-inout-smoke-working
```

Result:
`tmp/cuda-backend/persistent-pair-inout-smoke-working/persistent-graph_descriptor_pair_inout-repeat2-smoke-5028d521/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`,
`graph_task_arg_key=pair`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, and `input:tmp1,input:a,output_existing:out`, repeat
completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, and zero scheduler errors on both GPUs.
A100 reported per-launch device times `[40960,25600]`; H200 reported
`[29760,20928]`. Supplemental single-baseline samples under
`tmp/cuda-backend/pair-inout-single-benchmark/` reported A100
`device_wall_ns=49152`, `host_wall_ns=68485`, and H200
`device_wall_ns=28672`, `host_wall_ns=38506` for
`pto_persistent_dag_graph_pair_inout`.

The pair-shaped graph spelling is now also covered by the selected compact
A100/H200 benchmark matrix:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/pair-current-compact-working
```

