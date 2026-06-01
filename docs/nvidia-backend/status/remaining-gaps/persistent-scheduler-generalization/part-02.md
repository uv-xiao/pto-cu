# CUDA Backend Status: Persistent Scheduler Generalization Part 2

That run reported `2 passed, 341 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_submit_groups --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-submit-groups-smoke-working
```

The capture is under:

```text
tmp/cuda-backend/persistent-submit-groups-smoke-working/persistent-graph_descriptor_submit_groups-repeat2-smoke-dccd3bd9/
```

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,1,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, graph task arg key `submit_groups`, and graph task args
`task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2;`
`task2=input:tmp1,input:tmp2,output_existing:out`. Device times were
`64512 ns` on A100 and `43616 ns` on H200.

The named graph-callable descriptor spelling is now available in the paired
persistent-smoke workflow as `graph_descriptor_named_callable`. The TDD
selector first failed because `run_persistent_smoke` rejected the unknown DAG
shape and the paired runner rejected the CLI choice. After wiring the
add/mul/add descriptor, paired-runner expectations, graph fan-in/dependent
metadata, report-visible graph-node ops, and callable-name task-arg metadata,
the focused local tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_named_callable or named_callable_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 345 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_named_callable --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-named-callable-smoke-working
```

The capture is under:

```text
tmp/cuda-backend/persistent-named-callable-smoke-working/persistent-graph_descriptor_named_callable-repeat2-smoke-4b785a91/
```

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents
`[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, graph task arg key
`named_callable`, and graph task args
`task0=callable:add,input:a,input:b,output:tmp0;`
`task1=callable:mul,input:a,input:b,output:tmp1;`
`task2=callable:add,input:a,input:b,output:out`. Device times were
`66560 ns` on A100 and `42784 ns` on H200.

The same named-callable graph descriptor is now part of the selected paired
benchmark matrix as `pto_persistent_dag_graph_named_callable`. The compact
paired benchmark gate uses `--batch-tasks 0` as the no-batch sentinel, so the
paired runner and `compact-current` validator now normalize that value away
and require `96` A100/H200 samples: `48` selected non-batch rows on each
machine. The focused TDD selector first failed because the compact validator
still expected batch rows and `104` samples for this no-batch command. After
normalizing zero batch sweeps in `cuda_pair_benchmark.py` and splitting the
compact baseline list from the full paired-current list in
`cuda_validate_capture.py`, the selector passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
  -q -k 'compact_current_preset_matches_docs_gate or \
omits_empty_batch_sweeps or treats_zero_batch_sweeps_as_empty or \
validate_command_matches_configured_capture' --platform cuda
```

That run reported `4 passed, 292 deselected`. The compact A100/H200 capture
under:

```text
tmp/cuda-backend/persistent-named-callable-baseline-working/combined-current-95be2b5b/
```

passes:

```bash
PYTHONPATH=$PWD:$PWD/python \
  ROOT=tmp/cuda-backend/persistent-named-callable-baseline-working \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    "$ROOT"/combined-current-95be2b5b/cuda-benchmark.json \
    --preset compact-current
```

It validates source-paper provenance, sanitized command examples, graph
topology/task-argument reports, tensor-throughput reports, and zero scheduler
errors. The named-callable benchmark rows were:

- A100: `pto_persistent_dag_graph_named_callable`, `n=1024`,
  dispatch `1,2,1`, fan-in `0,0,2`, dependents `2,2`, task arg key
  `named_callable`, `33792 ns`.
- H200: `pto_persistent_dag_graph_named_callable`, `n=1024`,
  dispatch `1,2,1`, fan-in `0,0,2`, dependents `2,2`, task arg key
  `named_callable`, `25728 ns`.

The refreshed full paired-current capture under
`tmp/cuda-backend/current-head-full-submit-groups-working/`
`combined-current-c183d1ad/` validates the same selected row across
`N=1024,65536,1048576` with zero scheduler errors. Median named-callable
device times were `26624/135168/2361344 ns` on A100 and
`25344/132320/1907936 ns` on H200 for those sizes.

Needed:

