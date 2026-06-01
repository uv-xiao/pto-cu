# CUDA Backend Status: Latest Local Verification Part 2

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2
```

Result: `status=pass`, `device_scheduler_errors={"count": 0, "code": 0,
"task_id": 0}`, `completed_count=3`.

The synthetic invalid-dispatch shape was also run locally to verify that an
unsupported generated-dispatch `func_id` is surfaced before output mismatch
checks:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_func_id
```

Result: expected non-zero exit with `persistent dag scheduler error code=1
task_id=0 count=1`.

The synthetic invalid-dependent shape was run locally to verify that a runtime
graph descriptor cannot release a task ID outside the descriptor array:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_dependent
```

Result: expected non-zero exit with `persistent dag scheduler error code=2
task_id=7 count=1`.

The synthetic invalid-dependent-range shape was run locally to verify that a
runtime graph descriptor cannot read outside the dependents array:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_dependent_range
```

Result: expected non-zero exit with `persistent dag scheduler error code=3
task_id=0 count=1`.

The synthetic fan-in-underflow shape was run locally to verify that a runtime
graph descriptor cannot decrement a dependent task's fan-in below zero:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape bad_fanin_underflow
```

Result: expected non-zero exit with `persistent dag scheduler error code=4
task_id=2 count=1`.

The synthetic duplicate-dependent shape was run locally to verify that a
runtime graph descriptor cannot list the same dependent task twice for one
completed task. Without this check, one producer could decrement the same
dependent fan-in twice and make it ready without two distinct predecessors:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 2 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape bad_duplicate_dependent
```

Result: expected non-zero exit with `persistent dag scheduler error code=8
task_id=1 count=1`.

The synthetic self-dependent shape was run locally to verify that a runtime
graph descriptor cannot make a completed task release itself:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_self_dependent
```

Result: expected non-zero exit with `persistent dag scheduler error code=9
task_id=0 count=1`.

The synthetic initial-fan-in mismatch shape was run locally to verify that a
runtime graph descriptor cannot start from fan-in counters that disagree with
task metadata:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_initial_fanin
```

Result: expected non-zero exit with `persistent dag scheduler error code=5
task_id=0 count=1`.

The synthetic no-root shape was run locally to verify that a runtime graph
descriptor cannot deadlock workers by declaring no zero-fan-in task:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_no_root
```

Result: expected non-zero exit with `persistent dag scheduler error code=6
task_id=0 count=1`.

The synthetic unreachable-task shape was run locally to verify that a runtime
graph descriptor cannot deadlock workers by publishing one root but leaving
another task behind a dangling fan-in counter:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 2 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_unreachable \
    --worker-blocks 2
```

Result: expected non-zero exit with `persistent dag scheduler error code=7
task_id=1 count=1`.

The same scheduler-diagnostic slice was verified on the remote H200 checkout
after pushing this change:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   git fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
     --device 0 --task-count 3 --n 1024 --arch compute_90 \
     --mode dag --queue-capacity 2'
```

Result: `status=pass`, `ptx_arch=compute_90`,
`device_scheduler_errors={"count": 0, "code": 0, "task_id": 0}`,
`completed_count=3`.

The H200 invalid-dispatch check returned the expected diagnostic:
`persistent dag scheduler error code=1 task_id=0 count=1`.

The H200 invalid-dependent check returned the expected diagnostic:
`persistent dag scheduler error code=2 task_id=7 count=1`.

The H200 invalid-dependent-range check returned the expected diagnostic:
`persistent dag scheduler error code=3 task_id=0 count=1`.

The H200 fan-in-underflow check returned the expected diagnostic:
`persistent dag scheduler error code=4 task_id=2 count=1`.

After syncing this working tree to H200, the duplicate-dependent check returned
the expected diagnostic:
`persistent dag scheduler error code=8 task_id=1 count=1`.

The H200 self-dependent check returned the expected diagnostic:
`persistent dag scheduler error code=9 task_id=0 count=1`.

The H200 initial-fan-in mismatch check returned the expected diagnostic:
`persistent dag scheduler error code=5 task_id=0 count=1`.

The H200 no-root check returned the expected diagnostic:
`persistent dag scheduler error code=6 task_id=0 count=1`.

The H200 unreachable-task check returned the expected diagnostic:
`persistent dag scheduler error code=7 task_id=1 count=1`.

The current scheduler-diagnostic matrix was then captured as paired A100/H200
JSON, Markdown, and SVG evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_scheduler_error_matrix.py \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/scheduler-error-matrix-working
```

Result:
`tmp/cuda-backend/scheduler-error-matrix-working/scheduler-error-matrix-35de3303/`
contains `cuda-scheduler-error-matrix.json`,
`cuda-scheduler-error-matrix.md`, and
`cuda-scheduler-error-matrix.svg`. The matrix has `18` rows covering A100 and
H200 for unsupported `func_id`, invalid dependent ID, invalid dependent
range, fan-in underflow, duplicate dependent, self dependent, initial fan-in
mismatch, no root, and unreachable task. Every row reported `status=pass`
with the expected scheduler code, task ID, and `count=1`.

The matrix also passes the scheduler-error matrix validator with
`--preset default`, which checks the required cases, A100/H200 coverage,
source-paper provenance, command examples, and generated report files.

The current unreachable-task slice was also checked on H200 through pytest
after syncing the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
     -q -rs -k "unreachable or smoke_runs_dispatch_dag" --platform cuda'
```

Result: `10 passed, 25 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The persistent callable lifecycle path has repeat-run smoke support that
prepares the callable once and launches it multiple times. Direct mode reuses
the prepared callable, queue mode resets the ready queue counters and flags,
and DAG mode resets the graph state between launches:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 5 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape chain --repeat-runs 2
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 4096 --arch compute_80 \
    --mode queue --queue-capacity 2 --repeat-runs 2
```

