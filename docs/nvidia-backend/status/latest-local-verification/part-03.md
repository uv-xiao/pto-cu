# CUDA Backend Status: Latest Local Verification Part 3

Result: `tmp/cuda-backend/persistent-chain-repeat2-smoke-4bcd56c4/`
contains A100 and H200 JSON plus Markdown/SVG reports. Local A100 returned
`launch_completed_counts=[5,5]`, `launch_device_wall_ns=[44032,41984]`, and
zero scheduler errors. Remote H200 returned `launch_completed_counts=[5,5]`,
`launch_device_wall_ns=[45760,39616]`, and zero scheduler errors.

The queue-mode lifecycle capture at
`tmp/cuda-backend/persistent-queue-repeat2-smoke-0a4447c0/` verifies that the
ready-queue counters and flags are reset between prepared-callable launches.
Local A100 returned `launch_completed_counts=[4,4]` and
`launch_device_wall_ns=[22528,14336]`. Remote H200 returned
`launch_completed_counts=[4,4]` and `launch_device_wall_ns=[25280,14272]`.

After adding invalid-dependent, dependent-range, fan-in-underflow,
duplicate-dependent, self-dependent, initial-fan-in, no-root, and
unreachable-task scheduler diagnostics, the CUDA backend/codegen/report tests
were rerun locally:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
    -q --platform cuda

.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_benchmark_report.py -q
```

Result: the CUDA backend suite reported `47 passed`, and the codegen/report
suite reported `276 passed`.

After adding the third-tensor persistent DAG scene-test arg builder, the new
ctypes-backed real-data scene test was checked on remote H200 without requiring
`torch`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PTO_ISA_ROOT=<remote-pto-cu>/build/pto-isa \
   PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest \
     tests/ut/py/test_cuda_scene_test.py::test_scene_test_runs_cuda_persistent_device_triad_with_ctypes_data -q'
```

Result: `1 passed`.

After adding shared CUDA preflight skip reporting, the local A100-focused test
set was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_preflight.py \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_scene_test.py -q
```

Result: `25 passed`.

After adding the tensor-tile persistent DAG scene-test arg builder, the
focused CUDA suite was rerun locally on A100:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_kernel_compiler.py -q
```

Result: `39 passed`.

The same branch tip was checked on the remote H200 checkout. The scene-test
file passed its compile/plumbing cases and skipped the real-data cases because
the remote Python environment still lacks `torch`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   git fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py -q'
```

Result: `2 passed, 3 skipped`.

The no-torch tensor-tile persistent DAG smoke was also run on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
     --device 0 --task-count 4 --n 4096 --arch compute_90 \
     --mode dag --queue-capacity 2 --dag-shape tensor_tile'
```

Result: `status=pass`, `ptx_arch=compute_90`,
`dispatch_func_ids=[3, 1, 2, 1]`, `completed_count=4`,
`device_scheduler_errors={"count": 0, "code": 0, "task_id": 0}`.

After adding chain and reuse persistent DAG scene-test arg builders, the
focused local CUDA scene/codegen set was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_persistent_codegen.py -q
```

Result: `27 passed`. The new real-data chain and reuse scene tests both ran
through the local A100 CUDA L2 Worker path.

The same chain and reuse DAG shapes were checked on the remote H200 through
the no-torch persistent smoke path:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 5 --n 1024 --arch compute_90 \
    --mode dag --queue-capacity 3 --dag-shape chain

CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_90 \
    --mode dag --queue-capacity 3 --dag-shape scratch_reuse
```

Result: both returned `status=pass` with zero device scheduler errors.

After adding paired persistent-smoke automation, the chain DAG smoke was
captured on local A100 and remote H200 with tree sync and compact report
generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape chain --task-count 5 --queue-capacity 3 --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-chain-smoke-e1fa429b/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `device_wall_ns=29696`; the H200 row returned
`status=pass`, `ptx_arch=compute_90`, `device_wall_ns=33152`.

After adding tensor descriptor flags to the paired persistent-smoke runner,
the non-square tensor-tile DAG smoke was captured on local A100 and remote
H200 with tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_tile --task-count 4 --queue-capacity 2 \
    --n 4096 --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-tensor_tile-8x4x12-smoke-ad45b69c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `device_wall_ns=82944`; the H200 row returned
`status=pass`, `ptx_arch=compute_90`, `device_wall_ns=49536`. Both rows
reported `dispatch_func_ids=[3,1,2,1]`, tensor shape `8x4x12`, `128` tiles,
and zero device scheduler errors.

After adding tensor descriptor validation to the smoke validator, the same
non-square tensor-tile DAG was captured with prepared-callable repeat reuse:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_tile --task-count 4 --queue-capacity 2 \
    --n 768 --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-tensor_tile-8x4x12-repeat2-smoke-223425b6/`
contains A100/H200 JSON plus Markdown/SVG reports. The paired runner then
validated both artifacts with `expected-tensor-tile=8x4x12`,
`repeat_runs=2`, `launch_completed_counts=[4,4]`, zero scheduler errors, and
the generated report files. The A100 row reported
`launch_device_wall_ns=[48128,36864]`; H200 reported
`launch_device_wall_ns=[54944,27040]`.

The preflight and CUDA scene-test subset was also run on the remote H200
checkout after pushing this change:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && \
   git fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest \
     tests/ut/py/test_cuda_preflight.py \
     tests/ut/py/test_cuda_scene_test.py -q'
```

Result: `6 passed, 2 skipped`. The skips are the real-data scene cases that
still require `torch` in the remote Python environment; the compile/plumbing
and preflight checks passed.

After adding native CUDA `device` build roles, both CUDA runtimes were rebuilt
locally and on H200. `RuntimeBuilder(platform="cuda")` reported
`libcuda_device_runtime.so` for the `device` role in `host_schedule` and
`persistent_device`. The later role-cleanup slice removed the CUDA
`aicpu_path` and `aicore_path` aliases, so CUDA consumers now read the device
artifact through `path_for_role("device")` or `role_paths["device"]`.

The local A100 no-build host-schedule Worker smoke passed after the rebuild:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 --no-build
```

Result: `status=pass`, `mode=worker/add`, `ptx_arch=compute_80`,
`device_wall_ns=40960`.

The two-scalar affine host-schedule Worker smoke was captured on local A100
and remote H200 after adding the ABI:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op affine --sync-remote-tree --build-runtime
```

