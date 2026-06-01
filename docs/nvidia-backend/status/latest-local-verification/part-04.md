# CUDA Backend Status: Latest Local Verification Part 4

Result: `tmp/cuda-backend/worker-affine-smoke-dd026085/` contains A100 and
H200 JSON plus Markdown/SVG reports. A100 reported `status=pass`,
`ptx_arch=compute_80`, and `device_wall_ns=16384`; H200 reported
`status=pass`, `ptx_arch=compute_90`, and `device_wall_ns=41760`.

The three-input triad host-schedule Worker smoke was captured on local A100
and remote H200 after adding the `(a, b, c, out, n)` ABI:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op triad --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-triad-smoke-1af18449/` contains A100 and
H200 JSON plus Markdown/SVG reports. A100 reported `status=pass`,
`ptx_arch=compute_80`, and `device_wall_ns=22528`; H200 reported
`status=pass`, `ptx_arch=compute_90`, and `device_wall_ns=42496`.

The local A100 and remote H200 persistent-device DAG smokes also passed after
building the native `device` role:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2
```

Result: A100 `status=pass`, `device_wall_ns=30720`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`; H200
`status=pass`, `ptx_arch=compute_90`, `device_wall_ns=24896`, zero scheduler
errors.

After promoting the scalar AXPY DAG to a benchmark baseline, the benchmark
report tests passed locally and the new single-baseline path was checked on
both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_axpy \
    --sizes 1024 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`dispatch_func_ids=[4,2,1]`, `scalar_args={"scalar0":1.5}`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`; H200
`status=pass`, `ptx_arch=compute_90`, `dispatch_func_ids=[4,2,1]`,
`scalar_args={"scalar0":1.5}`, zero scheduler errors.

After promoting the unary host-schedule ABI to a benchmark baseline, the new
single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_unary_square \
    --sizes 1024 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`ptx_source=kernel-compiler-task-body-wrapper-unary-square-compute_80`,
`device_wall_ns=8192`; H200 `status=pass`, `ptx_arch=compute_90`,
`ptx_source=kernel-compiler-task-body-wrapper-unary-square-compute_90`,
`device_wall_ns=13888`.

After adding the unary row to the default paired benchmark, the validator was
updated to compare CUDA `float32` square results. The previous exact Python
integer-square comparison failed at `N=65536` because CUDA stores
single-precision output:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_unary_square \
    --sizes 65536 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`ptx_source=kernel-compiler-task-body-wrapper-unary-square-compute_80`,
`device_wall_ns=848864`.

After adding a two-scalar persistent DAG descriptor, the focused local CUDA
test set was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_benchmark_report.py \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_backend.py -q
```

Result: `133 passed`.

The two-scalar affine DAG smoke was captured on local A100 and remote H200
with tree sync and compact report generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape scalar_affine --task-count 3 --queue-capacity 2 \
    --n 4096 --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-scalar_affine-smoke-469f55cd/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[5,2,1]`,
`scalar_args={"scalar0":1.5,"scalar1":0.5}`, `device_wall_ns=28672`;
the H200 row returned `status=pass`, `ptx_arch=compute_90`,
`dispatch_func_ids=[5,2,1]`, the same scalar args, and
`device_wall_ns=35584`. Both rows reported zero device scheduler errors.

After adding a third tensor pointer to the persistent DAG descriptor, the
triad DAG smoke was captured on local A100 and remote H200 with tree sync and
compact report generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape triad --task-count 3 --queue-capacity 2 \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-triad-smoke-3a3bcdb1/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[6,2,1]`,
`tensor_args={"c":"tmp0"}`, `device_wall_ns=27648`; the H200 row returned
`status=pass`, `ptx_arch=compute_90`, the same dispatch IDs and tensor args,
and `device_wall_ns=24832`. Both rows reported zero device scheduler errors.

After adding a unary generated-dispatch task body, the unary-square DAG smoke
was captured on local A100 and remote H200 with tree sync and compact report
generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape unary_square --task-count 3 --queue-capacity 2 \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-unary_square-smoke-cb01f013/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[7,1,1]`, and
`device_wall_ns=30720`; the H200 row returned `status=pass`,
`ptx_arch=compute_90`, the same dispatch IDs, and `device_wall_ns=31136`.
Both rows reported zero device scheduler errors.

After promoting the unary-square DAG to a benchmark baseline, the new
single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_unary_square \
    --sizes 4096 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`dispatch_func_ids=[7,1,1]`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`, and
`device_wall_ns=43008`; H200 `status=pass`, `ptx_arch=compute_90`, the same
dispatch IDs, zero scheduler errors, and `device_wall_ns=45184`.

After promoting the triad DAG to a benchmark baseline, the new single-baseline
path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_triad \
    --sizes 4096 --arch compute_80
```

Result: `tmp/cuda-backend/persistent-triad-baseline/` contains A100 and H200
JSON plus Markdown/SVG reports. A100 returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[6,2,1]`,
`tensor_args={"c":"tmp0"}`, and `device_wall_ns=34816`; H200 returned
`status=pass`, `ptx_arch=compute_90`, the same dispatch IDs and tensor args,
and `device_wall_ns=33536`. Both rows reported zero device scheduler errors.

After promoting the two-scalar affine DAG to a benchmark baseline, the focused
report tests passed locally and the new single-baseline path was checked on
both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_affine \
    --sizes 4096 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`dispatch_func_ids=[5,2,1]`,
`scalar_args={"scalar0":1.5,"scalar1":0.5}`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`,
`device_wall_ns=31744`; H200 `status=pass`, `ptx_arch=compute_90`,
`dispatch_func_ids=[5,2,1]`, the same scalar args, zero scheduler errors,
and `device_wall_ns=30560`.

The full paired benchmark was then refreshed with scalar affine, triad, quad,
and unary-square rows in the default persistent baseline set:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-ba99b593/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`, and
`cuda-benchmark-ratios.svg`. The combined JSON has `666` samples, including
`18` `pto_persistent_dag_scalar_affine` samples and `18`
`pto_persistent_dag_triad` samples, `18`
`pto_persistent_dag_quad` samples, and `18`
`pto_persistent_dag_unary_square` samples. The compact DAG table reports
quad ratios versus `pto_persistent_dag` of `1.05x`, `1.19x`, and `1.19x` on
A100 for `N=1024,65536,1048576`, and `0.98x`, `1.07x`, and `1.01x` on H200
for the same sizes. The quad smoke and paired benchmark golden path now
matches NVCC's generated `mul.f32` plus `fma.rn.f32` sequence for
`a * b + c * d`, so the fourth tensor descriptor row is recorded as
correctness and scheduler-shape evidence. Throughput conclusions still need a
tuned tensor workload.

The combined capture was validated with:

