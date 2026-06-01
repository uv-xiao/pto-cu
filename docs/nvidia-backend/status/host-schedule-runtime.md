# CUDA Backend Status: Host-Schedule Runtime

## Host-Schedule Runtime

The CUDA `host_schedule` runtime is implemented as a real CUDA host runtime
slice. It supports:

- device initialization/finalization;
- device allocation and free;
- host-to-device and device-to-host copies;
- PTX module loading through the CUDA Driver API;
- prepared callable registration and unregistration;
- vector-add launch through `run_prepared`;
- a non-blocking stream pool using callable `stream_id` metadata. The pool
  defaults to four streams and can be sized at device init with
  `PTO_CUDA_STREAM_POOL_SIZE` for host-schedule concurrency experiments.

`KernelCompiler(platform="cuda").compile_cuda_host_schedule()` now compiles a
user-authored CUDA task body through the shared wrapper generator and writes a
cached host-schedule callable artifact under
`build/cache/cuda/onboard/host_schedule/callables/`. The generated host
wrapper can lower a task context into the current vector-add launch ABI, so
the artifact can be passed to `prepare_callable` and run with real device
data.

Evidence:

- `tests/ut/py/test_cuda_backend.py` validates vector-add with real CUDA
  device data.
- The stream concurrency smoke validates two independent prepared callables
  on distinct streams.
- The stream-pool sizing smoke validates a callable on `stream_id=5` with
  `PTO_CUDA_STREAM_POOL_SIZE=6` on both local A100 and remote H200.
- `simpler_setup.cuda_callable_compiler.prepare_cuda_host_schedule_callable()`
  builds the shared ctypes manifest for host-schedule compiler artifacts and
  preserves PTX/entry-name buffer lifetimes for `prepare_callable`.
- `PreparedCudaCallable` exposes `buffer_ptr()` / `buffer_size()`, and the L2
  Python `Worker.register(...)` path can prepare those raw CUDA callable
  blobs through `prepare_callable_from_blob`.
- L2 Python `Worker.run(...)` can launch backend-specific raw CUDA argument
  structs that expose `buffer_ptr()` / `buffer_size()`, so the host-schedule
  vector-add path no longer has to call `run_prepared` through `ctypes`.
- `tests/ut/py/test_cuda_kernel_compiler.py` covers the CUDA `KernelCompiler`
  entry point for host-schedule task bodies.
- `tests/ut/py/test_cuda_backend.py` runs one host-schedule callable compiled
  by `KernelCompiler` through `prepare_callable` and validates real CUDA output
  data.
- `tests/ut/py/test_cuda_backend.py` also runs the compiler-backed
  host-schedule vector-add through `Worker(level=2, platform="cuda")`,
  `Worker.register(...)`, device allocation/copy helpers, and `Worker.run(...)`
  with a real `CudaVectorAddArgs` struct.
- `SceneTestCase` L2 compilation accepts `CALLABLE["cuda"]` specs for
  `host_schedule`, compiles them through `KernelCompiler(platform="cuda")`,
  registers the prepared raw callable through the normal L2 `Worker`, builds
  `CudaVectorAddArgs`, `CudaVectorUnaryArgs`, `CudaVectorScaleArgs`,
  `CudaVectorAxpyArgs`, `CudaVectorAffineArgs`, and `CudaVectorTernaryArgs`
  from normal `TaskArgsBuilder` CPU tensors/scalars, and validates real
  copied-back CUDA output data.

Focused stream-pool verification first failed because the fixed four-stream
pool rejected a callable manifest with `stream_id=5`. After adding
`PTO_CUDA_STREAM_POOL_SIZE`, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py -q \
  -k 'stream_pool_size_env or independent_callables_on_multiple_streams' \
  --platform cuda
```

Result: `2 passed, 37 deselected`. The same stream-pool sizing selector
passed on remote H200 after syncing the tree and rebuilding the runtime:
`1 passed, 38 deselected`, with the known PTO-ISA SSH refresh warning printed
before pytest.

The stream-concurrency benchmark now accepts `--stream-pool-size` and records
the host stream-pool setting in report metadata. A paired A100/H200 capture
used `--stream-pool-size 6`, two repeats, and the stream-concurrency
microbenchmark:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
  --stream-concurrency --device 0 --repeats 2 --arch compute_80 \
  --stream-pool-size 6 --label a100-stream-pool6-working \
  --output-dir tmp/cuda-backend/a100-stream-pool6-working
```

The H200 command used `--arch compute_90` after syncing the working tree.
The merged report is under `tmp/cuda-backend/stream-pool6-working/` and
contains JSON, Markdown, and SVG report files. A100 reported median
`pto_stream_parallel/pto_stream_serial = 0.51x`; H200 reported `0.48x`.
The merged Markdown report includes `Host stream pool size: 6`, making the
concurrency configuration explicit in the visual artifact set.

The stream-concurrency workflow now has a paired runner,
`cuda_pair_stream_benchmark.py`, so the A100/H200 capture is repeatable with
the same validation discipline as the selected persistent benchmark gates. The
TDD check first failed because the paired stream runner did not exist, then
passed after adding command construction, sanitized merge examples, and a
capture validation command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'pair_stream_benchmark'
```

Result: `2 passed, 266 deselected`.

The paired A100/H200 run used `--stream-pool-size 6`, two repeats, local
`compute_80`, remote `compute_90`, and tree sync to the H200 host:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_stream_benchmark.py \
    --repeats 2 --stream-pool-size 6 --sync-remote-tree \
    --output-root tmp/cuda-backend/stream-pair-working
```

Artifacts:

- `tmp/cuda-backend/stream-pair-working/a100-stream-pool6-a36d137b/`
- `tmp/cuda-backend/stream-pair-working/h200-stream-pool6-a36d137b/`
- `tmp/cuda-backend/stream-pair-working/combined-stream-pool6-a36d137b/`
- `tmp/cuda-backend/stream-pair-working/index.md`

The validator required machines `hina` and `dasys-h200x8`, baselines
`pto_stream_serial` and `pto_stream_parallel`, size `2`, two repeats, eight
total rows, source-paper provenance, command examples, and generated report
files.

| GPU | Serial ns | Parallel ns | Parallel/serial |
| --- | --------- | ----------- | --------------- |
| A100 | 113791963 | 58041653 | 0.51x |
| H200 | 89717007 | 46203624 | 0.51x |

