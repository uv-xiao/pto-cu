# CUDA Backend Status: Fourth-Tensor Persistent DAG Verification

## Fourth-Tensor Persistent DAG Verification

After adding the fourth tensor pointer to the persistent DAG descriptor, the
new quad DAG shape was verified with both Python coverage and real CUDA data.
The quad graph uses generated-dispatch `func_id` sequence `[8, 2, 1]`; the
first task computes `a * b + c * d`, and the final task adds an independent
`a * b` branch.

Focused local tests:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_benchmark_report.py \
  -q -m "not requires_hardware"

.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k quad_with_ctypes --platform cuda

.venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
  -q -k dispatch_dag_quad --platform cuda
```

Results: `148` non-hardware tests passed, the CUDA ctypes scene test passed,
and the CUDA standalone smoke test passed on the local A100.

The same standalone smoke was run on the remote H200 with a tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_90 \
    --mode dag --queue-capacity 2 --dag-shape quad \
    --output-json tmp/cuda-backend/persistent-quad-smoke-working/h200.json
```

Result artifacts:

- `tmp/cuda-backend/persistent-quad-smoke-working/a100.json`
- `tmp/cuda-backend/persistent-quad-smoke-working/h200.json`
- `tmp/cuda-backend/persistent-quad-smoke-working/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-quad-smoke-working/cuda-smoke-report.svg`

Both A100 and H200 runs reported zero scheduler errors and tensor arguments
`c=tmp0,d=tmp3`.

