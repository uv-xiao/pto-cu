# Gluon H200 Performance Evidence

This note records the first review-facing Gluon microbenchmark evidence for
the representative kernels that already had H200 correctness coverage.

The benchmark harness is `examples/cuda/gluon_benchmark.py`. It always emits
structured JSON, generates repo-relative artifact paths, and treats local
non-H200 environments as skip-safe.

## Non-Claims

- Microbenchmark timings are not serving throughput.
- Skipped runs are not H200 performance evidence.
- Results cover only the listed shapes and dtypes.
- This does not claim production readiness, multi-GPU behavior, or DeepSeek
  serving behavior.
- PyTorch timings are local reference timings for the same tiny shapes, not a
  broad framework comparison.

## Harness Contract

The JSON payload records:

- `schema_version`, overall `status`, `machine_class`, and exact command;
- one entry per generated Gluon kernel;
- shape, dtype, tolerance, artifact paths, and source digest;
- correctness status and maximum absolute error when measured;
- warmup count, iteration count, and CUDA-event timings when measured;
- explicit non-claims.

The harness rejects absolute `--output-dir` values so stdout JSON does not
contain private local paths. Raw outputs remain under `tmp/` and are not
committed.

## Local Skip Verification

Local command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_benchmark.py \
  --output-dir tmp/gluon-performance-local \
  --warmup 1 --iterations 2
```

Result:

- exit code: `0`;
- overall status: `skipped`;
- reason on the local machine: `expected H200 capability 9.x, got (8, 0)`;
- all four kernel entries included repo-relative generated artifact paths;
- correctness and timing were marked `measured: false`.

## H200 Command

The H200 run used the generic tree-sync remote CUDA runner and the preserved
remote H200 venv with Torch 2.8.0+cu128 and Triton 3.7.1:

```bash
set -o pipefail; \
REMOTE_PTO_CU=/tmp/pto-cu-gluon-gemm \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    examples/cuda/gluon_benchmark.py \
    --output-dir tmp/gluon-performance-h200 \
    --warmup 3 --iterations 10 --require-cuda' \
  | tee tmp/gluon-performance-h200-remote.json
```

Result:

- exit code: `0`;
- overall status: `passed`;
- machine class: `H200`;
- device class observed before the run: `NVIDIA H200 NVL`, capability `(9, 0)`;
- raw local capture: `tmp/gluon-performance-h200-remote.json`.

## Distilled Results

All timings are CUDA-event milliseconds over 3 warmup launches and 10 measured
iterations. `generated_mean_ms` is the generated Gluon kernel timing.
`torch_mean_ms` is the PyTorch reference timing for the same shape.

| Kernel | Shape | Dtype | Tolerance | Correctness | Max abs err | generated_mean_ms | torch_mean_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemm_f32` | `m=128,n=128,k=64` | FP32 inputs/output | `atol=1e-4,rtol=1e-4` | passed | `1.1444091796875e-05` | `0.06264320015907288` | `0.02270719986408949` |
| `gemm_tensor_core_f16_f32` | `m=64,n=32,k=32` | FP16 inputs, FP32 accumulator/output | `atol=1e-3,rtol=1e-1` | passed | `0.0073032379150390625` | `0.09045439884066582` | `0.028335999883711337` |
| `gemm_tensor_core_tiled_f16_f32` | `m=256,n=256,k=64` | FP16 inputs, FP32 accumulator/output | `atol=1e-3,rtol=1e-1` | passed | `0.009243011474609375` | `0.09350399971008301` | `0.027286400273442268` |
| `flashattention_fwd_f32` | `seqlen_q=32,seqlen_k=32,head_dim=32` | FP32 Q/K/V/output | `atol=1e-3,rtol=1e-2` | passed | `2.384185791015625e-07` | `0.02705280017107725` | `0.050297600030899045` |

## Follow-Up Gaps

- Increase benchmark iteration counts only after the harness shape and JSON
  contract settle.
- Add larger tensor-core and FlashAttention shapes in separate PR-sized slices.
- Keep serving, multi-GPU, and persistent-runtime claims out of this
  microbenchmark evidence.
