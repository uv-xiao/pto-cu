# Gluon GEMM H200 Evidence

This note records H200 execution evidence for generated Triton/Gluon
`gemm_f32`. It is scalar correctness evidence only; it is not tensor-core
evidence.

Short form: scalar, not tensor-core.

## Command

The local working tree was synced to a temporary checkout on a remote H200 host
through the generic CUDA remote runner:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_gemm_f32.py \
      --output-dir tmp/gluon-gemm-h200 \
      --m 16 --n 16 --k 16 \
      --arch compute_90 \
      --require-cuda'
```

## Result

The remote H200 venv used for the run provided:

- GPU: H200, `NVIDIA H200 NVL`, compute capability `(9, 0)`
- PyTorch: `2.8.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- Triton: `3.7.1`

```text
status: passed
kernel_name: gemm_f32
arch: compute_90
compiler_role: pto-isa-replacement
shape: m=16, n=16, k=16
tile_shape: [64, 128, 32]
source_sha256: 6b090607d76b5cad6446052298ed75635050e12ca2c9d072f121142a41cae99b
max_abs_error: 9.5367431640625e-07
atol: 0.0001
rtol: 0.0001
```

This proves that `examples/cuda/gluon_gemm_f32.py` can generate and execute
the scalar `gemm_f32` Gluon source on H200 with correct output under the stated
tolerance for the recorded shape.

## Boundary

The generated `gemm_f32` kernel is deliberately scalar, with one Gluon program
per output element. This evidence does not prove tensor-core codegen,
flash-attention coverage, MoE or distributed behavior, serving integration, or
performance. Generated source and JSON manifests stay under `tmp/gluon-*`.
