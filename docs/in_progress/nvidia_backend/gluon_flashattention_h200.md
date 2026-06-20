# Gluon FlashAttention H200 Correctness

This note tracks the first `gluon-gen` FlashAttention forward correctness
milestone. It is correctness evidence, not performance evidence.

## Current Contract

`KernelCompiler(platform="cuda").generate_gluon_kernel(...)` supports
`flashattention_fwd_f32` as a representative single-tile attention kernel.

The generated source uses Gluon `dot_fma` with explicit `BlockedLayout` and
`DotOperandLayout` operands for both matmul stages:

- score formation: `q @ k.T`
- softmax normalization over the key dimension
- value accumulation: `softmax(q @ k.T * scale) @ v`

The harness at `examples/cuda/gluon_flashattention_fwd.py` builds the generated
artifact, loads it, and runs one `32x32x32` FP32 correctness case against
PyTorch when CUDA and a Gluon build with `dot_fma` support are available.

## H200 Evidence

On 2026-06-20, the harness passed on a remote H200 host through the generic
tree-sync CUDA runner. The remote temporary checkout used a project-local
`.venv` configured with the preserved CUDA Python packages needed for Torch
and Triton/Gluon `dot_fma`.

```bash
REMOTE_PTO_CU=/tmp/pto-cu-gluon-flashattention \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-h200 \
      --arch compute_90 \
      --require-cuda'
```

Distilled result:

```text
status: passed
shape: seqlen_q=32, seqlen_k=32, head_dim=32
tolerance: atol=0.001, rtol=0.01
max_abs_error: 2.384185791015625e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
tile_shape: 32x32x32
```

## Boundary

This milestone covers single-tile FlashAttention correctness only. It is not
benchmark evidence and does not cover block streaming, causal masking,
varlen/page tables, persistent scheduling, MoE, distributed communication,
serving, or DeepSeek integration.
