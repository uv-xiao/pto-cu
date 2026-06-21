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
- PyTorch reference expression: `softmax((q @ k.T) * scale) @ v`

The harness at `examples/cuda/gluon_flashattention_fwd.py` builds the generated
artifact, loads it, and runs one `32x32x32` FP32 correctness case against
PyTorch when CUDA and a Gluon build with `dot_fma` support are available.
The stdout JSON uses `schema_version: 1`, keeps artifact paths repo-relative,
rejects absolute `--output-dir` values, and sanitizes exception text so
private absolute paths are not recorded.

## H200 Evidence

On 2026-06-21, the harness passed on a remote H200 host through the generic
tree-sync CUDA runner. The remote checkout used the preserved CUDA Python
environment needed for Torch and Triton/Gluon `dot_fma`.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-review-hygiene-h200 \
      --require-cuda --arch compute_90'
```

Distilled result:

```text
schema_version: 1
status: passed
shape: seqlen_q=32, seqlen_k=32, head_dim=32
tolerance: atol=0.001, rtol=0.01
max_abs_error: 2.384185791015625e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 8be2d57903aa58cfb3e2f58eb583dbfdc9f84611b85b90318a3ae68e642330fd
artifact paths are repo-relative
tile_shape: 32x32x32
machine class: H200
private absolute paths are not recorded
```

## Boundary

This milestone covers single-tile FlashAttention correctness only. It is not
benchmark evidence and does not cover block streaming, causal masking,
varlen/page tables, persistent scheduling, MoE, distributed communication,
serving, or DeepSeek integration.

Non-claims:

- not production serving readiness
- not FlashInfer integration evidence
- not DeepSeek semantic correctness
- not performance, throughput, or latency evidence
- not multi-tile attention coverage
- not fused attention integration
- not KV-cache integration
- not vLLM/simpler-nv integration evidence
