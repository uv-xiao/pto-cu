# Gluon FlashAttention H200 Correctness

This note tracks the `gluon-gen` FlashAttention forward correctness
shape-coverage milestone. It is correctness evidence, not performance
evidence.

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
artifact, loads it, and runs FP32 correctness cases against PyTorch when CUDA
and a Gluon build with `dot_fma` support are available. The default path still
runs one `32x32x32` single-tile case. The single-case path also accepts
`--tile-shape MxNxD` for bounded repros such as `32x32x64`. The `--sweep` path
emits aggregate
structured JSON with `schema_version: 1`, aggregate status and case counts,
and per-case shape, provenance, artifact metadata, status, and max error when
run. Artifact paths are repo-relative, absolute `--output-dir` values are
rejected, and exception text is sanitized so private absolute paths are not
recorded.

The sweep includes:

- `existing_32x32x32`: `seqlen_q=32`, `seqlen_k=32`, `head_dim=32`;
- `q16_k64_head_dim64`: `seqlen_q=16`, `seqlen_k=64`, `head_dim=64`,
  provenance `common serving attention head dimension; selected after
  32x32x64 failed H200 correctness`.

The `head_dim=64` case uses the smallest bounded H200-passing representative
shape found during this slice. The narrower `32x32x64` candidate generated and
ran, but failed H200 correctness. It is now a first-class single-case repro
and is not promoted as passing evidence.

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
      --output-dir tmp/gluon-flashattention-shape-coverage-h200 \
      --sweep --require-cuda --arch compute_90'
```

Distilled result:

```text
schema_version: 1
status: passed
case_count: 2
passed_cases: 2
failed_cases: 0
skipped_cases: 0
shape: seqlen_q=32, seqlen_k=32, head_dim=32
provenance: existing 32x32x32 correctness fixture
tolerance: atol=0.001, rtol=0.01
max_abs_error: 2.384185791015625e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 8be2d57903aa58cfb3e2f58eb583dbfdc9f84611b85b90318a3ae68e642330fd
tile_shape: 32x32x32
shape: seqlen_q=16, seqlen_k=64, head_dim=64
provenance: common serving attention head dimension; selected after
  32x32x64 failed H200 correctness
tolerance: atol=0.001, rtol=0.01
max_abs_error: 2.682209014892578e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 1078903f5b4aae7f524685e39b5898f8bc9a68e1732250c3b77f1a84b28c3685
tile_shape: 16x64x64
artifact paths are repo-relative
per-case artifact paths are repo-relative
machine class: H200
private absolute paths are not recorded
```

## Current Blocker

On 2026-06-21, the explicit `32x32x64` repro still generated and launched on
the same H200 class machine but failed correctness against the PyTorch
reference. The command used the generic tree-sync CUDA runner, repo-relative
artifact paths, and the preserved CUDA Python environment.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-32x32x64-blocker-h200 \
      --tile-shape 32x32x64 --seed 0 --require-cuda --arch compute_90'
```

Distilled failed result:

```text
schema_version: 1
status: failed
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax((q @ k.T) * scale) @ v
tolerance: atol=0.001, rtol=0.01
max_abs_error: 1.362005591392517
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 475e4c60be3bd660db8a8b4483889ceda87df4e0c20349bcd644d2e72255435c
tile_shape: 32x32x64
machine class: H200
artifact paths are repo-relative
private absolute paths are not recorded
```

The failure is bounded to the generated single-kernel correctness case. A
minimal row-major K/V load experiment was not promoted because it regressed the
two H200-passing sweep shapes, so the next fix needs a Gluon dot-operand layout
change that preserves `32x32x32` and `16x64x64` while making `32x32x64` pass.

## Boundary

This milestone covers a small single-tile FlashAttention correctness sweep
only. It is not benchmark evidence and does not cover block streaming, causal
masking, varlen/page tables, persistent scheduling, MoE, distributed
communication, serving, or DeepSeek integration.

Non-claims:

- not production serving readiness
- not FlashInfer integration evidence
- not DeepSeek semantic correctness
- not performance, throughput, or latency evidence
- not multi-tile attention coverage
- not fused attention integration
- not KV-cache integration
- not vLLM/simpler-nv integration evidence
