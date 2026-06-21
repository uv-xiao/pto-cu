# Gluon FlashAttention H200 Correctness

This note tracks the `gluon-gen` FlashAttention forward correctness
shape-coverage milestone. It is correctness evidence, not performance
evidence.

## Current Contract

`KernelCompiler(platform="cuda").generate_gluon_kernel(...)` supports
`flashattention_fwd_f32` as a representative single-tile attention kernel.

The generated source uses explicit Gluon `BlockedLayout` and
`DotOperandLayout` operands. The square score path and the value accumulation
path use `dot_fma`; rectangular score shapes where `seqlen_k != head_dim` use
a scalar FMA score loop so row-major K storage can still represent `k.T`
correctly:

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

The `head_dim=64` sweep case remains the smallest bounded H200-passing
representative shape selected during the earlier blocker slice. The narrower
`32x32x64` candidate is now a first-class single-case repro and passes H200
correctness, but it remains separate from the two-case promoted sweep.

## H200 Evidence

On 2026-06-21, the harness passed on a remote H200 host through the generic
CUDA runner. The remote checkout used the preserved CUDA Python environment
needed for Torch and Triton/Gluon `dot_fma`.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
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
source_sha256: 9fab42d2a6cf44d24eedb00f47576e2f9d09f477f1cacf8f3903dbc5ef83a03e
tile_shape: 32x32x32
shape: seqlen_q=16, seqlen_k=64, head_dim=64
provenance: common serving attention head dimension; selected after
  32x32x64 failed H200 correctness
tolerance: atol=0.001, rtol=0.01
max_abs_error: 2.682209014892578e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 4684a32f6b09462a051de5bc611e0cef58b0914e7b34cf91687ba87a45c65bce
tile_shape: 16x64x64
artifact paths are repo-relative
per-case artifact paths are repo-relative
machine class: H200
private absolute paths are not recorded
```

## Resolved `32x32x64` Repro

On 2026-06-21, the explicit `32x32x64` repro generated and launched on the
same H200 class machine and passed correctness against the PyTorch reference.
The command used the generic CUDA runner, repo-relative artifact paths, and
the preserved CUDA Python environment.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-32x32x64-fixed-h200 \
      --tile-shape 32x32x64 --seed 0 --require-cuda --arch compute_90'
```

Distilled passed result:

```text
schema_version: 1
status: passed
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax((q @ k.T) * scale) @ v
tolerance: atol=0.001, rtol=0.01
max_abs_error: 3.5762786865234375e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: c611666d3b527e615f2d8e4658b57f10865f1547fd370e8bb45639353682a06e
tile_shape: 32x32x64
machine class: H200
artifact paths are repo-relative
private absolute paths are not recorded
```

The fix is bounded to the generated single-kernel correctness case. The
rectangular score path avoids the current Gluon `dot_fma` RHS-layout boundary
for row-major K storage; the value accumulation path still uses `dot_fma`.

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
