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
- optional causal gate: `causal: true` applies a lower-triangular
  key-index/query-index mask before softmax, with PyTorch reference
  `softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v`
- prefill-shaped causal gate: same-length multi-query causal attention is
  marked as `phase: prefill` and keeps the lower-triangular PyTorch reference
  instead of the shifted decode/append reference
- decode-shaped causal gate: a single query over a longer KV prefix is marked
  as `phase: decode` and offsets the causal query index by
  `seqlen_k - seqlen_q`, with PyTorch reference
  `softmax(masked_fill((q @ k.T) * scale, key_index > query_index + (seqlen_k - seqlen_q), -inf)) @ v`
- append-shaped causal gate: multiple queries over a longer KV prefix are
  marked as `phase: append` and use the same `seqlen_k - seqlen_q` causal
  query offset. The bounded `4x32x64` gate records
  `causal_query_offset: 28` and uses the same shifted PyTorch reference
  formula as the decode-shaped gate.

The harness at `examples/cuda/gluon_flashattention_fwd.py` builds the generated
artifact, loads it, and runs FP32 correctness cases against PyTorch when CUDA
and a Gluon build with `dot_fma` support are available. The default path still
runs one `32x32x32` single-tile case. The single-case path also accepts
`--tile-shape MxNxD` for bounded repros such as `32x32x64`, `1x32x64`, and
`4x32x64`, with `--causal` for bounded causal gates. The single-case path
also accepts `--kv-cache-boundary paged` and `--kv-cache-boundary ragged` to
emit explicit unsupported-boundary JSON before CUDA availability checks. A
single-case path also accepts `--sequence-boundary varlen` to emit explicit
varlen unsupported-boundary JSON before CUDA availability checks. A
single-case path also accepts `--attention-variant mla` to emit explicit MLA
attention unsupported-boundary JSON before CUDA availability checks. A
single-case path also accepts `--attention-variant cascade` to emit explicit
Cascade Attention unsupported-boundary JSON before CUDA availability checks. A
single-case path also accepts `--attention-variant sparse` to emit explicit
Sparse Attention unsupported-boundary JSON before CUDA availability checks. A
single-case path also accepts `--attention-variant pod` to emit explicit
POD-Attention unsupported-boundary JSON before CUDA availability checks. A
causal same-length `32x32x64` single-case run is reported as
`phase: prefill` to distinguish same-length multi-query prefill-shaped
evidence from decode/append evidence. A causal `1x32x64` single-case run is
reported as `phase: decode` to distinguish single-query decode-shaped
evidence from full decode coverage. A causal `4x32x64` single-case run is
reported as `phase: append` to distinguish small multi-query append-shaped
evidence from full append KV-cache coverage.
The `--sweep` path emits aggregate structured JSON with `schema_version: 1`,
aggregate status and case counts, and per-case shape, provenance, artifact
metadata, status, phase, causal flag, and max error when run. Artifact paths
are repo-relative, absolute `--output-dir` values are rejected, and exception
text is sanitized so private absolute paths are not recorded.

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

## Prefill-Shaped Same-Length Gate

On 2026-06-22, the same-length multi-query prefill-shaped causal gate
generated and launched on the same H200 class machine and passed correctness
against the lower-triangular masked PyTorch reference. The remote run used
tree sync into `<remote-pto-cu>` through the generic CUDA runner, and the
preserved remote Gluon Python environment because the remote default Python
lacked Torch and Triton/Gluon. The earlier
`--output-dir tmp/gluon-flashattention-causal-boundary-h200` label refers to
the same bounded shape before the harness classified it as `phase: prefill`.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-prefill-boundary-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal --require-cuda'
```

Distilled passed result:

```text
schema_version: 1
status: passed
phase: prefill
causal: true
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
max_abs_error: 4.172325134277344e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: f9f0ff900d33023c462579063be9aa8560a82c63d43aae2bd851369cfcfb58a4
tile_shape: 32x32x64
artifact paths are repo-relative
machine class: H200
private absolute paths are not recorded
```

This result is bounded prefill-shaped evidence for one generated same-length
multi-query FP32 causal FlashAttention source. It is not full prefill
coverage, not FlashInfer integration evidence, not vLLM/simpler-nv
integration evidence, not production serving readiness, not performance,
throughput, or latency evidence, not paged/ragged KV-cache coverage, not full
decode, full append, or append KV-cache coverage, and not DeepSeek semantic
correctness.

## Decode-Shaped Single-Query Gate

On 2026-06-22, the decode-shaped single-query causal gate generated and
launched on the same H200 class machine and passed correctness against the
offset masked PyTorch reference. The remote run used tree sync into
`<remote-pto-cu>` through the generic CUDA runner, and the preserved remote
Gluon Python environment because the remote default Python lacked Torch and
Triton/Gluon.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-decode-boundary-h200 \
      --arch compute_90 --tile-shape 1x32x64 --causal --require-cuda'
```

Distilled passed result:

```text
schema_version: 1
status: passed
phase: decode
causal: true
shape: seqlen_q=1, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index + (seqlen_k - seqlen_q), -inf)) @ v
tolerance: atol=0.001, rtol=0.01
max_abs_error: 2.384185791015625e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 43a91756bc237b2c8240a7a61dcec448ceb7d335f915cd62630d7fa8c8a1e4c2
tile_shape: 1x32x64
artifact paths are repo-relative
machine class: H200
private absolute paths are not recorded
```

This result is bounded decode-shaped evidence for one generated single-query
FP32 causal FlashAttention source. It is not FlashInfer integration evidence,
not FlashInfer parity, not vLLM/simpler-nv integration evidence, not
production serving readiness, not performance, throughput, or latency
evidence, not paged/ragged KV-cache coverage, not full decode, prefill, or
append coverage, not MLA/cascade/sparse/POD attention coverage, and not
DeepSeek semantic correctness.

## Append-Shaped Multi-Query Gate

On 2026-06-22, the small multi-query append-shaped causal gate generated and
launched on the same H200 class machine and passed correctness against the
offset masked PyTorch reference. The remote run used tree sync into
`<remote-pto-cu>` through the generic CUDA runner, and the preserved remote
Gluon Python environment because the remote default Python lacked Torch and
Triton/Gluon.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-append-boundary-h200 \
      --arch compute_90 --tile-shape 4x32x64 --causal --require-cuda'
```

Distilled passed result:

```text
schema_version: 1
status: passed
phase: append
causal: true
shape: seqlen_q=4, seqlen_k=32, head_dim=64
causal_query_offset: 28
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index + (seqlen_k - seqlen_q), -inf)) @ v
tolerance: atol=0.001, rtol=0.01
max_abs_error: 1.7881393432617188e-07
artifact: arch=compute_90, compiler_role=pto-isa-replacement
source_sha256: 953f398978a0b8241e56273ca138912f02a6363e4ef27529037d4a5ec5e5632a
tile_shape: 4x32x64
artifact paths are repo-relative
machine class: H200
private absolute paths are not recorded
```

This result is bounded append-shaped evidence for one generated small
multi-query FP32 causal FlashAttention source. It is not FlashInfer
integration evidence, not FlashInfer parity, not vLLM/simpler-nv integration
evidence, not production serving readiness, not performance, throughput, or
latency evidence, not paged/ragged KV-cache coverage, not full decode,
prefill, full append, or append KV-cache coverage, not
MLA/cascade/sparse/POD attention coverage, and not DeepSeek semantic
correctness.

## Paged/Ragged KV-Cache Unsupported Boundary

On 2026-06-22, the paged and ragged KV-cache boundary probes reported
structured unsupported JSON on the same H200 class machine. The remote runs
used tree sync into `<remote-pto-cu>` through the generic CUDA runner and the
preserved remote Gluon Python environment. These commands do not generate
correctness artifacts and do not require CUDA availability before reporting
the unsupported boundary.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-kvcache-paged-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --kv-cache-boundary paged --require-cuda'
```

Distilled paged result:

```text
schema_version: 1
status: skipped
phase: prefill
causal: true
kv_cache_boundary: paged
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: paged_kv_cache
unsupported_boundary.boundary: paged
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention paged KV-cache boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-kvcache-ragged-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --kv-cache-boundary ragged --require-cuda'
```

Distilled ragged result:

```text
schema_version: 1
status: skipped
phase: prefill
causal: true
kv_cache_boundary: ragged
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: ragged_kv_cache
unsupported_boundary.boundary: ragged
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention ragged KV-cache boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

These results are unsupported-boundary evidence only. They are not
paged/ragged KV-cache correctness, not full prefill/decode/append coverage,
not FlashInfer integration evidence, not vLLM/simpler-nv integration
evidence, not serving readiness, and not performance, throughput, or latency
evidence.

## Varlen Unsupported Boundary

On 2026-06-22, the varlen sequence boundary probe reported structured
unsupported JSON on the same H200 class machine. The remote run used tree
sync into `<remote-pto-cu>` through the generic CUDA runner and the preserved
remote Gluon Python environment. This command does not generate correctness
artifacts and does not require CUDA availability before reporting the
unsupported boundary.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-varlen-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --sequence-boundary varlen --require-cuda'
```

Distilled result:

```text
exit_code: 2 under --require-cuda structured skip
schema_version: 1
status: skipped
phase: prefill
causal: true
sequence_boundary: varlen
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: varlen_attention
unsupported_boundary.boundary: varlen
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention varlen attention boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

This result is unsupported-boundary evidence only. It is not varlen attention
correctness, not paged/ragged KV-cache correctness, not full
prefill/decode/append coverage, not FlashInfer integration evidence, not
vLLM/simpler-nv integration evidence, not serving readiness, and not
performance, throughput, or latency evidence.

## MLA Unsupported Boundary

On 2026-06-22, the MLA attention boundary probe reported structured
unsupported JSON on the same H200 class machine. The remote run used tree
sync into `<remote-pto-cu>` through the generic CUDA runner and the preserved
remote Gluon Python environment. This command does not generate correctness
artifacts and does not require CUDA availability before reporting the
unsupported boundary.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-mla-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --attention-variant mla --require-cuda'
```

Distilled result:

```text
exit_code: 2 under --require-cuda structured skip
schema_version: 1
status: skipped
phase: prefill
causal: true
attention_variant: mla
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: mla_attention
unsupported_boundary.boundary: mla
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention MLA attention boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

This result is unsupported-boundary evidence only. It is not MLA attention
correctness, not varlen attention correctness, not paged/ragged KV-cache
correctness, not full prefill/decode/append coverage, not FlashInfer
integration evidence, not vLLM/simpler-nv integration evidence, not serving
readiness, and not performance, throughput, or latency evidence.

## Cascade Attention Unsupported Boundary

On 2026-06-22, the Cascade Attention boundary probe reported structured
unsupported JSON on the same H200 class machine. The remote run used tree
sync into `<remote-pto-cu>` through the generic CUDA runner and the preserved
remote Gluon Python environment. This command does not generate correctness
artifacts and does not require CUDA availability before reporting the
unsupported boundary.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-cascade-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --attention-variant cascade --require-cuda'
```

Distilled result:

```text
exit_code: 2 under --require-cuda structured skip
schema_version: 1
status: skipped
phase: prefill
causal: true
attention_variant: cascade
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: cascade_attention
unsupported_boundary.boundary: cascade
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention Cascade Attention boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

This result is unsupported-boundary evidence only. It is not Cascade
Attention correctness, not MLA attention correctness, not varlen attention
correctness, not paged/ragged KV-cache correctness, not full
prefill/decode/append coverage, not FlashInfer integration evidence, not
vLLM/simpler-nv integration evidence, not serving readiness, and not
performance, throughput, or latency evidence.

## Sparse Attention Unsupported Boundary

On 2026-06-22, the Sparse Attention boundary probe reported structured
unsupported JSON on the same H200 class machine. The remote run used tree
sync into `<remote-pto-cu>` through the generic CUDA runner and the preserved
remote Gluon Python environment. This command does not generate correctness
artifacts and does not require CUDA availability before reporting the
unsupported boundary.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-sparse-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --attention-variant sparse --require-cuda'
```

Distilled result:

```text
exit_code: 2 under --require-cuda structured skip
schema_version: 1
status: skipped
phase: prefill
causal: true
attention_variant: sparse
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: sparse_attention
unsupported_boundary.boundary: sparse
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention Sparse Attention boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

This result is unsupported-boundary evidence only. It is not Sparse
Attention correctness, not Cascade Attention correctness, not MLA attention
correctness, not varlen attention correctness, not paged/ragged KV-cache
correctness, not full prefill/decode/append coverage, not FlashInfer
integration evidence, not vLLM/simpler-nv integration evidence, not serving
readiness, and not performance, throughput, or latency evidence.

## POD-Attention Unsupported Boundary

On 2026-06-22, the POD-Attention boundary probe reported structured
unsupported JSON on the same H200 class machine. The remote run used tree
sync into `<remote-pto-cu>` through the generic CUDA runner and the preserved
remote Gluon Python environment. This command does not generate correctness
artifacts and does not require CUDA availability before reporting the
unsupported boundary.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_flashattention_fwd.py \
      --output-dir tmp/gluon-flashattention-pod-unsupported-h200 \
      --arch compute_90 --tile-shape 32x32x64 --causal \
      --attention-variant pod --require-cuda'
```

Distilled result:

```text
exit_code: 2 under --require-cuda structured skip
schema_version: 1
status: skipped
phase: prefill
causal: true
attention_variant: pod
shape: seqlen_q=32, seqlen_k=32, head_dim=64
reference: softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v
tolerance: atol=0.001, rtol=0.01
unsupported_boundary.kind: pod_attention
unsupported_boundary.boundary: pod
unsupported_boundary.operator: flashattention_fwd_f32
reason: Gluon FlashAttention POD-Attention boundary is unsupported; this is unsupported-boundary evidence only
machine class: H200
private absolute paths are not recorded
```

This result is unsupported-boundary evidence only. It is not POD-Attention
correctness, not Sparse Attention correctness, not Cascade Attention
correctness, not MLA attention correctness, not varlen attention correctness,
not paged/ragged KV-cache correctness, not full prefill/decode/append
coverage, not FlashInfer integration evidence, not vLLM/simpler-nv
integration evidence, not serving readiness, and not performance, throughput,
or latency evidence.

## Boundary

This milestone covers a small single-tile FlashAttention correctness sweep
and bounded same-length multi-query prefill-shaped, single-query
decode-shaped, small multi-query append-shaped gates, and explicit
paged/ragged KV-cache, varlen, MLA, Cascade Attention, Sparse Attention, and
POD-Attention unsupported-boundary reporting only. It is not benchmark
evidence and does not cover block streaming, varlen correctness, MLA
attention correctness, Cascade Attention correctness, Sparse Attention
correctness, POD-Attention correctness, page tables, persistent scheduling,
MoE, distributed communication, serving, or DeepSeek integration.

Non-claims:

- not production serving readiness
- not FlashInfer integration evidence
- not DeepSeek semantic correctness
- not performance, throughput, or latency evidence
- not paged/ragged KV-cache correctness
- not varlen attention correctness
- not MLA attention correctness
- not Cascade Attention correctness
- not Sparse Attention correctness
- not POD-Attention correctness
- not full prefill, full decode, full append, or append coverage
- not full append or append KV-cache coverage
- not multi-tile attention coverage
- not fused attention integration
- not KV-cache integration
- not vLLM/simpler-nv integration evidence
