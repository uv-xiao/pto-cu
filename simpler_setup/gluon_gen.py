# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Triton/Gluon source generation for the CUDA backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

from .environment import PROJECT_ROOT

_DEFAULT_CACHE_ROOT = Path("build") / "cache" / "cuda" / "onboard" / "gluon_gen"
_SUPPORTED_KERNELS = {
    "gemm_f32",
    "gemm_tensor_core_f16_f32",
    "gemm_tensor_core_tiled_f16_f32",
    "gemm_tensor_core_tiled_bf16_f32",
    "gemm_tensor_core_tiled_fp8e4nv_f32",
    "flashattention_fwd_f32",
    "moe_expert_affine_f32",
    "rmsnorm_f32",
    "gemma_fused_rmsnorm_f32",
    "layernorm_f32",
    "rope_f32",
    "silu_f32",
    "gelu_f32",
    "gated_silu_f32",
    "topk_sampling_f32",
    "topp_sampling_f32",
    "minp_sampling_f32",
    "speculative_accept_f32",
}


@dataclass(frozen=True)
class GluonKernelArtifact:
    kernel_name: str
    compiler_role: str
    arch: str
    source_path: Path
    manifest_path: Path
    source_sha256: str
    tile_shape: tuple[int, int, int]


@dataclass(frozen=True)
class GluonPersistentTaskBodyArtifact:
    kernel_name: str
    task_name: str
    body: str
    source_kind: str
    source_sha256: str


def default_gluon_cache_root() -> Path:
    return PROJECT_ROOT / _DEFAULT_CACHE_ROOT


def generate_gluon_kernel(
    kernel_name: str,
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    tile_shape: tuple[int, int, int] = (64, 128, 32),
) -> GluonKernelArtifact:
    if kernel_name not in _SUPPORTED_KERNELS:
        supported = ", ".join(sorted(_SUPPORTED_KERNELS))
        raise ValueError(f"unsupported Gluon kernel {kernel_name!r}; expected one of: {supported}")

    resolved_output_dir = default_gluon_cache_root() if output_dir is None else Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    source = _render_source(kernel_name, tile_shape)
    source_digest = sha256(source.encode("utf-8")).hexdigest()
    source_path = resolved_output_dir / f"{kernel_name}.gluon.py"
    manifest_path = resolved_output_dir / f"{kernel_name}.gluon.json"
    source_path.write_text(source, encoding="utf-8")

    manifest = {
        "arch": arch,
        "compiler_role": "pto-isa-replacement",
        "kernel_name": kernel_name,
        "source_kind": "triton-gluon-python",
        "source_path": source_path.name,
        "source_sha256": source_digest,
        "tile_shape": list(tile_shape),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return GluonKernelArtifact(
        kernel_name=kernel_name,
        compiler_role="pto-isa-replacement",
        arch=arch,
        source_path=source_path,
        manifest_path=manifest_path,
        source_sha256=source_digest,
        tile_shape=tile_shape,
    )


def generate_gluon_persistent_task_body(kernel_name: str) -> GluonPersistentTaskBodyArtifact:
    if kernel_name != "moe_expert_affine_f32":
        raise ValueError(
            "persistent task-body bridge is only available for "
            "'moe_expert_affine_f32'"
        )

    body = _render_moe_expert_affine_persistent_body()
    return GluonPersistentTaskBodyArtifact(
        kernel_name=kernel_name,
        task_name="gluon_moe_expert_affine_f32",
        body=body,
        source_kind="gluon-persistent-task-body-bridge",
        source_sha256=sha256(body.encode("utf-8")).hexdigest(),
    )


def _render_source(kernel_name: str, tile_shape: tuple[int, int, int]) -> str:
    if kernel_name == "gemm_f32":
        return _render_gemm_f32_source()
    if kernel_name == "gemm_tensor_core_f16_f32":
        return _render_tensor_core_gemm_source(tile_shape)
    if kernel_name == "gemm_tensor_core_tiled_f16_f32":
        return _render_tiled_tensor_core_gemm_source(
            tile_shape,
            kernel_name="gemm_tensor_core_tiled_f16_f32",
            input_gl_dtype="float16",
        )
    if kernel_name == "gemm_tensor_core_tiled_bf16_f32":
        return _render_tiled_tensor_core_gemm_source(
            tile_shape,
            kernel_name="gemm_tensor_core_tiled_bf16_f32",
            input_gl_dtype="bfloat16",
        )
    if kernel_name == "gemm_tensor_core_tiled_fp8e4nv_f32":
        return _render_tiled_tensor_core_gemm_source(
            tile_shape,
            kernel_name="gemm_tensor_core_tiled_fp8e4nv_f32",
            input_gl_dtype="float8e4nv",
        )
    if kernel_name == "flashattention_fwd_f32":
        return _render_flashattention_source(tile_shape)
    if kernel_name == "moe_expert_affine_f32":
        return _render_moe_expert_affine_source()
    if kernel_name == "rmsnorm_f32":
        return _render_rmsnorm_f32_source()
    if kernel_name == "gemma_fused_rmsnorm_f32":
        return _render_gemma_fused_rmsnorm_f32_source()
    if kernel_name == "layernorm_f32":
        return _render_layernorm_f32_source()
    if kernel_name == "rope_f32":
        return _render_rope_f32_source()
    if kernel_name == "silu_f32":
        return _render_silu_f32_source()
    if kernel_name == "gelu_f32":
        return _render_gelu_f32_source()
    if kernel_name == "gated_silu_f32":
        return _render_gated_silu_f32_source()
    if kernel_name == "topk_sampling_f32":
        return _render_topk_sampling_f32_source()
    if kernel_name == "topp_sampling_f32":
        return _render_topp_sampling_f32_source()
    if kernel_name == "minp_sampling_f32":
        return _render_minp_sampling_f32_source()
    if kernel_name == "speculative_accept_f32":
        return _render_speculative_accept_f32_source()
    raise AssertionError(f"unhandled Gluon kernel: {kernel_name}")


def _render_gemm_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def gemm_f32_kernel(a_ptr, b_ptr, c_ptr, m: gl.constexpr, n: gl.constexpr, k: gl.constexpr):
            row = gl.program_id(0)
            col = gl.program_id(1)
            acc = 0.0
            for kk in range(0, k):
                a = gl.load(a_ptr + row * k + kk)
                b = gl.load(b_ptr + kk * n + col)
                acc += a * b
            gl.store(c_ptr + row * n + col, acc)
        """
    ).lstrip()


def _render_moe_expert_affine_persistent_body() -> str:
    return dedent(
        """
        const PtoCudaPersistentDagTask *task = ctx->task;
        unsigned long long i = ctx->i;
        task->out[i] = task->scalar0 * task->a[i] + task->scalar1 * task->b[i];
        """
    ).strip()


def _render_tensor_core_gemm_source(tile_shape: tuple[int, int, int]) -> str:
    block_m, block_n, block_k = tile_shape
    return dedent(
        f"""
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
        from triton.experimental.gluon.language.nvidia.hopper import (
            fence_async_shared,
            mbarrier,
            tma,
            warpgroup_mma,
            warpgroup_mma_wait,
        )


        @gluon.jit
        def gemm_tensor_core_f16_f32_kernel(a_desc, b_desc, c_desc, d_desc, instr_shape_n: gl.constexpr, num_warps: gl.constexpr):
            bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(bar, count=1)
            a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
            b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
            c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
            tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
            tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
            mbarrier.wait(bar, phase=0)
            mbarrier.invalidate(bar)

            m: gl.constexpr = 16
            k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
            n: gl.constexpr = instr_shape_n
            warps_per_cta: gl.constexpr = [num_warps, 1]
            c_layout: gl.constexpr = gl.NVMMADistributedLayout(
                version=[3, 0],
                warps_per_cta=warps_per_cta,
                instr_shape=[m, n, k],
            )

            c = c_smem.load(c_layout)
            d = warpgroup_mma(a_smem, b_smem, c, is_async=True, use_acc=True)
            d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

            d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
            d_smem.store(d)
            fence_async_shared()
            tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
            tma.store_wait(pendings=0)


        def run_gemm_tensor_core_f16_f32(a, b, c, d, instr_shape_n=16, num_warps=4):
            expected_shapes = (({block_m}, {block_k}), ({block_k}, {block_n}), ({block_m}, {block_n}), ({block_m}, {block_n}))
            actual_shapes = (tuple(a.shape), tuple(b.shape), tuple(c.shape), tuple(d.shape))
            if actual_shapes != expected_shapes:
                raise ValueError(f"expected tensor shapes {{expected_shapes}}, got {{actual_shapes}}")

            a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
            b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
            cd_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
            a_desc = TensorDescriptor.from_tensor(a, a.shape, a_layout)
            b_desc = TensorDescriptor.from_tensor(b, b.shape, b_layout)
            c_desc = TensorDescriptor.from_tensor(c, c.shape, cd_layout)
            d_desc = TensorDescriptor.from_tensor(d, d.shape, cd_layout)
            gemm_tensor_core_f16_f32_kernel[(1,)](
                a_desc,
                b_desc,
                c_desc,
                d_desc,
                instr_shape_n,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_tiled_tensor_core_gemm_source(
    tile_shape: tuple[int, int, int],
    *,
    kernel_name: str,
    input_gl_dtype: str,
) -> str:
    block_m, block_n, block_k = tile_shape
    return dedent(
        f"""
        import triton

        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
        from triton.experimental.gluon.language.nvidia.hopper import (
            fence_async_shared,
            mbarrier,
            tma,
            warpgroup_mma,
            warpgroup_mma_wait,
        )


        @gluon.jit
        def {kernel_name}_kernel(a_desc, b_desc, c_desc, d_desc, instr_shape_n: gl.constexpr, num_warps: gl.constexpr):
            pid_m = gl.program_id(axis=0)
            pid_n = gl.program_id(axis=1)
            off_m = pid_m * {block_m}
            off_n = pid_n * {block_n}

            bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(bar, count=1)
            a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
            b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
            c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

            phase = 0
            mbarrier.expect(bar, c_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(c_desc, [off_m, off_n], bar, c_smem)
            mbarrier.wait(bar, phase=phase)
            phase ^= 1

            m: gl.constexpr = 16
            k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
            n: gl.constexpr = instr_shape_n
            warps_per_cta: gl.constexpr = [num_warps, 1]
            c_layout: gl.constexpr = gl.NVMMADistributedLayout(
                version=[3, 0],
                warps_per_cta=warps_per_cta,
                instr_shape=[m, n, k],
            )

            acc = c_smem.load(c_layout)
            K = a_desc.shape[1]
            for k_offset in range(0, K, {block_k}):
                mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(a_desc, [off_m, k_offset], bar, a_smem)
                tma.async_copy_global_to_shared(b_desc, [k_offset, off_n], bar, b_smem)
                mbarrier.wait(bar, phase=phase)
                phase ^= 1

                acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True, use_acc=True)
                acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

            mbarrier.invalidate(bar)

            d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
            d_smem.store(acc)
            fence_async_shared()
            tma.async_copy_shared_to_global(d_desc, [off_m, off_n], d_smem)
            tma.store_wait(pendings=0)


        def run_{kernel_name}(a, b, c, d, instr_shape_n=16, num_warps=4):
            expected_rank = 2
            for name, tensor in [("a", a), ("b", b), ("c", c), ("d", d)]:
                if len(tensor.shape) != expected_rank:
                    raise ValueError(f"expected {{name}} to be rank-2, got shape {{tuple(tensor.shape)}}")
            if a.shape[1] % {block_k} != 0 or b.shape[0] != a.shape[1]:
                raise ValueError(f"expected shared K dimension divisible by {block_k}, got a.shape={{tuple(a.shape)}}, b.shape={{tuple(b.shape)}}")
            if a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1] or d.shape != c.shape:
                raise ValueError(f"expected c/d shape to match a@b, got a={{tuple(a.shape)}}, b={{tuple(b.shape)}}, c={{tuple(c.shape)}}, d={{tuple(d.shape)}}")
            if c.shape[0] % {block_m} != 0 or c.shape[1] % {block_n} != 0:
                raise ValueError(f"expected output shape divisible by tile ({block_m}, {block_n}), got {{tuple(c.shape)}}")

            a_layout = gl.NVMMASharedLayout.get_default_for([{block_m}, {block_k}], gl.{input_gl_dtype})
            b_layout = gl.NVMMASharedLayout.get_default_for([{block_k}, {block_n}], gl.{input_gl_dtype})
            cd_layout = gl.NVMMASharedLayout.get_default_for([{block_m}, {block_n}], gl.float32)
            a_desc = TensorDescriptor.from_tensor(a, [{block_m}, {block_k}], a_layout)
            b_desc = TensorDescriptor.from_tensor(b, [{block_k}, {block_n}], b_layout)
            c_desc = TensorDescriptor.from_tensor(c, [{block_m}, {block_n}], cd_layout)
            d_desc = TensorDescriptor.from_tensor(d, [{block_m}, {block_n}], cd_layout)
            grid = (triton.cdiv(c.shape[0], {block_m}), triton.cdiv(c.shape[1], {block_n}))
            {kernel_name}_kernel[grid](
                a_desc,
                b_desc,
                c_desc,
                d_desc,
                instr_shape_n,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_flashattention_source(tile_shape: tuple[int, int, int]) -> str:
    block_m, block_n, block_d = tile_shape
    if block_n == block_d:
        score_source = f"""
            offs_k_col = gl.arange(0, {block_d}, layout=gl.SliceLayout(0, layout))[None, :]
            offs_k_row = gl.arange(0, {block_d}, layout=gl.SliceLayout(1, layout))[:, None]

            q = gl.load(q_ptr + offs_m * head_dim + offs_k_col)
            k_t = gl.load(k_ptr + offs_k_row * head_dim + offs_n)
            q = gl.convert_layout(q, lhs_layout)
            k_t = gl.convert_layout(k_t, rhs_layout)
            score_acc = gl.full(({block_m}, {block_n}), 0.0, gl.float32, layout=layout)
            scores = gl.dot_fma(q, k_t, score_acc) * scale
"""
    else:
        score_source = f"""
            score_acc = gl.full(({block_m}, {block_n}), 0.0, gl.float32, layout=layout)
            for k_offset in gl.static_range(0, {block_d}):
                q_value = gl.load(q_ptr + offs_m * head_dim + k_offset)
                k_value = gl.load(k_ptr + offs_n * head_dim + k_offset)
                score_acc += q_value * k_value
            scores = score_acc * scale
"""
    return dedent(
        f"""
        import math

        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def flashattention_fwd_f32_kernel(q_ptr, k_ptr, v_ptr, out_ptr, seqlen_q: gl.constexpr, seqlen_k: gl.constexpr, head_dim: gl.constexpr, scale: gl.constexpr, causal: gl.constexpr):
            layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
            lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
            rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

            offs_m = gl.arange(0, {block_m}, layout=gl.SliceLayout(1, layout))[:, None]
            offs_n = gl.arange(0, {block_n}, layout=gl.SliceLayout(0, layout))[None, :]
{score_source}
            if causal:
                causal_mask = offs_n <= offs_m
                scores = gl.where(causal_mask, scores, -float("inf"))

            row_max = gl.max(scores, axis=1)
            probs = gl.exp(scores - row_max[:, None])
            row_sum = gl.sum(probs, axis=1)
            probs = probs / row_sum[:, None]

            offs_d = gl.arange(0, {block_d}, layout=gl.SliceLayout(0, layout))[None, :]
            offs_vn = gl.arange(0, {block_n}, layout=gl.SliceLayout(1, layout))[:, None]
            probs = gl.convert_layout(probs, lhs_layout)
            v = gl.load(v_ptr + offs_d * seqlen_k + offs_vn)
            v = gl.convert_layout(v, rhs_layout)
            out_acc = gl.full(({block_m}, {block_d}), 0.0, gl.float32, layout=layout)
            out = gl.dot_fma(probs, v, out_acc)
            gl.store(out_ptr + offs_m * head_dim + offs_d, out)


        def run_flashattention_fwd_f32(q, k, v, out, scale=None, causal=False, num_warps=4):
            expected_shapes = (({block_m}, {block_d}), ({block_n}, {block_d}), ({block_n}, {block_d}), ({block_m}, {block_d}))
            actual_shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape), tuple(out.shape))
            if actual_shapes != expected_shapes:
                raise ValueError(f"expected tensor shapes {{expected_shapes}}, got {{actual_shapes}}")

            resolved_scale = 1.0 / math.sqrt({block_d}) if scale is None else scale
            flashattention_fwd_f32_kernel[(1,)](
                q,
                k,
                v,
                out,
                {block_m},
                {block_n},
                {block_d},
                resolved_scale,
                causal,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_moe_expert_affine_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def moe_expert_affine_f32_kernel(a_ptr, b_ptr, out_ptr, n: gl.constexpr, scale_a: gl.constexpr, scale_b: gl.constexpr):
            idx = gl.program_id(0)
            a = gl.load(a_ptr + idx)
            b = gl.load(b_ptr + idx)
            gl.store(out_ptr + idx, scale_a * a + scale_b * b)


        def run_moe_expert_affine_f32(a, b, out, scale_a=1.25, scale_b=0.5, num_warps=4):
            expected_rank = 1
            for name, tensor in [("a", a), ("b", b), ("out", out)]:
                if len(tensor.shape) != expected_rank:
                    raise ValueError(f"expected {name} to be rank-1, got shape {tuple(tensor.shape)}")
            if a.numel() != b.numel() or out.numel() != a.numel():
                raise ValueError(f"expected equal vector lengths, got a={a.numel()}, b={b.numel()}, out={out.numel()}")

            moe_expert_affine_f32_kernel[(a.numel(),)](
                a,
                b,
                out,
                a.numel(),
                scale_a,
                scale_b,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_rmsnorm_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def rmsnorm_f32_kernel(x_ptr, weight_ptr, out_ptr, rows: gl.constexpr, hidden: gl.constexpr, eps: gl.constexpr):
            row = gl.program_id(0)
            sum_sq = 0.0
            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                sum_sq += x * x
            mean_sq = sum_sq / hidden
            inv_rms = gl.rsqrt(mean_sq + eps)
            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                weight = gl.load(weight_ptr + col)
                gl.store(out_ptr + row * hidden + col, x * inv_rms * weight)


        def run_rmsnorm_f32(x, weight, out, eps=1.0e-5, num_warps=4):
            expected_rank = 2
            if len(x.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected x/out to be rank-2, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if len(weight.shape) != 1:
                raise ValueError(f"expected weight to be rank-1, got shape {tuple(weight.shape)}")
            if out.shape != x.shape or weight.numel() != x.shape[1]:
                raise ValueError(f"expected out shape to match x and weight length to match hidden, got x={tuple(x.shape)}, weight={tuple(weight.shape)}, out={tuple(out.shape)}")

            rmsnorm_f32_kernel[(x.shape[0],)](
                x,
                weight,
                out,
                x.shape[0],
                x.shape[1],
                eps,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_gemma_fused_rmsnorm_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def gemma_fused_rmsnorm_f32_kernel(x_ptr, weight_ptr, out_ptr, rows: gl.constexpr, hidden: gl.constexpr, eps: gl.constexpr):
            row = gl.program_id(0)
            sum_sq = 0.0
            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                sum_sq += x * x
            mean_sq = sum_sq / hidden
            inv_rms = gl.rsqrt(mean_sq + eps)
            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                weight = gl.load(weight_ptr + col)
                gl.store(out_ptr + row * hidden + col, x * inv_rms * (1.0 + weight))


        def run_gemma_fused_rmsnorm_f32(x, weight, out, eps=1.0e-5, num_warps=4):
            expected_rank = 2
            if len(x.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected x/out to be rank-2, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if len(weight.shape) != 1:
                raise ValueError(f"expected weight to be rank-1, got shape {tuple(weight.shape)}")
            if out.shape != x.shape or weight.numel() != x.shape[1]:
                raise ValueError(f"expected out shape to match x and weight length to match hidden, got x={tuple(x.shape)}, weight={tuple(weight.shape)}, out={tuple(out.shape)}")

            gemma_fused_rmsnorm_f32_kernel[(x.shape[0],)](
                x,
                weight,
                out,
                x.shape[0],
                x.shape[1],
                eps,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_layernorm_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def layernorm_f32_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, rows: gl.constexpr, hidden: gl.constexpr, eps: gl.constexpr):
            row = gl.program_id(0)
            mean = 0.0
            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                mean += x
            mean = mean / hidden

            var = 0.0
            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                centered = x - mean
                var += centered * centered
            var = var / hidden
            inv_std = gl.rsqrt(var + eps)

            for col in range(0, hidden):
                x = gl.load(x_ptr + row * hidden + col)
                weight = gl.load(weight_ptr + col)
                bias = gl.load(bias_ptr + col)
                gl.store(out_ptr + row * hidden + col, (x - mean) * inv_std * weight + bias)


        def run_layernorm_f32(x, weight, bias, out, eps=1.0e-5, num_warps=4):
            expected_rank = 2
            if len(x.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected x/out to be rank-2, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if len(weight.shape) != 1 or len(bias.shape) != 1:
                raise ValueError(f"expected weight/bias to be rank-1, got weight={tuple(weight.shape)}, bias={tuple(bias.shape)}")
            if out.shape != x.shape or weight.numel() != x.shape[1] or bias.numel() != x.shape[1]:
                raise ValueError(f"expected out shape to match x and weight/bias lengths to match hidden, got x={tuple(x.shape)}, weight={tuple(weight.shape)}, bias={tuple(bias.shape)}, out={tuple(out.shape)}")

            layernorm_f32_kernel[(x.shape[0],)](
                x,
                weight,
                bias,
                out,
                x.shape[0],
                x.shape[1],
                eps,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_rope_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def rope_f32_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, batch: gl.constexpr, seq: gl.constexpr, head_dim: gl.constexpr):
            pid = gl.program_id(0)
            half_dim: gl.constexpr = head_dim // 2
            pair = pid % half_dim
            token = (pid // half_dim) % seq
            batch_idx = pid // (seq * half_dim)
            base = batch_idx * seq * head_dim + token * head_dim + pair * 2
            trig_base = token * half_dim + pair
            x_even = gl.load(x_ptr + base)
            x_odd = gl.load(x_ptr + base + 1)
            cos = gl.load(cos_ptr + trig_base)
            sin = gl.load(sin_ptr + trig_base)
            out_even = x_even * cos - x_odd * sin
            out_odd = x_even * sin + x_odd * cos
            gl.store(out_ptr + base, out_even)
            gl.store(out_ptr + base + 1, out_odd)


        def run_rope_f32(x, cos, sin, out, num_warps=4):
            expected_rank = 3
            if len(x.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected x/out to be rank-3, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if out.shape != x.shape:
                raise ValueError(f"expected out shape to match x, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if x.shape[2] % 2 != 0:
                raise ValueError(f"expected even head_dim, got {x.shape[2]}")
            expected_trig_shape = (x.shape[1], x.shape[2] // 2)
            if tuple(cos.shape) != expected_trig_shape or tuple(sin.shape) != expected_trig_shape:
                raise ValueError(f"expected cos/sin shape {expected_trig_shape}, got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")

            total_pairs = x.shape[0] * x.shape[1] * (x.shape[2] // 2)
            rope_f32_kernel[(total_pairs,)](
                x,
                cos,
                sin,
                out,
                x.shape[0],
                x.shape[1],
                x.shape[2],
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_silu_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def silu_f32_kernel(x_ptr, out_ptr, n: gl.constexpr):
            idx = gl.program_id(0)
            x = gl.load(x_ptr + idx)
            gl.store(out_ptr + idx, x / (1.0 + gl.exp(-x)))


        def run_silu_f32(x, out, num_warps=4):
            expected_rank = 1
            if len(x.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected x/out to be rank-1, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if out.numel() != x.numel():
                raise ValueError(f"expected out length to match x, got x={x.numel()}, out={out.numel()}")

            silu_f32_kernel[(x.numel(),)](
                x,
                out,
                x.numel(),
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_gelu_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def gelu_f32_kernel(x_ptr, out_ptr, n: gl.constexpr):
            idx = gl.program_id(0)
            x = gl.load(x_ptr + idx)
            gl.store(out_ptr + idx, 0.5 * x * (1.0 + gl.erf(x * 0.7071067811865476)))


        def run_gelu_f32(x, out, num_warps=4):
            expected_rank = 1
            if len(x.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected x/out to be rank-1, got x={tuple(x.shape)}, out={tuple(out.shape)}")
            if out.numel() != x.numel():
                raise ValueError(f"expected out length to match x, got x={x.numel()}, out={out.numel()}")

            gelu_f32_kernel[(x.numel(),)](
                x,
                out,
                x.numel(),
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_gated_silu_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def gated_silu_f32_kernel(gate_ptr, value_ptr, out_ptr, n: gl.constexpr):
            idx = gl.program_id(0)
            gate = gl.load(gate_ptr + idx)
            value = gl.load(value_ptr + idx)
            gl.store(out_ptr + idx, value * gate / (1.0 + gl.exp(-gate)))


        def run_gated_silu_f32(gate, value, out, num_warps=4):
            expected_rank = 1
            if len(gate.shape) != expected_rank or len(value.shape) != expected_rank or len(out.shape) != expected_rank:
                raise ValueError(f"expected gate/value/out to be rank-1, got gate={tuple(gate.shape)}, value={tuple(value.shape)}, out={tuple(out.shape)}")
            if value.numel() != gate.numel() or out.numel() != gate.numel():
                raise ValueError(f"expected value/out lengths to match gate, got gate={gate.numel()}, value={value.numel()}, out={out.numel()}")

            gated_silu_f32_kernel[(gate.numel(),)](
                gate,
                value,
                out,
                gate.numel(),
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_topk_sampling_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def topk_sampling_f32_kernel(logits_ptr, top_values_ptr, top_indices_ptr, rows: gl.constexpr, vocab: gl.constexpr, k: gl.constexpr):
            row = gl.program_id(0)
            for rank in range(0, k):
                best_value = -3.4028234663852886e38
                best_index = vocab
                for col in range(0, vocab):
                    already_selected = False
                    for previous_rank in range(0, rank):
                        previous_index = gl.load(top_indices_ptr + row * k + previous_rank)
                        already_selected = already_selected or previous_index == col

                    value = gl.load(logits_ptr + row * vocab + col)
                    better_value = value > best_value
                    tie_lower_index = value == best_value and col < best_index
                    if (not already_selected) and (better_value or tie_lower_index):
                        best_value = value
                        best_index = col

                gl.store(top_values_ptr + row * k + rank, best_value)
                gl.store(top_indices_ptr + row * k + rank, best_index)


        def run_topk_sampling_f32(logits, top_values, top_indices, k=3, num_warps=4):
            expected_rank = 2
            if len(logits.shape) != expected_rank:
                raise ValueError(f"expected logits to be rank-2, got shape {tuple(logits.shape)}")
            if len(top_values.shape) != expected_rank or len(top_indices.shape) != expected_rank:
                raise ValueError(f"expected top_values/top_indices to be rank-2, got top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if logits.shape[0] != top_values.shape[0] or logits.shape[0] != top_indices.shape[0]:
                raise ValueError(f"expected matching row counts, got logits={tuple(logits.shape)}, top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if top_values.shape[1] != k or top_indices.shape[1] != k:
                raise ValueError(f"expected top-k output width {k}, got top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if k <= 0 or k > logits.shape[1]:
                raise ValueError(f"expected 0 < k <= vocab, got k={k}, vocab={logits.shape[1]}")

            topk_sampling_f32_kernel[(logits.shape[0],)](
                logits,
                top_values,
                top_indices,
                logits.shape[0],
                logits.shape[1],
                k,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_topp_sampling_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def topp_sampling_f32_kernel(probabilities_ptr, top_values_ptr, top_indices_ptr, selected_counts_ptr, cumulative_probs_ptr, rows: gl.constexpr, vocab: gl.constexpr, max_k: gl.constexpr, p: gl.constexpr):
            row = gl.program_id(0)
            cumulative = 0.0
            selected_count = 0
            done = False

            for rank in range(0, max_k):
                if done:
                    gl.store(top_values_ptr + row * max_k + rank, 0.0)
                    gl.store(top_indices_ptr + row * max_k + rank, -1)
                else:
                    best_value = -3.4028234663852886e38
                    best_index = vocab
                    for col in range(0, vocab):
                        already_selected = False
                        for previous_rank in range(0, rank):
                            previous_index = gl.load(top_indices_ptr + row * max_k + previous_rank)
                            already_selected = already_selected or previous_index == col

                        value = gl.load(probabilities_ptr + row * vocab + col)
                        better_value = value > best_value
                        tie_lower_index = value == best_value and col < best_index
                        if (not already_selected) and (better_value or tie_lower_index):
                            best_value = value
                            best_index = col

                    cumulative += best_value
                    selected_count += 1
                    gl.store(top_values_ptr + row * max_k + rank, best_value)
                    gl.store(top_indices_ptr + row * max_k + rank, best_index)
                    done = cumulative >= p

            gl.store(selected_counts_ptr + row, selected_count)
            gl.store(cumulative_probs_ptr + row, cumulative)


        def run_topp_sampling_f32(probabilities, top_values, top_indices, selected_counts, cumulative_probs, p=0.75, max_k=5, num_warps=4):
            expected_rank = 2
            if len(probabilities.shape) != expected_rank:
                raise ValueError(f"expected probabilities to be rank-2, got shape {tuple(probabilities.shape)}")
            if len(top_values.shape) != expected_rank or len(top_indices.shape) != expected_rank:
                raise ValueError(f"expected top_values/top_indices to be rank-2, got top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if len(selected_counts.shape) != 1 or len(cumulative_probs.shape) != 1:
                raise ValueError(f"expected selected_counts/cumulative_probs to be rank-1, got selected_counts={tuple(selected_counts.shape)}, cumulative_probs={tuple(cumulative_probs.shape)}")
            if probabilities.shape[0] != top_values.shape[0] or probabilities.shape[0] != top_indices.shape[0]:
                raise ValueError(f"expected matching row counts, got probabilities={tuple(probabilities.shape)}, top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if probabilities.shape[0] != selected_counts.shape[0] or probabilities.shape[0] != cumulative_probs.shape[0]:
                raise ValueError(f"expected matching metadata row counts, got probabilities={tuple(probabilities.shape)}, selected_counts={tuple(selected_counts.shape)}, cumulative_probs={tuple(cumulative_probs.shape)}")
            if top_values.shape[1] != max_k or top_indices.shape[1] != max_k:
                raise ValueError(f"expected top-p output width {max_k}, got top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if max_k <= 0 or max_k > probabilities.shape[1]:
                raise ValueError(f"expected 0 < max_k <= vocab, got max_k={max_k}, vocab={probabilities.shape[1]}")
            if p <= 0.0 or p > 1.0:
                raise ValueError(f"expected 0 < p <= 1, got p={p}")

            topp_sampling_f32_kernel[(probabilities.shape[0],)](
                probabilities,
                top_values,
                top_indices,
                selected_counts,
                cumulative_probs,
                probabilities.shape[0],
                probabilities.shape[1],
                max_k,
                p,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_minp_sampling_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def minp_sampling_f32_kernel(probabilities_ptr, top_values_ptr, top_indices_ptr, selected_counts_ptr, rows: gl.constexpr, vocab: gl.constexpr, max_k: gl.constexpr, min_p: gl.constexpr):
            row = gl.program_id(0)
            row_max = -3.4028234663852886e38
            for col in range(0, vocab):
                value = gl.load(probabilities_ptr + row * vocab + col)
                if value > row_max:
                    row_max = value

            threshold = min_p * row_max
            selected_count = 0
            done = False

            for rank in range(0, max_k):
                if done:
                    gl.store(top_values_ptr + row * max_k + rank, 0.0)
                    gl.store(top_indices_ptr + row * max_k + rank, -1)
                else:
                    best_value = -3.4028234663852886e38
                    best_index = vocab
                    for col in range(0, vocab):
                        already_selected = False
                        for previous_rank in range(0, rank):
                            previous_index = gl.load(top_indices_ptr + row * max_k + previous_rank)
                            already_selected = already_selected or previous_index == col

                        value = gl.load(probabilities_ptr + row * vocab + col)
                        passes_threshold = value >= threshold
                        better_value = value > best_value
                        tie_lower_index = value == best_value and col < best_index
                        if (not already_selected) and passes_threshold and (better_value or tie_lower_index):
                            best_value = value
                            best_index = col

                    if best_index == vocab:
                        gl.store(top_values_ptr + row * max_k + rank, 0.0)
                        gl.store(top_indices_ptr + row * max_k + rank, -1)
                        done = True
                    else:
                        selected_count += 1
                        gl.store(top_values_ptr + row * max_k + rank, best_value)
                        gl.store(top_indices_ptr + row * max_k + rank, best_index)

            gl.store(selected_counts_ptr + row, selected_count)


        def run_minp_sampling_f32(probabilities, top_values, top_indices, selected_counts, min_p=0.5, max_k=5, num_warps=4):
            expected_rank = 2
            if len(probabilities.shape) != expected_rank:
                raise ValueError(f"expected probabilities to be rank-2, got shape {tuple(probabilities.shape)}")
            if len(top_values.shape) != expected_rank or len(top_indices.shape) != expected_rank:
                raise ValueError(f"expected top_values/top_indices to be rank-2, got top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if len(selected_counts.shape) != 1:
                raise ValueError(f"expected selected_counts to be rank-1, got selected_counts={tuple(selected_counts.shape)}")
            if probabilities.shape[0] != top_values.shape[0] or probabilities.shape[0] != top_indices.shape[0]:
                raise ValueError(f"expected matching row counts, got probabilities={tuple(probabilities.shape)}, top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if probabilities.shape[0] != selected_counts.shape[0]:
                raise ValueError(f"expected matching metadata row counts, got probabilities={tuple(probabilities.shape)}, selected_counts={tuple(selected_counts.shape)}")
            if top_values.shape[1] != max_k or top_indices.shape[1] != max_k:
                raise ValueError(f"expected min-p output width {max_k}, got top_values={tuple(top_values.shape)}, top_indices={tuple(top_indices.shape)}")
            if max_k <= 0 or max_k > probabilities.shape[1]:
                raise ValueError(f"expected 0 < max_k <= vocab, got max_k={max_k}, vocab={probabilities.shape[1]}")
            if min_p <= 0.0 or min_p > 1.0:
                raise ValueError(f"expected 0 < min_p <= 1, got min_p={min_p}")

            minp_sampling_f32_kernel[(probabilities.shape[0],)](
                probabilities,
                top_values,
                top_indices,
                selected_counts,
                probabilities.shape[0],
                probabilities.shape[1],
                max_k,
                min_p,
                num_warps=num_warps,
            )
        """
    ).lstrip()


def _render_speculative_accept_f32_source() -> str:
    return dedent(
        """
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def speculative_accept_f32_kernel(draft_token_ids_ptr, draft_probabilities_ptr, target_probabilities_ptr, thresholds_ptr, accepted_token_ids_ptr, accept_mask_ptr, accepted_counts_ptr, rows: gl.constexpr, max_draft: gl.constexpr):
            row = gl.program_id(0)
            accepting = True
            accepted_count = 0

            for pos in range(0, max_draft):
                offset = row * max_draft + pos
                draft_probability = gl.load(draft_probabilities_ptr + offset)
                target_probability = gl.load(target_probabilities_ptr + offset)
                threshold = gl.load(thresholds_ptr + offset)
                ratio = target_probability / draft_probability
                accept_probability = ratio
                if accept_probability > 1.0:
                    accept_probability = 1.0

                should_accept = accepting and threshold <= accept_probability
                if should_accept:
                    token_id = gl.load(draft_token_ids_ptr + offset)
                    gl.store(accepted_token_ids_ptr + offset, token_id)
                    gl.store(accept_mask_ptr + offset, 1)
                    accepted_count += 1
                else:
                    gl.store(accepted_token_ids_ptr + offset, -1)
                    gl.store(accept_mask_ptr + offset, 0)
                    accepting = False

            gl.store(accepted_counts_ptr + row, accepted_count)


        def run_speculative_accept_f32(draft_token_ids, draft_probabilities, target_probabilities, thresholds, accepted_token_ids, accept_mask, accepted_counts, max_draft=4, num_warps=4):
            expected_rank = 2
            if len(draft_token_ids.shape) != expected_rank:
                raise ValueError(f"expected draft_token_ids to be rank-2, got shape {tuple(draft_token_ids.shape)}")
            if len(draft_probabilities.shape) != expected_rank or len(target_probabilities.shape) != expected_rank or len(thresholds.shape) != expected_rank:
                raise ValueError(f"expected probability and threshold inputs to be rank-2, got draft_probabilities={tuple(draft_probabilities.shape)}, target_probabilities={tuple(target_probabilities.shape)}, thresholds={tuple(thresholds.shape)}")
            if len(accepted_token_ids.shape) != expected_rank or len(accept_mask.shape) != expected_rank:
                raise ValueError(f"expected accepted_token_ids/accept_mask to be rank-2, got accepted_token_ids={tuple(accepted_token_ids.shape)}, accept_mask={tuple(accept_mask.shape)}")
            if len(accepted_counts.shape) != 1:
                raise ValueError(f"expected accepted_counts to be rank-1, got accepted_counts={tuple(accepted_counts.shape)}")

            input_shapes = (tuple(draft_token_ids.shape), tuple(draft_probabilities.shape), tuple(target_probabilities.shape), tuple(thresholds.shape))
            if len(set(input_shapes)) != 1:
                raise ValueError(f"expected matching input shapes, got {input_shapes}")
            if tuple(accepted_token_ids.shape) != tuple(draft_token_ids.shape) or tuple(accept_mask.shape) != tuple(draft_token_ids.shape):
                raise ValueError(f"expected output matrices to match draft_token_ids, got draft_token_ids={tuple(draft_token_ids.shape)}, accepted_token_ids={tuple(accepted_token_ids.shape)}, accept_mask={tuple(accept_mask.shape)}")
            if accepted_counts.shape[0] != draft_token_ids.shape[0]:
                raise ValueError(f"expected accepted_counts rows to match inputs, got draft_token_ids={tuple(draft_token_ids.shape)}, accepted_counts={tuple(accepted_counts.shape)}")
            if max_draft <= 0 or max_draft != draft_token_ids.shape[1]:
                raise ValueError(f"expected max_draft to match input width and be positive, got max_draft={max_draft}, input_width={draft_token_ids.shape[1]}")

            speculative_accept_f32_kernel[(draft_token_ids.shape[0],)](
                draft_token_ids,
                draft_probabilities,
                target_probabilities,
                thresholds,
                accepted_token_ids,
                accept_mask,
                accepted_counts,
                draft_token_ids.shape[0],
                max_draft,
                num_warps=num_warps,
            )
        """
    ).lstrip()
