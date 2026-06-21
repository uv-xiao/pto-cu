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
    "flashattention_fwd_f32",
    "moe_expert_affine_f32",
    "rmsnorm_f32",
    "rope_f32",
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
        return _render_tiled_tensor_core_gemm_source(tile_shape)
    if kernel_name == "flashattention_fwd_f32":
        return _render_flashattention_source(tile_shape)
    if kernel_name == "moe_expert_affine_f32":
        return _render_moe_expert_affine_source()
    if kernel_name == "rmsnorm_f32":
        return _render_rmsnorm_f32_source()
    if kernel_name == "rope_f32":
        return _render_rope_f32_source()
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


def _render_tiled_tensor_core_gemm_source(tile_shape: tuple[int, int, int]) -> str:
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
        def gemm_tensor_core_tiled_f16_f32_kernel(a_desc, b_desc, c_desc, d_desc, instr_shape_n: gl.constexpr, num_warps: gl.constexpr):
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


        def run_gemm_tensor_core_tiled_f16_f32(a, b, c, d, instr_shape_n=16, num_warps=4):
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

            a_layout = gl.NVMMASharedLayout.get_default_for([{block_m}, {block_k}], gl.float16)
            b_layout = gl.NVMMASharedLayout.get_default_for([{block_k}, {block_n}], gl.float16)
            cd_layout = gl.NVMMASharedLayout.get_default_for([{block_m}, {block_n}], gl.float32)
            a_desc = TensorDescriptor.from_tensor(a, [{block_m}, {block_k}], a_layout)
            b_desc = TensorDescriptor.from_tensor(b, [{block_k}, {block_n}], b_layout)
            c_desc = TensorDescriptor.from_tensor(c, [{block_m}, {block_n}], cd_layout)
            d_desc = TensorDescriptor.from_tensor(d, [{block_m}, {block_n}], cd_layout)
            grid = (triton.cdiv(c.shape[0], {block_m}), triton.cdiv(c.shape[1], {block_n}))
            gemm_tensor_core_tiled_f16_f32_kernel[grid](
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
    return dedent(
        f"""
        import math

        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl


        @gluon.jit
        def flashattention_fwd_f32_kernel(q_ptr, k_ptr, v_ptr, out_ptr, seqlen_q: gl.constexpr, seqlen_k: gl.constexpr, head_dim: gl.constexpr, scale: gl.constexpr):
            layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
            lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
            rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

            offs_m = gl.arange(0, {block_m}, layout=gl.SliceLayout(1, layout))[:, None]
            offs_n = gl.arange(0, {block_n}, layout=gl.SliceLayout(0, layout))[None, :]
            offs_k_col = gl.arange(0, {block_d}, layout=gl.SliceLayout(0, layout))[None, :]
            offs_k_row = gl.arange(0, {block_d}, layout=gl.SliceLayout(1, layout))[:, None]

            q = gl.load(q_ptr + offs_m * head_dim + offs_k_col)
            k_t = gl.load(k_ptr + offs_k_row * head_dim + offs_n)
            q = gl.convert_layout(q, lhs_layout)
            k_t = gl.convert_layout(k_t, rhs_layout)
            score_acc = gl.full(({block_m}, {block_n}), 0.0, gl.float32, layout=layout)
            scores = gl.dot_fma(q, k_t, score_acc) * scale

            row_max = gl.max(scores, axis=1)
            probs = gl.exp(scores - row_max[:, None])
            row_sum = gl.sum(probs, axis=1)
            probs = probs / row_sum[:, None]

            offs_d = gl.arange(0, {block_d}, layout=gl.SliceLayout(0, layout))[None, :]
            offs_vn = gl.arange(0, {block_n}, layout=gl.SliceLayout(1, layout))[:, None]
            probs = gl.convert_layout(probs, lhs_layout)
            v = gl.load(v_ptr + offs_d * head_dim + offs_vn)
            v = gl.convert_layout(v, rhs_layout)
            out_acc = gl.full(({block_m}, {block_d}), 0.0, gl.float32, layout=layout)
            out = gl.dot_fma(probs, v, out_acc)
            gl.store(out_ptr + offs_m * head_dim + offs_d, out)


        def run_flashattention_fwd_f32(q, k, v, out, scale=None, num_warps=4):
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
