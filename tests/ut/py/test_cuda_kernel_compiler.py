# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CUDA KernelCompiler integration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from simpler_setup.gluon_gen import generate_gluon_persistent_task_body
from simpler_setup import kernel_compiler
from simpler_setup.kernel_compiler import KernelCompiler


def _load_gluon_gemm_example():
    module_path = "examples/cuda/gluon_gemm_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_gemm_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_tensor_core_example():
    module_path = "examples/cuda/gluon_gemm_tensor_core.py"
    spec = importlib.util.spec_from_file_location("gluon_gemm_tensor_core_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_tensor_core_tiled_example():
    module_path = "examples/cuda/gluon_gemm_tensor_core_tiled.py"
    spec = importlib.util.spec_from_file_location("gluon_gemm_tensor_core_tiled_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_flashattention_example():
    module_path = "examples/cuda/gluon_flashattention_fwd.py"
    spec = importlib.util.spec_from_file_location("gluon_flashattention_fwd_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_moe_expert_example():
    module_path = "examples/cuda/gluon_moe_expert_affine.py"
    spec = importlib.util.spec_from_file_location("gluon_moe_expert_affine_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_rmsnorm_example():
    module_path = "examples/cuda/gluon_rmsnorm_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_rmsnorm_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_gemma_fused_rmsnorm_example():
    module_path = "examples/cuda/gluon_gemma_fused_rmsnorm_f32.py"
    spec = importlib.util.spec_from_file_location(
        "gluon_gemma_fused_rmsnorm_f32_example", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_layernorm_example():
    module_path = "examples/cuda/gluon_layernorm_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_layernorm_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_rope_example():
    module_path = "examples/cuda/gluon_rope_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_rope_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_silu_example():
    module_path = "examples/cuda/gluon_silu_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_silu_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_gelu_example():
    module_path = "examples/cuda/gluon_gelu_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_gelu_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_gated_silu_example():
    module_path = "examples/cuda/gluon_gated_silu_f32.py"
    spec = importlib.util.spec_from_file_location("gluon_gated_silu_f32_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gluon_benchmark_example():
    module_path = "examples/cuda/gluon_benchmark.py"
    spec = importlib.util.spec_from_file_location("gluon_benchmark_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_persistent_moe_dispatch_example():
    module_path = "examples/cuda/persistent_moe_dispatch_combine.py"
    spec = importlib.util.spec_from_file_location("persistent_moe_dispatch_combine_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_gluon_gemm_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gemm_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(16, 16, 16),
    )

    source = artifact.source_path.read_text()
    manifest = json.loads(artifact.manifest_path.read_text())

    assert artifact.kernel_name == "gemm_f32"
    assert artifact.compiler_role == "pto-isa-replacement"
    assert artifact.arch == "compute_90"
    assert artifact.tile_shape == (16, 16, 16)
    assert artifact.source_path.name == "gemm_f32.gluon.py"
    assert artifact.manifest_path.name == "gemm_f32.gluon.json"
    assert "from triton.experimental import gluon" in source
    assert "def gemm_f32_kernel" in source
    assert "gl.program_id(0)" in source
    assert "acc += a * b" in source
    assert manifest["kernel_name"] == "gemm_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "gemm_f32.gluon.py"
    assert manifest["tile_shape"] == [16, 16, 16]
    assert manifest["source_sha256"] == artifact.source_sha256


def test_gluon_generation_is_cuda_only(tmp_path):
    compiler = KernelCompiler(platform="a2a3sim")

    try:
        compiler.generate_gluon_kernel("gemm_f32", output_dir=tmp_path)
    except ValueError as exc:
        assert "only available for platform='cuda'" in str(exc)
    else:
        raise AssertionError("expected generate_gluon_kernel to reject non-CUDA platforms")


def test_gluon_generation_rejects_unknown_kernels(tmp_path):
    compiler = KernelCompiler(platform="cuda")

    try:
        compiler.generate_gluon_kernel("unknown_cuda_kernel", output_dir=tmp_path)
    except ValueError as exc:
        assert "unsupported Gluon kernel" in str(exc)
        assert "flashattention_fwd_f32" in str(exc)
        assert "gemm_f32" in str(exc)
    else:
        raise AssertionError("expected generator to reject unknown kernels")


def test_generate_gluon_tensor_core_gemm_writes_wgmma_source(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gemm_tensor_core_f16_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(64, 32, 32),
    )

    source = artifact.source_path.read_text()
    manifest = json.loads(artifact.manifest_path.read_text())

    assert artifact.kernel_name == "gemm_tensor_core_f16_f32"
    assert artifact.tile_shape == (64, 32, 32)
    assert artifact.source_path.name == "gemm_tensor_core_f16_f32.gluon.py"
    assert manifest["kernel_name"] == "gemm_tensor_core_f16_f32"
    assert manifest["tile_shape"] == [64, 32, 32]
    assert "TensorDescriptor" in source
    assert "NVMMADistributedLayout" in source
    assert "warpgroup_mma" in source
    assert "tma.async_copy_global_to_shared" in source
    assert "def gemm_tensor_core_f16_f32_kernel" in source
    assert "def run_gemm_tensor_core_f16_f32" in source


def test_generate_gluon_tiled_tensor_core_gemm_writes_tiled_source(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gemm_tensor_core_tiled_f16_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(64, 128, 32),
    )

    source = artifact.source_path.read_text()
    manifest = json.loads(artifact.manifest_path.read_text())

    assert artifact.kernel_name == "gemm_tensor_core_tiled_f16_f32"
    assert manifest["tile_shape"] == [64, 128, 32]
    assert "import triton" in source
    assert "pid_m = gl.program_id(axis=0)" in source
    assert "pid_n = gl.program_id(axis=1)" in source
    assert "for k_offset in range(0, K, 32):" in source
    assert "tma.async_copy_global_to_shared(a_desc, [off_m, k_offset]" in source
    assert "tma.async_copy_shared_to_global(d_desc, [off_m, off_n]" in source
    assert "TensorDescriptor.from_tensor(a, [64, 32], a_layout)" in source
    assert "TensorDescriptor.from_tensor(b, [32, 128], b_layout)" in source
    assert "triton.cdiv(c.shape[0], 64)" in source
    assert "def run_gemm_tensor_core_tiled_f16_f32" in source


def test_gluon_gemm_example_reports_skip_json_and_relative_artifacts(tmp_path, monkeypatch):
    example = _load_gluon_gemm_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_gemm_correctness(
        output_dir=Path("gluon-artifacts"),
        arch="compute_90",
        tile_shape=(16, 16, 16),
        m=16,
        n=16,
        k=16,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["shape"] == {"m": 16, "n": 16, "k": 16}
    assert result["artifact"]["source_path"] == "gluon-artifacts/gemm_f32.gluon.py"
    assert result["artifact"]["manifest_path"] == "gluon-artifacts/gemm_f32.gluon.json"
    assert result["artifact"]["tile_shape"] == [16, 16, 16]


def test_gluon_gemm_example_main_requires_cuda_on_skip(tmp_path, capsys, monkeypatch):
    example = _load_gluon_gemm_example()
    monkeypatch.setattr(example, "gluon_gemm_skip_reason", lambda: "missing triton")

    code = example.main(
        [
            "--output-dir",
            str(tmp_path),
            "--m",
            "16",
            "--n",
            "16",
            "--k",
            "16",
            "--require-cuda",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing triton"


def test_gluon_tensor_core_example_reports_skip_json_and_relative_artifacts(tmp_path, monkeypatch):
    example = _load_gluon_tensor_core_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_tensor_core_correctness(
        output_dir=Path("tensor-core-artifacts"),
        arch="compute_90",
        skip_reason=lambda: "triton Gluon WGMMA import failed: missing primitive",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "triton Gluon WGMMA import failed: missing primitive"
    assert result["kernel_name"] == "gemm_tensor_core_f16_f32"
    assert result["shape"] == {"m": 64, "n": 32, "k": 32}
    assert result["artifact"]["source_path"] == (
        "tensor-core-artifacts/gemm_tensor_core_f16_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tensor-core-artifacts/gemm_tensor_core_f16_f32.gluon.json"
    )
    assert result["artifact"]["tile_shape"] == [64, 32, 32]


def test_gluon_tensor_core_example_main_requires_cuda_on_skip(tmp_path, capsys, monkeypatch):
    example = _load_gluon_tensor_core_example()
    monkeypatch.setattr(example, "tensor_core_skip_reason", lambda: "torch.cuda is not available")

    code = example.main(["--output-dir", str(tmp_path), "--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "skipped"
    assert payload["reason"] == "torch.cuda is not available"


def test_gluon_tensor_core_example_main_reports_generation_failure(capsys, monkeypatch):
    example = _load_gluon_tensor_core_example()

    def fail_generation(**_kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(example, "build_tensor_core_artifact", fail_generation)

    code = example.main(["--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["kernel_name"] == "gemm_tensor_core_f16_f32"
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "generation failed"


def test_gluon_tiled_tensor_core_example_reports_skip_and_validates_shape(tmp_path):
    example = _load_gluon_tensor_core_tiled_example()

    result = example.run_tiled_tensor_core_correctness(
        output_dir=tmp_path,
        m=256,
        n=256,
        k=64,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["kernel_name"] == "gemm_tensor_core_tiled_f16_f32"
    assert result["artifact"]["tile_shape"] == [64, 128, 32]

    try:
        example.run_tiled_tensor_core_correctness(
            output_dir=tmp_path,
            m=224,
            n=256,
            k=64,
            skip_reason=lambda: "not reached",
        )
    except ValueError as exc:
        assert "expected m,n divisible by tile" in str(exc)
    else:
        raise AssertionError("expected tiled tensor-core harness to validate shapes")


def test_gluon_tiled_tensor_core_example_main_requires_cuda_on_skip(tmp_path, capsys, monkeypatch):
    example = _load_gluon_tensor_core_tiled_example()
    monkeypatch.setattr(example, "tensor_core_skip_reason", lambda: "missing WGMMA APIs")

    code = example.main(["--output-dir", str(tmp_path), "--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing WGMMA APIs"


def test_gluon_tiled_tensor_core_example_main_reports_generation_failure(capsys, monkeypatch):
    example = _load_gluon_tensor_core_tiled_example()

    def fail_generation(**_kwargs):
        raise RuntimeError("tiled generation failed")

    monkeypatch.setattr(example, "build_tiled_tensor_core_artifact", fail_generation)

    code = example.main(["--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["kernel_name"] == "gemm_tensor_core_tiled_f16_f32"
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "tiled generation failed"


def test_generate_gluon_flashattention_writes_dot_fma_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "flashattention_fwd_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(32, 32, 32),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "flashattention_fwd_f32"
    assert artifact.tile_shape == (32, 32, 32)
    assert artifact.source_path.name == "flashattention_fwd_f32.gluon.py"
    assert manifest["kernel_name"] == "flashattention_fwd_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "flashattention_fwd_f32.gluon.py"
    assert manifest["tile_shape"] == [32, 32, 32]
    assert "def flashattention_fwd_f32_kernel" in source
    assert "q_ptr" in source
    assert "k_ptr" in source
    assert "v_ptr" in source
    assert "out_ptr" in source
    assert "gl.DotOperandLayout" in source
    assert "gl.dot_fma(q, k_t, score_acc)" in source
    assert "k_ptr + offs_k_row * head_dim + offs_n" in source
    assert "v_ptr + offs_d * head_dim + offs_vn" in source
    assert "gl.softmax" not in source
    assert "gl.dot(" not in source
    assert "def run_flashattention_fwd_f32" in source


def test_gluon_flashattention_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_flashattention_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_flashattention_correctness(
        output_dir=Path("flashattention-artifacts"),
        arch="compute_90",
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "flashattention_fwd_f32"
    assert result["shape"] == {"seqlen_q": 32, "seqlen_k": 32, "head_dim": 32}
    assert result["artifact"]["source_path"] == (
        "flashattention-artifacts/flashattention_fwd_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "flashattention-artifacts/flashattention_fwd_f32.gluon.json"
    )
    assert result["artifact"]["tile_shape"] == [32, 32, 32]


def test_gluon_flashattention_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_flashattention_example()
    monkeypatch.setattr(
        example,
        "flashattention_skip_reason",
        lambda: "triton Gluon import failed: missing gl.dot_fma",
    )

    code = example.main(["--output-dir", str(tmp_path), "--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "skipped"
    assert payload["reason"] == "triton Gluon import failed: missing gl.dot_fma"


def test_gluon_flashattention_example_main_reports_generation_failure(capsys, monkeypatch):
    example = _load_gluon_flashattention_example()

    def fail_generation(**_kwargs):
        raise RuntimeError("flashattention generation failed")

    monkeypatch.setattr(example, "build_flashattention_artifact", fail_generation)

    code = example.main(["--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["kernel_name"] == "flashattention_fwd_f32"
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "flashattention generation failed"


def test_gluon_flashattention_skip_reason_checks_required_gluon_apis(monkeypatch):
    example = _load_gluon_flashattention_example()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    fake_triton = ModuleType("triton")
    fake_experimental = ModuleType("triton.experimental")
    fake_gluon = ModuleType("triton.experimental.gluon")
    fake_gl = ModuleType("triton.experimental.gluon.language")
    fake_gl.exp = object()
    fake_gl.max = object()
    fake_gl.sum = object()
    fake_experimental.gluon = fake_gluon
    fake_gluon.language = fake_gl

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.experimental", fake_experimental)
    monkeypatch.setitem(sys.modules, "triton.experimental.gluon", fake_gluon)
    monkeypatch.setitem(sys.modules, "triton.experimental.gluon.language", fake_gl)

    assert example.flashattention_skip_reason() == (
        "triton Gluon import failed: missing gl.dot_fma"
    )


def test_generate_gluon_moe_expert_affine_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "moe_expert_affine_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "moe_expert_affine_f32"
    assert artifact.source_path.name == "moe_expert_affine_f32.gluon.py"
    assert artifact.manifest_path.name == "moe_expert_affine_f32.gluon.json"
    assert manifest["kernel_name"] == "moe_expert_affine_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "moe_expert_affine_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def moe_expert_affine_f32_kernel" in source
    assert "scale_a * a + scale_b * b" in source
    assert "def run_moe_expert_affine_f32" in source


def test_generate_gluon_rmsnorm_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "rmsnorm_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "rmsnorm_f32"
    assert artifact.source_path.name == "rmsnorm_f32.gluon.py"
    assert artifact.manifest_path.name == "rmsnorm_f32.gluon.json"
    assert manifest["kernel_name"] == "rmsnorm_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "rmsnorm_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def rmsnorm_f32_kernel" in source
    assert "sum_sq += x * x" in source
    assert "gl.rsqrt(mean_sq + eps)" in source
    assert "def run_rmsnorm_f32" in source


def test_generate_gluon_gemma_fused_rmsnorm_f32_writes_source_and_manifest(
    tmp_path,
):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gemma_fused_rmsnorm_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "gemma_fused_rmsnorm_f32"
    assert artifact.source_path.name == "gemma_fused_rmsnorm_f32.gluon.py"
    assert artifact.manifest_path.name == "gemma_fused_rmsnorm_f32.gluon.json"
    assert manifest["kernel_name"] == "gemma_fused_rmsnorm_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "gemma_fused_rmsnorm_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def gemma_fused_rmsnorm_f32_kernel" in source
    assert "sum_sq += x * x" in source
    assert "gl.rsqrt(mean_sq + eps)" in source
    assert "x * inv_rms * (1.0 + weight)" in source
    assert "def run_gemma_fused_rmsnorm_f32" in source


def test_generate_gluon_layernorm_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "layernorm_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "layernorm_f32"
    assert artifact.source_path.name == "layernorm_f32.gluon.py"
    assert artifact.manifest_path.name == "layernorm_f32.gluon.json"
    assert manifest["kernel_name"] == "layernorm_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "layernorm_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def layernorm_f32_kernel" in source
    assert "mean += x" in source
    assert "var += centered * centered" in source
    assert "gl.rsqrt(var + eps)" in source
    assert "def run_layernorm_f32" in source


def test_generate_gluon_rope_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "rope_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "rope_f32"
    assert artifact.source_path.name == "rope_f32.gluon.py"
    assert artifact.manifest_path.name == "rope_f32.gluon.json"
    assert manifest["kernel_name"] == "rope_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "rope_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def rope_f32_kernel" in source
    assert "out_even = x_even * cos - x_odd * sin" in source
    assert "out_odd = x_even * sin + x_odd * cos" in source
    assert "def run_rope_f32" in source


def test_generate_gluon_silu_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "silu_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "silu_f32"
    assert artifact.source_path.name == "silu_f32.gluon.py"
    assert artifact.manifest_path.name == "silu_f32.gluon.json"
    assert manifest["kernel_name"] == "silu_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "silu_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def silu_f32_kernel" in source
    assert "x / (1.0 + gl.exp(-x))" in source
    assert "def run_silu_f32" in source


def test_generate_gluon_gelu_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gelu_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "gelu_f32"
    assert artifact.source_path.name == "gelu_f32.gluon.py"
    assert artifact.manifest_path.name == "gelu_f32.gluon.json"
    assert manifest["kernel_name"] == "gelu_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "gelu_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def gelu_f32_kernel" in source
    assert "0.5 * x * (1.0 + gl.erf(x * 0.7071067811865476))" in source
    assert "def run_gelu_f32" in source


def test_generate_gluon_gated_silu_f32_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "gated_silu_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "gated_silu_f32"
    assert artifact.source_path.name == "gated_silu_f32.gluon.py"
    assert artifact.manifest_path.name == "gated_silu_f32.gluon.json"
    assert manifest["kernel_name"] == "gated_silu_f32"
    assert manifest["compiler_role"] == "pto-isa-replacement"
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "gated_silu_f32.gluon.py"
    assert manifest["tile_shape"] == [1, 1, 1]
    assert "def gated_silu_f32_kernel" in source
    assert "value * gate / (1.0 + gl.exp(-gate))" in source
    assert "def run_gated_silu_f32" in source


def test_generate_gluon_persistent_moe_expert_task_body_bridge():
    artifact = generate_gluon_persistent_task_body("moe_expert_affine_f32")

    assert artifact.kernel_name == "moe_expert_affine_f32"
    assert artifact.task_name == "gluon_moe_expert_affine_f32"
    assert artifact.source_kind == "gluon-persistent-task-body-bridge"
    assert "task->scalar0 * task->a[i] + task->scalar1 * task->b[i]" in artifact.body
    assert artifact.source_sha256


def test_persistent_moe_dispatch_source_uses_gluon_expert_and_weighted_combine():
    example = _load_persistent_moe_dispatch_example()

    source = example.rendered_dispatch_source(example.build_task_body_specs())

    assert "pto_dag_task_gluon_moe_expert_affine_f32" in source
    assert "pto_dag_task_weighted_combine_f32" in source
    assert "case 12U:" in source
    assert "case 13U:" in source
    assert "task->scalar_args[3] * task->d[i]" in source


def test_persistent_moe_dispatch_reports_gluon_expert_bridge_on_skip():
    example = _load_persistent_moe_dispatch_example()
    expected = generate_gluon_persistent_task_body("moe_expert_affine_f32")

    result = example.run_moe_dispatch_combine(
        n=8,
        arch="compute_90",
        skip_reason=lambda: "CUDA unavailable",
    )

    assert result["status"] == "skipped"
    assert result["gluon_expert_bridge"] == {
        "func_id": 12,
        "kernel_name": "moe_expert_affine_f32",
        "task_name": "gluon_moe_expert_affine_f32",
        "source_kind": "gluon-persistent-task-body-bridge",
        "source_sha256": expected.source_sha256,
    }
    expert_body = next(
        body for body in result["task_bodies"] if body["func_id"] == 12
    )
    assert expert_body["name"] == result["gluon_expert_bridge"]["task_name"]
    assert expert_body["source_kind"] == result["gluon_expert_bridge"]["source_kind"]
    assert expert_body["source_sha256"] == result["gluon_expert_bridge"]["source_sha256"]


def test_gluon_moe_expert_affine_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_moe_expert_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_moe_expert_correctness(
        output_dir=Path("tmp/gluon-moe-expert-local"),
        arch="compute_90",
        n=4096,
        scale_a=1.25,
        scale_b=0.5,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "moe_expert_affine_f32"
    assert result["shape"] == {"n": 4096}
    assert result["scalars"] == {"scale_a": 1.25, "scale_b": 0.5}
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-moe-expert-local/moe_expert_affine_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-moe-expert-local/moe_expert_affine_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_rmsnorm_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_rmsnorm_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_rmsnorm_correctness(
        output_dir=Path("tmp/gluon-rmsnorm-local"),
        arch="compute_90",
        rows=2,
        hidden=16,
        eps=1e-5,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "rmsnorm_f32"
    assert result["shape"] == {"rows": 2, "hidden": 16}
    assert result["epsilon"] == 1e-5
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-rmsnorm-local/rmsnorm_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-rmsnorm-local/rmsnorm_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_rmsnorm_f32_sweep_reports_fixed_cases_and_provenance(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_rmsnorm_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_rmsnorm_sweep(
        output_dir=Path("tmp/gluon-rmsnorm-sweep-local"),
        arch="compute_90",
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "rmsnorm_f32"
    assert result["status"] == "skipped"
    assert result["case_count"] == 2
    assert result["passed_cases"] == 0
    assert result["failed_cases"] == 0
    assert result["skipped_cases"] == 2

    cases = result["cases"]
    assert [case["shape"] for case in cases] == [
        {"rows": 2, "hidden": 16},
        {"rows": 1, "hidden": 7168},
    ]
    assert [case["epsilon"] for case in cases] == [1e-5, 1e-5]
    assert [case["provenance"] for case in cases] == [
        "existing-smoke",
        "DeepSeek-V4-Flash config hidden_size",
    ]
    assert {case["status"] for case in cases} == {"skipped"}
    for case in cases:
        assert case["artifact"]["source_path"].startswith(
            "tmp/gluon-rmsnorm-sweep-local/"
        )
        assert not Path(case["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_gemma_fused_rmsnorm_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_gemma_fused_rmsnorm_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_gemma_fused_rmsnorm_correctness(
        output_dir=Path("tmp/gluon-gemma-fused-rmsnorm-local"),
        arch="compute_90",
        rows=2,
        hidden=16,
        eps=1e-5,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "gemma_fused_rmsnorm_f32"
    assert result["shape"] == {"rows": 2, "hidden": 16}
    assert result["epsilon"] == 1e-5
    assert result["reference"] == (
        "out[row, col] = x[row, col] * rsqrt(mean(x[row, :]^2) + eps) "
        "* (1.0 + weight[col])"
    )
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-gemma-fused-rmsnorm-local/gemma_fused_rmsnorm_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-gemma-fused-rmsnorm-local/gemma_fused_rmsnorm_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_layernorm_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_layernorm_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_layernorm_correctness(
        output_dir=Path("tmp/gluon-layernorm-local"),
        arch="compute_90",
        rows=2,
        hidden=16,
        eps=1e-5,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "layernorm_f32"
    assert result["shape"] == {"rows": 2, "hidden": 16}
    assert result["epsilon"] == 1e-5
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-layernorm-local/layernorm_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-layernorm-local/layernorm_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_layernorm_f32_sweep_reports_fixed_cases_and_provenance(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_layernorm_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_layernorm_sweep(
        output_dir=Path("tmp/gluon-layernorm-sweep-local"),
        arch="compute_90",
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "layernorm_f32"
    assert result["status"] == "skipped"
    assert result["case_count"] == 2
    assert result["passed_cases"] == 0
    assert result["failed_cases"] == 0
    assert result["skipped_cases"] == 2

    cases = result["cases"]
    assert [case["shape"] for case in cases] == [
        {"rows": 2, "hidden": 16},
        {"rows": 1, "hidden": 7168},
    ]
    assert [case["epsilon"] for case in cases] == [1e-5, 1e-5]
    assert [case["provenance"] for case in cases] == [
        "existing-smoke",
        "DeepSeek-V4-Flash config hidden_size",
    ]
    assert [case["reference"] for case in cases] == [
        "out = (x - mean) * rsqrt(var + eps) * weight + bias",
        "out = (x - mean) * rsqrt(var + eps) * weight + bias",
    ]
    assert {case["status"] for case in cases} == {"skipped"}
    for case in cases:
        assert case["artifact"]["source_path"].startswith(
            "tmp/gluon-layernorm-sweep-local/"
        )
        assert not Path(case["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_rope_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_rope_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_rope_correctness(
        output_dir=Path("tmp/gluon-rope-local"),
        arch="compute_90",
        batch=1,
        seq=2,
        head_dim=8,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "rope_f32"
    assert result["shape"] == {"batch": 1, "seq": 2, "head_dim": 8}
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-rope-local/rope_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-rope-local/rope_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_silu_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_silu_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_silu_correctness(
        output_dir=Path("tmp/gluon-silu-local"),
        arch="compute_90",
        n=32,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "silu_f32"
    assert result["shape"] == {"n": 32}
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-silu-local/silu_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-silu-local/silu_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_gelu_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_gelu_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_gelu_correctness(
        output_dir=Path("tmp/gluon-gelu-local"),
        arch="compute_90",
        n=32,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "gelu_f32"
    assert result["shape"] == {"n": 32}
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-gelu-local/gelu_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-gelu-local/gelu_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_gated_silu_f32_example_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_gated_silu_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_gated_silu_correctness(
        output_dir=Path("tmp/gluon-gated-silu-local"),
        arch="compute_90",
        n=32,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["kernel_name"] == "gated_silu_f32"
    assert result["shape"] == {"n": 32}
    assert result["reference"] == "out = value * gate / (1.0 + exp(-gate))"
    assert result["artifact"]["source_path"] == (
        "tmp/gluon-gated-silu-local/gated_silu_f32.gluon.py"
    )
    assert result["artifact"]["manifest_path"] == (
        "tmp/gluon-gated-silu-local/gated_silu_f32.gluon.json"
    )
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_rmsnorm_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_rmsnorm_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "rmsnorm_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-rmsnorm-local",
            "--rows",
            "2",
            "--hidden",
            "16",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_rmsnorm_f32_sweep_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_rmsnorm_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "rmsnorm_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-rmsnorm-sweep-local",
            "--sweep",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["case_count"] == 2
    assert payload["skipped_cases"] == 2
    assert [case["provenance"] for case in payload["cases"]] == [
        "existing-smoke",
        "DeepSeek-V4-Flash config hidden_size",
    ]


def test_gluon_gemma_fused_rmsnorm_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_gemma_fused_rmsnorm_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        example,
        "gemma_fused_rmsnorm_skip_reason",
        lambda: "missing CUDA",
    )

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-gemma-fused-rmsnorm-local",
            "--rows",
            "2",
            "--hidden",
            "16",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_gemma_fused_rmsnorm_f32_rejects_absolute_output_dir(tmp_path):
    example = _load_gluon_gemma_fused_rmsnorm_example()

    try:
        example.run_gemma_fused_rmsnorm_correctness(
            output_dir=tmp_path,
            rows=2,
            hidden=16,
            skip_reason=lambda: "torch.cuda is not available",
        )
    except ValueError as exc:
        assert "--output-dir must be repo-relative" in str(exc)
    else:
        raise AssertionError("expected absolute output directory to be rejected")


def test_gluon_layernorm_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_layernorm_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "layernorm_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-layernorm-local",
            "--rows",
            "2",
            "--hidden",
            "16",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_layernorm_f32_sweep_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_layernorm_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "layernorm_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-layernorm-sweep-local",
            "--sweep",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["case_count"] == 2
    assert payload["skipped_cases"] == 2
    assert [case["provenance"] for case in payload["cases"]] == [
        "existing-smoke",
        "DeepSeek-V4-Flash config hidden_size",
    ]


def test_gluon_rope_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_rope_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "rope_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-rope-local",
            "--batch",
            "1",
            "--seq",
            "2",
            "--head-dim",
            "8",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_silu_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_silu_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "silu_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-silu-local",
            "--n",
            "32",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_gelu_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_gelu_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "gelu_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-gelu-local",
            "--n",
            "32",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_gated_silu_f32_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_gated_silu_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "gated_silu_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-gated-silu-local",
            "--n",
            "32",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_rmsnorm_f32_rejects_absolute_output_dir(capsys):
    example = _load_gluon_rmsnorm_example()

    code = example.main(["--output-dir", "/tmp/private-output", "--sweep"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_layernorm_f32_rejects_absolute_output_dir(capsys):
    example = _load_gluon_layernorm_example()

    code = example.main(["--output-dir", "/tmp/private-output", "--sweep"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_rope_f32_rejects_absolute_output_dir(capsys):
    example = _load_gluon_rope_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_silu_f32_rejects_absolute_output_dir(capsys):
    example = _load_gluon_silu_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_gelu_f32_rejects_absolute_output_dir(capsys):
    example = _load_gluon_gelu_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_gated_silu_f32_rejects_absolute_output_dir(capsys):
    example = _load_gluon_gated_silu_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_moe_expert_affine_example_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_moe_expert_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "moe_expert_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-moe-expert-local",
            "--n",
            "4096",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_moe_expert_affine_rejects_absolute_output_dir(capsys):
    example = _load_gluon_moe_expert_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_benchmark_cli_skips_all_kernels_with_structured_json(
    tmp_path,
    capsys,
    monkeypatch,
):
    benchmark = _load_gluon_benchmark_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        benchmark,
        "h200_skip_reason",
        lambda: "torch.cuda is not available",
    )

    code = benchmark.main(
        [
            "--output-dir",
            "tmp/gluon-performance-local",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["schema_version"] == 1
    assert payload["status"] == "skipped"
    assert payload["machine_class"] == "H200"
    assert payload["command"] == (
        "examples/cuda/gluon_benchmark.py --output-dir "
        "tmp/gluon-performance-local --warmup 1 --iterations 2"
    )
    assert payload["non_claims"] == [
        "microbenchmark timings are not serving throughput",
        "skipped runs are not H200 performance evidence",
        "results cover only the listed shapes and dtypes",
    ]
    assert [entry["kernel_name"] for entry in payload["benchmarks"]] == [
        "gemm_f32",
        "gemm_tensor_core_f16_f32",
        "gemm_tensor_core_tiled_f16_f32",
        "flashattention_fwd_f32",
    ]
    for entry in payload["benchmarks"]:
        assert entry["status"] == "skipped"
        assert entry["reason"] == "torch.cuda is not available"
        assert entry["machine_class"] == "H200"
        assert entry["correctness"] == {"measured": False, "status": "skipped"}
        assert entry["timing"]["measured"] is False
        assert entry["timing"]["warmup"] == 1
        assert entry["timing"]["iterations"] == 2
        assert not Path(entry["artifact"]["source_path"]).is_absolute()
        assert not Path(entry["artifact"]["manifest_path"]).is_absolute()
        assert str(tmp_path) not in json.dumps(payload)


def test_gluon_benchmark_cli_require_cuda_returns_nonzero_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    benchmark = _load_gluon_benchmark_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "h200_skip_reason", lambda: "expected H200 GPU")

    code = benchmark.main(["--output-dir", "tmp/gluon-performance-local", "--require-cuda"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "skipped"
    assert {entry["reason"] for entry in payload["benchmarks"]} == {"expected H200 GPU"}


def test_gluon_benchmark_rejects_absolute_output_dir(capsys):
    benchmark = _load_gluon_benchmark_example()

    code = benchmark.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)


def test_gluon_benchmark_cli_parse_errors_are_json(capsys):
    benchmark = _load_gluon_benchmark_example()

    code = benchmark.main(["--not-a-real-flag"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert captured.err == ""
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert "unrecognized arguments: --not-a-real-flag" in payload["error"]


def test_cuda_kernel_compiler_compiles_host_schedule_task_body(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_host_schedule(task_body, arch, cache_root=None, nvcc="nvcc"):
        seen["task_body"] = task_body
        seen["arch"] = arch
        seen["cache_root"] = cache_root
        seen["nvcc"] = nvcc
        return SimpleNamespace(
            cache_key="fake-key",
            cache_hit=False,
            source_path=tmp_path / "generated_host_wrapper.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"fake-ptx",
            entry_name="pto_kernel_vector_add",
            persistent_entry_name="pto_task_vector_add",
            arch=arch,
            source_kind="task-body-wrapper",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_host_schedule", fake_compile_cuda_host_schedule)
    source_path = tmp_path / "vector_add.pto.cu"
    source_path.write_text("out[i] = a[i] + b[i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_host_schedule(
        str(source_path),
        task_name="vector_add",
        arch="compute_80",
        cache_root=tmp_path / "cache",
        nvcc="nvcc-test",
    )

    assert artifact.ptx == b"fake-ptx"
    assert seen["task_body"].name == "vector_add"
    assert seen["task_body"].body == "out[i] = a[i] + b[i];\n"
    assert seen["task_body"].context_type == "PtoTaskContext"
    assert seen["arch"] == "compute_80"
    assert seen["cache_root"] == tmp_path / "cache"
    assert seen["nvcc"] == "nvcc-test"


def test_cuda_kernel_compiler_compiles_persistent_device_task_sources(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        seen["arch"] = arch
        seen["cache_root"] = cache_root
        seen["nvcc"] = nvcc
        return SimpleNamespace(
            cache_key="persistent-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"persistent-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    add_src = tmp_path / "add.pto.cu"
    mul_src = tmp_path / "mul.pto.cu"
    add_src.write_text("task->out[i] = task->a[i] + task->b[i];\n")
    mul_src.write_text("task->out[i] = task->a[i] * task->b[i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_persistent_device(
        [
            {"func_id": 2, "task_name": "mul_f32", "source_path": str(mul_src)},
            {"func_id": 1, "task_name": "add_f32", "source_path": str(add_src)},
        ],
        arch="compute_90",
        cache_root=tmp_path / "cache",
        nvcc="/usr/local/cuda/bin/nvcc",
    )

    assert artifact.ptx == b"persistent-ptx"
    assert seen["arch"] == "compute_90"
    assert seen["cache_root"] == tmp_path / "cache"
    assert seen["nvcc"] == "/usr/local/cuda/bin/nvcc"
    assert [task.func_id for task in seen["task_functions"]] == [2, 1]
    assert [task.name for task in seen["task_functions"]] == ["mul_f32", "add_f32"]
    assert [task.body for task in seen["task_functions"]] == [
        "task->out[i] = task->a[i] * task->b[i];\n",
        "task->out[i] = task->a[i] + task->b[i];\n",
    ]
    assert [task.threading for task in seen["task_functions"]] == ["element", "element"]


def test_cuda_kernel_compiler_accepts_persistent_device_source_alias(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        return SimpleNamespace(
            cache_key="persistent-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"persistent-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    add_src = tmp_path / "add.pto.cu"
    add_src.write_text("task->out[i] = task->a[i] + task->b[i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_persistent_device(
        [{"func_id": 1, "task_name": "add_f32", "source": str(add_src)}],
        arch="compute_80",
    )

    assert artifact.ptx == b"persistent-ptx"
    assert seen["task_functions"][0].body == "task->out[i] = task->a[i] + task->b[i];\n"


def test_cuda_kernel_compiler_preserves_persistent_task_threading(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        return SimpleNamespace(
            cache_key="persistent-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"persistent-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    src = tmp_path / "wmma.pto.cu"
    src.write_text("if (threadIdx.x < 32) { task->out[threadIdx.x] = 1.0f; }\n")

    KernelCompiler(platform="cuda").compile_cuda_persistent_device(
        [
            {
                "func_id": 10,
                "task_name": "wmma_tile_f32",
                "source_path": str(src),
                "threading": "block",
            }
        ],
        arch="compute_90",
    )

    assert seen["task_functions"][0].threading == "block"


def test_cuda_kernel_compiler_compiles_persistent_device_task_bodies(tmp_path, monkeypatch):
    seen = {}

    def fake_compile_cuda_persistent_device(task_functions, arch, cache_root=None, nvcc="nvcc"):
        seen["task_functions"] = task_functions
        seen["arch"] = arch
        seen["cache_root"] = cache_root
        seen["nvcc"] = nvcc
        return SimpleNamespace(
            cache_key="task-body-key",
            cache_hit=False,
            source_path=tmp_path / "generated_dispatch.cu",
            ptx_path=tmp_path / "pto_callable.ptx",
            manifest_path=tmp_path / "pto_callable.json",
            ptx=b"task-body-ptx",
            entry_name="pto_persistent_dag_f32_executor",
            arch=arch,
            source_kind="generated-dispatch",
        )

    monkeypatch.setattr(kernel_compiler, "compile_cuda_persistent_device", fake_compile_cuda_persistent_device)
    add_src = tmp_path / "add.pto.cu"
    add_src.write_text("ctx->task->out[ctx->i] = ctx->task->a[ctx->i] + ctx->task->b[ctx->i];\n")

    compiler = KernelCompiler(platform="cuda")
    artifact = compiler.compile_cuda_persistent_device(
        [
            {
                "func_id": 1,
                "task_name": "add_f32",
                "source_path": str(add_src),
                "body_style": "task_body",
                "context_definition": """
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""".strip(),
            }
        ],
        arch="compute_90",
        cache_root=tmp_path / "cache",
        nvcc="nvcc-test",
    )

    assert artifact.ptx == b"task-body-ptx"
    assert seen["arch"] == "compute_90"
    assert seen["cache_root"] == tmp_path / "cache"
    assert seen["nvcc"] == "nvcc-test"
    assert seen["task_functions"][0].func_id == 1
    assert seen["task_functions"][0].task_body.name == "add_f32"
    assert seen["task_functions"][0].task_body.context_type == "PtoTaskContext"
    assert "PtoCudaPersistentDagTask" in seen["task_functions"][0].task_body.context_definition
    assert seen["task_functions"][0].task_body.body == (
        "ctx->task->out[ctx->i] = ctx->task->a[ctx->i] + ctx->task->b[ctx->i];\n"
    )
