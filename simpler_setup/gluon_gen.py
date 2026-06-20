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
_SUPPORTED_KERNELS = {"gemm_f32"}


@dataclass(frozen=True)
class GluonKernelArtifact:
    kernel_name: str
    compiler_role: str
    arch: str
    source_path: Path
    manifest_path: Path
    source_sha256: str
    tile_shape: tuple[int, int, int]


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

    source = _render_gemm_f32_source()
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
