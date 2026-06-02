"""Fallback PTX resources for CUDA benchmark flows."""

from __future__ import annotations

from pathlib import Path

_RESOURCE_DIR = Path(__file__).resolve().parent / "fallback_ptx"


def _read_ptx(name: str) -> bytes:
    return (_RESOURCE_DIR / name).read_bytes()


SLOW_VECTOR_ADD_PTX = _read_ptx("slow_vector_add_sm80.ptx")
