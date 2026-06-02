"""Fallback PTX resources used when nvcc is unavailable."""

from __future__ import annotations

from pathlib import Path

_RESOURCE_DIR = Path(__file__).resolve().parent / "fallback_ptx"


def _read_ptx(name: str) -> bytes:
    return (_RESOURCE_DIR / name).read_bytes()


FALLBACK_PERSISTENT_VECTOR_ADD_PTX = _read_ptx("persistent_vector_add_sm80.ptx")
FALLBACK_PERSISTENT_QUEUE_VECTOR_ADD_PTX = _read_ptx("persistent_queue_vector_add_sm80.ptx")
FALLBACK_PERSISTENT_DAG_F32_PTX = _read_ptx("persistent_dag_f32_sm80.ptx")
