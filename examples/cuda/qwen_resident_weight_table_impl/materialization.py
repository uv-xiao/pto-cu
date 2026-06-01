"""Bridge a live pointer table into persistent task materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_resident_weight_table_impl.common import ROOT, load_module


def materialize_with_pointer_table(
    *,
    weight_args_json: Path | None,
    weight_binding_json: Path | None,
    pointer_table: dict[str, Any],
) -> dict[str, Any]:
    module = load_module(
        ROOT / "examples" / "cuda" / "qwen_persistent_weight_materialization.py",
        "qwen_persistent_weight_materialization_for_resident_table",
    )
    return module.build_materialization_manifest(
        weight_args_json=weight_args_json,
        weight_binding_json=weight_binding_json,
        pointer_table=pointer_table,
    )
