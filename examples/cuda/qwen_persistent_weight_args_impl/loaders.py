"""Weight-binding artifact loading for Qwen persistent weight arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_WEIGHT_BINDING,
    ROOT,
    load_json,
    load_python_payload,
    repo_relative,
)


def load_or_build_weight_binding(
    weight_binding_json: Path | None,
) -> tuple[dict[str, Any], str]:
    if weight_binding_json is not None:
        return load_json(weight_binding_json), repo_relative(weight_binding_json)
    if DEFAULT_WEIGHT_BINDING.is_file():
        return load_json(DEFAULT_WEIGHT_BINDING), repo_relative(DEFAULT_WEIGHT_BINDING)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py",
        "qwen_cuda_weight_binding",
        "build_weight_binding",
        no_cuda_probe=True,
    )
    return payload, "generated_from_examples/cuda/qwen_cuda_weight_binding.py"


def binding_map(weight_binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = weight_binding.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("weight binding artifact has no bindings list")
    result = {}
    for item in bindings:
        if not isinstance(item, dict) or not isinstance(item.get("tensor"), str):
            raise ValueError("weight binding contains a malformed binding record")
        result[item["tensor"]] = item
    return result
