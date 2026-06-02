"""Active vocabulary-window policy for resource-backed Qwen logits."""

from __future__ import annotations

import copy
from typing import Any


def apply_logits_active_cols_override(
    descriptors: list[dict[str, Any]],
    active_cols: str | int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = normalize_logits_active_cols(active_cols)
    if policy["mode"] == "descriptor_default":
        return descriptors, policy

    updated = []
    applied_values = []
    for descriptor in descriptors:
        if descriptor.get("callable") != "qwen_logits":
            updated.append(descriptor)
            continue
        item = copy.deepcopy(descriptor)
        shape = dict(item.get("task_shape_fields", {}))
        if policy["mode"] == "full_descriptor_cols":
            if "cols" not in shape:
                raise ValueError("qwen_logits descriptor lacks cols for full override")
            active = int(shape["cols"])
        else:
            active = int(policy["requested_active_cols"])
        shape["scalar1"] = active
        item["task_shape_fields"] = shape
        updated.append(item)
        applied_values.append(active)

    return updated, {**policy, "applied_scalar1_values": applied_values}


def normalize_logits_active_cols(active_cols: str | int | None) -> dict[str, Any]:
    if active_cols is None:
        return {
            "mode": "descriptor_default",
            "requested_active_cols": None,
        }
    text = str(active_cols).strip()
    if text == "":
        return {
            "mode": "descriptor_default",
            "requested_active_cols": None,
        }
    if text == "full":
        return {
            "mode": "full_descriptor_cols",
            "requested_active_cols": "full",
        }
    requested = int(text)
    if requested <= 0:
        raise ValueError("resource-backed logits active cols must be positive")
    return {
        "mode": "explicit_active_cols",
        "requested_active_cols": requested,
    }
