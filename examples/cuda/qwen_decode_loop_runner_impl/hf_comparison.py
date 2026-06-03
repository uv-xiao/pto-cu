"""Compare resource-backed Qwen logits with a Hugging Face reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_hf_token_comparison(
    pto_payload: dict[str, Any],
    hf_payload: dict[str, Any],
    *,
    pto_artifact: str,
    hf_reference: str,
) -> dict[str, Any]:
    workload = selected_workload(pto_payload)
    repeat = selected_repeat(workload)
    logits_summary = repeat.get("logits_summary", {})
    topk = logits_summary.get("topk", [])
    pto_top = topk[0] if topk else {}
    pto_top_token_id = int(pto_top.get("token_id", -1))
    hf_top_token_id = int(hf_payload.get("top_token_id", -1))
    token_match = (
        pto_top_token_id >= 0
        and hf_top_token_id >= 0
        and pto_top_token_id == hf_top_token_id
    )
    prompt_prefill = workload.get("prompt_prefill", {})
    prompt_prefill_executed = (
        isinstance(prompt_prefill, dict)
        and prompt_prefill.get("status") == "prompt_prefill_executed"
    )
    model_equivalent_ready = prompt_prefill_executed
    blocking_reasons = blocking_reason_list(
        token_match=token_match,
        model_equivalent_ready=model_equivalent_ready,
    )
    diagnostic_reference = logits_summary.get("diagnostic_reference", {})
    return {
        "schema_version": 1,
        "status": "pass" if token_match and model_equivalent_ready else "fail",
        "scope": "diagnostic_full_prefix_vs_hf_reference_top_token",
        "comparison_scope": (
            "model_equivalent_decode"
            if model_equivalent_ready
            else "diagnostic_decode_without_prompt_prefill"
        ),
        "model_equivalent_ready": model_equivalent_ready,
        "blocking_reasons": blocking_reasons,
        "pto_artifact": pto_artifact,
        "hf_reference": hf_reference,
        "workload_id": str(workload.get("workload_id", "")),
        "decode_position": int(repeat.get("decode_position", -1)),
        "hf_logits_position": str(hf_payload.get("logits_position", "")),
        "token_match": token_match,
        "pto_top_token_id": pto_top_token_id,
        "pto_top_logit": float(pto_top.get("logit", 0.0)),
        "hf_top_token_id": hf_top_token_id,
        "hf_top_logit": float(hf_payload.get("top_logit", 0.0)),
        "pto_logits_diagnostic_reference": diagnostic_reference,
        "pto_prompt_prefill": prompt_prefill,
        "pto_scheduler_counters": workload.get("scheduler_counters", {}),
        "hf_status": hf_payload.get("status"),
    }


def selected_workload(pto_payload: dict[str, Any]) -> dict[str, Any]:
    execution = pto_payload.get("resource_backed_execution", {})
    workloads = execution.get("workloads", [])
    if not isinstance(workloads, list) or not workloads:
        return {}
    for workload in workloads:
        if isinstance(workload, dict) and workload.get("status") == "pass":
            return workload
    return workloads[0] if isinstance(workloads[0], dict) else {}


def selected_repeat(workload: dict[str, Any]) -> dict[str, Any]:
    repeats = workload.get("repeat_results", [])
    if isinstance(repeats, list) and repeats:
        last = repeats[-1]
        if isinstance(last, dict):
            return last
    return {}


def blocking_reason_list(
    *,
    token_match: bool,
    model_equivalent_ready: bool,
) -> list[str]:
    reasons = []
    if not model_equivalent_ready:
        reasons.append("prompt_prefill_not_executed")
    if not token_match:
        reasons.append("token_mismatch")
    return reasons


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
