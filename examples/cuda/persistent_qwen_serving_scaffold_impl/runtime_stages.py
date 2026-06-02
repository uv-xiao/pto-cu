"""Runtime and review stage construction for the Qwen scaffold."""

from __future__ import annotations

from typing import Any

from .stages import stage
def runtime_stages(artifacts: dict[str, dict[str, Any]], flags: dict[str, Any]) -> list[dict[str, Any]]:
    lifecycle_plan = artifacts["lifecycle_plan"]
    kv_cache = artifacts["kv_cache_binding"]
    return [
        stage(
            stage_id="kv_cache_lifecycle",
            title="KV-cache allocation and token-position lifecycle",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="partial" if flags["kv_cache_binding_ready"] and kv_cache.get("mode") == "dry_run_pointer_lifecycle"
            else "pass" if flags["kv_cache_binding_ready"] else "missing"
            if not lifecycle_plan.get("workload_plans") else "partial",
            evidence="examples/cuda/qwen_kv_cache_binding.py",
            next_action=(
                "Run the KV-cache owner in cuda_live mode from the decode-loop "
                "runner and consume c/d from Qwen attention kernels."
            ),
        ),
        stage(
            stage_id="decode_loop_runner",
            title="Decode loop runner",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="partial" if flags["decode_loop_runner_ready"] else "missing",
            evidence="examples/cuda/qwen_decode_loop_runner.py",
            next_action=(
                "Replace the dry-run submission plan with cuda_live resource "
                "owners and numerically correct Qwen kernels."
            ),
        ),
        stage(
            stage_id="qwen_persistent_task_bodies",
            title="Qwen persistent task body source generation",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="partial" if flags["task_bodies_ready"] else "missing",
            evidence="examples/cuda/qwen_persistent_task_bodies.py",
            next_action=(
                "Use the controlled proxy oracle only as deterministic scaffold "
                "evidence. Replace review-oriented task bodies with numerically "
                "correct Qwen kernels, then validate them in cuda_live decode-loop execution."
            ),
        ),
        stage(
            stage_id="viewer_result_import",
            title="Full-serving viewer result import",
            owner="paper_evaluation",
            required_for_full_serving=True,
            status="missing",
            evidence="evaluations/nvidia/benchmark-viewer/data/results.json",
            next_action=(
                "Import Qwen/Qwen3-8B PTO rows with serving_coverage="
                "full_serving or full_serving_latency_caveat."
            ),
        ),
    ]

