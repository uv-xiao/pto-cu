"""Build paper serving command plan records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from paper_serving_command_plan_impl.commands_kernel import mpk_commands
from paper_serving_command_plan_impl.commands_kernel import thunderkittens_commands
from paper_serving_command_plan_impl.commands_kernel import vdcores_commands
from paper_serving_command_plan_impl.commands_serving import sglang_commands
from paper_serving_command_plan_impl.commands_serving import vllm_commands
from paper_serving_command_plan_impl.errors import fail
from paper_serving_command_plan_impl.paths import DEFAULT_RUNS
from paper_serving_command_plan_impl.paths import DEFAULT_SERVING
from paper_serving_command_plan_impl.paths import ROOT
from paper_serving_command_plan_impl.paths import artifact_dir
from paper_serving_command_plan_impl.plan_ids import plan_id
from paper_serving_command_plan_impl.plan_ids import selected_model
from paper_serving_command_plan_impl.vcs import git_commit


COMMAND_BUILDERS = {
    "mpk": mpk_commands,
    "vdcores": vdcores_commands,
    "vllm": vllm_commands,
    "sglang": sglang_commands,
    "thunderkittens": thunderkittens_commands,
}


def filter_commands_for_run(
    commands: list[dict[str, str]],
    run: dict[str, Any],
) -> list[dict[str, str]]:
    selected = run.get("serving_command_kinds")
    if selected is None:
        return commands
    if not isinstance(selected, list) or not all(
        isinstance(item, str) and item for item in selected
    ):
        fail(f"{run['id']} has invalid serving_command_kinds")
    allowed = set(selected)
    return [command for command in commands if command.get("kind") in allowed]


def build_plan(
    serving: dict[str, Any],
    runs: dict[str, Any],
    *,
    artifact_root: str,
    model_tier: str,
) -> dict[str, Any]:
    workloads = {item["id"]: item for item in serving["serving_workloads"]}
    records = []
    for run in runs["paper_baseline_runs"]:
        if run.get("paper_evaluation_id") != "llm_serving_paper_baselines":
            continue
        baseline_id = run["paper_baseline_id"]
        builder = COMMAND_BUILDERS.get(baseline_id)
        if builder is None:
            continue
        for policy_id in run.get("serving_workload_ids", []):
            policy = workloads.get(policy_id)
            if policy is None:
                fail(f"{run['id']} references unknown serving workload {policy_id}")
            model = selected_model(policy, model_tier)
            out_dir = artifact_dir(artifact_root, baseline_id, policy_id)
            for batch_size in policy["decode_policy"]["batch_sizes"]:
                commands = builder(
                    model=model,
                    policy=policy,
                    batch_size=int(batch_size),
                    out_dir=out_dir,
                )
                commands = filter_commands_for_run(commands, run)
                if not commands:
                    fail(f"{run['id']} selected no serving commands")
                records.append(
                    {
                        "id": plan_id(run["id"], policy_id, int(batch_size)),
                        "paper_baseline_run_id": run["id"],
                        "paper_baseline_id": baseline_id,
                        "serving_workload_id": policy_id,
                        "model_tier": model_tier,
                        "model": model,
                        "batch_size": int(batch_size),
                        "prompt_tokens": policy["prompt_policy"][
                            "target_prompt_tokens"
                        ],
                        "decode_tokens": policy["decode_policy"]["decode_tokens"],
                        "traffic_mode": policy["decode_policy"]["traffic_mode"],
                        "commands": commands,
                    }
                )
    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pto_commit": git_commit(),
            "artifact_root": artifact_root,
            "model_tier": model_tier,
            "source_files": [
                str(DEFAULT_SERVING.relative_to(ROOT)),
                str(DEFAULT_RUNS.relative_to(ROOT)),
            ],
        },
        "serving_command_plans": records,
    }
