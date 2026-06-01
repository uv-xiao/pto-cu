from __future__ import annotations

from pathlib import Path
from typing import Any

from .checks import artifact_checks
from .checks import blocking_gaps
from .checks import check_entrypoints
from .checks import command_checks
from .checks import environment_checks
from .checks import metric_checks
from .checks import probe_checks
from .checks import readiness_status
from .errors import fail
from .io import repo_relative
from .records import environment_plan_by_baseline
from .records import probe_by_baseline
from .records import require_records
from .records import source_by_baseline


def build_run_readiness(
    *,
    baselines: dict[str, Any],
    runs: dict[str, Any],
    probes: dict[str, Any],
    env_plans: dict[str, Any],
    output_root: Path,
    commit: str,
) -> dict[str, Any]:
    sources = source_by_baseline(baselines)
    probes_by_baseline = probe_by_baseline(probes)
    env_plans_by_baseline = environment_plan_by_baseline(env_plans)
    records: list[dict[str, Any]] = []
    for run in require_records(runs, "paper_baseline_runs"):
        run_id = run.get("id")
        baseline_id = run.get("paper_baseline_id")
        if not isinstance(baseline_id, str):
            fail(f"{run_id} missing paper_baseline_id")
        source_root = sources.get(baseline_id)
        if source_root is None:
            fail(f"{run_id} references baseline without local_tmp_path")

        checks = [
            {
                "kind": "source_path",
                "path": repo_relative(source_root),
                "status": "pass" if source_root.is_dir() else "fail",
                "why": "Baseline source checkout used by run_commands.",
            },
            *command_checks(run),
            *check_entrypoints(run, source_root),
            *artifact_checks(run),
            *metric_checks(run),
            *probe_checks(baseline_id, probes_by_baseline),
            *environment_checks(
                run,
                baseline_id,
                source_root,
                env_plans_by_baseline,
            ),
        ]
        gaps = blocking_gaps(checks)
        records.append(
            {
                "id": f"{run_id}_readiness",
                "paper_baseline_run_id": run_id,
                "paper_baseline_id": baseline_id,
                "title": f"{run.get('title', run_id)} Readiness",
                "latest_status": readiness_status(checks),
                "latest_artifact_root": repo_relative(output_root) + "/",
                "checks": checks,
                "blocking_gaps": gaps,
                "next_action": (
                    "Resolve blocking gaps, execute the run_commands, then "
                    "import measured raw JSON through "
                    "paper_baseline_results_update.py."
                ),
            }
        )
    return {
        "schema_version": 1,
        "metadata": {
            "pto_commit": commit,
            "source_files": [
                "docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json",
                "docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json",
                "docs/nvidia-backend/benchmark-viewer/data/paper_baseline_probes.json",
                (
                    "docs/nvidia-backend/benchmark-viewer/data/"
                    "paper_baseline_environment_plans.json"
                ),
            ],
        },
        "paper_baseline_run_readiness": records,
    }
