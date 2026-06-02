from __future__ import annotations

from pathlib import Path

from .baseline_probes import validate_paper_baseline_probes
from .baseline_runs import (
    validate_paper_baseline_execution_attempts,
    validate_paper_baseline_run_readiness,
    validate_paper_baseline_runs,
)
from .basic import (
    validate_benchmarks,
    validate_methods,
    validate_paper_baselines,
)
from .common import ROOT, load_json
from .environments import (
    validate_paper_baseline_environment_attempts,
    validate_paper_baseline_environment_plans,
)
from .goal_progress import validate_goal_progress
from .matrix import validate_paper_evaluation_matrix
from .paper_readiness import (
    validate_paper_readiness_audit,
    validate_paper_readiness_work_queue,
)
from .persistent_scheduler_coverage import (
    validate_persistent_scheduler_coverage,
)
from .results import validate_capture_imports, validate_results
from .scene_builder_coverage import validate_scene_builder_coverage
from .serving import (
    validate_serving_command_plan,
    validate_serving_workload_run_refs,
    validate_serving_workloads,
)
from .tensor_workload_coverage import validate_tensor_workload_coverage


def validate_viewer_data(root: Path = ROOT) -> None:
    benchmarks = load_json(root, "benchmarks.json")
    methods = load_json(root, "methods.json")
    scene_builder_coverage = load_json(root, "scene_builder_coverage.json")
    persistent_scheduler_coverage = load_json(
        root, "persistent_scheduler_coverage.json"
    )
    tensor_workload_coverage = load_json(root, "tensor_workload_coverage.json")
    paper_baselines = load_json(root, "paper_baselines.json")
    paper_baseline_runs = load_json(root, "paper_baseline_runs.json")
    paper_baseline_probes = load_json(root, "paper_baseline_probes.json")
    paper_baseline_environment_plans = load_json(
        root, "paper_baseline_environment_plans.json"
    )
    paper_baseline_environment_attempts = load_json(
        root, "paper_baseline_environment_attempts.json"
    )
    paper_baseline_run_readiness = load_json(
        root, "paper_baseline_run_readiness.json"
    )
    paper_baseline_execution_attempts = load_json(
        root, "paper_baseline_execution_attempts.json"
    )
    serving_command_plan = load_json(root, "serving_command_plan.json")
    serving_workloads = load_json(root, "serving_workloads.json")
    paper_evaluation_matrix = load_json(root, "paper_evaluation_matrix.json")
    paper_readiness_audit = load_json(root, "paper_readiness_audit.json")
    paper_readiness_work_queue = load_json(
        root, "paper_readiness_work_queue.json"
    )
    goal_progress = load_json(root, "goal_progress.json")
    capture_imports = load_json(root, "capture_imports.json")
    results = load_json(root, "results.json")
    benchmark_ids = validate_benchmarks(benchmarks, root)
    method_ids = validate_methods(methods, root)
    validate_scene_builder_coverage(scene_builder_coverage, root)
    validate_persistent_scheduler_coverage(
        persistent_scheduler_coverage,
        root,
    )
    baseline_ids = validate_paper_baselines(paper_baselines)
    serving_workload_ids = validate_serving_workloads(serving_workloads, root)
    validate_paper_baseline_probes(paper_baseline_probes, baseline_ids, root)
    validate_paper_baseline_environment_plans(
        paper_baseline_environment_plans,
        baseline_ids,
        root,
    )
    environment_plan_ids = {
        record["id"]
        for record in paper_baseline_environment_plans[
            "paper_baseline_environment_plans"
        ]
    }
    validate_paper_baseline_environment_attempts(
        paper_baseline_environment_attempts,
        baseline_ids,
        environment_plan_ids,
        root,
    )
    run_ids = {
        record["id"]
        for record in paper_baseline_runs["paper_baseline_runs"]
    }
    planned_run_ids = {
        record["id"]
        for record in paper_baseline_runs["paper_baseline_runs"]
        if record.get("status", "planned_not_run") != "imported_to_viewer"
    }
    validate_paper_baseline_run_readiness(
        paper_baseline_run_readiness,
        run_ids,
        planned_run_ids,
        baseline_ids,
        root,
    )
    validate_paper_baseline_execution_attempts(
        paper_baseline_execution_attempts,
        baseline_ids,
        run_ids,
        root,
    )
    validate_serving_command_plan(
        serving_command_plan,
        paper_baseline_runs,
        serving_workloads,
    )
    validate_capture_imports(capture_imports, benchmark_ids, method_ids)
    validate_results(results, benchmark_ids, method_ids, root)
    validate_tensor_workload_coverage(
        tensor_workload_coverage,
        results,
        method_ids,
        root,
    )
    paper_evaluation_ids = validate_paper_evaluation_matrix(
        paper_evaluation_matrix,
        benchmark_ids,
        method_ids,
        baseline_ids,
        serving_workload_ids,
        results,
        root,
    )
    validate_paper_baseline_runs(
        paper_baseline_runs,
        baseline_ids,
        paper_evaluation_ids,
        serving_workload_ids,
        root,
    )
    validate_paper_readiness_audit(
        paper_readiness_audit,
        matrix=paper_evaluation_matrix,
        runs=paper_baseline_runs,
        probes=paper_baseline_probes,
        run_readiness=paper_baseline_run_readiness,
        execution_attempts=paper_baseline_execution_attempts,
        results=results,
        serving_workload_ids=serving_workload_ids,
    )
    validate_paper_readiness_work_queue(
        paper_readiness_work_queue,
        audit=paper_readiness_audit,
        serving_workload_ids=serving_workload_ids,
    )
    validate_goal_progress(
        goal_progress,
        audit=paper_readiness_audit,
        work_queue=paper_readiness_work_queue,
        matrix=paper_evaluation_matrix,
        baselines=paper_baselines,
    )
    validate_serving_workload_run_refs(
        serving_workloads,
        {
            record["id"]
            for record in paper_baseline_runs["paper_baseline_runs"]
        },
    )


def main() -> None:
    validate_viewer_data()
    print("benchmark viewer data validation passed")


if __name__ == "__main__":
    main()
