from __future__ import annotations

from .common import *  # noqa: F403


def check_evidence_refs(records: list[dict], owner: str) -> None:
    for record in records:
        for ref in record.get("evidence_refs", []):
            path = ROOT / ref["path"]
            text = logical_text(path)
            for symbol in ref.get("symbols", []):
                if symbol not in text:
                    fail(f"{owner} {record['id']} missing symbol {symbol} in {ref['path']}")


def check_viewer_data() -> None:
    require_file(VIEWER_ROOT / "index.html")
    require_file(VIEWER_ROOT / "styles.css")
    require_file(VIEWER_ROOT / "viewer.js")
    viewer_files = [VIEWER_ROOT / "viewer.js", *sorted((VIEWER_ROOT / "viewer").glob("*.js"))]
    viewer_script = "\n".join(path.read_text(encoding="utf-8") for path in viewer_files)
    for needle in [
        "run.inputs.shape",
        "run.inputs.dtype",
        "run.inputs.repeat_policy",
        "method.category",
        "method.launch_model",
        "paperBaselineRuns",
        "paper_baseline_runs",
        "paperBaselineProbes",
        "paper_baseline_probes",
        "paperBaselineEnvironmentPlans",
        "paper_baseline_environment_plans",
        "paperBaselineRunReadiness",
        "paper_baseline_run_readiness",
        "servingWorkloads",
        "serving_workloads",
        "latest_artifact_root",
        "latest_machine_status",
        "paperEvaluation",
        "paper_evaluation_matrix",
        "paperReadinessAudit",
        "paper_readiness_audit",
        "paperReadinessWorkQueue",
        "paper_readiness_work_queue",
        "item.promotion_gate",
        "paperBaselineRunTitle",
        "Work Item",
        "Execution Attempt",
        "Promotion Gate",
        "goalProgress",
        "goal_progress",
        "paper_baseline_run_readiness_statuses",
        "ready_for_paper_claim",
        "result_records",
        "raw_artifact",
        "correctness",
    ]:
        if needle not in viewer_script:
            fail(f"viewer.js does not render contract field: {needle}")

    benchmarks = load_json(VIEWER_ROOT / "data" / "benchmarks.json")
    methods = load_json(VIEWER_ROOT / "data" / "methods.json")
    capture_imports = load_json(VIEWER_ROOT / "data" / "capture_imports.json")
    paper_baselines = load_json(VIEWER_ROOT / "data" / "paper_baselines.json")
    paper_baseline_runs = load_json(
        VIEWER_ROOT / "data" / "paper_baseline_runs.json"
    )
    paper_baseline_probes = load_json(
        VIEWER_ROOT / "data" / "paper_baseline_probes.json"
    )
    paper_baseline_run_readiness = load_json(
        VIEWER_ROOT / "data" / "paper_baseline_run_readiness.json"
    )
    serving_workloads = load_json(VIEWER_ROOT / "data" / "serving_workloads.json")
    paper_evaluation = load_json(
        VIEWER_ROOT / "data" / "paper_evaluation_matrix.json"
    )
    paper_readiness_audit = load_json(
        VIEWER_ROOT / "data" / "paper_readiness_audit.json"
    )
    goal_progress = load_json(VIEWER_ROOT / "data" / "goal_progress.json")
    results = load_json(VIEWER_ROOT / "data" / "results.json")

    benchmark_ids = {item["id"] for item in benchmarks.get("benchmarks", [])}
    required_benchmarks = {
        "host_schedule_vector_ops",
        "graph_layered_cross",
        "tensor_core_tile",
        "llm_serving_decode",
    }
    if not required_benchmarks <= benchmark_ids:
        fail(f"missing benchmark ids: {sorted(required_benchmarks - benchmark_ids)}")
    for benchmark in benchmarks["benchmarks"]:
        for key in ("description", "math", "code"):
            if not benchmark.get(key):
                fail(f"benchmark {benchmark['id']} has empty {key}")
        if not benchmark.get("run", {}).get("command"):
            fail(f"benchmark {benchmark['id']} has no run command")
    check_evidence_refs(benchmarks["benchmarks"], "benchmark")

    method_ids = {item["id"] for item in methods.get("methods", [])}
    required_methods = {
        "pto_host_schedule",
        "pto_persistent_device",
        "direct_runtime",
        "direct_driver",
        "direct_driver_graph",
        "cublas_sgemm_graph",
        "triton",
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    }
    if not required_methods <= method_ids:
        fail(f"missing method ids: {sorted(required_methods - method_ids)}")
    check_evidence_refs(methods["methods"], "method")

    import_rules = capture_imports.get("capture_imports", [])
    import_baselines = {item["baseline"] for item in import_rules}
    required_import_baselines = {
        "pto_host_schedule",
        "direct_runtime",
        "direct_driver",
        "direct_driver_graph",
        "pto_persistent_dag_graph_layered_cross",
        "cublas_sgemm_graph",
    }
    if not required_import_baselines <= import_baselines:
        missing = sorted(required_import_baselines - import_baselines)
        fail(f"missing capture import baselines: {missing}")

    paper_baseline_ids = {
        item["id"] for item in paper_baselines.get("paper_baselines", [])
    }
    required_paper_baselines = {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    }
    if not required_paper_baselines <= paper_baseline_ids:
        missing = sorted(required_paper_baselines - paper_baseline_ids)
        fail(f"missing paper baseline ids: {missing}")
    for baseline in paper_baselines["paper_baselines"]:
        for key in ("status", "paper_role", "next_action"):
            if not baseline.get(key):
                fail(f"paper baseline {baseline['id']} has empty {key}")
        source = baseline.get("source", {})
        if not source.get("upstream_url") or not source.get("local_tmp_path"):
            fail(f"paper baseline {baseline['id']} has incomplete source")
        if baseline["status"] != "source_cloned_for_survey":
            fail(f"paper baseline {baseline['id']} is not cloned for survey")
        if len(source.get("commit", "")) != 40:
            fail(f"paper baseline {baseline['id']} has no pinned commit")

    serving_workload_ids = {
        item["id"] for item in serving_workloads.get("serving_workloads", [])
    }
    required_serving_workloads = {"mpk_offline_decode", "vdcores_offline_decode"}
    if not required_serving_workloads <= serving_workload_ids:
        missing = sorted(required_serving_workloads - serving_workload_ids)
        fail(f"missing serving workload ids: {missing}")
    check_evidence_refs(serving_workloads["serving_workloads"], "serving workload")

    run_ids = {
        item["id"] for item in paper_baseline_runs.get("paper_baseline_runs", [])
    }
    required_run_ids = {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_qwen3_8b_decode_preflight",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_tile_kernel",
        "thunderkittens_full_sweep",
        "thunderkittens_decode_attention_tile",
    }
    if not required_run_ids <= run_ids:
        missing = sorted(required_run_ids - run_ids)
        fail(f"missing paper baseline run ids: {missing}")

    probe_baseline_ids = {
        item["paper_baseline_id"]
        for item in paper_baseline_probes.get("paper_baseline_probes", [])
    }
    if not required_paper_baselines <= probe_baseline_ids:
        missing = sorted(required_paper_baselines - probe_baseline_ids)
        fail(f"missing paper baseline probe coverage: {missing}")
    readiness_run_ids = {
        item["paper_baseline_run_id"]
        for item in paper_baseline_run_readiness.get(
            "paper_baseline_run_readiness", []
        )
    }
    required_run_readiness = {
        item["id"]
        for item in paper_baseline_runs.get("paper_baseline_runs", [])
        if item.get("status", "planned_not_run") != "imported_to_viewer"
    }
    if not required_run_readiness <= readiness_run_ids:
        missing = sorted(required_run_readiness - readiness_run_ids)
        fail(f"missing paper baseline run readiness coverage: {missing}")

    matrix_ids = {
        item["id"] for item in paper_evaluation.get("paper_evaluation_matrix", [])
    }
    required_matrix_ids = {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    }
    if not required_matrix_ids <= matrix_ids:
        missing = sorted(required_matrix_ids - matrix_ids)
        fail(f"missing paper evaluation matrix ids: {missing}")
    if paper_readiness_audit.get("overall_status") != "not_paper_ready":
        fail("paper readiness audit must not claim paper-ready status yet")
    claim_audits = paper_readiness_audit.get("claim_audits", [])
    if not isinstance(claim_audits, list) or not claim_audits:
        fail("paper readiness audit has no claim audits")
    if not any(item.get("blockers") for item in claim_audits):
        fail("paper readiness audit must expose blockers")
    if goal_progress.get("overall_status") != "in_progress":
        fail("goal progress must remain in_progress until paper results finish")
    progress_by_id = {
        item.get("id"): item
        for item in goal_progress.get("acceptance_criteria", [])
        if isinstance(item, dict)
    }
    paper_results = progress_by_id.get("paper_grade_results")
    if not paper_results:
        fail("goal progress missing paper_grade_results criterion")
    backend_closure = progress_by_id.get("backend_implementation_closure")
    if not backend_closure:
        fail("goal progress missing backend_implementation_closure criterion")
    if backend_closure.get("status") not in {"met", "in_progress"}:
        fail("backend implementation closure has invalid status")
    if backend_closure.get("status") == "in_progress" and not backend_closure.get("gaps"):
        fail("backend implementation closure must expose status gaps")
    if paper_results.get("status") != "in_progress":
        fail("paper_grade_results criterion must remain in_progress")
    if paper_results.get("paper_readiness_status") != "not_paper_ready":
        fail("paper_grade_results must reflect current audit status")
    matrix_baselines = {
        baseline_id
        for item in paper_evaluation["paper_evaluation_matrix"]
        for baseline_id in item.get("paper_baseline_ids", [])
    }
    if not required_paper_baselines <= matrix_baselines:
        missing = sorted(required_paper_baselines - matrix_baselines)
        fail(f"paper evaluation matrix missing baseline ids: {missing}")

    snapshot = results.get("snapshot", {})
    if snapshot.get("commit") != "743709f3":
        fail("viewer snapshot commit must be 743709f3")
    if snapshot.get("full_capture", {}).get("samples") != 1350:
        fail("viewer full capture sample count must be 1350")
    if snapshot.get("compact_capture", {}).get("samples") != 108:
        fail("viewer compact capture sample count must be 108")
