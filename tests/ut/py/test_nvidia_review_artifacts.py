import ctypes
import json
import importlib.util
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "viewer"
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"


def load_pto_serving_preflight_module():
    script_path = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "pto_serving_preflight.py"
    )
    spec = importlib.util.spec_from_file_location("pto_serving_preflight", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_viewer_collection(path):
    if path.is_dir():
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        base = path
    elif path.suffix == ".json" and path.with_suffix("").is_dir():
        index = json.loads(
            (path.with_suffix("") / "index.json").read_text(encoding="utf-8")
        )
        base = path.with_suffix("")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "collection" not in payload:
            return payload
        index = payload
        base = path.parent
    if "collection" in index:
        record_files = index.get("record_files")
        if record_files is None:
            record_files = json.loads(
                (base / index["record_files_path"]).read_text(encoding="utf-8")
            )
        records = [
            expand_viewer_record(
                base,
                json.loads((base / relpath).read_text(encoding="utf-8")),
            )
            for relpath in record_files
        ]
        return {
            **{
                key: value
                for key, value in index.items()
                if key not in {"collection", "record_files", "record_files_path"}
            },
            index["collection"]: records,
        }
    return index


def expand_viewer_record(base, record):
    payload = dict(record)
    for field in (
        "current_evidence_refs",
        "missing_evidence_details",
        "paper_baseline_run_statuses",
        "paper_baseline_run_readiness_statuses",
        "execution_attempt_statuses",
        "probe_statuses",
        "next_actions",
    ):
        path_key = f"{field}_path"
        relpath = payload.pop(path_key, None)
        if relpath is not None:
            payload[field] = load_viewer_sidecar_list(base / relpath)
    return payload


def load_viewer_sidecar_list(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    return [
        json.loads((path / relpath).read_text(encoding="utf-8"))
        for relpath in index["item_files"]
    ]


def qwen_one_layer_bindings():
    tensors = [
        ("model.embed_tokens.weight", "embedding"),
        ("model.layers.0.input_layernorm.weight", "attention_norms"),
        ("model.layers.0.self_attn.q_proj.weight", "attention_qkv_o"),
        ("model.layers.0.self_attn.k_proj.weight", "attention_qkv_o"),
        ("model.layers.0.self_attn.v_proj.weight", "attention_qkv_o"),
        ("model.layers.0.self_attn.q_norm.weight", "attention_norms"),
        ("model.layers.0.self_attn.k_norm.weight", "attention_norms"),
        ("model.layers.0.self_attn.o_proj.weight", "attention_qkv_o"),
        ("model.layers.0.post_attention_layernorm.weight", "attention_norms"),
        ("model.layers.0.mlp.gate_proj.weight", "mlp_gate_up_down"),
        ("model.layers.0.mlp.up_proj.weight", "mlp_gate_up_down"),
        ("model.layers.0.mlp.down_proj.weight", "mlp_gate_up_down"),
        ("model.norm.weight", "norm_and_logits"),
        ("lm_head.weight", "norm_and_logits"),
    ]
    return [
        {
            "slot_id": slot_id,
            "tensor": tensor,
            "binding_group": binding_group,
            "persistent_arg_role": "readonly_weight_tensor",
            "shape": [4],
            "dtype": "bfloat16",
            "size_bytes": 8,
        }
        for slot_id, (tensor, binding_group) in enumerate(tensors)
    ]


def write_qwen_binding_fixture(path, bindings):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pto_qwen_cuda_weight_binding",
                "status": "binding_plan_ready",
                "tensor_count": len(bindings),
                "planned_binding_count": len(bindings),
                "cuda_probe": {"mode": "full_residency", "status": "pass"},
                "bindings": bindings,
            }
        ),
        encoding="utf-8",
    )


def write_qwen_weight_args_fixture(tmp_path, binding):
    weight_args = tmp_path / "qwen-persistent-weight-args.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_persistent_weight_args.py",
            "--weight-binding-json",
            str(binding),
            "--num-hidden-layers",
            "1",
            "--output-json",
            str(weight_args),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return weight_args


def test_nvidia_review_guard_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/check_nvidia_review_ready.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_benchmark_viewer_schema_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_benchmark_viewer_data.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_probe_machine_status_must_match_raw_artifact(tmp_path):
    script_path = ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    artifact = tmp_path / "tmp" / "probes" / "a100-probe.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "paper_baseline_id": baseline_id,
                        "status": "pass",
                        "blocking_gaps": [],
                    }
                    for baseline_id in [
                        "mpk",
                        "vdcores",
                        "vllm",
                        "sglang",
                        "thunderkittens",
                    ]
                ]
            }
        ),
        encoding="utf-8",
    )
    def probe_record(baseline_id, checks):
        return {
            "id": f"{baseline_id}_source_entrypoints",
            "paper_baseline_id": baseline_id,
            "title": f"{baseline_id} source entrypoints",
            "latest_status": "pass",
            "latest_artifact_root": "tmp/probes/",
            "latest_machine_status": [
                {
                    "gpu": "A100",
                    "status": "pass",
                    "artifact": "tmp/probes/a100-probe.json",
                    "blocking_gaps": [],
                },
                {
                    "gpu": "H200",
                    "status": "pass",
                    "artifact": "tmp/probes/a100-probe.json",
                    "blocking_gaps": [],
                },
            ],
            "checks": checks,
            "next_action": "fixture",
        }

    data = {
        "paper_baseline_probes": [
            probe_record(
                "mpk",
                [
                    {
                        "kind": "python_module",
                        "module": "transformers",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "vdcores",
                [
                    {
                        "kind": "python_module",
                        "module": "transformers",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "vllm",
                [
                    {
                        "kind": "python_module",
                        "module": "vllm",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "sglang",
                [
                    {
                        "kind": "python_module",
                        "module": "sglang",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "thunderkittens",
                [
                    {
                        "kind": "python_module",
                        "module": module_name,
                        "why": "fixture",
                    }
                    for module_name in [
                        "torch",
                        "pybind11",
                        "numpy",
                        "pandas",
                        "matplotlib",
                        "tqdm",
                    ]
                ],
            ),
        ]
    }

    baseline_ids = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    module.validate_paper_baseline_probes(data, baseline_ids, tmp_path)
    data["paper_baseline_probes"][0]["latest_machine_status"][1][
        "status"
    ] = "partial"
    try:
        module.validate_paper_baseline_probes(data, baseline_ids, tmp_path)
    except SystemExit as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("machine status drift was not rejected")


def test_imported_paper_baseline_run_rejects_missing_expected_artifact(tmp_path):
    script_path = ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    (tmp_path / "tmp" / "paper").mkdir(parents=True)
    (tmp_path / "tmp" / "paper" / "present.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    def run_record(run_id, baseline_id, paper_evaluation_id, serving_ids=None):
        return {
            "id": run_id,
            "paper_baseline_id": baseline_id,
            "paper_evaluation_id": paper_evaluation_id,
            "title": run_id,
            "status": "planned_not_run",
            "hardware_targets": ["H200"],
            "serving_workload_ids": serving_ids or [],
            "workload": {
                "model": "fixture",
                "input_policy": "fixture",
                "output_policy": "fixture",
                "batch_or_concurrency": "fixture",
            },
            "setup_commands": ["true"],
            "run_commands": ["true"],
            "expected_artifacts": ["tmp/paper/future.json"],
            "required_metrics": [
                "correctness",
                "raw_artifacts",
                "model_and_prompt_shape",
                "batch_or_concurrency_policy",
            ],
            "import_target": {
                "viewer_file": (
                    "evaluations/nvidia/benchmark-viewer/data/results.json"
                ),
                "result_kind": "paper_baseline_result_record",
                "notes": "fixture",
            },
        }

    imported = run_record(
        "mpk_qwen3_native_vs_persistent",
        "mpk",
        "llm_serving_paper_baselines",
        ["mpk_offline_decode"],
    )
    imported["status"] = "imported_to_viewer"
    imported["expected_artifacts"] = [
        "tmp/paper/present.json",
        "tmp/paper/missing.json",
    ]
    data = {
        "paper_baseline_runs": [
            imported,
            run_record(
                "vdcores_qwen3_8b_decode_preflight",
                "vdcores",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
            ),
            run_record(
                "mpk_persistent_scheduler_trace",
                "mpk",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vdcores_resource_policy_trace",
                "vdcores",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vllm_serving_and_throughput",
                "vllm",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
            ),
            run_record(
                "sglang_serving_and_offline",
                "sglang",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
            ),
            run_record(
                "thunderkittens_tile_kernel",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_full_sweep",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_decode_attention_tile",
                "thunderkittens",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
            ),
        ]
    }

    try:
        module.validate_paper_baseline_runs(
            data,
            {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"},
            {
                "llm_serving_paper_baselines",
                "persistent_device_scheduler_overhead",
                "tensor_core_tile_baselines",
            },
            {"mpk_offline_decode", "vdcores_offline_decode"},
            root=tmp_path,
        )
    except SystemExit as exc:
        assert "expected artifact path missing" in str(exc)
    else:
        raise AssertionError("missing expected artifact was accepted")


def test_llm_serving_paper_baseline_run_requires_shape_and_concurrency(tmp_path):
    script_path = ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def run_record(
        run_id,
        baseline_id,
        paper_evaluation_id,
        serving_ids=None,
        required_metrics=None,
    ):
        return {
            "id": run_id,
            "paper_baseline_id": baseline_id,
            "paper_evaluation_id": paper_evaluation_id,
            "title": run_id,
            "status": "planned_not_run",
            "hardware_targets": ["H200"],
            "serving_workload_ids": serving_ids or [],
            "workload": {
                "model": "fixture",
                "input_policy": "fixture",
                "output_policy": "fixture",
                "batch_or_concurrency": "fixture",
            },
            "setup_commands": ["true"],
            "run_commands": ["true"],
            "expected_artifacts": ["tmp/paper/future.json"],
            "required_metrics": required_metrics
            or ["correctness", "raw_artifacts"],
            "import_target": {
                "viewer_file": (
                    "evaluations/nvidia/benchmark-viewer/data/results.json"
                ),
                "result_kind": "paper_baseline_result_record",
                "notes": "fixture",
            },
        }

    serving_metrics = [
        "correctness",
        "raw_artifacts",
        "model_and_prompt_shape",
        "batch_or_concurrency_policy",
    ]
    data = {
        "paper_baseline_runs": [
            run_record(
                "mpk_qwen3_native_vs_persistent",
                "mpk",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
                ["correctness", "raw_artifacts", "model_and_prompt_shape"],
            ),
            run_record(
                "vdcores_qwen3_8b_decode_preflight",
                "vdcores",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
                serving_metrics,
            ),
            run_record(
                "mpk_persistent_scheduler_trace",
                "mpk",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vdcores_resource_policy_trace",
                "vdcores",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vllm_serving_and_throughput",
                "vllm",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
                serving_metrics,
            ),
            run_record(
                "sglang_serving_and_offline",
                "sglang",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
                serving_metrics,
            ),
            run_record(
                "thunderkittens_tile_kernel",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_full_sweep",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_decode_attention_tile",
                "thunderkittens",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
                serving_metrics,
            ),
        ]
    }

    try:
        module.validate_paper_baseline_runs(
            data,
            {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"},
            {
                "llm_serving_paper_baselines",
                "persistent_device_scheduler_overhead",
                "tensor_core_tile_baselines",
            },
            {"mpk_offline_decode", "vdcores_offline_decode"},
            root=tmp_path,
        )
    except SystemExit as exc:
        assert "missing LLM serving required metrics" in str(exc)
        assert "batch_or_concurrency_policy" in str(exc)
    else:
        raise AssertionError("LLM serving run without concurrency policy was accepted")


def test_pto_serving_preflight_captures_current_full_serving_gap(tmp_path):
    output = tmp_path / "pto-serving-preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    preflight = json.loads(output.read_text(encoding="utf-8"))
    assert preflight["kind"] == "pto_persistent_device_full_serving_preflight"
    assert preflight["status"] == "partial"

    checks = {check["id"]: check for check in preflight["checks"]}
    assert checks["persistent_device_task_descriptor_abi"]["status"] == "pass"
    assert checks["persistent_dag_source_codegen"]["status"] == "pass"
    assert checks["pto_controlled_serving_proxy_imported"]["status"] == "pass"
    assert checks["qwen_serving_lifecycle_scaffold"]["status"] == "pass"
    assert checks["qwen_serving_lifecycle_plan"]["status"] == "pass"
    assert checks["qwen_prompt_accounting"]["status"] == "pass"
    assert checks["qwen_runtime_input_binding"]["status"] == "pass"
    assert checks["qwen_cuda_token_buffer_binding"]["status"] == "pass"
    assert checks["qwen_persistent_decode_args"]["status"] == "pass"
    assert checks["qwen_token_pointer_table_owner"]["status"] == "pass"
    assert checks["qwen_weight_inventory"]["status"] == "pass"
    assert checks["qwen_safetensors_shard_plan"]["status"] == "pass"
    assert checks["qwen_safetensors_shards_present"]["status"] == "pass"
    assert checks["qwen_safetensors_metadata_probe"]["status"] == "pass"
    assert checks["qwen_actual_safetensors_metadata"]["status"] == "pass"
    assert checks["qwen_cuda_weight_binding_plan"]["status"] == "pass"
    assert checks["qwen_persistent_weight_materialization_plan"]["status"] == "pass"
    assert checks["qwen_resident_weight_table_owner"]["status"] == "pass"
    assert checks["qwen_kv_cache_binding"]["status"] == "pass"
    assert checks["qwen_decode_loop_runner"]["status"] == "pass"
    assert checks["qwen_persistent_task_bodies"]["status"] == "pass"
    assert checks["qwen3_8b_full_serving_rows_imported"]["status"] == "fail"
    assert checks["qwen3_8b_full_serving_rows_imported"]["missing_workload_ids"] == [
        "mpk_offline_decode",
        "vdcores_offline_decode",
    ]
    qwen_row_statuses = checks["qwen3_8b_full_serving_rows_imported"][
        "row_statuses"
    ]
    assert qwen_row_statuses
    assert all(status["status"] == "fail" for status in qwen_row_statuses)
    assert any(
        "statistic.serving_coverage=full_serving"
        in status["missing_requirements"]
        for status in qwen_row_statuses
    )
    assert checks["qwen_model_loader_or_token_loop"]["status"] == "pass"
    assert preflight["blocking_gaps"] == [
        checks["qwen3_8b_full_serving_rows_imported"]["why"],
    ]
    lifecycle = preflight["serving_lifecycle"]
    assert lifecycle["kind"] == "pto_qwen_persistent_serving_scaffold"
    assert lifecycle["status"] == "partial"
    assert (
        lifecycle["lifecycle_plan"]["kind"]
        == "pto_qwen_persistent_serving_lifecycle_plan"
    )
    assert lifecycle["prompt_accounting"]["kind"] == "pto_qwen_prompt_accounting"
    assert lifecycle["runtime_input_binding"]["kind"] == (
        "pto_qwen_runtime_input_binding"
    )
    assert lifecycle["runtime_input_binding"]["status"] == (
        "runtime_input_binding_plan_ready"
    )
    assert lifecycle["cuda_token_buffer_binding"]["kind"] == (
        "pto_qwen_cuda_token_buffer_binding"
    )
    assert lifecycle["cuda_token_buffer_binding"]["status"] == (
        "token_buffer_binding_plan_ready"
    )
    assert lifecycle["persistent_decode_args"]["kind"] == (
        "pto_qwen_persistent_decode_args"
    )
    assert lifecycle["persistent_decode_args"]["status"] == (
        "persistent_decode_args_plan_ready"
    )
    assert lifecycle["token_pointer_table"]["kind"] == (
        "pto_qwen_cuda_token_pointer_table_lifecycle"
    )
    assert lifecycle["token_pointer_table"]["status"] == (
        "token_pointer_table_lifecycle_ready"
    )
    assert lifecycle["weight_inventory"]["kind"] == "pto_qwen_weight_inventory"
    assert (
        lifecycle["safetensors_shards"]["kind"]
        == "pto_qwen_safetensors_shard_status"
    )
    assert lifecycle["safetensors_shards"]["status"] == "ready_for_metadata_probe"
    assert (
        lifecycle["safetensors_metadata"]["kind"]
        == "pto_qwen_safetensors_metadata_probe"
    )
    assert lifecycle["safetensors_metadata"]["status"] == "metadata_validated"
    assert lifecycle["cuda_weight_binding"]["kind"] == (
        "pto_qwen_cuda_weight_binding"
    )
    assert lifecycle["cuda_weight_binding"]["status"] == "binding_plan_ready"
    assert lifecycle["persistent_weight_args"]["kind"] == (
        "pto_qwen_persistent_weight_args"
    )
    assert lifecycle["persistent_weight_args"]["status"] == (
        "persistent_weight_args_ready"
    )
    assert lifecycle["persistent_weight_materialization"]["kind"] == (
        "pto_qwen_persistent_weight_materialization"
    )
    assert lifecycle["persistent_weight_materialization"]["status"] == (
        "persistent_weight_materialization_plan_ready"
    )
    assert lifecycle["resident_weight_table"]["kind"] == (
        "pto_qwen_resident_weight_table_lifecycle"
    )
    assert lifecycle["resident_weight_table"]["status"] == (
        "resident_weight_table_lifecycle_ready"
    )
    assert checks["qwen_persistent_weight_arg_manifest"]["status"] == "pass"
    assert lifecycle["kv_cache_binding"]["kind"] == "pto_qwen_cuda_kv_cache_lifecycle"
    assert lifecycle["kv_cache_binding"]["status"] == "kv_cache_lifecycle_ready"
    assert lifecycle["decode_loop_runner"]["kind"] == "pto_qwen_decode_loop_runner"
    assert lifecycle["decode_loop_runner"]["status"] == "decode_loop_runner_plan_ready"
    assert lifecycle["persistent_task_bodies"]["kind"] == (
        "pto_qwen_persistent_task_bodies"
    )
    assert lifecycle["persistent_task_bodies"]["status"] == (
        "generated_task_bodies_ready"
    )
    assert {
        "qwen_tokenizer",
        "qwen_cuda_token_buffer_binding",
        "qwen_persistent_decode_args",
        "qwen_token_pointer_table_owner",
        "qwen_weight_loader",
        "qwen_persistent_weight_materialization",
        "qwen_resident_weight_table_owner",
        "kv_cache_lifecycle",
        "decode_loop_runner",
        "qwen_persistent_task_bodies",
        "viewer_result_import",
    } <= set(lifecycle["missing_stage_ids"])
    assert any(
        row["shape"].startswith("controlled serving-equivalent")
        for row in preflight["pto_serving_rows"]
    )
    assert {workload["id"] for workload in preflight["serving_workloads"]} == {
        "mpk_offline_decode",
        "vdcores_offline_decode",
    }
    assert "Qwen/Qwen3-8B" in preflight["next_action"]


def test_pto_full_serving_row_gate_rejects_diagnostic_qwen_row():
    module = load_pto_serving_preflight_module()
    row = {
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "inputs": {
            "shape": "Qwen/Qwen3-8B resource-backed diagnostic vdcores_offline_decode"
        },
        "statistic": {
            "serving_coverage": "diagnostic_resource_backed_qwen_dag",
            "workload_id": "vdcores_offline_decode",
            "sample_count": 64,
            "host_wall_ns": 9273351357,
            "device_wall_ns": 9138154496,
        },
        "raw_artifact": "tmp/cuda-backend/diagnostic/qwen-decode-loop-runner.json",
        "correctness": "pass",
    }

    status = module.full_serving_qwen_row_status(row)

    assert status["status"] == "fail"
    assert status["workload_id"] == "vdcores_offline_decode"
    assert "statistic.serving_coverage=full_serving" in status[
        "missing_requirements"
    ]
    assert "statistic.end_to_end_latency_ns>0" in status["missing_requirements"]
    assert module.full_serving_qwen_rows([row]) == []


def test_pto_full_serving_row_gate_rejects_diagnostic_comparison_scope():
    module = load_pto_serving_preflight_module()
    row = {
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "inputs": {
            "shape": (
                "mpk_offline_decode,Qwen/Qwen3-8B,batch=8,"
                "prompt_tokens=64,decode_tokens=1024"
            )
        },
        "statistic": {
            "serving_coverage": "full_serving",
            "workload_id": "mpk_offline_decode",
            "sample_count": 3,
            "host_wall_ns": 6_000_000_000,
            "device_wall_ns": 5_900_000_000,
            "end_to_end_latency_ns": 6_000_000_000,
            "inter_token_latency_ns": 5_800_000,
            "time_to_first_token_ns": 50_000_000,
            "throughput_tokens_per_s": 1300.0,
            "batch_size": 8,
            "decode_tokens": 1024,
            "comparison_scope": "diagnostic_decode_without_prompt_prefill",
            "model_equivalent_ready": False,
        },
        "correctness_details": {
            "comparison_scope": "diagnostic_decode_without_prompt_prefill",
            "model_equivalent_ready": False,
        },
        "raw_artifact": "tmp/cuda-backend/full-serving/mpk_offline_decode/",
        "correctness": "pass",
    }

    status = module.full_serving_qwen_row_status(row)

    assert status["status"] == "fail"
    assert "correctness_details.model_equivalent_ready=true" in status[
        "missing_requirements"
    ]
    assert module.full_serving_qwen_rows([row]) == []


def test_pto_full_serving_row_gate_accepts_both_policy_rows():
    module = load_pto_serving_preflight_module()

    def row_for(workload_id):
        return {
            "benchmark_id": "llm_serving_decode",
            "method_id": "pto_persistent_device",
            "inputs": {
                "shape": (
                    f"{workload_id},Qwen/Qwen3-8B,batch=8,"
                    "prompt_tokens=64,decode_tokens=1024"
                )
            },
            "statistic": {
                "serving_coverage": "full_serving",
                "workload_id": workload_id,
                "sample_count": 3,
                "host_wall_ns": 6_000_000_000,
                "device_wall_ns": 5_900_000_000,
                "end_to_end_latency_ns": 6_000_000_000,
                "inter_token_latency_ns": 5_800_000,
                "time_to_first_token_ns": 50_000_000,
                "throughput_tokens_per_s": 1300.0,
                "batch_size": 8,
                "decode_tokens": 1024,
                "comparison_scope": "model_equivalent_decode",
                "model_equivalent_ready": True,
            },
            "correctness_details": {
                "comparison_scope": "model_equivalent_decode",
                "model_equivalent_ready": True,
            },
            "raw_artifact": f"tmp/cuda-backend/full-serving/{workload_id}/",
            "correctness": "pass",
        }

    rows = [row_for("mpk_offline_decode"), row_for("vdcores_offline_decode")]
    statuses = [module.full_serving_qwen_row_status(row) for row in rows]

    assert all(status["status"] == "pass" for status in statuses)
    assert {module.row_workload_id(row) for row in rows} == {
        "mpk_offline_decode",
        "vdcores_offline_decode",
    }
    assert module.full_serving_qwen_rows(rows) == rows


def test_persistent_qwen_serving_scaffold_is_reviewable(tmp_path):
    output = tmp_path / "qwen-serving-scaffold.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/persistent_qwen_serving_scaffold.py",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    scaffold = json.loads(output.read_text(encoding="utf-8"))
    assert scaffold["kind"] == "pto_qwen_persistent_serving_scaffold"
    assert scaffold["benchmark_id"] == "llm_serving_decode"
    assert scaffold["method_id"] == "pto_persistent_device"
    assert scaffold["runtime"] == "cuda/persistent_device"
    assert {item["id"] for item in scaffold["serving_workloads"]} == {
        "mpk_offline_decode",
        "vdcores_offline_decode",
    }
    stages = {item["id"]: item for item in scaffold["stages"]}
    assert stages["persistent_device_task_abi"]["status"] == "pass"
    assert stages["persistent_dag_codegen"]["status"] == "pass"
    assert stages["qwen_serving_lifecycle_plan"]["status"] == "pass"
    assert stages["qwen_tokenizer"]["status"] == "partial"
    assert stages["qwen_runtime_input_binding"]["status"] == "pass"
    assert stages["qwen_cuda_token_buffer_binding"]["status"] == "partial"
    assert stages["qwen_persistent_decode_args"]["status"] == "partial"
    assert stages["qwen_token_pointer_table_owner"]["status"] == "partial"
    assert stages["qwen_weight_loader"]["status"] == "partial"
    assert "cuda_live table ownership" in stages[
        "qwen_weight_loader"
    ]["next_action"]
    assert stages["qwen_cuda_weight_binding"]["status"] == "pass"
    assert stages["qwen_persistent_weight_args"]["status"] == "pass"
    assert stages["qwen_persistent_weight_materialization"]["status"] == "partial"
    assert stages["qwen_resident_weight_table_owner"]["status"] == "partial"
    assert stages["qwen_safetensors_shards"]["status"] == "pass"
    assert "rerun the metadata probe" in stages["qwen_safetensors_shards"][
        "next_action"
    ]
    assert scaffold["safetensors_shards"]["kind"] == (
        "pto_qwen_safetensors_shard_status"
    )
    assert scaffold["safetensors_shards"]["status"] == "ready_for_metadata_probe"
    assert scaffold["safetensors_metadata"]["kind"] == (
        "pto_qwen_safetensors_metadata_probe"
    )
    assert scaffold["safetensors_metadata"]["status"] == "metadata_validated"
    assert scaffold["cuda_weight_binding"]["kind"] == "pto_qwen_cuda_weight_binding"
    assert scaffold["cuda_weight_binding"]["status"] == "binding_plan_ready"
    assert scaffold["persistent_weight_args"]["kind"] == (
        "pto_qwen_persistent_weight_args"
    )
    assert scaffold["persistent_weight_args"]["status"] == (
        "persistent_weight_args_ready"
    )
    assert scaffold["persistent_weight_materialization"]["kind"] == (
        "pto_qwen_persistent_weight_materialization"
    )
    assert scaffold["persistent_weight_materialization"]["status"] == (
        "persistent_weight_materialization_plan_ready"
    )
    assert scaffold["resident_weight_table"]["kind"] == (
        "pto_qwen_resident_weight_table_lifecycle"
    )
    assert scaffold["resident_weight_table"]["status"] == (
        "resident_weight_table_lifecycle_ready"
    )
    assert scaffold["resident_weight_table"]["mode"] == "dry_run_pointer_lifecycle"
    assert scaffold["runtime_input_binding"]["kind"] == (
        "pto_qwen_runtime_input_binding"
    )
    assert scaffold["runtime_input_binding"]["status"] == (
        "runtime_input_binding_plan_ready"
    )
    assert scaffold["cuda_token_buffer_binding"]["kind"] == (
        "pto_qwen_cuda_token_buffer_binding"
    )
    assert scaffold["cuda_token_buffer_binding"]["status"] == (
        "token_buffer_binding_plan_ready"
    )
    assert scaffold["persistent_decode_args"]["kind"] == (
        "pto_qwen_persistent_decode_args"
    )
    assert scaffold["persistent_decode_args"]["status"] == (
        "persistent_decode_args_plan_ready"
    )
    assert scaffold["token_pointer_table"]["kind"] == (
        "pto_qwen_cuda_token_pointer_table_lifecycle"
    )
    assert scaffold["token_pointer_table"]["status"] == (
        "token_pointer_table_lifecycle_ready"
    )
    assert scaffold["token_pointer_table"]["mode"] == "dry_run_pointer_lifecycle"
    assert scaffold["kv_cache_binding"]["kind"] == "pto_qwen_cuda_kv_cache_lifecycle"
    assert scaffold["kv_cache_binding"]["status"] == "kv_cache_lifecycle_ready"
    assert scaffold["kv_cache_binding"]["mode"] == "dry_run_pointer_lifecycle"
    assert scaffold["decode_loop_runner"]["kind"] == "pto_qwen_decode_loop_runner"
    assert scaffold["decode_loop_runner"]["status"] == "decode_loop_runner_plan_ready"
    assert scaffold["decode_loop_runner"]["mode"] == "dry_run_submission_plan"
    assert scaffold["persistent_task_bodies"]["kind"] == (
        "pto_qwen_persistent_task_bodies"
    )
    assert scaffold["persistent_task_bodies"]["status"] == (
        "generated_task_bodies_ready"
    )
    assert stages["kv_cache_lifecycle"]["status"] == "partial"
    assert stages["decode_loop_runner"]["status"] == "partial"
    assert stages["qwen_persistent_task_bodies"]["status"] == "partial"


def test_persistent_qwen_serving_lifecycle_plan_is_reviewable(tmp_path):
    output = tmp_path / "qwen-serving-lifecycle-plan.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_serving_lifecycle_plan.py",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["kind"] == "pto_qwen_persistent_serving_lifecycle_plan"
    assert plan["benchmark_id"] == "llm_serving_decode"
    assert plan["method_id"] == "pto_persistent_device"
    assert plan["runtime"] == "cuda/persistent_device"
    model = plan["model_shape"]
    assert model["model_id"] == "Qwen/Qwen3-8B"
    assert model["config_revision"] == "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
    assert model["num_hidden_layers"] == 36
    assert model["num_key_value_heads"] == 8
    assert model["head_dim"] == 128

    workload_plans = {item["id"]: item for item in plan["workload_plans"]}
    assert set(workload_plans) == {"mpk_offline_decode", "vdcores_offline_decode"}
    mpk_batch1 = workload_plans["mpk_offline_decode"]["kv_cache_plans"][0]
    assert mpk_batch1["batch_size"] == 1
    assert mpk_batch1["sequence_capacity_tokens"] == 64 + 1024
    expected_elements = 1 * (64 + 1024) * 36 * 2 * 8 * 128
    assert mpk_batch1["element_count"] == expected_elements
    assert mpk_batch1["element_dtype"] == "float32"
    assert mpk_batch1["model_compute_dtype"] == "bfloat16"
    assert mpk_batch1["bytes"] == expected_elements * 4
    assert any(
        item["callable"] == "qwen_layer_attention"
        for item in plan["persistent_task_mapping"]
    )
    assert "decode_loop_execution" in plan["remaining_runtime_gaps"]


def test_qwen_kv_cache_binding_maps_key_value_fields():
    script_path = ROOT / "examples" / "cuda" / "qwen_kv_cache_binding.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_kv_cache_binding",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    lifecycle = module.build_kv_cache_lifecycle()

    assert lifecycle["kind"] == "pto_qwen_cuda_kv_cache_lifecycle"
    assert lifecycle["status"] == "kv_cache_lifecycle_ready"
    assert lifecycle["mode"] == "dry_run_pointer_lifecycle"
    assert lifecycle["abi"]["kv_pointer_fields"] == {
        "c": "key_cache",
        "d": "value_cache",
    }
    assert lifecycle["pointer_count"] == 20
    assert lifecycle["closed_pointer_table"]["freed_pointer_count"] == 20
    records = {
        (item["workload_id"], item["batch_size"]): item
        for item in lifecycle["kv_cache_bindings"]
    }
    mpk = records[("mpk_offline_decode", 16)]
    assert mpk["status"] == "ready"
    assert mpk["sequence_capacity_tokens"] == 1088
    assert mpk["key_cache"]["field"] == "c"
    assert mpk["value_cache"]["field"] == "d"
    assert mpk["key_cache"]["byte_count"] == 2566914048
    assert mpk["value_cache"]["byte_count"] == 2566914048
    assert mpk["key_cache"]["element_dtype"] == "float32"
    assert mpk["value_cache"]["element_dtype"] == "float32"
    assert "decode step t writes position prompt_tokens + t" in mpk[
        "token_position_lifecycle"
    ]
    assert "kv_cache_key_value_field_binding" in lifecycle[
        "implemented_contracts"
    ]
    assert lifecycle["remaining_runtime_gaps"] == [
        "cuda_live_kv_cache_owner_in_decode_loop",
        "numerically_correct_qwen_attention_kv_cache_use",
        "decode_loop_execution",
    ]


def test_qwen_decode_loop_runner_orders_resource_lifetimes():
    script_path = ROOT / "examples" / "cuda" / "qwen_decode_loop_runner.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_decode_loop_runner",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    runner = module.build_decode_loop_runner(mode="mock")

    assert runner["kind"] == "pto_qwen_decode_loop_runner"
    assert runner["status"] == "decode_loop_runner_plan_ready"
    assert runner["mode"] == "dry_run_submission_plan"
    assert runner["resource_lifecycle_status"] == {
        "token_pointer_table": "token_pointer_table_lifecycle_ready",
        "kv_cache": "kv_cache_lifecycle_ready",
        "resident_weight_table": "resident_weight_table_lifecycle_ready",
        "activation_workspace": "activation_workspace_lifecycle_planned",
    }
    assert runner["total_decode_iterations"] == 1088
    plans = {item["workload_id"]: item for item in runner["dag_submission_plans"]}
    mpk = plans["mpk_offline_decode"]
    assert mpk["status"] == "submission_plan_ready"
    assert mpk["decode_steps"] == 1024
    assert mpk["max_batch_size"] == 16
    assert mpk["owner_lifetime_order"] == [
        "open_token_pointer_table",
        "open_kv_cache",
        "open_resident_weight_table",
        "open_activation_workspace",
        "materialize_decode_args",
        "materialize_weight_args",
        "submit_persistent_dag",
        "close_activation_workspace",
        "close_resident_weight_table",
        "close_kv_cache",
        "close_token_pointer_table",
    ]
    assert mpk["task_argument_fields"] == {
        "a": "input_ids",
        "b": "attention_mask",
        "out": "output_ids",
        "c": "key_cache",
        "d": "value_cache",
        "tensor_args": "resident_weight_tensors",
    }
    assert mpk["output_token_accounting"]["output_buffer"] == "output_ids"
    assert "decode_loop_owner_lifetime_order" in runner["implemented_contracts"]
    assert runner["remaining_runtime_gaps"] == [
        "numerically_correct_qwen_kernel_bodies",
        "full_cuda_live_decode_loop_execution",
        "viewer_result_import",
    ]


def test_qwen_persistent_task_bodies_render_generated_source():
    script_path = ROOT / "examples" / "cuda" / "qwen_persistent_task_bodies.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_task_bodies",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    manifest = module.build_task_body_manifest(num_hidden_layers=1)

    assert manifest["kind"] == "pto_qwen_persistent_task_bodies"
    assert manifest["status"] == "generated_task_bodies_ready"
    assert manifest["source_kind"] == "generated-dispatch"
    assert manifest["task_body_count"] == 10
    assert manifest["rendered_source"]["entry_name"] == (
        "pto_persistent_dag_f32_executor"
    )
    callable_names = {item["callable"] for item in manifest["task_bodies"]}
    assert {
        "qwen_embedding_lookup",
        "qwen_attention_qkv",
        "qwen_attention_o",
        "qwen_mlp_gate_up",
        "qwen_logits",
    } <= callable_names
    qkv = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_attention_qkv"
    )
    assert qkv["consumes_fields"] == ["a", "b", "out", "c", "d", "tensor_args"]
    assert {"key_cache", "value_cache"} <= set(qkv["consumes_roles"])
    attention_o = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_attention_o"
    )
    assert attention_o["consumes_fields"] == [
        "a",
        "out",
        "c",
        "d",
        "tensor_args",
        "scalar_args",
    ]
    assert {"key_cache", "value_cache"} <= set(attention_o["consumes_roles"])
    assert "kv_page_table" in attention_o["consumes_roles"]
    assert manifest["coverage"]["token_fields"] == ["a", "b", "out"]
    assert manifest["coverage"]["kv_fields"] == ["c", "d"]
    assert (
        manifest["coverage"]["kv_write_policy"]
        == "slot_mapped_kv_cache_writeback_ready"
    )
    assert manifest["coverage"]["weight_fields"] == ["tensor_args"]
    source = manifest["rendered_source"]["preview"]
    assert "__device__ void pto_task_qwen_attention_qkv" in source
    assert "task->c" in source
    assert "task->d" in source
    assert "kv_write_index" in source
    assert "task->tensor_args[0]" in source
    assert "task->tensor_args[1]" in source
    assert "const unsigned int *kv_page_table" in source
    assert "const unsigned int logical_page = decode_position / kv_page_size;" in (
        source
    )
    assert "const unsigned int projection_input_count =" in source
    assert "for (unsigned int tile_begin = 0U;" in source
    assert "physical_page) * kv_page_size +" in source
    assert "generated_qwen_kernel_bodies" in manifest["implemented_contracts"]
    assert "controlled_proxy_numeric_oracle" in manifest["implemented_contracts"]
    assert "qwen_unit_math_oracle" in manifest["implemented_contracts"]
    assert "qwen_unit_math_source_coverage" in manifest["implemented_contracts"]
    assert (
        "qwen_kernel_kv_cache_writeback_field_contract"
        in manifest["implemented_contracts"]
    )
    assert (
        "qwen_bounded_decode_attention_reduction_source"
        in manifest["implemented_contracts"]
    )
    assert (
        "qwen_gqa_decode_attention_head_grouping_source"
        in manifest["implemented_contracts"]
    )
    assert (
        "qwen_paged_kv_attention_index_source"
        in manifest["implemented_contracts"]
    )
    assert (
        "qwen_tiled_decode_attention_softmax_source"
        in manifest["implemented_contracts"]
    )
    oracle = manifest["numeric_oracle"]
    assert oracle["status"] == "controlled_proxy_numeric_oracle_ready"
    assert oracle["scope"] == "controlled_proxy_not_full_qwen"
    assert oracle["checked_callables"] == manifest["task_body_count"]
    assert oracle["max_abs_error"] == 0.0
    qkv_oracle = next(
        item
        for item in oracle["sample_outputs"]
        if item["callable"] == "qwen_attention_qkv"
    )
    assert qkv_oracle["expected_out"] == [13.0, 15.0, 17.0, 19.0]
    assert qkv_oracle["expected_c"] == [11.0, 13.0, 15.0, 17.0]
    assert qkv_oracle["expected_d"] == qkv_oracle["expected_out"]
    unit_oracle = manifest["qwen_unit_math_oracle"]
    assert unit_oracle["status"] == "qwen_unit_math_oracle_ready"
    assert unit_oracle["steps"]["mlp_swiglu"] == [
        0.054983,
        -0.05402,
        0.060482,
        -0.063023,
    ]
    attention_oracle = manifest["qwen_decode_attention_oracle"]
    assert attention_oracle["status"] == "qwen_decode_attention_oracle_ready"
    assert attention_oracle["head_grouping"]["kv_heads"] == 2
    assert attention_oracle["steps"]["attention_context"] == [
        5.659033,
        11.318066,
        5.5,
        11.0,
        15.689571,
        20.919428,
        18.162065,
        24.216087,
    ]
    assert manifest["remaining_runtime_gaps"] == [
        "numerically_correct_qwen_kernel_bodies",
        "cuda_live_qwen_unit_math_execution",
        "cuda_live_decode_loop_execution",
        "viewer_result_import",
    ]


def test_qwen_persistent_proxy_live_plan_maps_qkv_to_single_task_dag():
    script_path = ROOT / "examples" / "cuda" / "qwen_persistent_proxy_live.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_proxy_live",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    plan = module.build_live_proxy_plan()

    assert plan["kind"] == "pto_qwen_proxy_live_execution_plan"
    assert plan["status"] == "ready_to_run"
    assert plan["scope"] == "controlled_proxy_not_full_qwen"
    assert plan["runtime"] == "cuda/persistent_device"
    assert plan["callable"] == "qwen_attention_qkv"
    assert plan["func_id"] == 7102
    assert plan["dag"]["task_count"] == 1
    assert plan["dag"]["scheduler_blocks"] == 1
    assert plan["dag"]["worker_blocks"] == 1
    assert plan["dag"]["queue_capacity"] == 4
    assert plan["task_argument_fields"] == {
        "a": "hidden_state",
        "b": "attention_mask",
        "out": "attention_output",
        "c": "key_cache_mutable",
        "d": "value_cache_mutable",
        "tensor_args[0]": "q_proj_weight",
    }
    assert plan["inputs"]["a"] == [10.0, 11.0, 12.0, 13.0]
    assert plan["inputs"]["weights"][0] == [1.0, 2.0, 3.0, 4.0]
    assert plan["expected"]["out"] == [13.0, 15.0, 17.0, 19.0]
    assert plan["expected"]["c"] == [11.0, 13.0, 15.0, 17.0]
    assert plan["expected"]["d"] == plan["expected"]["out"]
    assert plan["remaining_runtime_gaps"] == [
        "numerically_correct_qwen_kernel_bodies",
        "full_qwen_decode_loop_execution",
        "viewer_result_import",
    ]


def test_qwen_persistent_microdecode_live_plan_maps_proxy_chain_dag():
    script_path = ROOT / "examples" / "cuda" / "qwen_persistent_microdecode_live.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_microdecode_live",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    plan = module.build_live_microdecode_plan()

    assert plan["kind"] == "pto_qwen_microdecode_live_execution_plan"
    assert plan["status"] == "ready_to_run"
    assert plan["scope"] == "controlled_proxy_not_full_qwen"
    assert plan["runtime"] == "cuda/persistent_device"
    assert plan["dag"]["task_count"] == 3
    assert plan["dag"]["dependent_count"] == 2
    assert plan["dag"]["dependency_edges"] == [
        ["qwen_attention_qkv", "qwen_attention_o"],
        ["qwen_attention_o", "qwen_logits"],
    ]
    assert [task["func_id"] for task in plan["tasks"]] == [7102, 7104, 7109]
    assert [task["initial_fanin"] for task in plan["tasks"]] == [0, 1, 1]
    assert plan["expected"]["attention_qkv_out"] == [13.0, 15.0, 17.0, 19.0]
    assert plan["expected"]["attention_o_out"] == [13.5, 16.5, 19.5, 22.5]
    assert plan["expected"]["logits_out"] == [14.5, 18.5, 22.5, 26.5]
    assert plan["expected"]["c"] == [11.0, 13.0, 15.0, 17.0]
    assert plan["expected"]["d"] == plan["expected"]["attention_qkv_out"]
    assert "controlled_proxy_live_microdecode_plan" in plan["implemented_contracts"]
    assert plan["remaining_runtime_gaps"] == [
        "numerically_correct_qwen_kernel_bodies",
        "full_qwen_decode_loop_execution",
        "viewer_result_import",
    ]


def test_persistent_qwen_prompt_accounting_is_reviewable(tmp_path):
    output = tmp_path / "qwen-prompt-accounting.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_prompt_accounting.py",
            "--mode",
            "mock",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    accounting = json.loads(output.read_text(encoding="utf-8"))
    assert accounting["kind"] == "pto_qwen_prompt_accounting"
    assert accounting["status"] == "pass"
    assert accounting["model_id"] == "Qwen/Qwen3-8B"
    assert accounting["model_revision"] == "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
    records = {
        item["workload_id"]: item
        for item in accounting["prompt_records"]
    }
    assert set(records) == {"mpk_offline_decode", "vdcores_offline_decode"}
    assert records["mpk_offline_decode"]["target_prompt_tokens"] == 64
    assert records["vdcores_offline_decode"]["target_prompt_tokens"] == 128
    assert records["mpk_offline_decode"]["chat_prompt_tokens"] > 0
    assert records["mpk_offline_decode"]["padding_or_regeneration_required"]
    assert "decode_loop_consumes_token_ids" in accounting["remaining_runtime_gaps"]


def test_qwen_runtime_input_binding_materializes_token_buffers():
    script_path = ROOT / "examples" / "cuda" / "qwen_runtime_input_binding.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_runtime_input_binding",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    binding = module.build_runtime_input_binding(mode="mock")

    assert binding["kind"] == "pto_qwen_runtime_input_binding"
    assert binding["status"] == "runtime_input_binding_plan_ready"
    assert "tokenizer_to_runtime_input_ids" in binding["implemented_contracts"]
    assert "runtime_token_id_binding" not in binding["remaining_runtime_gaps"]
    records = {item["workload_id"]: item for item in binding["workload_records"]}
    mpk = records["mpk_offline_decode"]
    assert mpk["prompt_token_ids"][:3] == [0, 1, 2]
    assert mpk["prompt_token_count"] > 0
    assert mpk["target_prompt_alignment"]["status"] == "padded_to_target"
    assert mpk["runtime_prompt_token_count"] == mpk["target_prompt_tokens"]
    assert mpk["input_ids_buffer"]["dtype"] == "int32"
    assert mpk["input_ids_buffer"]["shape"] == [
        max(mpk["batch_sizes"]),
        mpk["target_prompt_tokens"],
    ]
    assert mpk["attention_mask_buffer"]["shape"] == [
        max(mpk["batch_sizes"]),
        mpk["target_prompt_tokens"],
    ]
    assert mpk["output_ids_buffer"]["shape"] == [
        max(mpk["batch_sizes"]),
        mpk["decode_tokens"],
    ]
    assert mpk["scalar_bindings"]["first_decode_position"] == (
        mpk["target_prompt_tokens"]
    )
    assert mpk["scalar_bindings"]["active_prompt_token_count"] == (
        mpk["prompt_token_count"]
    )
    assert mpk["scalar_bindings"]["first_logits_position"] == (
        mpk["prompt_token_count"] - 1
    )
    assert mpk["device_binding_state"] == "host_materialized_not_cuda_allocated"
    assert "target_prompt_shape_alignment" not in binding["remaining_runtime_gaps"]
    assert "cuda_token_buffer_allocation" in binding["remaining_runtime_gaps"]
    assert "decode_loop_consumes_token_ids" in binding["remaining_runtime_gaps"]


def test_qwen_cuda_token_buffer_binding_plans_device_buffers():
    script_path = ROOT / "examples" / "cuda" / "qwen_cuda_token_buffer_binding.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_cuda_token_buffer_binding",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    binding = module.build_cuda_token_buffer_binding(
        mode="mock",
        no_cuda_probe=True,
    )

    assert binding["kind"] == "pto_qwen_cuda_token_buffer_binding"
    assert binding["status"] == "token_buffer_binding_plan_ready"
    assert "cuda_token_buffer_plan" in binding["implemented_contracts"]
    records = {item["workload_id"]: item for item in binding["workload_records"]}
    mpk = records["mpk_offline_decode"]
    assert mpk["input_ids_device_buffer"]["shape"] == [16, 64]
    assert mpk["attention_mask_device_buffer"]["shape"] == [16, 64]
    assert mpk["output_ids_device_buffer"]["shape"] == [16, 1024]
    assert mpk["copy_plan"]["host_to_device_buffers"] == [
        "input_ids",
        "attention_mask",
    ]
    assert mpk["copy_plan"]["device_to_host_buffers"] == ["output_ids"]
    assert binding["cuda_probe"]["reason"] == "disabled_by_no_cuda_probe"
    assert "cuda_token_buffer_allocation" in binding["remaining_runtime_gaps"]
    assert "decode_loop_consumes_token_ids" in binding["remaining_runtime_gaps"]


def test_qwen_persistent_decode_args_bind_token_pointer_roles(tmp_path):
    token_binding = {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_token_buffer_binding",
        "status": "cuda_token_buffer_binding_ready",
        "workload_records": [
            {
                "workload_id": "mpk_offline_decode",
                "input_ids_device_buffer": {
                    "name": "input_ids",
                    "shape": [16, 64],
                    "byte_count": 4096,
                },
                "attention_mask_device_buffer": {
                    "name": "attention_mask",
                    "shape": [16, 64],
                    "byte_count": 4096,
                },
                "output_ids_device_buffer": {
                    "name": "output_ids",
                    "shape": [16, 1024],
                    "byte_count": 65536,
                },
                "scalar_bindings": {
                    "prompt_token_count": 64,
                    "active_prompt_token_count": 18,
                    "decode_tokens": 1024,
                    "max_batch_size": 16,
                    "first_logits_position": 17,
                    "first_decode_position": 64,
                },
            }
        ],
    }
    token_binding_path = tmp_path / "qwen-cuda-token-buffer-binding.json"
    token_binding_path.write_text(json.dumps(token_binding), encoding="utf-8")
    pointer_table = tmp_path / "qwen-token-buffer-pointers.json"
    pointer_table.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pto_qwen_cuda_token_pointer_table",
                "status": "cuda_token_pointer_table_ready",
                "pointers": [
                    {
                        "workload_id": "mpk_offline_decode",
                        "buffer": "input_ids",
                        "device_ptr": 0x20000000,
                        "byte_count": 4096,
                    },
                    {
                        "workload_id": "mpk_offline_decode",
                        "buffer": "attention_mask",
                        "device_ptr": 0x20001000,
                        "byte_count": 4096,
                    },
                    {
                        "workload_id": "mpk_offline_decode",
                        "buffer": "output_ids",
                        "device_ptr": 0x20002000,
                        "byte_count": 65536,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "qwen-persistent-decode-args.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_persistent_decode_args.py",
            "--cuda-token-buffer-json",
            str(token_binding_path),
            "--token-pointer-table-json",
            str(pointer_table),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["kind"] == "pto_qwen_persistent_decode_args"
    assert manifest["status"] == "persistent_decode_args_ready"
    assert manifest["abi"]["token_pointer_fields"] == {
        "a": "input_ids",
        "b": "attention_mask",
        "out": "output_ids",
    }
    assert manifest["abi"]["field_offsets"]["a"] > 0
    record = manifest["workload_decode_args"][0]
    assert record["workload_id"] == "mpk_offline_decode"
    assert record["status"] == "ready"
    assert record["pointer_bindings"] == [
        {
            "field": "a",
            "buffer": "input_ids",
            "device_ptr": 0x20000000,
            "device_ptr_hex": "0x20000000",
            "byte_count": 4096,
        },
        {
            "field": "b",
            "buffer": "attention_mask",
            "device_ptr": 0x20001000,
            "device_ptr_hex": "0x20001000",
            "byte_count": 4096,
        },
        {
            "field": "out",
            "buffer": "output_ids",
            "device_ptr": 0x20002000,
            "device_ptr_hex": "0x20002000",
            "byte_count": 65536,
        },
    ]
    assert record["scalar_fields"]["n"] == 64
    assert record["scalar_fields"]["rows"] == 16
    assert record["scalar_fields"]["cols"] == 1024
    assert record["scalar_fields"]["inner"] == 17
    assert record["scalar_fields"]["runtime_prompt_tokens"] == 64
    assert record["scalar_fields"]["active_prompt_tokens"] == 18
    assert record["scalar_fields"]["output_start_position"] == 64
    assert "persistent_decode_token_arg_binding" in manifest[
        "implemented_contracts"
    ]
    assert "decode_loop_consumes_token_ids" not in manifest[
        "remaining_runtime_gaps"
    ]
    assert (
        "numerically_correct_qwen_token_consumption"
        in manifest["remaining_runtime_gaps"]
    )


def test_qwen_token_pointer_table_materializes_decode_args_during_lifetime():
    script_path = ROOT / "examples" / "cuda" / "qwen_token_pointer_table.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_token_pointer_table",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    lifecycle = module.build_token_pointer_table_lifecycle(mode="mock")

    assert lifecycle["kind"] == "pto_qwen_cuda_token_pointer_table_lifecycle"
    assert lifecycle["status"] == "token_pointer_table_lifecycle_ready"
    assert lifecycle["mode"] == "dry_run_pointer_lifecycle"
    assert lifecycle["pointer_table"]["status"] == "cuda_token_pointer_table_ready"
    assert lifecycle["pointer_count"] == 6
    assert lifecycle["closed_pointer_table"]["status"] == (
        "cuda_token_pointer_table_closed"
    )
    assert lifecycle["freed_pointer_count"] == 6
    assert lifecycle["decode_args"]["kind"] == "pto_qwen_persistent_decode_args"
    assert lifecycle["decode_args"]["status"] == "persistent_decode_args_ready"
    records = {
        item["workload_id"]: item
        for item in lifecycle["decode_args"]["workload_decode_args"]
    }
    assert records["mpk_offline_decode"]["status"] == "ready"
    assert {
        item["field"]: item["buffer"]
        for item in records["mpk_offline_decode"]["pointer_bindings"]
    } == {
        "a": "input_ids",
        "b": "attention_mask",
        "out": "output_ids",
    }
    assert "live_token_pointer_table_owner" in lifecycle["implemented_contracts"]
    assert lifecycle["remaining_runtime_gaps"] == [
        "numerically_correct_qwen_token_consumption",
        "decode_loop_execution",
    ]


def test_persistent_qwen_weight_inventory_is_reviewable(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 4,
                "intermediate_size": 8,
                "vocab_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 552},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": (
                        "model-00001-of-00002.safetensors"
                    ),
                    "model.layers.0.self_attn.k_proj.weight": (
                        "model-00001-of-00002.safetensors"
                    ),
                    "model.layers.0.self_attn.v_proj.weight": (
                        "model-00001-of-00002.safetensors"
                    ),
                    "model.layers.0.self_attn.o_proj.weight": (
                        "model-00001-of-00002.safetensors"
                    ),
                    "model.layers.0.mlp.gate_proj.weight": (
                        "model-00002-of-00002.safetensors"
                    ),
                    "model.layers.0.mlp.up_proj.weight": (
                        "model-00002-of-00002.safetensors"
                    ),
                    "model.layers.0.mlp.down_proj.weight": (
                        "model-00002-of-00002.safetensors"
                    ),
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "qwen-weight-inventory.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_weight_inventory.py",
            "--index-json",
            str(index),
            "--config-json",
            str(config),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    inventory = json.loads(output.read_text(encoding="utf-8"))
    assert inventory["kind"] == "pto_qwen_weight_inventory"
    assert inventory["status"] == "partial_inventory"
    assert inventory["model_id"] == "Qwen/Qwen3-8B"
    assert inventory["tensor_count"] == 10
    assert inventory["shard_count"] == 2
    shape_contract = inventory["weight_shape_contract"]
    assert shape_contract["status"] == "complete_for_index"
    assert shape_contract["dtype"] == "bfloat16"
    assert shape_contract["expected_total_size_bytes"] == 552
    assert shape_contract["index_total_size_bytes"] == 552
    assert shape_contract["size_matches_index"] is True
    tensor_shapes = {
        item["name"]: item for item in shape_contract["tensor_shapes"]
    }
    assert tensor_shapes["model.embed_tokens.weight"]["shape"] == [16, 4]
    assert tensor_shapes["model.layers.0.self_attn.q_proj.weight"]["shape"] == [
        4,
        4,
    ]
    assert tensor_shapes["model.layers.0.self_attn.k_proj.weight"]["shape"] == [
        2,
        4,
    ]
    assert tensor_shapes["model.layers.0.mlp.down_proj.weight"]["shape"] == [
        4,
        8,
    ]
    groups = {item["id"]: item for item in inventory["binding_groups"]}
    assert groups["embedding"]["tensor_count"] == 1
    assert groups["attention_qkv_o"]["tensor_count"] == 4
    assert groups["mlp_gate_up_down"]["tensor_count"] == 3
    assert groups["norm_and_logits"]["tensor_count"] == 2
    assert "expected_shape_dtype_contract" in inventory["implemented_contracts"]
    assert "safetensors_tensor_open" in inventory["remaining_runtime_gaps"]
    assert "cuda_device_weight_binding" in inventory["remaining_runtime_gaps"]


def test_persistent_qwen_safetensors_metadata_probe_validates_headers(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 4,
                "intermediate_size": 8,
                "vocab_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 264},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "model.norm.weight": "model-00001-of-00001.safetensors",
                    "lm_head.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "qwen-weight-inventory.json"
    inventory_result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_weight_inventory.py",
            "--index-json",
            str(index),
            "--config-json",
            str(config),
            "--output-json",
            str(inventory),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert inventory_result.returncode == 0, inventory_result.stdout

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    header = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [16, 4],
            "data_offsets": [0, 128],
        },
        "model.norm.weight": {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [128, 136],
        },
        "lm_head.weight": {
            "dtype": "BF16",
            "shape": [16, 4],
            "data_offsets": [136, 264],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard_path = shard_dir / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + (b"\0" * 264)
    )
    output = tmp_path / "qwen-safetensors-metadata.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_safetensors_metadata.py",
            "--index-json",
            str(index),
            "--weight-inventory-json",
            str(inventory),
            "--shard-dir",
            str(shard_dir),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["kind"] == "pto_qwen_safetensors_metadata_probe"
    assert metadata["status"] == "metadata_validated"
    assert metadata["opened_shard_count"] == 1
    assert metadata["validated_tensor_count"] == 3
    assert metadata["mismatch_count"] == 0
    assert "safetensors_header_parse" in metadata["implemented_contracts"]
    assert (
        "actual_safetensors_shape_dtype_validation"
        in metadata["implemented_contracts"]
    )
    assert "cuda_device_weight_binding" in metadata["remaining_runtime_gaps"]


def test_persistent_qwen_cuda_weight_binding_is_reviewable(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 4,
                "intermediate_size": 8,
                "vocab_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 264},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "model.norm.weight": "model-00001-of-00001.safetensors",
                    "lm_head.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "qwen-weight-inventory.json"
    inventory_result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_weight_inventory.py",
            "--index-json",
            str(index),
            "--config-json",
            str(config),
            "--output-json",
            str(inventory),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert inventory_result.returncode == 0, inventory_result.stdout

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    header = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [16, 4],
            "data_offsets": [0, 128],
        },
        "model.norm.weight": {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [128, 136],
        },
        "lm_head.weight": {
            "dtype": "BF16",
            "shape": [16, 4],
            "data_offsets": [136, 264],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard_path = shard_dir / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(
        struct.pack("<Q", len(header_bytes))
        + header_bytes
        + bytes(range(256))
        + bytes(range(8))
    )
    metadata = tmp_path / "qwen-safetensors-metadata.json"
    metadata_result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_safetensors_metadata.py",
            "--index-json",
            str(index),
            "--weight-inventory-json",
            str(inventory),
            "--shard-dir",
            str(shard_dir),
            "--output-json",
            str(metadata),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert metadata_result.returncode == 0, metadata_result.stdout

    output = tmp_path / "qwen-cuda-weight-binding.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_cuda_weight_binding.py",
            "--index-json",
            str(index),
            "--weight-inventory-json",
            str(inventory),
            "--metadata-json",
            str(metadata),
            "--shard-dir",
            str(shard_dir),
            "--cuda-probe-mode",
            "full",
            "--no-cuda-probe",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    binding = json.loads(output.read_text(encoding="utf-8"))
    assert binding["kind"] == "pto_qwen_cuda_weight_binding"
    assert binding["status"] == "binding_plan_ready"
    assert binding["tensor_count"] == 3
    assert binding["planned_binding_count"] == 3
    assert binding["total_weight_bytes"] == 264
    assert binding["metadata_status"] == "metadata_validated"
    assert binding["cuda_probe"]["status"] == "skipped"
    assert binding["cuda_probe"]["mode"] == "full_residency"
    assert binding["cuda_probe"]["reason"] == "disabled_by_no_cuda_probe"
    assert "safetensors_tensor_data_offsets" in binding["implemented_contracts"]
    assert "persistent_task_weight_arg_binding_plan" in binding[
        "implemented_contracts"
    ]
    assert "full_cuda_weight_residency" in binding["remaining_runtime_gaps"]

    records = {item["tensor"]: item for item in binding["bindings"]}
    norm = records["model.norm.weight"]
    assert norm["binding_group"] == "norm_and_logits"
    assert norm["dtype"] == "bfloat16"
    assert norm["shape"] == [4]
    assert norm["size_bytes"] == 8
    assert norm["file_data_offsets"] == [128, 136]
    assert norm["file_absolute_offsets"][0] == 8 + len(header_bytes) + 128
    assert norm["persistent_arg_role"] == "readonly_weight_tensor"
    assert norm["cuda_binding_state"] == "planned_not_resident"


def test_persistent_qwen_weight_arg_manifest_is_reviewable(tmp_path):
    bindings = []

    def add(slot_id, tensor, binding_group):
        bindings.append(
            {
                "slot_id": slot_id,
                "tensor": tensor,
                "binding_group": binding_group,
                "persistent_arg_role": "readonly_weight_tensor",
                "shape": [4],
                "dtype": "bfloat16",
                "size_bytes": 8,
            }
        )

    slot = 0
    add(slot, "model.embed_tokens.weight", "embedding")
    slot += 1
    for name, group in [
        ("model.layers.0.input_layernorm.weight", "attention_norms"),
        ("model.layers.0.self_attn.q_proj.weight", "attention_qkv_o"),
        ("model.layers.0.self_attn.k_proj.weight", "attention_qkv_o"),
        ("model.layers.0.self_attn.v_proj.weight", "attention_qkv_o"),
        ("model.layers.0.self_attn.q_norm.weight", "attention_norms"),
        ("model.layers.0.self_attn.k_norm.weight", "attention_norms"),
        ("model.layers.0.self_attn.o_proj.weight", "attention_qkv_o"),
        ("model.layers.0.post_attention_layernorm.weight", "attention_norms"),
        ("model.layers.0.mlp.gate_proj.weight", "mlp_gate_up_down"),
        ("model.layers.0.mlp.up_proj.weight", "mlp_gate_up_down"),
        ("model.layers.0.mlp.down_proj.weight", "mlp_gate_up_down"),
        ("model.norm.weight", "norm_and_logits"),
        ("lm_head.weight", "norm_and_logits"),
    ]:
        add(slot, name, group)
        slot += 1

    binding = tmp_path / "qwen-cuda-weight-binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pto_qwen_cuda_weight_binding",
                "status": "binding_plan_ready",
                "tensor_count": len(bindings),
                "planned_binding_count": len(bindings),
                "cuda_probe": {"mode": "full_residency", "status": "pass"},
                "bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "qwen-persistent-weight-args.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_persistent_weight_args.py",
            "--weight-binding-json",
            str(binding),
            "--num-hidden-layers",
            "1",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["kind"] == "pto_qwen_persistent_weight_args"
    assert manifest["status"] == "persistent_weight_args_ready"
    assert manifest["abi"]["task_struct"] == "PtoCudaPersistentDagTask"
    assert manifest["abi"]["tensor_arg_capacity"] == 5
    assert manifest["covered_tensor_count"] == len(bindings)
    assert manifest["missing_tensor_count"] == 0
    assert manifest["max_tensor_args_per_task"] == 5
    assert "qwen_weight_task_decomposition" in manifest["implemented_contracts"]
    assert "qwen_rope_table_tensor_arg_contract" in manifest[
        "implemented_contracts"
    ]
    assert "persistent_task_weight_arg_runtime_binding" in manifest[
        "remaining_runtime_gaps"
    ]

    descriptors = {
        item["id"]: item for item in manifest["task_arg_descriptors"]
    }
    qkv = descriptors["layer_0_attention_qkv"]
    assert qkv["callable"] == "qwen_attention_qkv"
    assert qkv["tensor_arg_count"] == 4
    assert qkv["tensor_args"] == [
        {
            "arg": "tensor_args[0]",
            "slot_id": 2,
            "tensor": "model.layers.0.self_attn.q_proj.weight",
        },
        {
            "arg": "tensor_args[1]",
            "slot_id": 3,
            "tensor": "model.layers.0.self_attn.k_proj.weight",
        },
        {
            "arg": "tensor_args[2]",
            "slot_id": 4,
            "tensor": "model.layers.0.self_attn.v_proj.weight",
        },
        {
            "arg": "tensor_args[3]",
            "tensor": "kv_page_table",
            "role": "kv_page_table",
            "status": "runtime_generated_tensor",
            "device_ptr_source": "runtime_buffers.kv_page_table",
        },
    ]
    qk_norm = descriptors["layer_0_attention_qk_norm"]
    assert qk_norm["tensor_arg_count"] == 5
    assert qk_norm["tensor_args"][2:] == [
        {
            "arg": "tensor_args[2]",
            "tensor": "rope_cos_table",
            "role": "rope_cos_table",
            "status": "runtime_generated_tensor",
            "device_ptr_source": "runtime_buffers.rope_cos_table",
        },
        {
            "arg": "tensor_args[3]",
            "tensor": "rope_sin_table",
            "role": "rope_sin_table",
            "status": "runtime_generated_tensor",
            "device_ptr_source": "runtime_buffers.rope_sin_table",
        },
        {
            "arg": "tensor_args[4]",
            "tensor": "kv_page_table",
            "role": "kv_page_table",
            "status": "runtime_generated_tensor",
            "device_ptr_source": "runtime_buffers.kv_page_table",
        },
    ]
    assert descriptors["layer_0_mlp_gate_up"]["tensor_arg_count"] == 2
    mlp_down = descriptors["layer_0_mlp_down"]
    assert mlp_down["tensor_arg_count"] == 2
    assert mlp_down["tensor_args"][1] == {
        "arg": "tensor_args[1]",
        "tensor": "mlp_residual",
        "role": "mlp_residual",
        "status": "runtime_generated_tensor",
        "device_ptr_source": "runtime_buffers.mlp_residual",
    }
    assert descriptors["logits"]["tensor_args"][0]["tensor"] == "lm_head.weight"


def test_persistent_qwen_weight_materialization_binds_resident_pointers(tmp_path):
    bindings = qwen_one_layer_bindings()
    binding = tmp_path / "qwen-cuda-weight-binding.json"
    write_qwen_binding_fixture(binding, bindings)
    weight_args = write_qwen_weight_args_fixture(tmp_path, binding)

    pointer_table = tmp_path / "qwen-resident-weight-pointers.json"
    pointer_table.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pto_qwen_resident_weight_pointer_table",
                "status": "resident_weight_pointer_table_ready",
                "pointers": [
                    {
                        "slot_id": item["slot_id"],
                        "tensor": item["tensor"],
                        "device_ptr": 0x10000000 + item["slot_id"] * 0x1000,
                        "size_bytes": item["size_bytes"],
                    }
                    for item in bindings
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "qwen-persistent-weight-materialization.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_persistent_weight_materialization.py",
            "--weight-args-json",
            str(weight_args),
            "--weight-binding-json",
            str(binding),
            "--pointer-table-json",
            str(pointer_table),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    materialization = json.loads(output.read_text(encoding="utf-8"))
    assert materialization["kind"] == (
        "pto_qwen_persistent_weight_materialization"
    )
    assert materialization["status"] == "persistent_weight_materialization_ready"
    assert "persistent_task_weight_arg_runtime_materializer" in materialization[
        "implemented_contracts"
    ]
    assert "live_decode_loop_pointer_table" in materialization[
        "remaining_runtime_gaps"
    ]
    assert materialization["abi"]["task_struct"] == "CudaPersistentDagTask"
    assert materialization["abi"]["field_offsets"]["tensor_args"] > 0
    assert materialization["materialized_task_count"] == 10
    assert materialization["bound_tensor_pointer_count"] == len(bindings)
    assert materialization["missing_pointer_count"] == 0

    materialized_descriptors = {
        item["id"]: item
        for item in materialization["materialized_task_descriptors"]
    }
    qkv = materialized_descriptors["layer_0_attention_qkv"]
    assert qkv["tensor_arg_count"] == 4
    assert qkv["tensor_args"][0] == {
        "arg": "tensor_args[0]",
        "slot_id": 2,
        "tensor": "model.layers.0.self_attn.q_proj.weight",
        "device_ptr": 0x10002000,
        "device_ptr_hex": "0x10002000",
        "size_bytes": 8,
    }
    assert qkv["tensor_args"][3] == {
        "arg": "tensor_args[3]",
        "tensor": "kv_page_table",
        "role": "kv_page_table",
        "device_ptr_source": "runtime_buffers.kv_page_table",
        "status": "requires_live_pointer",
    }
    qk_norm = materialized_descriptors["layer_0_attention_qk_norm"]
    assert qk_norm["tensor_arg_count"] == 5
    assert qk_norm["tensor_args"][2:] == [
        {
            "arg": "tensor_args[2]",
            "tensor": "rope_cos_table",
            "role": "rope_cos_table",
            "device_ptr_source": "runtime_buffers.rope_cos_table",
            "status": "requires_live_pointer",
        },
        {
            "arg": "tensor_args[3]",
            "tensor": "rope_sin_table",
            "role": "rope_sin_table",
            "device_ptr_source": "runtime_buffers.rope_sin_table",
            "status": "requires_live_pointer",
        },
        {
            "arg": "tensor_args[4]",
            "tensor": "kv_page_table",
            "role": "kv_page_table",
            "device_ptr_source": "runtime_buffers.kv_page_table",
            "status": "requires_live_pointer",
        },
    ]
    assert materialization["symbolic_tensor_pointer_count"] == 5
    assert materialized_descriptors["layer_0_mlp_gate_up"]["tensor_arg_count"] == 2
    assert materialized_descriptors["logits"]["tensor_args"][0]["tensor"] == (
        "lm_head.weight"
    )


def test_qwen_resident_weight_table_owner_materializes_during_lifetime(tmp_path):
    bindings = qwen_one_layer_bindings()
    binding = tmp_path / "qwen-cuda-weight-binding.json"
    write_qwen_binding_fixture(binding, bindings)
    weight_args = write_qwen_weight_args_fixture(tmp_path, binding)

    resident_spec = importlib.util.spec_from_file_location(
        "qwen_resident_weight_table",
        ROOT / "examples" / "cuda" / "qwen_resident_weight_table.py",
    )
    assert resident_spec is not None
    assert resident_spec.loader is not None
    resident_module = importlib.util.module_from_spec(resident_spec)
    sys.modules[resident_spec.name] = resident_module
    resident_spec.loader.exec_module(resident_module)

    materialization_spec = importlib.util.spec_from_file_location(
        "qwen_persistent_weight_materialization",
        ROOT / "examples" / "cuda" / "qwen_persistent_weight_materialization.py",
    )
    assert materialization_spec is not None
    assert materialization_spec.loader is not None
    materialization_module = importlib.util.module_from_spec(materialization_spec)
    sys.modules[materialization_spec.name] = materialization_module
    materialization_spec.loader.exec_module(materialization_module)

    events = []

    def allocate_and_copy(item):
        events.append(("allocate_and_copy", item["slot_id"]))
        return 0x20000000 + item["slot_id"] * 0x1000

    def free_pointer(ptr, item):
        events.append(("free", item["slot_id"], ptr))

    owner = resident_module.ResidentWeightTableOwner(
        bindings=bindings,
        allocate_and_copy=allocate_and_copy,
        free_pointer=free_pointer,
        device=7,
        source="unit-test",
    )
    with owner:
        pointer_table = owner.pointer_table()
        assert pointer_table["kind"] == "pto_qwen_resident_weight_pointer_table"
        assert pointer_table["status"] == "resident_weight_pointer_table_ready"
        assert pointer_table["lifetime"] == "valid_until_owner_close"
        assert pointer_table["pointer_count"] == len(bindings)
        assert pointer_table["pointers"][2]["device_ptr_hex"] == "0x20002000"
        materialization = materialization_module.build_materialization_manifest(
            weight_args_json=weight_args,
            weight_binding_json=binding,
            pointer_table=pointer_table,
        )
        assert materialization["status"] == "persistent_weight_materialization_ready"
        assert materialization["bound_tensor_pointer_count"] == len(bindings)
        assert materialization["missing_pointer_count"] == 0

    closed = owner.pointer_table()
    assert closed["status"] == "resident_weight_pointer_table_closed"
    assert closed["pointer_count"] == 0
    assert closed["freed_pointer_count"] == len(bindings)
    assert events[:2] == [("allocate_and_copy", 0), ("allocate_and_copy", 1)]
    assert events[-1] == ("free", 0, 0x20000000)


def test_persistent_qwen_safetensors_fetch_status_is_reviewable(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 64},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    shard_dir = tmp_path / "qwen-shards"
    shard_dir.mkdir()
    present = shard_dir / "model-00001-of-00002.safetensors"
    present.write_bytes(b"placeholder")
    output = tmp_path / "qwen-safetensors-shards.json"

    result = subprocess.run(
        [
            sys.executable,
            "examples/cuda/qwen_safetensors_fetch.py",
            "--index-json",
            str(index),
            "--shard-dir",
            str(shard_dir),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["kind"] == "pto_qwen_safetensors_shard_status"
    assert status["status"] == "shards_missing"
    assert status["model_id"] == "Qwen/Qwen3-8B"
    assert (
        status["source_base_url"]
        == "https://huggingface.co/Qwen/Qwen3-8B/resolve/"
        "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
    )
    assert status["expected_shard_count"] == 2
    assert status["present_shard_count"] == 1
    assert status["missing_shard_count"] == 1
    shards = {item["name"]: item for item in status["shards"]}
    assert shards["model-00001-of-00002.safetensors"]["status"] == "present"
    assert shards["model-00001-of-00002.safetensors"]["tensor_count"] == 2
    assert shards["model-00002-of-00002.safetensors"]["status"] == "missing"
    assert shards["model-00002-of-00002.safetensors"]["tensor_count"] == 1
    assert shards["model-00002-of-00002.safetensors"]["url"].endswith(
        "/model-00002-of-00002.safetensors"
    )
    assert (
        Path(shards["model-00002-of-00002.safetensors"]["target_path"])
        == shard_dir / "model-00002-of-00002.safetensors"
    )
    assert "curl -L -C -" in shards["model-00002-of-00002.safetensors"][
        "resume_command"
    ]
    assert "local_shard_presence_check" in status["implemented_contracts"]
    assert "qwen_safetensors_shard_download" in status["remaining_runtime_gaps"]
    assert (
        "actual_safetensors_shape_dtype_validation"
        in status["remaining_runtime_gaps"]
    )


def test_llm_serving_matrix_tracks_pto_preflight_blocker():
    matrix = load_viewer_collection(
        VIEWER_DATA / "paper_evaluation_matrix.json"
    )
    claim = next(
        item
        for item in matrix["paper_evaluation_matrix"]
        if item["id"] == "llm_serving_paper_baselines"
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-lifecycle-b95ff321/qwen-serving-lifecycle-plan.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-tokenizer-b95ff321/qwen-prompt-accounting.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-input-binding-2026-06-01/qwen-runtime-input-binding.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-token-buffer-2026-06-01/qwen-cuda-token-buffer-binding.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-decode-args-2026-06-01/qwen-persistent-decode-args.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-kv-cache-2026-06-01/qwen-kv-cache-binding.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-shards-a16851f6/qwen-safetensors-shards.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-safetensors-a16851f6/qwen-safetensors-metadata.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-weight-residency-1ae913c9/qwen-cuda-weight-residency.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-weight-args-21589e81/qwen-persistent-weight-args.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-weight-materialization-2026-06-01/qwen-persistent-weight-materialization.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-resident-weight-table-2026-06-01/qwen-resident-weight-table.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-task-bodies-2026-06-01/qwen-persistent-task-bodies.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-proxy-live-2026-06-01/qwen-proxy-live.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == (
            "tmp/cuda-backend/pto-serving-microdecode-live-2026-06-01/"
            "qwen-microdecode-live.json"
        )
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-scaffold-2026-06-01/qwen-serving-scaffold.json"
        for ref in claim["current_evidence_refs"]
    )
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/pto-serving-preflight-2026-06-01/pto-serving-preflight.json"
        for ref in claim["current_evidence_refs"]
    )
    pto_gap = next(
        item
        for item in claim["missing_evidence_details"]
        if item["id"] == "pto_full_serving_qwen3_8b"
    )
    assert pto_gap["status"] == "missing"
    assert len(pto_gap["action"]) < 400
    action = " ".join([pto_gap["action"], *pto_gap["evidence_summary"]])
    for phrase in [
        "controlled attention-tile proxy",
        "persistent_qwen_serving_scaffold",
        "qwen_serving_lifecycle_plan",
        "qwen_prompt_accounting",
        "qwen_runtime_input_binding",
        "qwen_cuda_token_buffer_binding",
        "qwen_persistent_decode_args",
        "qwen_kv_cache_binding",
        "qwen_weight_inventory",
        "qwen_safetensors_fetch",
        "qwen_safetensors_metadata",
        "qwen_cuda_weight_binding",
        "qwen_persistent_weight_args",
        "qwen_persistent_weight_materialization",
        "qwen_resident_weight_table",
        "qwen_persistent_task_bodies",
        "local Qwen shard placement",
        "actual safetensors shape/dtype validation for 399 tensors",
        "stable CUDA weight binding slots and file offsets",
        "full CUDA residency for 16.38 GB of weights",
        "copy-back verification for 16 small tensors",
        "persistent DAG tensor_args manifest",
        "resident_weight_ptrs[slot_id]",
        "process-scoped resident weight table owner",
        "padded target-length input_ids",
        "attention_mask",
        "persistent decode token argument binding",
        "preserving tensor_args for weights",
        "dry-run KV-cache key/value pointer binding",
        "persistent DAG c/d",
        "generated persistent-device Qwen task-body source",
        "mutable KV fields c/d",
        "controlled proxy numeric oracle",
        "controlled proxy live CUDA execution",
        "controlled proxy live microdecode DAG execution",
        "full logits-buffer diagnostic reference checking",
        "1024-step MPK-policy and 64-step VDCores-policy diagnostic "
        "decode-loop execution",
        "decode-loop execution",
    ]:
        assert phrase in action
    assert "mutable KV-cache writeback ABI" not in action


def test_vdcores_scheduler_trace_keeps_diagnostic_scope_separate():
    matrix = load_viewer_collection(
        VIEWER_DATA / "paper_evaluation_matrix.json"
    )
    claim = next(
        item
        for item in matrix["paper_evaluation_matrix"]
        if item["id"] == "persistent_device_scheduler_overhead"
    )
    assert claim["status"] == "ready_for_paper_claim"
    assert claim["missing_evidence"] == []
    assert any(
        ref.get("kind") == "raw_artifact"
        and ref.get("path")
        == "tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json"
        for ref in claim["current_evidence_refs"]
    )
    assert "measurement scope" in claim["promotion_gate"]

    runs = load_viewer_collection(
        VIEWER_DATA / "paper_baseline_runs.json"
    )
    vdcores_run = next(
        item
        for item in runs["paper_baseline_runs"]
        if item["id"] == "vdcores_resource_policy_trace"
    )
    scope = vdcores_run["measurement_scope"]
    assert "without profile-slot diagnostics" in scope["latency_correctness_policy"]
    assert "diagnostic-scope" in scope["scheduler_queue_policy"]
    assert "Do not treat diagnostic scheduler fields" in scope["paper_use_rule"]

    results = load_viewer_collection(VIEWER_DATA / "results.json")
    vdcores_result = next(
        item
        for item in results["result_records"]
        if item["benchmark_id"] == "llm_serving_decode"
        and item["method_id"] == "vdcores"
        and item["commit"] == "46872fa4"
    )
    statistic = vdcores_result["statistic"]
    assert statistic["measurement_scope"] == "diagnostic_scheduler_trace"
    trace = statistic["dispatch_trace"]
    assert "tmp-only diagnostics" in trace["trace_scope"]
    assert "stable VDCores baseline build" in trace["paper_use_rule"]
    assert (
        statistic["resource_policy"]["instrumentation_scope"]
        == "diagnostic_profile_slots"
    )


def test_nvidia_changelog_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_nvidia_changelog.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_cuda_example_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_cuda_examples.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_remote_evaluation_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_remote_evaluation.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_cuda_viewer_export_generates_contract_records(tmp_path):
    capture = {
        "metadata": {
            "git_commit": "abc1234",
            "label": "fixture-capture",
        },
        "results": [
            {
                "machine": "hina",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 120,
                "device_wall_ns": 80,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 160,
                "device_wall_ns": 100,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 200,
                "device_wall_ns": 120,
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "cublas_sgemm_graph",
                "n": 1024,
                "task_count": 1,
                "tensor_tile": {
                    "rows": 16,
                    "cols": 16,
                    "inner": 16,
                    "tile_count": 4,
                },
                "host_wall_ns": 60,
                "device_wall_ns": 40,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "direct_runtime_sgemm",
                "n": 1024,
                "task_count": 1,
                "tensor_tile": {
                    "rows": 16,
                    "cols": 16,
                    "inner": 16,
                    "tile_count": 4,
                },
                "host_wall_ns": 90,
                "device_wall_ns": 70,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 1024,
                "task_count": 4,
                "tensor_tile": {
                    "rows": 16,
                    "cols": 16,
                    "inner": 16,
                    "tile_count": 4,
                },
                "host_wall_ns": 110,
                "device_wall_ns": 85,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 1024,
                "task_count": 4,
                "tensor_tile": {
                    "rows": 16,
                    "cols": 64,
                    "inner": 128,
                    "tile_count": 1,
                },
                "host_wall_ns": 210,
                "device_wall_ns": 185,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "cublas_sgemm_graph",
                "n": 1024,
                "task_count": 1,
                "tensor_tile": {
                    "rows": 16,
                    "cols": 64,
                    "inner": 128,
                    "tile_count": 1,
                },
                "host_wall_ns": 70,
                "device_wall_ns": 45,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_stream_serial",
                "n": 2,
                "task_count": 1,
                "host_wall_ns": 300,
                "device_wall_ns": 300,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_stream_parallel",
                "n": 2,
                "task_count": 1,
                "host_wall_ns": 160,
                "device_wall_ns": 160,
                "status": "pass",
            },
        ],
    }
    capture_path = tmp_path / "cuda-benchmark.json"
    output_path = tmp_path / "viewer-records.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py",
            str(capture_path),
            "--artifact-root",
            "tmp/cuda-backend/fixture/",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))

    host_record = next(
        record for record in records if record["method_id"] == "pto_host_schedule"
    )
    assert host_record["benchmark_id"] == "host_schedule_vector_ops"
    assert host_record["hardware"]["gpu"] == "A100"
    assert host_record["hardware"]["compute_target"] == "compute_80"
    assert host_record["statistic"]["sample_count"] == 3
    assert host_record["statistic"]["host_wall_ns"] == 160
    assert host_record["statistic"]["device_wall_ns"] == 100
    assert host_record["statistic"]["host_wall_p50_ns"] == 160
    assert host_record["statistic"]["host_wall_p90_ns"] == 192
    assert host_record["statistic"]["host_wall_p99_ns"] == 199
    assert host_record["statistic"]["host_wall_mean_ns"] == 160
    assert host_record["statistic"]["host_wall_stdev_ns"] == 40
    assert host_record["statistic"]["host_wall_min_ns"] == 120
    assert host_record["statistic"]["host_wall_max_ns"] == 200
    assert host_record["statistic"]["device_wall_p50_ns"] == 100
    assert host_record["statistic"]["device_wall_p90_ns"] == 116
    assert host_record["statistic"]["device_wall_p99_ns"] == 119
    assert host_record["statistic"]["device_wall_mean_ns"] == 100
    assert host_record["statistic"]["device_wall_stdev_ns"] == 20
    assert host_record["statistic"]["device_wall_min_ns"] == 80
    assert host_record["statistic"]["device_wall_max_ns"] == 120
    assert host_record["raw_artifact"] == "tmp/cuda-backend/fixture/"
    assert host_record["correctness"] == "pass"

    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "cublas_sgemm_graph"
        and record["hardware"]["gpu"] == "H200"
        for record in records
    )
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "direct_runtime"
        and record["hardware"]["gpu"] == "A100"
        and record["inputs"]["dtype"] == "float32 naive SGEMM"
        for record in records
    )
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "pto_persistent_device"
        and record["hardware"]["gpu"] == "A100"
        and record["inputs"]["dtype"] == "tf32 WMMA tensor-core, f32 accumulator"
        and record["statistic"]["sample_count"] == 1
        for record in records
    )
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "pto_persistent_device"
        and record["hardware"]["gpu"] == "A100"
        and record["inputs"]["shape"] == "n=1024, tensor tile 16x64x128"
        and record["statistic"]["device_wall_ns"] == 185
        for record in records
    )
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "cublas_sgemm_graph"
        and record["hardware"]["gpu"] == "A100"
        and record["inputs"]["shape"] == "n=1024, tensor tile 16x64x128"
        and record["statistic"]["device_wall_ns"] == 45
        for record in records
    )
    assert any(
        record["benchmark_id"] == "host_schedule_stream_concurrency"
        and record["method_id"] == "pto_stream_serial"
        and record["inputs"]["shape"] == "two independent n=1 vector kernels"
        for record in records
    )
    assert any(
        record["benchmark_id"] == "host_schedule_stream_concurrency"
        and record["method_id"] == "pto_stream_parallel"
        and record["inputs"]["shape"] == "two independent n=1 vector kernels"
        for record in records
    )


def test_triton_tensor_tile_capture_exports_fixture_records(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
            "source": "fixture",
        },
        "hardware": {
            "gpu": "A100",
            "machine": "hina",
            "compute_target": "compute_80",
            "driver": "fixture-driver",
            "cuda_toolkit": "fixture-cuda",
            "clock_policy": "fixture-clock",
        },
        "inputs": {
            "shape": "n=1024, tensor tile 16x16x16",
            "dtype": "tf32 Triton tl.dot, f32 accumulator",
            "repeat_policy": "3-repeat fixture capture",
        },
        "samples": [
            {"host_wall_ns": 1000, "device_wall_ns": 700, "max_abs_error": 0.0},
            {"host_wall_ns": 1200, "device_wall_ns": 800, "max_abs_error": 0.0},
            {"host_wall_ns": 1400, "device_wall_ns": 900, "max_abs_error": 0.0},
        ],
    }
    raw_path = tmp_path / "triton-capture.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture.py",
            "--input-json",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/triton/fixture/",
            "--viewer-output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record["benchmark_id"] == "tensor_core_tile"
    assert record["method_id"] == "triton"
    assert record["hardware"]["gpu"] == "A100"
    assert record["inputs"]["shape"] == "n=1024, tensor tile 16x16x16"
    assert record["statistic"]["sample_count"] == 3
    assert record["statistic"]["host_wall_ns"] == 1200
    assert record["statistic"]["device_wall_ns"] == 800
    assert record["statistic"]["max_abs_error"] == 0.0
    assert record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/triton/fixture/"
    assert record["correctness"] == "pass"


def test_cutlass_tensor_tile_capture_exports_fixture_records(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
            "source": "fixture",
            "cutlass_commit": "1732ed7",
        },
        "hardware": {
            "gpu": "A100",
            "machine": "hina",
            "compute_target": "compute_80",
            "driver": "fixture-driver",
            "cuda_toolkit": "fixture-cuda",
            "clock_policy": "fixture-clock",
        },
        "inputs": {
            "shape": "n=1024, tensor tile 16x16x16",
            "dtype": "tf32 CUTLASS Gemm tensor op, f32 accumulator",
            "repeat_policy": "3-repeat fixture capture",
        },
        "samples": [
            {"host_wall_ns": 2000, "device_wall_ns": 1200, "max_abs_error": 0.0},
            {"host_wall_ns": 2200, "device_wall_ns": 1300, "max_abs_error": 0.0},
            {"host_wall_ns": 2400, "device_wall_ns": 1400, "max_abs_error": 0.0},
        ],
    }
    raw_path = tmp_path / "cutlass-capture.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cutlass_tensor_tile_capture.py",
            "--input-json",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/cutlass/fixture/",
            "--viewer-output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record["benchmark_id"] == "tensor_core_tile"
    assert record["method_id"] == "cutlass"
    assert record["hardware"]["gpu"] == "A100"
    assert record["inputs"]["shape"] == "n=1024, tensor tile 16x16x16"
    assert record["statistic"]["sample_count"] == 3
    assert record["statistic"]["host_wall_ns"] == 2200
    assert record["statistic"]["device_wall_ns"] == 1300
    assert record["statistic"]["max_abs_error"] == 0.0
    assert record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/cutlass/fixture/"
    assert record["correctness"] == "pass"


def test_generated_tensor_capture_scripts_label_model_shape_tiles():
    script_dir = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
    )
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    triton_spec = importlib.util.spec_from_file_location(
        "triton_tensor_tile_capture_for_shape_test",
        script_dir / "triton_tensor_tile_capture.py",
    )
    assert triton_spec is not None
    assert triton_spec.loader is not None
    triton_module = importlib.util.module_from_spec(triton_spec)
    sys.modules[triton_spec.name] = triton_module
    triton_spec.loader.exec_module(triton_module)

    cutlass_spec = importlib.util.spec_from_file_location(
        "cutlass_tensor_tile_capture_for_shape_test",
        script_dir / "cutlass_tensor_tile_capture.py",
    )
    assert cutlass_spec is not None
    assert cutlass_spec.loader is not None
    cutlass_module = importlib.util.module_from_spec(cutlass_spec)
    sys.modules[cutlass_spec.name] = cutlass_module
    cutlass_spec.loader.exec_module(cutlass_module)

    assert triton_module.tensor_tile_shape(16, 64, 128) == (
        "n=1024, tensor tile 16x64x128"
    )
    assert cutlass_module.tensor_tile_shape(16, 64, 256) == (
        "n=1024, tensor tile 16x64x256"
    )


def test_cutlass_tensor_capture_gencode_matches_requested_arch():
    script_dir = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
    )
    cutlass_spec = importlib.util.spec_from_file_location(
        "cutlass_tensor_tile_capture_for_gencode_test",
        script_dir / "cutlass_tensor_tile_capture.py",
    )
    assert cutlass_spec is not None
    assert cutlass_spec.loader is not None
    cutlass_module = importlib.util.module_from_spec(cutlass_spec)
    sys.modules[cutlass_spec.name] = cutlass_module
    cutlass_spec.loader.exec_module(cutlass_module)

    assert cutlass_module.nvcc_gencode("compute_80") == (
        "-gencode=arch=compute_80,code=compute_80"
    )
    assert cutlass_module.nvcc_gencode("compute_90") == (
        "-gencode=arch=compute_90,code=compute_90"
    )


def test_paper_baseline_viewer_export_generates_contract_records(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "vllm_serving_and_throughput",
                "benchmark_id": "llm_serving_decode",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                    "driver": "570.86.15",
                    "cuda_toolkit": "12.8",
                    "clock_policy": "application clocks locked",
                },
                "inputs": {
                    "shape": "model=fixture,prompt_tokens=128,decode_tokens=32",
                    "dtype": "bfloat16",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_capture",
                    "sample_count": 3,
                    "end_to_end_latency_ns": 1000000,
                    "time_to_first_token_ns": 250000,
                    "inter_token_latency_ns": 50000,
                    "throughput_tokens_per_s": 640.0,
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "paper-baseline.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/vllm/",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))

    assert records == [
        {
            "benchmark_id": "llm_serving_decode",
            "method_id": "vllm",
            "hardware": {
                "gpu": "H200",
                "machine": "dasys-h200x8",
                "compute_target": "compute_90",
                "driver": "570.86.15",
                "cuda_toolkit": "12.8",
                "clock_policy": "application clocks locked",
            },
            "commit": "abc1234",
            "inputs": {
                "shape": "model=fixture,prompt_tokens=128,decode_tokens=32",
                "dtype": "bfloat16",
                "repeat_policy": "warmup=1,repeat=3",
            },
            "statistic": {
                "kind": "paper_baseline_capture",
                "serving_coverage": "full_serving",
                "sample_count": 3,
                "host_wall_ns": 1000000,
                "device_wall_ns": 0,
                "end_to_end_latency_ns": 1000000,
                "time_to_first_token_ns": 250000,
                "inter_token_latency_ns": 50000,
                "throughput_tokens_per_s": 640.0,
            },
            "raw_artifact": "tmp/cuda-backend/paper-baselines/vllm/",
            "correctness": "pass",
        }
    ]


def test_paper_baseline_viewer_export_preserves_scheduler_trace_metadata(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "mpk_persistent_scheduler_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, nine-task layered-cross DAG",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 900000,
                    "device_wall_ns": 700000,
                    "scheduler_overhead_ns": 42000,
                    "dispatch_trace": {
                        "task_count": 9,
                        "func_ids": [1, 2, 11, 1, 2, 1, 6, 1, 1],
                    },
                    "resource_policy": {
                        "scheduler_blocks": 3,
                        "worker_blocks": 4,
                    },
                    "task_registry": {
                        "generated_kernel": "mirage_persistent_kernel",
                    },
                },
                "correctness": "pass",
            },
            {
                "paper_baseline_run_id": "vdcores_resource_policy_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, virtual-core resource trace",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 1100000,
                    "device_wall_ns": 850000,
                    "scheduler_overhead_ns": 51000,
                    "dispatch_trace": {
                        "task_count": 9,
                        "queue_ids": ["compute", "memory"],
                    },
                    "queue_pressure": {
                        "ready_queue_max": 4,
                        "memory_queue_max": 2,
                    },
                    "resource_policy": {
                        "virtual_cores": 8,
                        "memory_compute_split": "2:6",
                    },
                },
                "correctness": "pass",
            },
        ],
    }
    raw_path = tmp_path / "scheduler-baseline.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/scheduler-fixture/",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))
    by_method = {record["method_id"]: record for record in records}

    mpk_statistic = by_method["mpk"]["statistic"]
    assert mpk_statistic["scheduler_overhead_ns"] == 42000
    assert mpk_statistic["dispatch_trace"]["task_count"] == 9
    assert mpk_statistic["resource_policy"]["scheduler_blocks"] == 3
    assert mpk_statistic["task_registry"]["generated_kernel"]

    vdcores_statistic = by_method["vdcores"]["statistic"]
    assert vdcores_statistic["scheduler_overhead_ns"] == 51000
    assert vdcores_statistic["dispatch_trace"]["queue_ids"] == [
        "compute",
        "memory",
    ]
    assert vdcores_statistic["queue_pressure"]["ready_queue_max"] == 4
    assert vdcores_statistic["resource_policy"]["virtual_cores"] == 8


def test_thunderkittens_capture_builds_serving_decode_result():
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_mha_capture.py"
    )
    spec = importlib.util.spec_from_file_location("thunderkittens_mha_capture", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_raw_result_record(
        paper_baseline_run_id="thunderkittens_decode_attention_tile",
        benchmark_id="llm_serving_decode",
        machine="dasys-h200x8",
        cuda_toolkit="12.8",
        clock_policy="not recorded",
        gpu_metadata={
            "gpu": "NVIDIA H200 NVL",
            "driver": "580.126.20",
            "compute_target": "compute_90",
        },
        shape={
            "b": 4,
            "h": 1,
            "n": 256,
            "d": 64,
            "causal": True,
            "dtype": "bfloat16",
        },
        latency={
            "warmup": 5,
            "repeats": 20,
            "sample_count": 20,
            "p50_ns": 32000,
        },
        correctness={"status": "pass"},
        serving_workload_id="vdcores_offline_decode",
        prompt_tokens=128,
        decode_tokens=64,
    )

    assert result["paper_baseline_run_id"] == "thunderkittens_decode_attention_tile"
    assert result["benchmark_id"] == "llm_serving_decode"
    assert result["hardware"]["gpu"] == "H200"
    assert (
        result["inputs"]["shape"]
        == "vdcores_offline_decode,mha_h100,b=4,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64"
    )
    assert result["metrics"]["kind"] == "paper_baseline_serving_tile_capture"
    assert result["metrics"]["serving_coverage"] == (
        "controlled_attention_tile_proxy"
    )
    assert result["metrics"]["end_to_end_latency_ns"] == 32000
    assert result["metrics"]["time_to_first_token_ns"] == 32000
    assert result["metrics"]["inter_token_latency_ns"] == 500
    assert result["metrics"]["throughput_tokens_per_s"] == 8000000
    assert result["correctness"] == "pass"


def test_thunderkittens_full_sweep_capture_builds_importable_record():
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_full_sweep_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "thunderkittens_full_sweep_capture",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_raw_result_record(
        machine="dasys-h200x8",
        cuda_toolkit="12.8",
        clock_policy="not recorded",
        gpu_metadata={
            "gpu": "NVIDIA H200 NVL",
            "driver": "580.126.20",
            "compute_target": "compute_90",
        },
        shape={"b": 1, "h": 1, "n": 256, "d": 64, "causal": True, "dtype": "bfloat16"},
        latency={
            "warmup": 5,
            "repeats": 20,
            "sample_count": 20,
            "p50_ns": 32000,
        },
        correctness={"status": "pass", "max_abs_diff": 0.001953125},
    )

    assert result["paper_baseline_run_id"] == "thunderkittens_full_sweep"
    assert result["benchmark_id"] == "tensor_core_tile"
    assert result["hardware"]["gpu"] == "H200"
    assert result["inputs"]["shape"] == (
        "mha_h100_full_sweep,b=1,h=1,n=256,d=64,causal=True"
    )
    assert result["metrics"]["kind"] == "paper_baseline_full_sweep_capture"
    assert result["metrics"]["device_wall_ns"] == 32000
    assert result["metrics"]["throughput"] > 0
    assert result["metrics"]["max_abs_error"] == 0.001953125
    assert result["correctness"] == "pass"


def test_thunderkittens_rotary_capture_builds_importable_record():
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_rotary_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "thunderkittens_rotary_capture",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_result_record(
        metadata={
            "machine": "bizhaoh200",
            "cuda_toolkit": "12.8",
            "clock_policy": "not recorded",
            "gpu_metadata": {
                "gpu": "H200",
                "raw_gpu_name": "NVIDIA H200 NVL",
                "driver": "580.126.20",
                "compute_target": "compute_90",
            },
        },
        shape_result={
            "shape": {
                "b": 2,
                "h": 16,
                "n": 1024,
                "d": 64,
                "dtype": "bfloat16",
            },
            "latency": {
                "warmup": 5,
                "repeats": 20,
                "sample_count": 20,
                "p50_ns": 74015,
            },
            "correctness": {"status": "pass", "max_abs_diff": 0.150390625},
            "flops": module.rotary_flops(2, 16, 1024, 64),
        },
    )

    assert result["paper_baseline_run_id"] == "thunderkittens_non_mha_rotary"
    assert result["benchmark_id"] == "tensor_core_tile"
    assert result["hardware"]["gpu"] == "H200"
    assert result["inputs"]["shape"] == "rotary,b=2,h=16,n=1024,d=64"
    assert result["metrics"]["kind"] == "paper_baseline_non_mha_rotary_capture"
    assert result["metrics"]["device_wall_ns"] == 74015
    assert result["metrics"]["rotary_flops"] == 6291456
    assert result["metrics"]["throughput"] > 0
    assert result["metrics"]["max_abs_error"] == 0.150390625
    assert result["correctness"] == "pass"


def test_thunderkittens_gemm_compatibility_probe_marks_qwen_tiles_incompatible():
    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_gemm_compatibility_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "thunderkittens_gemm_compatibility_probe",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bf16_source = """
    using base_tile = st_bf<64, 64>;
    template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
    struct matmul_template;
    """
    int8_source = """
    static_assert(_Mb == 128);
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0);
    static_assert(_Kb >= 32 && _Kb % 32 == 0);
    """
    report = module.build_compatibility_report(
        baseline_dir="tmp/baselines/thunderkittens",
        bf16_source=bf16_source,
        int8_source=int8_source,
        targets=[
            module.TensorTileTarget(
                id="qwen_attention_projection_tile",
                rows=16,
                cols=64,
                inner=128,
            ),
            module.TensorTileTarget(
                id="qwen_mlp_projection_tile",
                rows=16,
                cols=64,
                inner=256,
            ),
        ],
    )

    assert report["status"] == (
        "exact_qwen_tile_not_supported_by_current_gemm_entrypoints"
    )
    by_name = {
        entry["entrypoint_id"]: entry for entry in report["entrypoints"]
    }
    bf16 = by_name["bf16_h100_gemm"]
    assert bf16["base_tile"] == {"rows": 64, "cols": 64}
    assert bf16["default_output_block"] == {"rows": 128, "cols": 256}
    assert {item["target_id"] for item in bf16["target_compatibility"]} == {
        "qwen_attention_projection_tile",
        "qwen_mlp_projection_tile",
    }
    assert all(
        item["exact_target_compatible"] is False
        for item in bf16["target_compatibility"]
    )
    assert all(
        "target rows 16 are smaller than BF16 base tile rows 64"
        in item["reason"]
        for item in bf16["target_compatibility"]
    )
    int8 = by_name["int8_h100_gemm"]
    assert int8["dtype"] == "int8"
    assert int8["comparability_scope"] == (
        "dtype_mismatch_for_current_qwen_float_tensor_claim"
    )
    assert all(
        item["exact_target_compatible"] is False
        for item in int8["target_compatibility"]
    )


def test_vdcores_instruction_window_plan_validator_guards_handoff_contract():
    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "vdcores_validate_instruction_window_plan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vdcores_validate_instruction_window_plan",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plan = {
        "schema_version": 1,
        "mode": "vdcores_qwen3_8b_shared_instruction_window_plan",
        "status": "analysis_only",
        "model": "Qwen/Qwen3-8B",
        "serving_workload_id": "vdcores_offline_decode",
        "shared_instruction_capacity": {
            "instructions_per_sm": 512,
            "num_sms": 132,
        },
        "observed_instruction_pressure": {
            "max_cinsts_per_sm": 2177,
            "max_minsts_per_sm": 15042,
            "overflow_cinst_sm_count": 132,
            "overflow_minst_sm_count": 132,
        },
        "minimum_window_lower_bound": {
            "compute_instruction_windows": 5,
            "memory_instruction_windows": 30,
            "worst_case_windows_per_sm": 30,
        },
        "segmented_window_manifest": {
            "manifest_kind": "per_sm_uniform_lower_bound",
            "compute_instruction_windows": [
                {
                    "index": 0,
                    "instruction_start": 0,
                    "instruction_end": 512,
                    "instruction_count": 512,
                    "capacity_ok": True,
                },
                {
                    "index": 1,
                    "instruction_start": 512,
                    "instruction_end": 1024,
                    "instruction_count": 512,
                    "capacity_ok": True,
                },
                {
                    "index": 2,
                    "instruction_start": 1024,
                    "instruction_end": 1536,
                    "instruction_count": 512,
                    "capacity_ok": True,
                },
                {
                    "index": 3,
                    "instruction_start": 1536,
                    "instruction_end": 2048,
                    "instruction_count": 512,
                    "capacity_ok": True,
                },
                {
                    "index": 4,
                    "instruction_start": 2048,
                    "instruction_end": 2177,
                    "instruction_count": 129,
                    "capacity_ok": True,
                },
            ],
            "memory_instruction_windows": [
                {
                    "index": index,
                    "instruction_start": index * 512,
                    "instruction_end": min((index + 1) * 512, 15042),
                    "instruction_count": min((index + 1) * 512, 15042)
                    - index * 512,
                    "capacity_ok": True,
                }
                for index in range(30)
            ],
            "max_compute_window_instruction_count": 512,
            "max_memory_window_instruction_count": 512,
        },
        "required_runtime_change": {
            "preferred_path": "segmented_token_windowed_shared_instruction_schedule",
            "builder_requirements": [
                "split the Qwen3-8B decode64 schedule into instruction windows",
                "emit per-window instruction tables no larger than max_insts",
                "preserve token, KV-cache, and stage dependency order",
                "record window metadata in the raw benchmark artifact",
            ],
            "runtime_requirements": [
                "reload or advance shared instruction windows without a new model load",
                "keep resident tensors and scheduler state live across windows",
                "report correctness and per-window timing before viewer import",
            ],
            "pre_import_checks": [
                "every emitted compute and memory window is <= max_insts",
                "all windows execute under one model residency and KV-cache owner",
                "window dependency handoff preserves ready queues and token state",
                "raw artifact records per-window timing and correctness status",
            ],
        },
    }

    assert module.validate_instruction_window_plan(plan) == []

    bad_plan = json.loads(json.dumps(plan))
    bad_plan["segmented_window_manifest"]["memory_instruction_windows"][0][
        "instruction_count"
    ] = 513
    assert (
        "memory window 0 exceeds shared capacity 512"
        in module.validate_instruction_window_plan(bad_plan)
    )

    importable_plan = json.loads(json.dumps(plan))
    importable_plan["status"] = "pass"
    assert (
        "window plan must remain analysis_only until a runnable baseline exists"
        in module.validate_instruction_window_plan(importable_plan)
    )


def test_mpk_native_token_capture_builds_importable_record():
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "mpk_native_token_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mpk_native_token_capture",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_result_record(
        token_payload={
            "prompt_length": 39,
            "generate_length": 2,
            "latency_ms_per_token": 238.19888305664062,
            "mode": "torch",
        },
        machine="bizhaoh200",
        pto_commit="61af3f01",
        gpu="H200",
        raw_gpu_name="NVIDIA H200",
        driver="see raw artifact",
        compute_target="compute_90",
        cuda_toolkit="see raw artifact",
        dtype="bfloat16",
        clock_policy="not recorded",
    )

    assert result["paper_baseline_run_id"] == "mpk_qwen3_native_token_bringup"
    assert result["benchmark_id"] == "llm_serving_decode"
    assert result["hardware"]["gpu"] == "H200"
    assert result["inputs"]["shape"] == (
        "model=Qwen/Qwen3-0.6B,prompt_tokens=39,"
        "decode_tokens=2,batch=1,mode=torch"
    )
    assert result["metrics"]["kind"] == "mpk_native_token_bringup"
    assert result["metrics"]["serving_coverage"] == "native_bringup"
    assert result["metrics"]["sample_count"] == 1
    assert result["metrics"]["end_to_end_latency_ns"] == 476397766
    assert result["metrics"]["inter_token_latency_ns"] == 238198883
    assert result["metrics"]["time_per_output_token_ns"] == 238198883
    assert result["metrics"]["throughput_tokens_per_s"] > 4.0
    assert result["metrics"]["completed_requests"] == 1
    assert result["metrics"]["failed_requests"] == 0
    assert result["correctness"] == "pass"


def test_mpk_qwen3_persistent_capture_builds_caveated_viewer_result(tmp_path):
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "mpk_qwen3_persistent_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mpk_qwen3_persistent_capture",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_result_record(
        persistent_payload={
            "token_ids": [151667, 198, 32313],
            "text": "decoded",
            "prompt_length": 39,
            "generate_length": 1024,
            "latency_ms_per_token": 0.00011056249786633998,
            "mode": "mpk",
        },
        persistent_path=tmp_path / "persistent.json",
        native_payload={
            "token_ids": [151667, 198],
            "text": "decoded",
            "prompt_length": 39,
            "generate_length": 2,
            "latency_ms_per_token": 52.96201705932617,
            "mode": "torch",
        },
        native_path=tmp_path / "native.json",
        model="Qwen/Qwen3-8B",
        machine="bizhaoh200",
        gpu="H200",
        raw_gpu_name="NVIDIA H200 NVL",
        driver="580.126.20",
        compute_target="compute_90a",
        cuda_toolkit="12.8",
        dtype="bfloat16",
        clock_policy="not pinned",
    )

    assert result["paper_baseline_run_id"] == "mpk_qwen3_native_vs_persistent"
    assert result["benchmark_id"] == "llm_serving_decode"
    assert result["inputs"]["shape"] == (
        "mpk_offline_decode,Qwen/Qwen3-8B,batch=1,"
        "target_prompt_tokens=64,actual_prompt_tokens=39,"
        "decode_tokens=1024,mode=mpk_persistent"
    )
    assert "asynchronously" in result["inputs"]["repeat_policy"]
    assert result["metrics"]["kind"] == (
        "mpk_qwen3_persistent_decode_async_timing_caveat"
    )
    assert result["metrics"]["serving_coverage"] == "full_serving_latency_caveat"
    assert result["metrics"]["sample_count"] == 1
    assert result["metrics"]["decode_tokens"] == 1024
    assert result["metrics"]["time_to_first_token_ns"] > 0
    assert result["metrics"]["inter_token_latency_ns"] > 0
    assert result["metrics"]["throughput_tokens_per_s"] > 0
    assert result["metrics"]["saved_debug_token_count"] == 3
    assert result["correctness"] == "pass"


def test_paper_baseline_results_update_marks_imported_run(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "mpk_persistent_scheduler_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, nine-task layered-cross DAG",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 900000,
                    "device_wall_ns": 700000,
                    "scheduler_overhead_ns": 42000,
                    "throughput_tokens_per_s": 12.5,
                    "throughput_tokens_per_s_stdev": 1.25,
                    "throughput_tokens_per_s_samples": [11.0, 12.5, 14.0],
                    "completed_requests": 3,
                    "failed_requests": 0,
                    "dispatch_trace": {"task_count": 9},
                    "resource_policy": {
                        "scheduler_blocks": 3,
                        "worker_blocks": 4,
                    },
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "mpk-scheduler.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    results_path = tmp_path / "results.json"
    runs_path = tmp_path / "paper_baseline_runs.json"
    audit_path = tmp_path / "paper_readiness_audit.json"
    viewer_path = tmp_path / "viewer-records.json"
    results_path.write_text(
        json.dumps(
            load_viewer_collection(VIEWER_DATA / "results.json"),
            indent=2,
        ),
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps(
            load_viewer_collection(
                VIEWER_DATA / "paper_baseline_runs.json"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/mpk/fixture/",
            "--results",
            str(results_path),
            "--runs",
            str(runs_path),
            "--viewer-output",
            str(viewer_path),
            "--audit-output",
            str(audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    viewer_records = json.loads(viewer_path.read_text(encoding="utf-8"))
    assert len(viewer_records) == 1
    assert viewer_records[0]["method_id"] == "mpk"
    assert viewer_records[0]["statistic"]["scheduler_overhead_ns"] == 42000
    assert viewer_records[0]["statistic"]["throughput_tokens_per_s_stdev"] == 1.25
    assert viewer_records[0]["statistic"]["throughput_tokens_per_s_samples"] == [
        11.0,
        12.5,
        14.0,
    ]
    assert viewer_records[0]["statistic"]["completed_requests"] == 3
    assert viewer_records[0]["statistic"]["failed_requests"] == 0

    updated_results = load_viewer_collection(results_path)
    assert any(
        record["method_id"] == "mpk"
        and record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/mpk/fixture/"
        for record in updated_results["result_records"]
    )

    updated_runs = json.loads(runs_path.read_text(encoding="utf-8"))
    run_by_id = {
        record["id"]: record
        for record in updated_runs["paper_baseline_runs"]
    }
    assert run_by_id["mpk_persistent_scheduler_trace"]["status"] == (
        "imported_to_viewer"
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    persistent_claim = {
        record["id"]: record for record in audit["claim_audits"]
    }["persistent_device_scheduler_overhead"]
    mpk_status = [
        record
        for record in persistent_claim["paper_baseline_run_statuses"]
        if record["id"] == "mpk_persistent_scheduler_trace"
    ][0]
    assert mpk_status["status"] == "imported_to_viewer"


def test_paper_baseline_results_update_rejects_missing_required_metric(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "mpk_persistent_scheduler_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, nine-task layered-cross DAG",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 900000,
                    "device_wall_ns": 700000,
                    "dispatch_trace": {"task_count": 9},
                    "resource_policy": {
                        "scheduler_blocks": 3,
                        "worker_blocks": 4,
                    },
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "missing-scheduler-overhead.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    results_path = tmp_path / "results.json"
    runs_path = tmp_path / "paper_baseline_runs.json"
    audit_path = tmp_path / "paper_readiness_audit.json"
    viewer_path = tmp_path / "viewer-records.json"
    results_path.write_text(
        json.dumps(
            load_viewer_collection(VIEWER_DATA / "results.json"),
            indent=2,
        ),
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps(
            load_viewer_collection(
                VIEWER_DATA / "paper_baseline_runs.json"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/mpk/fixture/",
            "--results",
            str(results_path),
            "--runs",
            str(runs_path),
            "--viewer-output",
            str(viewer_path),
            "--audit-output",
            str(audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "scheduler_overhead" in result.stdout
    assert not viewer_path.exists()
    assert not audit_path.exists()


def test_paper_baseline_run_readiness_probe_exports_run_blockers(tmp_path):
    baseline_root = tmp_path / "baselines"
    mpk_root = baseline_root / "mirage-mpk"
    vdcores_root = baseline_root / "vdcores"
    vllm_root = baseline_root / "vllm"
    sglang_root = baseline_root / "sglang"
    (mpk_root / "demo" / "qwen3").mkdir(parents=True)
    (mpk_root / "demo" / "qwen3" / "demo.py").write_text(
        "print('mpk fixture')\n",
        encoding="utf-8",
    )
    (vdcores_root / "app" / "python" / "llama3").mkdir(parents=True)
    (vdcores_root / "python" / "dae").mkdir(parents=True)
    (vdcores_root / "app" / "python" / "llama3" / "sched.py").write_text(
        "print('vdcores fixture')\n",
        encoding="utf-8",
    )
    vllm_root.mkdir(parents=True)
    sglang_root.mkdir(parents=True)

    baselines = {
        "paper_baselines": [
            {
                "id": "mpk",
                "source": {
                    "local_tmp_path": str(mpk_root),
                },
            },
            {
                "id": "vdcores",
                "source": {
                    "local_tmp_path": str(vdcores_root),
                },
            },
            {
                "id": "vllm",
                "source": {
                    "local_tmp_path": str(vllm_root),
                },
            },
            {
                "id": "sglang",
                "source": {
                    "local_tmp_path": str(sglang_root),
                },
            },
        ]
    }
    runs = {
        "paper_baseline_runs": [
            {
                "id": "mpk_persistent_scheduler_trace",
                "paper_baseline_id": "mpk",
                "title": "MPK Persistent Scheduler Trace",
                "run_commands": [
                    "cd tmp/baselines/mirage-mpk && python demo/qwen3/demo.py --use-mirage --profiling"
                ],
                "model_access": {
                    "requires_hf_token": False,
                    "public_models": ["Qwen/Qwen3-1.7B"],
                },
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json"
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
            {
                "id": "vdcores_resource_policy_trace",
                "paper_baseline_id": "vdcores",
                "title": "VDCores Resource Policy Trace",
                "run_commands": [
                    "cd tmp/baselines/vdcores && python app/python/llama3/sched.py --correctness"
                ],
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/vdcores/resource-policy-trace.json"
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
            {
                "id": "vllm_serving_and_throughput",
                "paper_baseline_id": "vllm",
                "title": "vLLM Serving And Throughput",
                "status": "planned_not_run",
                "run_commands": [
                    "vllm bench serve --backend vllm --model <model>",
                    "vllm bench throughput --model <model>",
                ],
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/vllm/bench-serve.json",
                    "tmp/cuda-backend/paper-baselines/vllm/bench-throughput.json",
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
            {
                "id": "sglang_serving_and_offline",
                "paper_baseline_id": "sglang",
                "title": "SGLang Serving And Offline",
                "status": "planned_not_run",
                "run_commands": [
                    "python -m sglang.bench_serving --backend sglang",
                    "python -m sglang.bench_offline_throughput --model-path <model>",
                ],
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/sglang/bench-serving.json",
                    "tmp/cuda-backend/paper-baselines/sglang/offline-throughput.json",
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
        ]
    }
    probes = {
        "paper_baseline_probes": [
            {
                "id": "mpk_source_entrypoints",
                "paper_baseline_id": "mpk",
                "latest_status": "pass",
                "latest_machine_status": [],
            },
            {
                "id": "vdcores_source_entrypoints",
                "paper_baseline_id": "vdcores",
                "latest_status": "pass",
                "latest_machine_status": [],
            },
            {
                "id": "vllm_source_entrypoints",
                "paper_baseline_id": "vllm",
                "latest_status": "partial",
                "latest_machine_status": [
                    {
                        "gpu": "H200",
                        "status": "partial",
                        "blocking_gaps": ["python_module failed: vllm"],
                    }
                ],
            },
            {
                "id": "sglang_source_entrypoints",
                "paper_baseline_id": "sglang",
                "latest_status": "partial",
                "latest_machine_status": [
                    {
                        "gpu": "H200",
                        "status": "partial",
                        "blocking_gaps": [
                            "python_import failed: sglang.bench_serving"
                        ],
                    }
                ],
            },
        ]
    }
    baselines_path = tmp_path / "paper_baselines.json"
    runs_path = tmp_path / "paper_baseline_runs.json"
    probes_path = tmp_path / "paper_baseline_probes.json"
    env_plans_path = tmp_path / "paper_baseline_environment_plans.json"
    output_root = tmp_path / "run-readiness"
    viewer_output = tmp_path / "paper_baseline_run_readiness.json"
    baselines_path.write_text(json.dumps(baselines), encoding="utf-8")
    runs_path.write_text(json.dumps(runs), encoding="utf-8")
    probes_path.write_text(json.dumps(probes), encoding="utf-8")
    env_plans_path.write_text(
        json.dumps(
            {
                "paper_baseline_environment_plans": [
                    {
                        "id": "vllm_runtime_environment",
                        "paper_baseline_id": "vllm",
                        "status": "plan_ready",
                        "raw_artifact": "tmp/cuda-backend/paper-baselines/environment-plans/run-readiness-fixture/environment-plans.json",
                    },
                    {
                        "id": "sglang_runtime_environment",
                        "paper_baseline_id": "sglang",
                        "status": "plan_ready",
                        "raw_artifact": "tmp/cuda-backend/paper-baselines/environment-plans/run-readiness-fixture/environment-plans.json",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py",
            "--runs",
            str(runs_path),
            "--baselines",
            str(baselines_path),
            "--probes",
            str(probes_path),
            "--env-plans",
            str(env_plans_path),
            "--output-root",
            str(output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    status = json.loads(viewer_output.read_text(encoding="utf-8"))
    records = {
        item["paper_baseline_run_id"]: item
        for item in status["paper_baseline_run_readiness"]
    }
    assert {
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
    } <= set(records)

    assert records["mpk_persistent_scheduler_trace"]["latest_status"] == "pass"
    assert not any(
        "HF_TOKEN" in gap
        for gap in records["mpk_persistent_scheduler_trace"]["blocking_gaps"]
    )
    assert any(
        check["kind"] == "environment"
        and check["name"] == "public_model_access"
        and check["status"] == "pass"
        for check in records["mpk_persistent_scheduler_trace"]["checks"]
    )
    vdcores = records["vdcores_resource_policy_trace"]
    assert vdcores["latest_status"] == "partial"
    assert any("dae.runtime" in gap for gap in vdcores["blocking_gaps"])
    assert records["vllm_serving_and_throughput"]["latest_status"] == "partial"
    assert any(
        "python_module failed: vllm" in gap
        for gap in records["vllm_serving_and_throughput"]["blocking_gaps"]
    )
    assert records["sglang_serving_and_offline"]["latest_status"] == "partial"
    assert any(
        "sglang.bench_serving" in gap
        for gap in records["sglang_serving_and_offline"]["blocking_gaps"]
    )
    assert any(
        check["kind"] == "environment_plan"
        and check["status"] == "pass"
        for check in records["vllm_serving_and_throughput"]["checks"]
    )
    assert (output_root / "run-readiness.json").is_file()
    artifact = json.loads((output_root / "run-readiness.json").read_text())
    assert artifact["metadata"]["pto_commit"] == "abc1234"


def test_paper_baseline_environment_plan_exports_isolated_serving_envs(tmp_path):
    baseline_root = tmp_path / "baselines"
    vllm_root = baseline_root / "vllm"
    sglang_root = baseline_root / "sglang"
    (vllm_root / "requirements").mkdir(parents=True)
    (vllm_root / "requirements" / "build").mkdir(parents=True)
    (vllm_root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["torch == 2.11.0", "setuptools-rust>=1.9.0", "setuptools-scm>=8.0"]
[project]
name = "vllm"
dependencies = ["uvloop", "pydantic", "cbor2"]
""",
        encoding="utf-8",
    )
    (vllm_root / "requirements" / "build" / "cuda.txt").write_text(
        "cmake>=3.26.1\nninja\nsetuptools-rust>=1.9.0\nsetuptools-scm>=8.0\n",
        encoding="utf-8",
    )
    (vllm_root / "requirements" / "common.txt").write_text(
        "pydantic >= 2.12.0\ncbor2\nuvloop==0.22.1\n",
        encoding="utf-8",
    )
    (vllm_root / "requirements" / "cuda.txt").write_text(
        "torch==2.11.0\ntorchvision==0.26.0\nflashinfer-python==0.6.11.post2\ntilelang==0.1.9\n",
        encoding="utf-8",
    )
    (sglang_root / "python").mkdir(parents=True)
    (sglang_root / "python" / "pyproject.toml").write_text(
        """
[project]
name = "sglang"
dependencies = [
  "sglang",
  "torch==2.11.0",
  "torchvision",
  "orjson",
  "uvloop",
  "flashinfer_python[cu13]==0.6.11.post1",
  "tilelang==0.1.8",
]
""",
        encoding="utf-8",
    )
    baselines = {
        "paper_baselines": [
            {
                "id": "vllm",
                "source": {
                    "local_tmp_path": str(vllm_root),
                    "commit": "1234567890abcdef1234567890abcdef12345678",
                },
            },
            {
                "id": "sglang",
                "source": {
                    "local_tmp_path": str(sglang_root),
                    "commit": "abcdef1234567890abcdef1234567890abcdef12",
                },
            },
        ]
    }
    baselines_path = tmp_path / "paper_baselines.json"
    output_root = tmp_path / "environment-plans"
    viewer_output = tmp_path / "paper_baseline_environment_plans.json"
    baselines_path.write_text(json.dumps(baselines), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py",
            "--baselines",
            str(baselines_path),
            "--output-root",
            str(output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(viewer_output.read_text(encoding="utf-8"))
    records = {
        item["paper_baseline_id"]: item
        for item in payload["paper_baseline_environment_plans"]
    }
    assert set(records) == {"vllm", "sglang"}
    assert records["vllm"]["status"] == "plan_ready"
    assert records["sglang"]["status"] == "plan_ready"
    assert records["vllm"]["environment_path"].startswith(
        "tmp/cuda-backend/paper-baselines/envs/vllm-12345678"
    )
    assert records["sglang"]["environment_path"].startswith(
        "tmp/cuda-backend/paper-baselines/envs/sglang-abcdef12"
    )
    assert all(
        "--system-site-packages" in command
        for command in [
            records["vllm"]["install_commands"][0],
            records["sglang"]["install_commands"][0],
        ]
    )
    assert all(
        command.startswith("REPO_ROOT=$PWD && cd ")
        or command.startswith("REPO_ROOT=$PWD && VLLM_VERSION_OVERRIDE=")
        for command in [
            command
            for command in (
                records["vllm"]["install_commands"]
                + records["sglang"]["install_commands"]
            )
            if command.startswith("REPO_ROOT=") or command.startswith("cd ")
        ]
    )
    assert ".venv" not in " ".join(
        records["vllm"]["install_commands"] + records["sglang"]["install_commands"]
    )
    assert not any(
        command.startswith("python -m pip")
        for command in (
            records["vllm"]["install_commands"]
            + records["sglang"]["install_commands"]
        )
    )
    assert all(
        "PYTHONNOUSERSITE=1" in command
        for command in (
            records["vllm"]["install_commands"]
            + records["sglang"]["install_commands"]
        )
        if "-m pip install" in command
    )
    assert all(
        "PATH=" in command and "/bin:$PATH" in command
        for command in (
            records["vllm"]["install_commands"]
            + records["sglang"]["install_commands"]
        )
        if "-m pip install" in command
    )
    assert "--user" not in " ".join(
        records["vllm"]["install_commands"] + records["sglang"]["install_commands"]
    )
    assert "PYTHONNOUSERSITE=1" in " ".join(
        records["sglang"]["validation_commands"]
    )
    vllm_packages = {
        package["name"]: package
        for package in records["vllm"]["critical_packages"]
    }
    assert vllm_packages["torch"]["declared"]
    assert vllm_packages["setuptools-rust"]["declared"]
    assert vllm_packages["setuptools-scm"]["declared"]
    assert vllm_packages["cbor2"]["declared"]
    vllm_manual_packages = {
        package["name"]: package
        for package in records["vllm"]["manual_packages"]
    }
    assert "old system SciPy" in vllm_manual_packages["scipy"]["why"]
    assert "vllm._aiter_ops" in vllm_manual_packages["pandas"]["why"]
    assert "system numexpr" in vllm_manual_packages["numexpr"]["why"]
    assert "system bottleneck" in vllm_manual_packages["bottleneck"]["why"]
    assert any(
        "requirements/build/cuda.txt" in command
        for command in records["vllm"]["install_commands"]
    )
    assert records["vllm"]["build_source_path"].startswith(
        "tmp/cuda-backend/paper-baselines/source-overlays/"
        "vllm-12345678-spinloop-cpython"
    )
    assert records["sglang"]["build_source_path"] == str(sglang_root)
    assert len(records["vllm"]["source_overlay_commands"]) == 1
    assert (
        "vllm_spinloop_source_overlay.py"
        in records["vllm"]["source_overlay_commands"][0]
    )
    assert records["vllm"]["source_path"] in records["vllm"][
        "source_overlay_commands"
    ][0]
    assert records["vllm"]["build_source_path"] in records["vllm"][
        "source_overlay_commands"
    ][0]
    assert any(
        records["vllm"]["build_source_path"] in command
        and "pip install --no-build-isolation -e ." in command
        and "VLLM_VERSION_OVERRIDE" in command
        and "setuptools_scm.get_version" in command
        and records["vllm"]["source_path"] in command
        for command in records["vllm"]["install_commands"]
    )
    assert any(
        "scipy>=1.15.0" in command
        and "pandas>=2.2.0" in command
        and "numexpr" in command
        and "bottleneck" in command
        for command in records["vllm"]["install_commands"]
    )
    assert all(
        records["vllm"]["build_source_path"] in command
        for command in records["vllm"]["validation_commands"]
    )
    assert any(
        "vllm.model_executor.models.qwen3" in command
        for command in records["vllm"]["validation_commands"]
    )
    assert records["vllm"]["preflight_after_install_steps"] == 6
    assert len(records["vllm"]["preflight_commands"]) == 1
    assert "vllm_spinloop_preflight.py" in records["vllm"]["preflight_commands"][0]
    assert records["vllm"]["build_source_path"] in records["vllm"][
        "preflight_commands"
    ][0]
    assert records["sglang"]["preflight_commands"] == []
    assert (output_root / "environment-plans.json").is_file()


def test_vllm_spinloop_preflight_rejects_python310_limited_api(tmp_path):
    source = tmp_path / "vllm"
    (source / "csrc").mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        """
define_extension_target(
  spinloop
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES csrc/spinloop.cpp
  USE_SABI 3.11
  WITH_SOABI)
""",
        encoding="utf-8",
    )
    (source / "csrc" / "spinloop.cpp").write_text(
        "void f() { Py_buffer buffer; PyBuffer_Release(&buffer); }\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[3, 10, 12]\\n'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/vllm_spinloop_preflight.py",
            "--source",
            str(source),
            "--env-python",
            str(fake_python),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert "Py_buffer" in payload["blocker"]
    assert "Python 3.10.12" in payload["blocker"]


def test_vllm_spinloop_source_overlay_patches_copy_only(tmp_path):
    source = tmp_path / "vllm"
    overlay = tmp_path / "overlays" / "vllm"
    (source / "csrc").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / ".deps" / "cutlass-subbuild").mkdir(parents=True)
    (source / ".deps" / "cutlass-subbuild" / "CMakeCache.txt").write_text(
        "stale cache\n",
        encoding="utf-8",
    )
    (source / "CMakeLists.txt").write_text(
        """
set(VLLM_SPINLOOP_EXT_SRC "csrc/spinloop.cpp")
set(SPINLOOP_COMPILE_FLAGS "")
define_extension_target(
  spinloop
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES ${VLLM_SPINLOOP_EXT_SRC}
  COMPILE_FLAGS ${SPINLOOP_COMPILE_FLAGS}
  USE_SABI 3.11
  WITH_SOABI)

#
# next section
#
""",
        encoding="utf-8",
    )
    (source / "csrc" / "spinloop.cpp").write_text(
        "void f() { Py_buffer buffer; PyBuffer_Release(&buffer); }\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[3, 10, 12]\\n'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/vllm_spinloop_source_overlay.py",
            "--source",
            str(source),
            "--overlay",
            str(overlay),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert not payload["upstream_checkout_mutated"]
    assert not (overlay / ".git").exists()
    assert not (overlay / ".deps").exists()
    assert "-UPy_LIMITED_API" not in (source / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    assert "-UPy_LIMITED_API" in (overlay / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    preflight = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/vllm_spinloop_preflight.py",
            "--source",
            str(overlay),
            "--env-python",
            str(fake_python),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert preflight.returncode == 0, preflight.stdout
    preflight_payload = json.loads(preflight.stdout)
    assert preflight_payload["status"] == "pass"
    assert preflight_payload["spinloop_unsets_limited_api"]


def test_paper_baseline_environment_attempt_captures_bounded_steps(tmp_path):
    env_path = tmp_path / "tmp" / "envs" / "vllm-fixture"
    marker = env_path / "marker.txt"
    plans_path = tmp_path / "paper_baseline_environment_plans.json"
    viewer_output = tmp_path / "paper_baseline_environment_attempts.json"
    output_root = tmp_path / "attempts"
    plans_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "pto_commit": "abc1234",
                    "artifact_root": "tmp/environment-plans/",
                    "source_files": [
                        "evaluations/nvidia/benchmark-viewer/data/paper_baselines.json"
                    ],
                },
                "paper_baseline_environment_plans": [
                    {
                        "id": "vllm_runtime_environment",
                        "paper_baseline_id": "vllm",
                        "title": "vLLM fixture env",
                        "status": "plan_ready",
                        "source_path": "tmp/baselines/vllm",
                        "source_commit": "1234567890abcdef",
                        "environment_path": str(env_path),
                        "python_policy": "fixture policy",
                        "dependency_sources": ["pyproject.toml"],
                        "critical_packages": [],
                        "manual_packages": [],
                        "install_commands": [
                            f"python3 -c \"import pathlib; pathlib.Path(r'{env_path}').mkdir(parents=True, exist_ok=True)\"",
                            f"python3 -c \"import pathlib; pathlib.Path(r'{marker}').write_text('installed')\"",
                        ],
                        "validation_commands": [
                            f"python3 -c \"import pathlib; assert pathlib.Path(r'{marker}').read_text() == 'installed'\""
                        ],
                        "execution_gaps": ["fixture gap"],
                        "notes": ["fixture note"],
                        "next_action": "fixture next",
                        "raw_artifact": "tmp/environment-plans/environment-plans.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py",
            "--plans",
            str(plans_path),
            "--baseline",
            "vllm",
            "--output-root",
            str(output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
            "--max-steps",
            "2",
            "--timeout-seconds",
            "10",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(viewer_output.read_text(encoding="utf-8"))
    [attempt] = payload["paper_baseline_environment_attempts"]
    assert attempt["paper_baseline_id"] == "vllm"
    assert attempt["environment_plan_id"] == "vllm_runtime_environment"
    assert attempt["status"] == "partial"
    assert attempt["steps_completed"] == 2
    assert attempt["steps_total"] == 3
    assert attempt["environment_path"] == str(env_path)
    assert all(step["status"] == "pass" for step in attempt["steps"])
    assert all(".venv" not in step["command"] for step in attempt["steps"])
    assert (output_root / "environment-attempt.json").is_file()
    assert (output_root / "step-01.log").is_file()
    assert marker.read_text(encoding="utf-8") == "installed"


def test_paper_baseline_environment_attempt_appends_resume_window(tmp_path):
    env_path = tmp_path / "tmp" / "envs" / "vllm-fixture"
    marker = env_path / "marker.txt"
    plans_path = tmp_path / "paper_baseline_environment_plans.json"
    viewer_output = tmp_path / "paper_baseline_environment_attempts.json"
    first_output_root = tmp_path / "attempts-first"
    second_output_root = tmp_path / "attempts-second"
    plans_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "pto_commit": "abc1234",
                    "artifact_root": "tmp/environment-plans/",
                    "source_files": [
                        "evaluations/nvidia/benchmark-viewer/data/paper_baselines.json"
                    ],
                },
                "paper_baseline_environment_plans": [
                    {
                        "id": "vllm_runtime_environment",
                        "paper_baseline_id": "vllm",
                        "title": "vLLM fixture env",
                        "status": "plan_ready",
                        "source_path": "tmp/baselines/vllm",
                        "source_commit": "1234567890abcdef",
                        "environment_path": str(env_path),
                        "python_policy": "fixture policy",
                        "dependency_sources": ["pyproject.toml"],
                        "critical_packages": [],
                        "manual_packages": [],
                        "install_commands": [
                            f"python3 -c \"import pathlib; pathlib.Path(r'{env_path}').mkdir(parents=True, exist_ok=True)\"",
                            f"python3 -c \"import pathlib; pathlib.Path(r'{marker}').write_text('installed')\"",
                        ],
                        "validation_commands": [
                            f"python3 -c \"import pathlib; assert pathlib.Path(r'{marker}').read_text() == 'installed'\""
                        ],
                        "execution_gaps": ["fixture gap"],
                        "notes": ["fixture note"],
                        "next_action": "fixture next",
                        "raw_artifact": "tmp/environment-plans/environment-plans.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py",
            "--plans",
            str(plans_path),
            "--baseline",
            "vllm",
            "--output-root",
            str(first_output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
            "--max-steps",
            "2",
            "--timeout-seconds",
            "10",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert first.returncode == 0, first.stdout

    second = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py",
            "--plans",
            str(plans_path),
            "--baseline",
            "vllm",
            "--output-root",
            str(second_output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
            "--start-step",
            "3",
            "--max-steps",
            "1",
            "--attempt-id-suffix",
            "step03",
            "--append-viewer",
            "--timeout-seconds",
            "10",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert second.returncode == 0, second.stdout

    payload = json.loads(viewer_output.read_text(encoding="utf-8"))
    attempts = payload["paper_baseline_environment_attempts"]
    assert [attempt["id"] for attempt in attempts] == [
        "vllm_environment_attempt_abc1234",
        "vllm_environment_attempt_abc1234_step03",
    ]
    resume = attempts[1]
    assert resume["start_step"] == 3
    assert resume["end_step"] == 3
    assert resume["steps_total"] == 3
    assert resume["steps_completed"] == 1
    assert resume["steps"][0]["index"] == 3
    assert resume["steps"][0]["kind"] == "validation"
    assert resume["steps"][0]["status"] == "pass"
    assert (second_output_root / "step-03.log").is_file()


def test_paper_baseline_viewer_export_rejects_bool_sample_count(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "vllm_serving_and_throughput",
                "benchmark_id": "llm_serving_decode",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "model=fixture",
                    "dtype": "bfloat16",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_capture",
                    "sample_count": True,
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "paper-baseline-invalid.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/vllm/",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid sample_count" in result.stdout


def test_paper_baseline_probe_collects_source_readiness(tmp_path):
    baseline_root = tmp_path / "fake-mpk"
    (baseline_root / "demo" / "qwen3").mkdir(parents=True)
    (baseline_root / "demo" / "qwen3" / "demo.py").write_text(
        "print('fixture')\n",
        encoding="utf-8",
    )
    (baseline_root / "python" / "fixture_pkg").mkdir(parents=True)
    (baseline_root / "python" / "fixture_pkg" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init"],
        cwd=baseline_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=baseline_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=baseline_root,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=baseline_root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=baseline_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=baseline_root,
        text=True,
    ).strip()

    baselines_path = tmp_path / "paper_baselines.json"
    probes_path = tmp_path / "paper_baseline_probes.json"
    output_path = tmp_path / "probe.json"

    baselines = {
        "paper_baselines": [
            {
                "id": "mpk",
                "name": "Mirage Persistent Kernel",
                "paper_role": "fixture",
                "status": "source_cloned_for_survey",
                "source": {
                    "upstream_url": "https://example.invalid/mpk",
                    "local_tmp_path": str(baseline_root),
                    "commit": commit,
                },
                "paper_baselines_to_reproduce": ["fixture"],
                "next_action": "fixture",
            }
        ]
    }
    probes = {
        "paper_baseline_probes": [
            {
                "id": "mpk_source_entrypoints",
                "paper_baseline_id": "mpk",
                "title": "MPK source entrypoints",
                "latest_status": "not_captured",
                "latest_artifact_root": "tmp/cuda-backend/paper-baselines/probes/",
                "latest_machine_status": [
                    {
                        "gpu": "A100",
                        "status": "pass",
                        "artifact": str(output_path),
                        "blocking_gaps": [],
                    },
                    {
                        "gpu": "H200",
                        "status": "partial",
                        "artifact": str(output_path),
                        "blocking_gaps": ["fixture gap"],
                    },
                ],
                "checks": [
                    {
                        "kind": "path_exists",
                        "path": "demo/qwen3/demo.py",
                        "why": "fixture entrypoint",
                    },
                    {
                        "kind": "py_compile",
                        "path": "demo/qwen3/demo.py",
                        "why": "fixture syntax",
                    },
                    {
                        "kind": "python_import",
                        "module": "fixture_pkg",
                        "pythonpath": "python",
                        "python_no_user_site": True,
                        "why": "fixture import through source path",
                    },
                ],
                "next_action": "fixture",
            }
        ]
    }
    baselines_path.write_text(json.dumps(baselines), encoding="utf-8")
    probes_path.write_text(json.dumps(probes), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py",
            "--baselines",
            str(baselines_path),
            "--probes",
            str(probes_path),
            "--output",
            str(output_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/probes/fixture/",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["artifact_root"] == (
        "tmp/cuda-backend/paper-baselines/probes/fixture/"
    )
    assert payload["probes"][0]["paper_baseline_id"] == "mpk"
    assert payload["probes"][0]["status"] == "pass"
    assert payload["probes"][0]["source_commit_actual"] == commit
    assert [check["status"] for check in payload["probes"][0]["checks"]] == [
        "pass",
        "pass",
        "pass",
    ]
    import_check = payload["probes"][0]["checks"][2]
    assert import_check["pythonpath"] == "python"
    assert import_check["python_no_user_site"] is True


def test_paper_probe_status_update_materializes_machine_status(tmp_path):
    paired_root = tmp_path / "tmp" / "paired-probe"
    paired_root.mkdir(parents=True)
    for filename, status, gaps in [
        ("a100-probe.json", "pass", []),
        ("h200-probe.json", "partial", ["python_module failed: fixture"]),
    ]:
        (paired_root / filename).write_text(
            json.dumps(
                {
                    "probes": [
                        {
                            "paper_baseline_id": "fixture",
                            "status": status,
                            "blocking_gaps": gaps,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    probes_path = tmp_path / "paper_baseline_probes.json"
    output_path = tmp_path / "updated-probes.json"
    probes_path.write_text(
        json.dumps(
            {
                "paper_baseline_probes": [
                    {
                        "id": "fixture_source_entrypoints",
                        "paper_baseline_id": "fixture",
                        "title": "Fixture Source Entrypoints",
                        "latest_status": "pass",
                        "latest_artifact_root": "tmp/old/",
                        "latest_machine_status": [],
                        "checks": [],
                        "next_action": "fixture",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_probe_status_update.py",
            "--probes",
            str(probes_path),
            "--paired-artifact-root",
            str(paired_root),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    record = payload["paper_baseline_probes"][0]

    assert record["latest_status"] == "partial"
    assert record["latest_artifact_root"].endswith("tmp/paired-probe/")
    assert record["latest_machine_status"] == [
        {
            "gpu": "A100",
            "status": "pass",
            "artifact": record["latest_artifact_root"] + "a100-probe.json",
            "blocking_gaps": [],
        },
        {
            "gpu": "H200",
            "status": "partial",
            "artifact": record["latest_artifact_root"] + "h200-probe.json",
            "blocking_gaps": ["python_module failed: fixture"],
        },
    ]


def test_paper_readiness_audit_matches_current_viewer_data(tmp_path):
    output_path = tmp_path / "paper-readiness-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_readiness_audit.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    generated = json.loads(output_path.read_text(encoding="utf-8"))
    committed = load_viewer_collection(
        VIEWER_DATA / "paper_readiness_audit.json"
    )
    assert generated == committed
    assert committed["overall_status"] == "not_paper_ready"
    assert committed["ready_claims"] == 3
    assert committed["blocked_claims"] == 1

    by_id = {claim["id"]: claim for claim in committed["claim_audits"]}
    host_claim = by_id["host_schedule_launch_overhead"]
    assert host_claim["ready_for_paper_claim"]
    assert host_claim["blockers"] == []
    assert host_claim["next_actions"] == []
    llm_claim = by_id["llm_serving_paper_baselines"]
    assert not llm_claim["ready_for_paper_claim"]
    assert llm_claim["matrix_status"] == "partial_current_capture"
    assert not any(
        "mpk_qwen3_native_vs_persistent is planned_not_run" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert not any(
        "MPK persistent-kernel" in blocker for blocker in llm_claim["blockers"]
    )
    assert not any(
        "SGLang MPK-policy" in blocker for blocker in llm_claim["blockers"]
    )
    assert not any(
        "vLLM, SGLang" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert not any(
        "Readiness probe for sglang is partial" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert not any(
        "Readiness probe for vllm is partial" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert not any(
        "Run readiness vllm_serving_and_throughput is partial" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert not any(
        action["source"] == "probe"
        and action["paper_baseline_id"] == "vllm"
        for action in llm_claim["next_actions"]
    )
    assert not any(
        action["source"] == "probe"
        and action["paper_baseline_id"] == "sglang"
        for action in llm_claim["next_actions"]
    )
    llm_readiness_ids = {
        item["paper_baseline_run_id"]
        for item in llm_claim["paper_baseline_run_readiness_statuses"]
    }
    assert "vdcores_qwen3_8b_decode_preflight" in llm_readiness_ids
    assert "mpk_qwen3_native_vs_persistent" not in llm_readiness_ids
    assert "vllm_serving_and_throughput" not in llm_readiness_ids
    assert "thunderkittens_decode_attention_tile" not in llm_readiness_ids
    assert not any(
        "No paper baseline run record is attached to this claim for thunderkittens"
        in blocker
        for blocker in llm_claim["blockers"]
    )
    persistent_claim = by_id["persistent_device_scheduler_overhead"]
    persistent_run_ids = {
        run["id"] for run in persistent_claim["paper_baseline_run_statuses"]
    }
    assert {
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
    } <= persistent_run_ids
    persistent_readiness = {
        item["paper_baseline_run_id"]: item
        for item in persistent_claim["paper_baseline_run_readiness_statuses"]
    }
    persistent_attempts = {
        item["paper_baseline_run_id"]: item
        for item in persistent_claim["execution_attempt_statuses"]
    }
    mpk_run_status = next(
        run
        for run in persistent_claim["paper_baseline_run_statuses"]
        if run["id"] == "mpk_persistent_scheduler_trace"
    )
    assert mpk_run_status["status"] == "imported_to_viewer"
    assert "mpk_persistent_scheduler_trace" not in persistent_attempts
    assert "vdcores_resource_policy_trace" not in persistent_attempts
    vdcores_run_status = next(
        run
        for run in persistent_claim["paper_baseline_run_statuses"]
        if run["id"] == "vdcores_resource_policy_trace"
    )
    assert vdcores_run_status["status"] == "imported_to_viewer"
    assert not any(
        "VDCores queue-pressure and scheduler-overhead metadata"
        in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert persistent_claim["ready_for_paper_claim"]
    assert persistent_claim["blockers"] == []
    assert persistent_claim["next_actions"] == []
    assert not any(
        "Latest execution attempt "
        "mpk_qwen3_0p6b_profile_termination_diagnostic_h200"
        in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        action["source"] == "execution_attempt"
        and action["paper_baseline_run_id"] == "mpk_persistent_scheduler_trace"
        for action in persistent_claim["next_actions"]
    )
    assert not any(
        "Readiness probe for mpk is partial" in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        action["source"] == "run_readiness"
        and action["paper_baseline_run_id"] == "vdcores_resource_policy_trace"
        for action in persistent_claim["next_actions"]
    )
    assert "mpk_persistent_scheduler_trace" not in persistent_readiness
    assert "vdcores_resource_policy_trace" not in persistent_readiness
    assert not any(
        "Scheduler-overhead breakdown" in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        "No paper baseline run record is attached" in blocker
        for blocker in persistent_claim["blockers"]
    )
    tensor_claim = by_id["tensor_core_tile_baselines"]
    tensor_run_ids = {
        run["id"] for run in tensor_claim["paper_baseline_run_statuses"]
    }
    assert {
        "thunderkittens_tile_kernel",
        "thunderkittens_full_sweep",
        "thunderkittens_non_mha_rotary",
    } <= tensor_run_ids
    assert tensor_claim["ready_for_paper_claim"]
    assert tensor_claim["matrix_status"] == "ready_for_paper_claim"
    assert tensor_claim["blockers"] == []
    assert tensor_claim["next_actions"] == []
    policy = tensor_claim["evidence_policy_exceptions"][0]
    assert policy["id"] == "thunderkittens_dense_pytorch_12288_oom_policy"
    assert policy["status"] == "accepted"
    assert "12288-token dense PyTorch reference" in policy["title"]
    assert "OOM/not-applicable footnotes" in policy["review_rule"]
    assert any(
        ref["path"].endswith("isolated-pt-reference-summary.json")
        for ref in policy["evidence_refs"]
    )
    assert not any(
        "thunderkittens_full_sweep is planned_not_run" in blocker
        for blocker in tensor_claim["blockers"]
    )
    llm_claim = by_id["llm_serving_paper_baselines"]
    llm_policies = {
        policy["id"]: policy
        for policy in llm_claim["evidence_policy_exceptions"]
    }
    tk_policy = llm_policies[
        "thunderkittens_llm_non_full_serving_policy_pending"
    ]
    assert tk_policy["status"] == "pending"
    assert tk_policy["missing_evidence_id"] == (
        "thunderkittens_full_serving_qwen3_8b"
    )
    assert "controlled attention-tile proxy" in tk_policy["decision"]
    assert "must remain non-paper-ready" in tk_policy["review_rule"]
    assert any(
        ref["path"].endswith("thunderkittens-gemm-compatibility-probe.md")
        for ref in tk_policy["evidence_refs"]
    )


def test_paper_readiness_work_queue_matches_current_audit(tmp_path):
    output_path = tmp_path / "work-queue.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_readiness_work_queue.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    generated = json.loads(output_path.read_text(encoding="utf-8"))
    committed = json.loads(
        (VIEWER_DATA / "paper_readiness_work_queue.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == committed
    assert committed["overall_status"] == "not_paper_ready"
    assert committed["summary"]["total_work_items"] == 4
    assert committed["summary"]["work_items_by_source"] == {
        "execution_attempt": 1,
        "matrix_missing_evidence": 3,
    }
    work_items = committed["work_items"]
    assert all(not item["ready_for_paper_claim"] for item in work_items)
    assert all(item["evidence_summary"] for item in work_items)
    llm_items = [
        item
        for item in work_items
        if item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "matrix_missing_evidence"
    ]
    assert {item["missing_evidence_id"] for item in llm_items} == {
        "pto_full_serving_qwen3_8b",
        "vdcores_full_serving_qwen3_8b",
        "thunderkittens_full_serving_qwen3_8b",
    }
    assert not any(
        item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "probe"
        and item["paper_baseline_id"] == "sglang"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "tensor_core_tile_baselines"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["paper_baseline_run_id"] == "mpk_persistent_scheduler_trace"
        and item["source"] == "run_readiness"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_run_id"] == "mpk_persistent_scheduler_trace"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_run_id"] == "vdcores_resource_policy_trace"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["source"] == "matrix_missing_evidence"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_id"] == "vllm"
        and item["paper_baseline_run_id"] == "vllm_serving_and_throughput"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "probe"
        and item["paper_baseline_id"] == "vllm"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_id"] == "sglang"
        and item["paper_baseline_run_id"] == "sglang_serving_and_offline"
        for item in work_items
    )
    vdcores_attempt_item = next(
        item
        for item in work_items
        if item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_run_id"] == "vdcores_qwen3_8b_decode_preflight"
        and item["execution_attempt_id"]
        == "vdcores_qwen3_8b_shared_instruction_window_plan_h200"
    )
    assert vdcores_attempt_item["serving_workload_ids"] == [
        "vdcores_offline_decode"
    ]
    assert vdcores_attempt_item["serving_command_plan_selectors"] == [
        "vdcores_qwen3_8b_decode_preflight:vdcores_offline_decode"
    ]
    assert any(
        "window_contract_validation=pass" in item
        for item in vdcores_attempt_item["evidence_summary"]
    )
    assert any(
        "runnable_handoff_contract_status=required_not_implemented" in item
        for item in vdcores_attempt_item["evidence_summary"]
    )
    vdcores_item = next(
        item
        for item in work_items
        if item["missing_evidence_id"] == "vdcores_full_serving_qwen3_8b"
    )
    assert any(
        "window_contract_validation=pass" in item
        for item in vdcores_item["evidence_summary"]
    )
    assert any(
        "runnable_handoff_contract_status=required_not_implemented" in item
        for item in vdcores_item["evidence_summary"]
    )
    assert any(
        "vdcores_validate_instruction_window_plan.py" in item
        for item in vdcores_item["evidence_summary"]
    )
    pto_item = next(
        item
        for item in work_items
        if item["missing_evidence_id"] == "pto_full_serving_qwen3_8b"
    )
    assert pto_item["serving_command_plan_selectors"] == [
        "pto_persistent_device_qwen3_8b_full_serving:mpk_offline_decode",
        "pto_persistent_device_qwen3_8b_full_serving:vdcores_offline_decode",
    ]
    assert len(pto_item["action"]) < 400
    assert pto_item["evidence_summary"]
    assert any(
        "resource-backed diagnostic execution" in item
        for item in pto_item["evidence_summary"]
    )


def test_paper_readiness_audit_rejects_malformed_policy_exception(tmp_path):
    matrix = json.loads(
        (VIEWER_DATA / "paper_evaluation_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    llm_claim = next(
        item
        for item in matrix["paper_evaluation_matrix"]
        if item["id"] == "llm_serving_paper_baselines"
    )
    llm_claim["evidence_policy_exceptions"] = [
        {
            "id": "bad_policy",
            "status": "pending",
        }
    ]
    matrix_path = tmp_path / "paper_evaluation_matrix.json"
    output_path = tmp_path / "paper_readiness_audit.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py",
            "--matrix",
            str(matrix_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode != 0
    assert "evidence_policy_exceptions" in result.stdout
    assert "review_rule" in result.stdout


def test_tensor_workload_coverage_records_multi_repeat_qwen_capture():
    coverage = json.loads(
        (VIEWER_DATA / "tensor_workload_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        item["id"]: item for item in coverage["model_shape_targets"]
    }

    for target_id in (
        "qwen_attention_projection_tile",
        "qwen_mlp_projection_tile",
    ):
        target = targets[target_id]
        capture = target["throughput_capture"]
        assert capture["status"] == "a100_h200_multi_repeat"
        captures = capture["captures"]
        by_gpu = {item["hardware"]["gpu"]: item for item in captures}
        assert set(by_gpu) == {"A100", "H200"}
        assert by_gpu["A100"]["hardware"]["compute_target"] == "compute_80"
        assert by_gpu["H200"]["hardware"]["compute_target"] == "compute_90"
        assert capture["sample_count"] >= 3
        assert set(capture["methods"]) == {
            "pto_persistent_device",
            "cublas_sgemm_graph",
        }
        for hardware_capture in captures:
            assert hardware_capture["artifact_root"].startswith(
                "tmp/cuda-backend/"
            )
            assert hardware_capture["exported_records_path"].startswith(
                hardware_capture["artifact_root"]
            )
            records = json.loads(
                (ROOT / hardware_capture["exported_records_path"]).read_text(
                    encoding="utf-8"
                )
            )
            record_methods = {record["method_id"] for record in records}
            assert set(capture["methods"]) <= record_methods
            for record in records:
                if record["method_id"] not in capture["methods"]:
                    continue
                assert record["statistic"]["sample_count"] == (
                    capture["sample_count"]
                )
                assert record["correctness"] == "pass"
                assert record["raw_artifact"] == hardware_capture["artifact_root"]
        generated = target["generated_kernel_capture"]
        assert generated["status"] == "a100_h200_multi_repeat"
        generated_captures = generated["captures"]
        generated_by_gpu = {
            item["hardware"]["gpu"]: item for item in generated_captures
        }
        assert set(generated_by_gpu) == {"A100", "H200"}
        assert (
            generated_by_gpu["A100"]["hardware"]["compute_target"]
            == "compute_80"
        )
        assert (
            generated_by_gpu["H200"]["hardware"]["compute_target"]
            == "compute_90"
        )
        assert generated["sample_count"] >= 3
        assert set(generated["methods"]) == {"triton", "cutlass"}
        for hardware_capture in generated_captures:
            assert set(hardware_capture["methods"]) == {"triton", "cutlass"}
            for exported_records_path in hardware_capture["exported_records_paths"]:
                records = json.loads(
                    (ROOT / exported_records_path).read_text(encoding="utf-8")
                )
                assert records
                for record in records:
                    assert record["method_id"] in hardware_capture["methods"]
                    assert record["statistic"]["sample_count"] == (
                        generated["sample_count"]
                    )
                    assert record["correctness"] == "pass"
                    assert exported_records_path.startswith(
                        record["raw_artifact"]
                    )
        thunderkittens = target["thunderkittens_proxy_capture"]
        assert thunderkittens["status"] == "h200_attention_proxy_imported"
        assert thunderkittens["sample_count"] >= 20
        assert thunderkittens["methods"] == ["thunderkittens"]
        assert thunderkittens["hardware"] == {
            "gpu": "H200",
            "compute_target": "compute_90",
        }
        assert thunderkittens["comparison_scope"] == (
            "attention_family_proxy_not_same_gemm_tile"
        )
        assert "does not close" in thunderkittens["remaining_scope"]
        compatibility = target["thunderkittens_gemm_compatibility_probe"]
        assert compatibility["status"] == (
            "exact_qwen_tile_not_supported_by_current_gemm_entrypoints"
        )
        assert compatibility["methods"] == ["thunderkittens"]
        assert compatibility["comparison_scope"] == (
            "source_entrypoint_probe_not_same_tile_result"
        )
        assert compatibility["artifact_path"].startswith(
            "tmp/cuda-backend/paper-baselines/thunderkittens/"
        )
        assert "compatibility.json" in compatibility["artifact_path"]
        report = json.loads(
            (ROOT / compatibility["artifact_path"]).read_text(encoding="utf-8")
        )
        assert report["baseline"] == "thunderkittens"
        assert report["status"] == compatibility["status"]
        report_targets = {item["id"]: item for item in report["targets"]}
        assert report_targets[target_id]["tensor_tile"] == target["tensor_tile"]
        assert {
            entry["entrypoint_id"] for entry in report["entrypoints"]
        } >= {"bf16_h100_gemm", "int8_h100_gemm"}
        assert "does not close" in compatibility["remaining_scope"]
        result_records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (
                VIEWER_DATA / "results" / "records"
            ).glob("*thunderkittens-h200*.json")
        ]
        for result_ref in thunderkittens["result_refs"]:
            matches = [
                record
                for record in result_records
                if record["benchmark_id"] == result_ref["benchmark_id"]
                and record["method_id"] == "thunderkittens"
                and record["hardware"]["gpu"] == "H200"
                and result_ref["shape_contains"] in record["inputs"]["shape"]
            ]
            assert matches
            assert all(record["correctness"] == "pass" for record in matches)
            assert all(
                record["statistic"]["sample_count"]
                == thunderkittens["sample_count"]
                for record in matches
            )


def test_cuda_persistent_smoke_dag_task_matches_compiler_abi():
    smoke_script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_persistent_smoke.py"
    )
    script_dir = smoke_script.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "cuda_persistent_smoke_for_abi_test",
        smoke_script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from simpler_setup.cuda_callable_compiler import (
        CudaPersistentDagTask as CompilerDagTask,
    )

    smoke_task = module.CudaPersistentDagTask
    smoke_fields = dict(smoke_task._fields_)
    compiler_fields = dict(CompilerDagTask._fields_)
    assert struct.calcsize("P") == 8
    assert smoke_fields["tensor_args"]._length_ == 5
    assert smoke_fields["tensor_arg_dtypes"]._length_ == 5
    assert ctypes.sizeof(smoke_fields["tensor_args"]) == ctypes.sizeof(
        compiler_fields["tensor_args"]
    )
    assert ctypes.sizeof(smoke_fields["tensor_arg_dtypes"]) == ctypes.sizeof(
        compiler_fields["tensor_arg_dtypes"]
    )
    assert smoke_task.scalar_args.offset == CompilerDagTask.scalar_args.offset
    assert smoke_task.tensor_arg_count.offset == (
        CompilerDagTask.tensor_arg_count.offset
    )
    assert smoke_task.scalar_arg_count.offset == (
        CompilerDagTask.scalar_arg_count.offset
    )
    assert ctypes.sizeof(smoke_task) == ctypes.sizeof(CompilerDagTask)


def test_pto_run_readiness_uses_repo_owned_entrypoints():
    readiness = json.loads(
        (VIEWER_DATA / "paper_baseline_run_readiness.json").read_text(
            encoding="utf-8"
        )
    )["paper_baseline_run_readiness"]
    by_run = {item["paper_baseline_run_id"]: item for item in readiness}
    pto_readiness = by_run["pto_persistent_device_qwen3_8b_full_serving"]
    pto_path_checks = {
        check["path"]: check["status"]
        for check in pto_readiness["checks"]
        if check["kind"] == "path_exists"
    }

    assert pto_path_checks["examples/cuda/qwen_decode_loop_runner.py"] == "pass"
    assert (
        pto_path_checks[
            ".agents/skills/cuda-backend-eval/scripts/"
            "pto_qwen_full_serving_viewer_import.py"
        ]
        == "pass"
    )
    assert not any(
        "examples/cuda/qwen_decode_loop_runner.py" in gap
        for gap in pto_readiness["blocking_gaps"]
    )


def test_pto_paper_baseline_probe_covers_repo_owned_entrypoints():
    probes = json.loads(
        (VIEWER_DATA / "paper_baseline_probes.json").read_text(
            encoding="utf-8"
        )
    )["paper_baseline_probes"]
    by_baseline = {item["paper_baseline_id"]: item for item in probes}
    pto_probe = by_baseline["pto_persistent_device"]
    pto_paths = {
        check["path"]
        for check in pto_probe["checks"]
        if check["kind"] in {"path_exists", "py_compile"}
    }

    assert pto_probe["latest_status"] == "pass"
    assert {
        "examples/cuda/qwen_decode_loop_runner.py",
        ".agents/skills/cuda-backend-eval/scripts/"
        "pto_qwen_full_serving_viewer_import.py",
    } <= pto_paths
    assert {
        status["gpu"]: status["status"]
        for status in pto_probe["latest_machine_status"]
    } == {"A100": "pass", "H200": "pass"}


def test_mpk_run_readiness_accepts_repo_relative_tmp_entrypoint():
    readiness = json.loads(
        (VIEWER_DATA / "paper_baseline_run_readiness.json").read_text(
            encoding="utf-8"
        )
    )["paper_baseline_run_readiness"]
    by_run = {item["paper_baseline_run_id"]: item for item in readiness}
    mpk_readiness = by_run["mpk_qwen3_native_vs_persistent"]
    mpk_path_checks = {
        check["path"]: check["status"]
        for check in mpk_readiness["checks"]
        if check["kind"] == "path_exists"
    }

    assert (
        mpk_path_checks["tmp/baselines/mirage-mpk/demo/qwen3/demo.py"]
        == "pass"
    )
    assert not any(
        "tmp/baselines/mirage-mpk/demo/qwen3/demo.py" in gap
        for gap in mpk_readiness["blocking_gaps"]
    )


def test_thunderkittens_tile_readiness_uses_selected_mha_kernel():
    readiness = json.loads(
        (VIEWER_DATA / "paper_baseline_run_readiness.json").read_text(
            encoding="utf-8"
        )
    )["paper_baseline_run_readiness"]
    by_run = {item["paper_baseline_run_id"]: item for item in readiness}
    tk_readiness = by_run["thunderkittens_tile_kernel"]
    tk_path_checks = {
        check["path"]: check["status"]
        for check in tk_readiness["checks"]
        if check["kind"] == "path_exists"
    }

    assert (
        tk_path_checks["kernels/attention/mha_h100/test_correctness.py"]
        == "pass"
    )
    assert tk_path_checks["kernels/attention/mha_h100/benchmark.py"] == "pass"
    assert tk_readiness["latest_status"] == "pass"
    assert tk_readiness["blocking_gaps"] == []


def test_nvidia_goal_progress_matches_current_artifacts(tmp_path):
    output_path = tmp_path / "goal-progress.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "nvidia_goal_progress.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    generated = json.loads(output_path.read_text(encoding="utf-8"))
    committed = json.loads(
        (VIEWER_DATA / "goal_progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == committed
    assert committed["overall_status"] == "in_progress"
    assert committed["summary"]["criteria_total"] == 9
    assert committed["summary"]["criteria_met"] >= 6
    assert committed["summary"]["criteria_in_progress"] >= 1
    by_id = {item["id"]: item for item in committed["acceptance_criteria"]}
    backend_closure = by_id["backend_implementation_closure"]
    status_remaining = (
        (DOC_ROOT / "status.md")
        .read_text(encoding="utf-8")
        .split("\n## Remaining Gaps\n", 1)[1]
        .split("\n## ", 1)[0]
    )
    status_gap_refs = {
        f"docs/nvidia-backend/{line.split('](', 1)[1].split(')', 1)[0]}"
        for line in status_remaining.splitlines()
        if line.startswith("- [")
    }
    assert status_gap_refs == {
        "docs/nvidia-backend/status/remaining-gaps/qwen-full-serving-correctness.md",
        "docs/nvidia-backend/status/remaining-gaps/tuned-tensor-workloads.md",
    }
    assert backend_closure["status"] == "in_progress"
    assert set(backend_closure["evidence_refs"]) == {
        "docs/nvidia-backend/status.md",
        *status_gap_refs,
    }
    assert any(
        "Close or reclassify every remaining-gap page" in gap
        for gap in backend_closure["gaps"]
    )
    assert any(
        "paper/evaluation readiness separate" in gap
        for gap in backend_closure["gaps"]
    )
    assert not any(
        "fourth-tensor-persistent-dag-verification" in ref
        for ref in backend_closure["evidence_refs"]
    )
    assert by_id["paper_grade_results"]["status"] == "in_progress"
    assert by_id["paper_grade_results"]["blocking_work_items"] == 4
    assert by_id["paper_grade_results"]["paper_readiness_status"] == (
        "not_paper_ready"
    )
    assert any(
        "remaining queued paper-readiness artifacts" in gap
        for gap in by_id["paper_grade_results"]["gaps"]
    )
    assert not any(
        "SGLang MPK-policy" in gap
        for gap in by_id["paper_grade_results"]["gaps"]
    )
    assert not any(
        "vLLM, SGLang" in gap
        for gap in by_id["paper_grade_results"]["gaps"]
    )
    assert by_id["remote_evaluation"]["status"] == "met"
    assert by_id["benchmark_viewer"]["status"] == "met"
    assert all(item["evidence_refs"] for item in committed["acceptance_criteria"])
    assert all(item["verification"] for item in committed["acceptance_criteria"])


def test_nvidia_review_artifact_refresh_regenerates_all_generated_json(tmp_path):
    output_dir = tmp_path / "viewer-data"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "refresh_nvidia_review_artifacts.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    for filename in [
        "paper_readiness_audit.json",
        "paper_readiness_work_queue.json",
        "goal_progress.json",
    ]:
        generated = load_viewer_collection(output_dir / filename)
        committed = load_viewer_collection(VIEWER_DATA / filename)
        if filename == "paper_readiness_work_queue.json":
            assert generated["source_file"].endswith(
                "/paper_readiness_audit.json"
            )
            generated = {
                **generated,
                "source_file": committed["source_file"],
            }
        assert generated == committed
    assert "paper_readiness_audit.json" in result.stdout
    assert "paper_readiness_work_queue.json" in result.stdout
    assert "goal_progress.json" in result.stdout


def test_paper_serving_command_plan_generates_policy_commands(tmp_path):
    output_path = tmp_path / "serving-plan.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_serving_command_plan.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    records = payload["serving_command_plans"]

    assert payload["metadata"]["model_tier"] == "primary"
    assert len(records) == 46
    by_id = {record["id"]: record for record in records}
    pto_mpk = by_id[
        "pto_persistent_device_qwen3_8b_full_serving:"
        "mpk_offline_decode:batch8"
    ]
    assert pto_mpk["paper_baseline_id"] == "pto_persistent_device"
    assert pto_mpk["model"] == "Qwen/Qwen3-8B"
    assert pto_mpk["prompt_tokens"] == 64
    assert pto_mpk["decode_tokens"] == 1024
    assert pto_mpk["batch_size"] == 8
    assert any(
        command["kind"] == "pto_qwen_full_serving"
        and "qwen_decode_loop_runner.py" in command["command"]
        and "--single-context-live-session" in command["command"]
        and "--run-resource-backed-smoke" in command["command"]
        and "--resource-backed-prefill-prompt" in command["command"]
        and "--resource-backed-workload mpk_offline_decode" in command["command"]
        and "--resource-backed-decode-steps 1024" in command["command"]
        for command in pto_mpk["commands"]
    )
    assert any(
        command["kind"] == "pto_viewer_import"
        and "pto_qwen_full_serving_viewer_import.py" in command["command"]
        for command in pto_mpk["commands"]
    )
    vllm_mpk = by_id[
        "vllm_serving_and_throughput:mpk_offline_decode:batch16"
    ]
    assert vllm_mpk["model"] == "Qwen/Qwen3-8B"
    assert vllm_mpk["prompt_tokens"] == 64
    assert vllm_mpk["decode_tokens"] == 1024
    assert vllm_mpk["batch_size"] == 16
    assert any(
        "--max-concurrency 16" in command["command"]
        and "--input-len 64" in command["command"]
        and "--output-len 1024" in command["command"]
        for command in vllm_mpk["commands"]
    )

    sglang_vdcores = by_id[
        "sglang_serving_and_offline:vdcores_offline_decode:batch8"
    ]
    assert sglang_vdcores["prompt_tokens"] == 128
    assert sglang_vdcores["decode_tokens"] == 64
    assert any(
        "--random-input-len 128" in command["command"]
        and "--random-output-len 64" in command["command"]
        and "--random-range-ratio 1.0" in command["command"]
        for command in sglang_vdcores["commands"]
    )
    assert any(
        command["kind"] == "online_serving"
        and "--dataset-name random-ids" in command["command"]
        and "--tokenize-prompt" in command["command"]
        for command in sglang_vdcores["commands"]
    )
    assert any(
        command["kind"] == "offline_throughput"
        and "--context-length 384" in command["command"]
        and "--skip-warmup" in command["command"]
        for command in sglang_vdcores["commands"]
    )
    assert any(
        command["kind"] == "one_batch"
        and "--disable-cuda-graph" in command["command"]
        for command in sglang_vdcores["commands"]
    )
    assert all(
        command["command"].startswith(
            "env PYTHONPATH=$PWD/tmp/baselines/sglang/python:$PYTHONPATH "
        )
        for command in sglang_vdcores["commands"]
    )
    thunderkittens_decode = by_id[
        "thunderkittens_decode_attention_tile:vdcores_offline_decode:batch4"
    ]
    assert thunderkittens_decode["prompt_tokens"] == 128
    assert thunderkittens_decode["decode_tokens"] == 64
    assert any(
        command["kind"] == "decode_attention_tile"
        and "thunderkittens_mha_capture.py" in command["command"]
        and "--paper-baseline-run-id thunderkittens_decode_attention_tile"
        in command["command"]
        and "--benchmark-id llm_serving_decode" in command["command"]
        and "--serving-workload-id vdcores_offline_decode" in command["command"]
        and "--prompt-tokens 128" in command["command"]
        and "--decode-tokens 64" in command["command"]
        and "--shape 4,1,256,64" in command["command"]
        for command in thunderkittens_decode["commands"]
    )
    mpk_native_bringup = by_id[
        "mpk_qwen3_native_token_bringup:"
        "mpk_native_qwen3_0p6b_token2_bringup:batch1"
    ]
    assert mpk_native_bringup["model"] == "Qwen/Qwen3-0.6B"
    assert mpk_native_bringup["prompt_tokens"] == 39
    assert mpk_native_bringup["decode_tokens"] == 2
    assert mpk_native_bringup["batch_size"] == 1
    assert [command["kind"] for command in mpk_native_bringup["commands"]] == [
        "native_demo"
    ]
    assert "--model Qwen/Qwen3-0.6B" in mpk_native_bringup["commands"][0][
        "command"
    ]
    assert "--max-new-tokens 2" in mpk_native_bringup["commands"][0]["command"]
    assert all(
        command.get("raw_artifact", "").startswith(
            "tmp/cuda-backend/paper-baselines/serving-runs"
        )
        for record in records
        for command in record["commands"]
        if command["kind"] != "server"
    )


def test_paper_baseline_pair_probe_uses_remote_fallback_contract():
    script_path = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_pair_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "paper_baseline_pair_probe",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = module.PairedPaperBaselineProbeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="goal/nvidia-paper-ready",
        output_root=Path("tmp/cuda-backend/paper-baselines/probes"),
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        refresh_remote=False,
        sync_remote_tree=True,
    )

    sync_command = module.build_remote_sync_command(config)
    assert sync_command[:3] == ["rsync", "-a", "--delete"]
    assert "--exclude=.venv" in sync_command
    assert "--exclude=build" in sync_command
    assert "--exclude=tmp" in sync_command
    assert sync_command[-1] == "h200-box:/remote/pto-cu/"

    source_sync_command = module.build_remote_baseline_source_sync_command(config)
    assert source_sync_command[:3] == ["rsync", "-a", "--delete"]
    assert "--exclude=build/" in source_sync_command
    assert "--exclude=*.egg-info/" in source_sync_command
    assert "--exclude=__pycache__/" in source_sync_command
    assert "--exclude=*.pyc" in source_sync_command
    assert "--exclude=*.cpython-*.so" in source_sync_command
    assert source_sync_command[-2] == "tmp/baselines/"
    assert source_sync_command[-1] == "h200-box:/remote/pto-cu/tmp/baselines/"

    remote_command = module.build_remote_probe_command(config, "abc123")
    assert remote_command[:5] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    assert remote_command[5] == "h200-box"
    shell = remote_command[-1]
    assert "cd /remote/pto-cu" in shell
    assert "CUDA_HOME=" in shell
    assert "PATH=" in shell
    assert "PYTHONPATH=$PWD:$PWD/python" in shell
    assert "paper_baseline_probe.py" in shell
    assert "--output" in shell
    assert "h200-probe.json" in shell
    assert "git fetch" not in shell
    assert "git checkout" not in shell

    default_config = module.PairedPaperBaselineProbeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="goal/nvidia-paper-ready",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
    )
    default_shell = module.build_remote_probe_command(default_config, "abc123")[-1]
    assert "fetch origin goal/nvidia-paper-ready" in default_shell
    assert "git checkout -B goal/nvidia-paper-ready FETCH_HEAD" in default_shell


def test_evaluation_docs_are_split_for_review():
    root_evaluation_docs = sorted(DOC_ROOT.glob("evaluation*.md"))
    assert {path.name for path in root_evaluation_docs} == {
        "evaluation-current.md",
        "evaluation.md",
    }
    for path in root_evaluation_docs:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 220, f"{path} has {len(lines)} lines"

    history_root = DOC_ROOT / "history"
    assert (history_root / "index.md").is_file()
    assert (history_root / "captures" / "current-head-layered-cross-743709f3.md").is_file()
    assert (history_root / "captures" / "legacy-captures.md").is_file()


def test_benchmark_viewer_has_json_backed_review_data():
    assert (VIEWER_ROOT / "index.html").is_file()
    assert (VIEWER_ROOT / "styles.css").is_file()
    assert (VIEWER_ROOT / "viewer.js").is_file()
    for filename in [
        "benchmarks.json",
        "methods.json",
        "paper_baselines.json",
        "paper_baseline_runs.json",
        "paper_baseline_probes.json",
        "paper_baseline_environment_plans.json",
        "paper_baseline_environment_attempts.json",
        "paper_baseline_run_readiness.json",
        "paper_baseline_execution_attempts.json",
        "serving_command_plan.json",
        "serving_workloads.json",
        "paper_evaluation_matrix.json",
        "paper_readiness_audit.json",
        "paper_readiness_work_queue.json",
        "goal_progress.json",
        "plan_history.json",
        "capture_imports.json",
    ]:
        assert (VIEWER_DATA / filename).is_file()
    assert (VIEWER_DATA / "results" / "index.json").is_file()
    assert (VIEWER_DATA / "results" / "record_files.json").is_file()
    viewer_files = [VIEWER_ROOT / "viewer.js", *sorted((VIEWER_ROOT / "viewer").glob("*.js"))]
    viewer_js = "\n".join(path.read_text(encoding="utf-8") for path in viewer_files)
    for required in [
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
        "Environment Plans",
        "preflight_commands",
        "Preflight",
        "paperBaselineEnvironmentAttempts",
        "paper_baseline_environment_attempts",
        "Environment Attempts",
        "paperBaselineRunReadiness",
        "paper_baseline_run_readiness",
        "Run Readiness",
        "paperBaselineExecutionAttempts",
        "paper_baseline_execution_attempts",
        "Execution Attempts",
        "next_actions",
        "Next Actions",
        "servingCommandPlan",
        "Serving Command Plan",
        "servingWorkloads",
        "serving_workloads",
        "Serving policies",
        "latest_artifact_root",
        "latest_machine_status",
        "paperEvaluation",
        "paper_evaluation_matrix",
        "paperReadinessAudit",
        "paper_readiness_audit",
        "paperReadinessWorkQueue",
        "paper_readiness_work_queue",
        "Paper Work Queue",
        "work_items_by_source",
        "item.promotion_gate",
        "paperBaselineRunTitle",
        "Work Item",
        "Execution Attempt",
        "Promotion Gate",
        "goalProgress",
        "goal_progress",
        "Goal Progress",
        "planHistory",
        "plan_history",
        "Recent Work Focus",
        "recent_slices",
        "reflection_log",
        "focus-segment",
        "acceptance_criteria",
        "paper_grade_results",
        "paper_baseline_run_readiness_statuses",
        "ready_for_paper_claim",
        "result_records",
        "raw_artifact",
        "correctness",
        "import_smoke",
        "Import Smoke",
        "Import Smoke Scope",
        "Import Smoke Commands",
        "Paper Target Commands",
    ]:
        assert required in viewer_js

    benchmarks = json.loads(
        (VIEWER_DATA / "benchmarks.json").read_text(encoding="utf-8")
    )
    methods = json.loads(
        (VIEWER_DATA / "methods.json").read_text(encoding="utf-8")
    )
    paper_baselines = json.loads(
        (VIEWER_DATA / "paper_baselines.json").read_text(
            encoding="utf-8"
        )
    )
    paper_baseline_runs = load_viewer_collection(
        VIEWER_DATA / "paper_baseline_runs.json"
    )
    paper_baseline_probes = load_viewer_collection(
        VIEWER_DATA / "paper_baseline_probes.json"
    )
    paper_baseline_run_readiness = load_viewer_collection(
        VIEWER_DATA / "paper_baseline_run_readiness.json"
    )
    paper_baseline_environment_attempts = load_viewer_collection(
        VIEWER_DATA / "paper_baseline_environment_attempts.json"
    )
    paper_baseline_execution_attempts = load_viewer_collection(
        VIEWER_DATA / "paper_baseline_execution_attempts.json"
    )
    serving_command_plan = load_viewer_collection(
        VIEWER_DATA / "serving_command_plan.json"
    )
    serving_workloads = json.loads(
        (VIEWER_DATA / "serving_workloads.json").read_text(
            encoding="utf-8"
        )
    )
    paper_evaluation = load_viewer_collection(
        VIEWER_DATA / "paper_evaluation_matrix.json"
    )
    paper_readiness_audit = load_viewer_collection(
        VIEWER_DATA / "paper_readiness_audit.json"
    )
    paper_readiness_work_queue = json.loads(
        (VIEWER_DATA / "paper_readiness_work_queue.json").read_text(
            encoding="utf-8"
        )
    )
    goal_progress = json.loads(
        (VIEWER_DATA / "goal_progress.json").read_text(
            encoding="utf-8"
        )
    )
    capture_imports = load_viewer_collection(
        VIEWER_DATA / "capture_imports.json"
    )
    results = load_viewer_collection(VIEWER_DATA / "results.json")

    benchmark_ids = {item["id"] for item in benchmarks["benchmarks"]}
    assert "llm_serving_decode" in benchmark_ids
    assert "graph_layered_cross" in benchmark_ids
    assert "tensor_core_tile" in benchmark_ids
    for benchmark in benchmarks["benchmarks"]:
        assert benchmark["description"]
        assert benchmark["math"]
        assert benchmark["code"]
        assert benchmark["run"]["command"]
        assert benchmark["run"]["inputs"]
        assert benchmark["evidence_refs"]

    method_ids = {item["id"] for item in methods["methods"]}
    assert {
        "pto_host_schedule",
        "pto_persistent_device",
        "direct_runtime",
        "direct_driver",
        "direct_driver_graph",
        "cublas_sgemm_graph",
        "triton",
        "cutlass",
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= method_ids
    for method in methods["methods"]:
        assert method["category"]
        assert method["launch_model"]

    import_baselines = {
        item["baseline"] for item in capture_imports["capture_imports"]
    }
    assert "direct_runtime" in import_baselines
    assert "direct_driver" in import_baselines
    assert "direct_driver_graph" in import_baselines
    assert "direct_driver_sgemm" in import_baselines
    assert "direct_runtime_sgemm" in import_baselines
    assert "direct_driver_graph_sgemm" in import_baselines
    assert "pto_persistent_dag_graph_tensor_core" in import_baselines

    paper_baseline_ids = {
        item["id"] for item in paper_baselines["paper_baselines"]
    }
    assert {"mpk", "vdcores"} <= paper_baseline_ids
    assert "vllm" in paper_baseline_ids
    assert "sglang" in paper_baseline_ids
    for baseline in paper_baselines["paper_baselines"]:
        assert baseline["status"]
        assert baseline["source"]["upstream_url"]
        assert baseline["paper_role"]
        assert baseline["next_action"]

    by_id = {item["id"]: item for item in paper_baselines["paper_baselines"]}
    for baseline_id in ["mpk", "vdcores", "vllm", "sglang", "thunderkittens"]:
        baseline = by_id[baseline_id]
        assert baseline["status"] == "source_cloned_for_survey"
        assert len(baseline["source"]["commit"]) == 40

    serving_by_id = {
        item["id"]: item for item in serving_workloads["serving_workloads"]
    }
    assert {
        "mpk_offline_decode",
        "mpk_native_qwen3_0p6b_token2_bringup",
        "vdcores_offline_decode",
    } <= set(serving_by_id)
    assert serving_by_id["mpk_offline_decode"]["model_policy"]["primary_model"] == (
        "Qwen/Qwen3-8B"
    )
    assert (
        serving_by_id["mpk_offline_decode"]["prompt_policy"][
            "target_prompt_tokens"
        ]
        == 64
    )
    assert serving_by_id["mpk_offline_decode"]["decode_policy"]["decode_tokens"] == 1024
    mpk_native_workload = serving_by_id["mpk_native_qwen3_0p6b_token2_bringup"]
    assert mpk_native_workload["model_policy"]["primary_model"] == "Qwen/Qwen3-0.6B"
    assert mpk_native_workload["prompt_policy"]["target_prompt_tokens"] == 39
    assert mpk_native_workload["decode_policy"]["decode_tokens"] == 2
    assert mpk_native_workload["decode_policy"]["batch_sizes"] == [1]
    assert mpk_native_workload["baseline_run_ids"] == [
        "mpk_qwen3_native_token_bringup"
    ]
    assert serving_by_id["vdcores_offline_decode"]["decode_policy"]["decode_tokens"] == 64
    assert serving_by_id["vdcores_offline_decode"]["status"] == (
        "partial_controlled_results"
    )
    assert serving_by_id["vdcores_offline_decode"]["prompt_policy"][
        "target_prompt_tokens"
    ] == 128
    assert any(
        ref["path"] == "evaluations/nvidia/benchmark-viewer/data/results.json"
        and "pto_controlled_serving_equivalent" in ref["symbols"]
        for ref in serving_by_id["vdcores_offline_decode"]["evidence_refs"]
    )
    for workload in serving_workloads["serving_workloads"]:
        assert workload["baseline_run_ids"]
        assert workload["required_metrics"]
        assert workload["evidence_refs"]

    allowed_serving_coverage = {
        "full_serving",
        "full_serving_latency_caveat",
        "controlled_attention_tile_proxy",
        "diagnostic_microdecode",
        "diagnostic_qwen_descriptor_smoke",
        "diagnostic_resource_backed_qwen_dag",
        "diagnostic_unit_math",
        "native_bringup",
    }
    llm_result_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "llm_serving_decode"
    ]
    assert llm_result_records
    assert all(
        record["statistic"].get("serving_coverage") in allowed_serving_coverage
        for record in llm_result_records
    )
    proxy_rows = [
        record
        for record in llm_result_records
        if record["statistic"]["serving_coverage"]
        == "controlled_attention_tile_proxy"
    ]
    assert proxy_rows
    assert all(
        "proxy" in record["inputs"]["shape"] or record["method_id"] == "thunderkittens"
        for record in proxy_rows
    )

    matrix_serving_refs = [
        ref
        for claim in paper_evaluation["paper_evaluation_matrix"]
        for ref in claim["current_evidence_refs"]
        if ref.get("kind") == "viewer_result"
        and ref.get("benchmark_id") == "llm_serving_decode"
    ]
    assert matrix_serving_refs
    assert all(
        ref.get("serving_coverage") in allowed_serving_coverage
        for ref in matrix_serving_refs
    )

    run_ids = {item["id"] for item in paper_baseline_runs["paper_baseline_runs"]}
    assert {
        "mpk_qwen3_native_vs_persistent",
        "mpk_qwen3_native_token_bringup",
        "vdcores_qwen3_8b_decode_preflight",
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_tile_kernel",
        "thunderkittens_full_sweep",
        "thunderkittens_non_mha_rotary",
        "thunderkittens_decode_attention_tile",
    } <= run_ids
    run_baselines = {
        item["paper_baseline_id"]
        for item in paper_baseline_runs["paper_baseline_runs"]
    }
    assert {
        "pto_persistent_device",
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= run_baselines
    for item in paper_baseline_runs["paper_baseline_runs"]:
        assert item["paper_baseline_id"] in paper_baseline_ids
        assert item["paper_evaluation_id"]
        assert item["hardware_targets"]
        assert item["setup_commands"]
        assert item["run_commands"]
        assert item["expected_artifacts"]
        assert item["import_target"]["viewer_file"].endswith("results.json")
        if item["paper_baseline_id"] in {"mpk", "vdcores"}:
            assert isinstance(item.get("model_access"), dict)
            assert isinstance(item["model_access"]["requires_hf_token"], bool)
        if item["paper_evaluation_id"] == "llm_serving_paper_baselines":
            assert item["serving_workload_ids"]
            assert {
                "model_and_prompt_shape",
                "batch_or_concurrency_policy",
            } <= set(item["required_metrics"])
        if item["id"] == "mpk_qwen3_native_vs_persistent":
            assert item["status"] == "imported_to_viewer"
            assert item["serving_workload_ids"] == ["mpk_offline_decode"]
            assert "Qwen/Qwen3-8B" in item["workload"]["model"]
            assert "batch=1 imported" in item["workload"]["batch_or_concurrency"]
            assert any(
                "mpk_qwen3_persistent_capture.py" in command
                for command in item["run_commands"]
            )
            assert any(
                path.endswith("persistent-batch1-decode1024.json")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("paper-baseline-results.json")
                for path in item["expected_artifacts"]
            )
        if item["id"] == "mpk_qwen3_native_token_bringup":
            assert item["paper_evaluation_id"] == "llm_serving_paper_baselines"
            assert item["status"] == "imported_to_viewer"
            assert item["serving_workload_ids"] == [
                "mpk_native_qwen3_0p6b_token2_bringup"
            ]
            assert item["serving_command_kinds"] == ["native_demo"]
            assert "native torch bring-up" in item["workload"][
                "batch_or_concurrency"
            ]
            assert any(
                "mpk_native_token_capture.py" in command
                for command in item["run_commands"]
            )
            assert any(
                path.endswith("native-token2-paper-results.json")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("native-token2-viewer-result-records.json")
                for path in item["expected_artifacts"]
            )
        if item["id"] == "thunderkittens_tile_kernel":
            assert item["status"] == "imported_to_viewer"
            assert any(
                "quick-smoke.json" in path for path in item["expected_artifacts"]
            )
            assert any("capture.json" in path for path in item["expected_artifacts"])
            assert any(
                "thunderkittens_mha_capture.py" in command
                for command in item["run_commands"]
            )
            assert not any(
                path.endswith("correctness.json") or path.endswith("benchmark.json")
                for path in item["expected_artifacts"]
            )
        if item["id"] == "thunderkittens_full_sweep":
            assert item["paper_evaluation_id"] == "tensor_core_tile_baselines"
            assert item["status"] == "imported_to_viewer"
            assert "Selected" in item["title"]
            assert item["serving_workload_ids"] == []
            assert any(
                path.endswith("correctness.json")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("benchmark.json") for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("upstream-summary.json")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("run-after-build.log")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("correctness.log")
                for path in item["expected_artifacts"]
            )
            assert any(
                "thunderkittens_full_sweep_capture.py" in command
                for command in item["run_commands"]
            )
        if item["id"] == "thunderkittens_non_mha_rotary":
            assert item["paper_evaluation_id"] == "tensor_core_tile_baselines"
            assert item["status"] == "imported_to_viewer"
            assert item["serving_workload_ids"] == []
            assert any(
                "thunderkittens_rotary_capture.py" in command
                for command in item["run_commands"]
            )
            assert any(
                path.endswith("capture.json") for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("correctness.log")
                for path in item["expected_artifacts"]
            )
        if item["id"] == "thunderkittens_decode_attention_tile":
            assert item["paper_evaluation_id"] == "llm_serving_paper_baselines"
            assert item["status"] == "imported_to_viewer"
            assert item["serving_workload_ids"] == ["vdcores_offline_decode"]
            assert any(
                "thunderkittens_mha_capture.py" in command
                and "--shape <batch>,1,256,64" in command
                and "--prompt-tokens 128" in command
                and "--decode-tokens 64" in command
                for command in item["run_commands"]
            )

    assert paper_readiness_work_queue["summary"]["total_work_items"] == sum(
        len(claim["next_actions"])
        for claim in paper_readiness_audit["claim_audits"]
    )
    assert goal_progress["overall_status"] == "in_progress"
    assert any(
        item["id"] == "paper_grade_results"
        and item["status"] == "in_progress"
        and item["blocking_work_items"]
        == paper_readiness_work_queue["summary"]["total_work_items"]
        for item in goal_progress["acceptance_criteria"]
    )
    probe_baselines = {
        item["paper_baseline_id"]
        for item in paper_baseline_probes["paper_baseline_probes"]
    }
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= probe_baselines
    for item in paper_baseline_probes["paper_baseline_probes"]:
        assert item["latest_artifact_root"].startswith("tmp/")
        probe_root = ROOT / item["latest_artifact_root"]
        assert probe_root.is_dir()
        assert any(path.suffix == ".json" for path in probe_root.iterdir())
        expected_probe_roots = {
            "pto_persistent_device": (
                "tmp/cuda-backend/paper-baselines/probes/"
                "pto-persistent-device-a100-h200-3755feab/"
            ),
            "vllm": (
                "tmp/cuda-backend/paper-baselines/probes/"
                "vllm-a100-h200-env-27fa5aa3/"
            ),
            "sglang": (
                "tmp/cuda-backend/paper-baselines/probes/"
                "sglang-a100-h200-env-7ed53d15/"
            ),
        }
        expected_probe_root = expected_probe_roots.get(
            item["paper_baseline_id"],
            "tmp/cuda-backend/paper-baselines/probes/"
            "paired-a100-h200-86ea3913/",
        )
        assert item["latest_artifact_root"] == expected_probe_root
        assert item["checks"]
        assert item["next_action"]
        machine_status = {
            status["gpu"]: status for status in item["latest_machine_status"]
        }
        assert set(machine_status) == {"A100", "H200"}
        for status in machine_status.values():
            assert status["status"] in {"pass", "partial", "fail", "not_captured"}
            artifact = ROOT / status["artifact"]
            assert artifact.is_file()
            assert artifact.suffix == ".json"
            assert isinstance(status["blocking_gaps"], list)
        if item["paper_baseline_id"] == "thunderkittens":
            probed_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_module"
            }
            assert {
                "torch",
                "pybind11",
                "numpy",
                "pandas",
                "matplotlib",
                "tqdm",
            } <= probed_modules
        if item["paper_baseline_id"] in {"mpk", "vdcores"}:
            probed_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_module"
            }
            assert "transformers" in probed_modules
        if item["paper_baseline_id"] == "mpk":
            imported_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_import"
            }
            assert "mirage.mpk.base_dynamic_shard_loader" in imported_modules
        if item["paper_baseline_id"] == "sglang":
            assert item["latest_status"] == "pass"
            assert machine_status["H200"]["status"] == "pass"
            assert machine_status["A100"]["status"] == "pass"
            assert machine_status["A100"]["blocking_gaps"] == []
            probed_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_module"
            }
            assert {"orjson", "torchvision"} <= probed_modules
            imported_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_import"
            }
            assert {
                "sglang.bench_serving",
                "sglang.bench_offline_throughput",
                "sglang.bench_one_batch",
            } <= imported_modules
            assert all(
                check.get("python_no_user_site") is True
                for check in item["checks"]
                if check["kind"] == "python_import"
            )
        if item["paper_baseline_id"] == "vllm":
            assert item["latest_status"] == "pass"
            assert machine_status["H200"]["status"] == "pass"
            assert machine_status["A100"]["status"] == "pass"
            assert machine_status["A100"]["blocking_gaps"] == []
            imported_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_import"
            }
            assert {
                "vllm",
                "vllm.entrypoints.cli.main",
                "vllm.entrypoints.openai.api_server",
                "vllm.engine.arg_utils",
            } <= imported_modules
            assert all(
                check.get("python_no_user_site") is True
                for check in item["checks"]
                if check["kind"] == "python_import"
            )
    readiness_by_run = {
        item["paper_baseline_run_id"]: item
        for item in paper_baseline_run_readiness["paper_baseline_run_readiness"]
    }
    execution_attempts = paper_baseline_execution_attempts[
        "paper_baseline_execution_attempts"
    ]
    attempts_by_id = {item["id"]: item for item in execution_attempts}
    assert execution_attempts
    assert len(attempts_by_id) == len(execution_attempts)
    assert {
        item["paper_baseline_id"] for item in execution_attempts
    } <= paper_baseline_ids
    assert {
        item["paper_baseline_run_id"] for item in execution_attempts
    } <= run_ids
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= {item["paper_baseline_id"] for item in execution_attempts}
    assert {
        "mpk_qwen3_8b_persistent_decode1024_h200",
        "vdcores_qwen3_8b_rebuild_correctness_token1_h200",
        "vdcores_qwen3_8b_global_instruction_capacity_h200",
        "vdcores_qwen3_8b_shared_instruction_window_plan_h200",
        "vllm_qwen3_8b_repeats_h200",
        "sglang_qwen3_8b_vdcores_fixedrange_repeats_h200",
        "thunderkittens_mha_h100_fa3_comparator_h200",
        "thunderkittens_mha_h100_pt_reference_isolated_h200",
        "thunderkittens_rotary_non_mha_h200",
    } <= set(attempts_by_id)

    referenced_attempt_ids = {
        item["execution_attempt_id"]
        for claim in paper_readiness_audit["claim_audits"]
        for item in claim.get("execution_attempt_statuses", [])
    }
    referenced_attempt_ids |= {
        action["execution_attempt_id"]
        for claim in paper_readiness_audit["claim_audits"]
        for action in claim.get("next_actions", [])
        if action.get("source") == "execution_attempt"
    }
    assert referenced_attempt_ids <= set(attempts_by_id)

    for attempt in execution_attempts:
        assert attempt["artifact_root"].startswith("tmp/")
        assert (ROOT / attempt["artifact_root"]).is_dir()
        assert attempt["artifacts"]
        assert any(path.endswith(".json") for path in attempt["artifacts"])
        for path in attempt["artifacts"]:
            assert path.startswith("tmp/")
            assert (ROOT / path).is_file()
    vdcores_window_attempt = attempts_by_id[
        "vdcores_qwen3_8b_shared_instruction_window_plan_h200"
    ]
    assert (
        ".agents/skills/cuda-backend-eval/scripts/"
        "vdcores_validate_instruction_window_plan.py"
        in vdcores_window_attempt["validation_scripts"]
    )
    vdcores_window_summary = vdcores_window_attempt["summary"]
    assert vdcores_window_summary["window_contract_validation"] == "pass"
    assert (
        vdcores_window_summary["runnable_handoff_contract_status"]
        == "required_not_implemented"
    )
    assert vdcores_window_summary["paper_row_importable"] is False
    assert vdcores_window_summary["minimum_worst_case_windows_per_sm"] == 30
    command_plan_records = serving_command_plan["serving_command_plans"]
    assert serving_command_plan["metadata"]["model_tier"] == "primary"
    assert len(command_plan_records) == 46
    command_plan_run_ids = {
        item["paper_baseline_run_id"] for item in command_plan_records
    }
    assert {
        "pto_persistent_device_qwen3_8b_full_serving",
        "mpk_qwen3_native_vs_persistent",
        "mpk_qwen3_native_token_bringup",
        "vdcores_qwen3_8b_decode_preflight",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_decode_attention_tile",
    } <= command_plan_run_ids
    command_plan_baselines = {
        item["paper_baseline_id"] for item in command_plan_records
    }
    assert {
        "pto_persistent_device",
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= command_plan_baselines
    for item in command_plan_records:
        assert item["serving_workload_id"] in serving_by_id
        assert item["batch_size"] in serving_by_id[item["serving_workload_id"]][
            "decode_policy"
        ]["batch_sizes"]
        assert item["prompt_tokens"] == serving_by_id[item["serving_workload_id"]][
            "prompt_policy"
        ]["target_prompt_tokens"]
        assert item["decode_tokens"] == serving_by_id[item["serving_workload_id"]][
            "decode_policy"
        ]["decode_tokens"]
        assert item["commands"]
        assert any(command.get("raw_artifact") for command in item["commands"])
        for command in item["commands"]:
            assert command["kind"]
            assert command["command"]
            if "raw_artifact" in command:
                assert command["raw_artifact"].startswith("tmp/")
    planned_run_ids = {
        item["id"]
        for item in paper_baseline_runs["paper_baseline_runs"]
        if item.get("status", "planned_not_run") != "imported_to_viewer"
    }
    assert planned_run_ids <= set(readiness_by_run)
    for item in readiness_by_run.values():
        assert item["latest_artifact_root"].startswith("tmp/")
        assert (ROOT / item["latest_artifact_root"]).is_dir()
        assert item["latest_status"] in {"pass", "partial", "fail"}
        assert item["checks"]
        assert isinstance(item["blocking_gaps"], list)
        if item["latest_status"] != "pass":
            assert item["blocking_gaps"]
    assert not any(
        "HF_TOKEN" in gap
        for gap in readiness_by_run[
            "mpk_qwen3_native_vs_persistent"
        ]["blocking_gaps"]
    )
    assert not any(
        "HF_TOKEN" in gap
        for gap in readiness_by_run[
            "mpk_persistent_scheduler_trace"
        ]["blocking_gaps"]
    )
    assert readiness_by_run["mpk_persistent_scheduler_trace"][
        "latest_status"
    ] == "pass"
    assert not any(
        "dae.runtime" in gap
        for gap in readiness_by_run[
            "vdcores_resource_policy_trace"
        ]["blocking_gaps"]
    )
    assert not any(
        "HF_TOKEN" in gap
        for gap in readiness_by_run[
            "vdcores_resource_policy_trace"
        ]["blocking_gaps"]
    )
    assert readiness_by_run["vdcores_resource_policy_trace"][
        "latest_status"
    ] == "pass"
    assert readiness_by_run["vllm_serving_and_throughput"][
        "latest_status"
    ] == "pass"
    assert readiness_by_run["vllm_serving_and_throughput"]["blocking_gaps"] == []
    assert readiness_by_run["sglang_serving_and_offline"][
        "latest_status"
    ] == "pass"
    assert readiness_by_run["sglang_serving_and_offline"]["blocking_gaps"] == []
    pto_readiness = readiness_by_run[
        "pto_persistent_device_qwen3_8b_full_serving"
    ]
    pto_path_checks = {
        check["path"]: check["status"]
        for check in pto_readiness["checks"]
        if check["kind"] == "path_exists"
    }
    assert pto_path_checks["examples/cuda/qwen_decode_loop_runner.py"] == "pass"
    assert (
        pto_path_checks[
            ".agents/skills/cuda-backend-eval/scripts/"
            "pto_qwen_full_serving_viewer_import.py"
        ]
        == "pass"
    )
    assert not any(
        "examples/cuda/qwen_decode_loop_runner.py" in gap
        for gap in pto_readiness["blocking_gaps"]
    )

    matrix_ids = {
        item["id"] for item in paper_evaluation["paper_evaluation_matrix"]
    }
    assert {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    } <= matrix_ids
    covered_baselines = {
        baseline_id
        for item in paper_evaluation["paper_evaluation_matrix"]
        for baseline_id in item["paper_baseline_ids"]
    }
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= covered_baselines
    for item in paper_evaluation["paper_evaluation_matrix"]:
        assert item["claim"]
        assert item["status"]
        assert item["workload_ids"]
        assert item["method_ids"]
        assert item["hardware_targets"]
        assert "correctness" in item["required_metrics"]
        assert "raw_artifacts" in item["required_metrics"]
        assert item["current_evidence_refs"]
        assert isinstance(item["missing_evidence"], list)
        assert item["promotion_gate"]
        if item["id"] == "host_schedule_launch_overhead":
            assert item["status"] == "ready_for_paper_claim"
            assert "host_schedule_stream_concurrency" in item["workload_ids"]
            assert {
                "pto_stream_serial",
                "pto_stream_parallel",
            } <= set(item["method_ids"])
            stream_refs = [
                ref
                for ref in item["current_evidence_refs"]
                if ref.get("benchmark_id") == "host_schedule_stream_concurrency"
            ]
            assert {
                (ref["gpu"], ref["method_id"]) for ref in stream_refs
            } == {
                ("A100", "pto_stream_serial"),
                ("A100", "pto_stream_parallel"),
                ("H200", "pto_stream_serial"),
                ("H200", "pto_stream_parallel"),
            }
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/combined-stream-pool6-02bca4df/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/graph-replay-sweep-01e30e99/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/direct-launch-sweep-626b8c75/"
                for ref in item["current_evidence_refs"]
            )
            assert item["missing_evidence"] == []
        if item["id"] == "persistent_device_scheduler_overhead":
            assert any(
                ref.get("kind") == "raw_artifact"
                and ref.get("path")
                == "tmp/cuda-backend/scheduler-breakdown-6f7a1040/persistent-scheduler-breakdown-6f7a1040/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "graph_layered_cross"
                and ref.get("method_id") == "mpk"
                and ref.get("gpu") == "H200"
                for ref in item["current_evidence_refs"]
            )
            assert not any(
                "Scheduler-overhead breakdown" in gap
                for gap in item["missing_evidence"]
            )
            assert not any(
                "MPK bounded-decode run" in gap
                for gap in item["missing_evidence"]
            )
        if item["id"] == "tensor_core_tile_baselines":
            assert item["status"] == "ready_for_paper_claim"
            assert item["missing_evidence"] == []
            policy = item["evidence_policy_exceptions"][0]
            assert policy["id"] == "thunderkittens_dense_pytorch_12288_oom_policy"
            assert policy["status"] == "accepted"
            assert "FA3 forward/backward comparator rows" in policy["decision"]
            assert any(
                ref["kind"] == "tmp_artifact"
                and ref["path"].endswith("isolated-pt-reference-summary.json")
                for ref in policy["evidence_refs"]
            )
        if item["id"] == "llm_serving_paper_baselines":
            policies = {
                policy["id"]: policy
                for policy in item["evidence_policy_exceptions"]
            }
            tk_policy = policies[
                "thunderkittens_llm_non_full_serving_policy_pending"
            ]
            assert tk_policy["status"] == "pending"
            assert tk_policy["missing_evidence_id"] == (
                "thunderkittens_full_serving_qwen3_8b"
            )
            assert "must remain non-paper-ready" in tk_policy["review_rule"]
            assert any(
                ref["kind"] == "changelog"
                and ref["path"].endswith(
                    "2026-06-03-thunderkittens-gemm-compatibility-probe.md"
                )
                for ref in tk_policy["evidence_refs"]
            )
            assert not any(
                "Selected shared model" in gap
                for gap in item["missing_evidence"]
            )
            assert not any(
                "PTO CUDA serving workload" in gap
                for gap in item["missing_evidence"]
            )
            assert any(
                ref.get("path")
                == "evaluations/nvidia/benchmark-viewer/data/serving_workloads.json"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "evaluations/nvidia/benchmark-viewer/data/serving_command_plan.json"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "pto_persistent_device"
                and ref.get("gpu") == "H200"
                and ref.get("shape_contains")
                == "vdcores_offline_decode attention tile proxy"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "mpk"
                and ref.get("gpu") == "H200"
                and ref.get("shape_contains") == "Qwen/Qwen3-0.6B"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "mpk"
                and ref.get("gpu") == "H200"
                and ref.get("shape_contains")
                == "mpk_offline_decode,Qwen/Qwen3-8B"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "vllm"
                and ref.get("gpu") == "H200"
                and ref.get("shape_contains")
                == "vdcores_offline_decode,Qwen/Qwen3-8B"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "vllm"
                and ref.get("gpu") == "H200"
                and ref.get("shape_contains")
                == "mpk_offline_decode,Qwen/Qwen3-8B"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "sglang"
                and ref.get("gpu") == "H200"
                and ref.get("shape_contains")
                == "vdcores_offline_decode,Qwen/Qwen3-8B"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("method_id") == "sglang"
                and ref.get("shape_contains")
                == "mpk_offline_decode,Qwen/Qwen3-8B"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/paper-baselines/mpk/bringup-qwen3-0.6b/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/paper-baselines/mpk/qwen3-8b-mpk-policy-072cc513/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-qwen3-8b-repeats-eb75a235/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-repeats-eb75a235/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-mpk-qwen3-8b-repeats-91ddaf86/"
                for ref in item["current_evidence_refs"]
            )
            assert not any(
                "vLLM, SGLang" in gap for gap in item["missing_evidence"]
            )
            assert not any(
                "SGLang MPK-policy" in gap for gap in item["missing_evidence"]
            )
            assert not any(
                "MPK persistent-kernel" in gap
                for gap in item["missing_evidence"]
            )
            missing_detail_ids = {
                detail["id"] for detail in item["missing_evidence_details"]
            }
            assert "mpk_persistent_qwen3_8b" not in missing_detail_ids

    assert paper_readiness_audit["overall_status"] == "not_paper_ready"
    assert paper_readiness_audit["ready_claims"] == 3
    assert paper_readiness_audit["blocked_claims"] == 1
    assert paper_readiness_audit["claim_audits"]
    for item in paper_readiness_audit["claim_audits"]:
        assert item["matrix_status"]
        assert "paper_baseline_run_readiness_statuses" in item
        if item["id"] == "persistent_device_scheduler_overhead":
            assert item["execution_attempt_statuses"] == []
            assert not any(
                action.get("source") == "execution_attempt"
                and action.get("paper_baseline_run_id")
                == "vdcores_resource_policy_trace"
                for action in item["next_actions"]
            )
            assert not any(
                "VDCores queue-pressure and scheduler-overhead"
                in action["action"]
                for action in item["next_actions"]
            )
        if item["id"] == "host_schedule_launch_overhead":
            assert item["matrix_status"] == "ready_for_paper_claim"
            assert item["ready_for_paper_claim"] is True
            assert item["blockers"] == []
        elif item["id"] == "persistent_device_scheduler_overhead":
            assert item["matrix_status"] == "ready_for_paper_claim"
            assert item["ready_for_paper_claim"] is True
            assert item["blockers"] == []
            assert item["next_actions"] == []
        elif item["id"] == "tensor_core_tile_baselines":
            assert item["matrix_status"] == "ready_for_paper_claim"
            assert item["ready_for_paper_claim"] is True
            assert item["blockers"] == []
            assert item["next_actions"] == []
            assert item["evidence_policy_exceptions"]
        else:
            assert item["ready_for_paper_claim"] is False
            assert item["blockers"]
        assert item["promotion_gate"]

    assert results["snapshot"]["commit"] == "743709f3"
    assert results["snapshot"]["committed_result_records"] == len(
        results["result_records"]
    )
    assert results["snapshot"]["compaction_note"]
    assert results["snapshot"]["full_capture"]["samples"] > (
        results["snapshot"]["compact_capture"]["samples"]
    )
    for capture in [
        results["snapshot"]["full_capture"],
        results["snapshot"]["compact_capture"],
    ]:
        artifact_root = ROOT / capture["artifact_root"]
        assert artifact_root.is_dir()
        assert any(path.suffix == ".json" for path in artifact_root.iterdir())

    result_records = results["result_records"]
    assert result_records
    result_groups = {
        (
            record["benchmark_id"],
            record["method_id"],
            record["hardware"]["gpu"],
        )
        for record in result_records
    }
    assert {
        ("graph_layered_cross", "mpk", "H200"),
        ("llm_serving_decode", "mpk", "H200"),
        ("llm_serving_decode", "vdcores", "H200"),
        ("llm_serving_decode", "pto_persistent_device", "H200"),
        ("llm_serving_decode", "vllm", "H200"),
        ("llm_serving_decode", "sglang", "H200"),
        ("llm_serving_decode", "thunderkittens", "H200"),
        ("tensor_core_tile", "pto_persistent_device", "A100"),
        ("tensor_core_tile", "pto_persistent_device", "H200"),
        ("tensor_core_tile", "triton", "A100"),
        ("tensor_core_tile", "triton", "H200"),
        ("tensor_core_tile", "cutlass", "A100"),
        ("tensor_core_tile", "cutlass", "H200"),
    } <= result_groups
    for gpu in {"A100", "H200"}:
        assert ("host_schedule_vector_ops", "direct_runtime", gpu) in result_groups
        assert ("host_schedule_vector_ops", "direct_driver", gpu) in result_groups
        assert (
            "host_schedule_vector_ops",
            "direct_driver_graph",
            gpu,
        ) in result_groups
        assert ("tensor_core_tile", "direct_runtime", gpu) in result_groups
        assert ("tensor_core_tile", "direct_driver", gpu) in result_groups
        assert ("tensor_core_tile", "direct_driver_graph", gpu) in result_groups

    full_serving_rows = [
        record
        for record in result_records
        if record["benchmark_id"] == "llm_serving_decode"
        and record["hardware"]["gpu"] == "H200"
        and record["statistic"].get("serving_coverage") == "full_serving"
    ]
    assert {
        (record["method_id"], record["statistic"]["decode_tokens"])
        for record in full_serving_rows
    } >= {
        ("vllm", 64),
        ("vllm", 1024),
        ("sglang", 64),
        ("sglang", 1024),
    }
    assert all(
        record["correctness"] == "pass"
        and record["statistic"]["sample_count"] >= 1
        and record["statistic"].get("failed_requests", 0) == 0
        and record["statistic"].get("throughput_tokens_per_s", 1) > 0
        for record in full_serving_rows
    )

    serving_coverages = {
        record["statistic"].get("serving_coverage")
        for record in result_records
        if record["benchmark_id"] == "llm_serving_decode"
    }
    assert {
        "full_serving",
        "controlled_attention_tile_proxy",
        "diagnostic_microdecode",
    } <= serving_coverages

    for record in results["result_records"]:
        assert record["benchmark_id"] in benchmark_ids
        assert record["method_id"] in method_ids
        assert record["hardware"]["gpu"]
        assert record["statistic"]["sample_count"] > 0
        assert record["correctness"] in {"pass", "caveat"}
        assert record["raw_artifact"].startswith("tmp/")
        raw_artifact = ROOT / record["raw_artifact"]
        assert raw_artifact.exists()
        if raw_artifact.is_dir():
            assert any(path.suffix == ".json" for path in raw_artifact.iterdir())
        else:
            assert raw_artifact.suffix == ".json"
    assert {"A100", "H200"} <= {
        item["gpu"] for item in results["headline_results"]
    }


def test_review_policy_changelog_and_examples_exist():
    assert (ROOT / ".agents" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "coding-guidance.md").is_file()
    assert (ROOT / ".agents" / "templates" / "ultimate-goal.md").is_file()
    assert (ROOT / ".agents" / "rules" / "core-development.md").is_file()
    assert (ROOT / ".agents" / "rules" / "requirements-first.md").is_file()
    assert (ROOT / ".agents" / "rules" / "testing-and-verification.md").is_file()
    assert (ROOT / ".agents" / "rules" / "ultimate-goal-dispatch.md").is_file()
    assert (ROOT / ".agents" / "rules" / "nvidia-backend-review.md").is_file()
    assert (ROOT / ".agents" / "rules" / "remote-evaluation.md").is_file()
    assert (ROOT / ".agents" / "agents" / "code-review" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "agents" / "documentation-sync" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "agents" / "testing" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_cuda_examples.py").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_remote_evaluation.py").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_nvidia_changelog.py").is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_viewer_export.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_viewer_export.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_results_update.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_run_readiness.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_mha_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_full_sweep_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "triton_tensor_tile_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cutlass_tensor_tile_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_scheduler_breakdown.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_probe.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_pair_probe.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_serving_command_plan.py"
    ).is_file()
    assert (ROOT / ".agents" / "skills" / "git-commit" / "SKILL.md").is_file()
    assert (ROOT / ".agents" / "skills" / "github-pr" / "SKILL.md").is_file()
    assert (DOC_ROOT / "changelog" / "index.md").is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-review-readiness.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-ultimate-goal.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-benchmark-viewer-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-viewer-result-export.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-changelog-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-cuda-example-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-evaluation-matrix.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-remote-evaluation-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-runs.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-importer.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-probes.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-paired-probe.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-thunderkittens-bounded-capture.md"
    ).is_file()
    assert (DOC_ROOT / "changelog" / "2026-05-31-serving-policy.md").is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-serving-command-plan.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paired-probe-dependencies.md"
    ).is_file()

    example_root = ROOT / "examples" / "cuda"
    assert (example_root / "README.md").is_file()
    assert (example_root / "manifest.json").is_file()
    assert (example_root / "host_schedule_vector_ops.py").is_file()
    assert (example_root / "persistent_layered_cross.py").is_file()


def test_ultimate_goal_ci_is_closed_and_avoids_ascend_jobs():
    workflow_paths = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    workflow_paths += sorted((ROOT / ".github" / "workflows").glob("*.yaml"))
    assert workflow_paths == []

    workflow = (
        ROOT / "docs" / "ci" / "nvidia-manual-review.workflow.yml"
    ).read_text(encoding="utf-8")
    assert "NVIDIA Manual Review" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "nvidia-manual-review:" in workflow

    ci_doc = (ROOT / "docs" / "ci.md").read_text(encoding="utf-8")
    assert "No runnable workflow YAML" in ci_doc
    assert "closed-CI policy" in ci_doc
    assert "a2a3/a5 CI" in ci_doc


def test_ultimate_goal_artifacts_define_paper_ready_cuda_path():
    goal_root = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready"
    goal_file = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready.md"

    assert goal_file.is_file()
    assert (goal_root / "dispatch_log.md").is_file()
    assert (goal_root / "work_preparation.md").is_file()
    assert (goal_root / "shared_contracts.md").is_file()
    assert (goal_root / "evaluation_plan.md").is_file()
    assert (goal_root / "baseline_survey.md").is_file()

    goal_text = goal_file.read_text(encoding="utf-8")
    for required in [
        "standalone pto-cu",
        "human-reviewable benchmark viewer",
        "MPK",
        "VDCores",
        "remote evaluation fallback",
        "code evidence",
    ]:
        assert required in goal_text

    evaluation_text = (goal_root / "evaluation_plan.md").read_text(
        encoding="utf-8"
    )
    for required in [
        "paper-ready",
        "Mirage Persistent Kernel",
        "VDCores",
        "CUDA Graph",
        "cuBLAS",
        "A100",
        "H200",
    ]:
        assert required in evaluation_text

    contracts_text = (goal_root / "shared_contracts.md").read_text(
        encoding="utf-8"
    )
    for required in [
        "benchmark_id",
        "method_id",
        "paper_baseline_id",
        "evidence_refs",
        "changelog report",
        "source notes",
    ]:
        assert required in contracts_text

    baseline_text = (goal_root / "baseline_survey.md").read_text(
        encoding="utf-8"
    )
    for required in [
        "mirage-project/mirage",
        "vdcores/vdcores",
        "vLLM",
        "SGLang",
        "ThunderKittens",
        "tmp/baselines/mirage-mpk",
        "tmp/baselines/vdcores",
        "tmp/baselines/vllm",
        "tmp/baselines/sglang",
        "tmp/baselines/thunderkittens",
        "bench_serving",
        "bench throughput",
    ]:
        assert required in baseline_text
