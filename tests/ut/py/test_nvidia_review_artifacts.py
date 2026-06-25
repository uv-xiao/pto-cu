import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = DOC_ROOT / "benchmark-viewer"
REMOVED_EVAL_SCRIPT_RE = re.compile(
    r"cuda_(smoke|persistent_smoke|benchmark)\.py|cuda-backend-eval/scripts"
)
ACTIVE_REVIEW_SURFACES = [
    VIEWER_ROOT / "data" / "benchmarks.json",
    VIEWER_ROOT / "data" / "methods.json",
    VIEWER_ROOT / "data" / "results.json",
    ROOT / "examples" / "cuda" / "README.md",
    ROOT / "examples" / "cuda" / "host_schedule_vector_ops.py",
    ROOT / "examples" / "cuda" / "persistent_layered_cross.py",
    ROOT / "examples" / "cuda" / "persistent_moe_dispatch_combine.py",
]
GLUON_GEMM_REVIEW_DOCS = [
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_gen_adapter.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_gemm_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_tensor_core_gemm.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_rmsnorm_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_layernorm_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_rope_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_silu_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_gelu_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_gated_silu_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_gemma_fused_rmsnorm_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_topk_sampling_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_topp_sampling_h200.md",
    ROOT / "docs" / "in_progress" / "nvidia_backend" / "gluon_minp_sampling_h200.md",
    ROOT
    / "docs"
    / "in_progress"
    / "nvidia_backend"
    / "gluon_speculative_decoding_h200.md",
]
UCCL_PRIVATE_PATH_RE = re.compile(
    r"/" + "home/"
    r"|/" + "Users/"
    r"|/" + "tmp/" + "pto" + "-cu"
    r"|/" + "tmp/uccl-"
    r"|" + "uv" + "xiao"
    r"|" + "bi" + "zhao" + "h200"
    r"|" + "hi" + "na"
    r"|" + "da" + "sys"
)


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

    benchmarks = json.loads(
        (VIEWER_ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8")
    )
    methods = json.loads(
        (VIEWER_ROOT / "data" / "methods.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (VIEWER_ROOT / "data" / "results.json").read_text(encoding="utf-8")
    )

    benchmark_ids = {item["id"] for item in benchmarks["benchmarks"]}
    assert "graph_layered_cross" in benchmark_ids
    assert "tensor_core_tile" in benchmark_ids
    for benchmark in benchmarks["benchmarks"]:
        assert benchmark["description"]
        assert benchmark["math"]
        assert benchmark["code"]
        assert benchmark["run"]["command"]
        assert benchmark["evidence_refs"]

    method_ids = {item["id"] for item in methods["methods"]}
    assert {"pto_host_schedule", "pto_persistent_device", "cublas_sgemm_graph"} <= method_ids

    assert results["snapshot"]["commit"] == "743709f3"
    assert results["snapshot"]["full_capture"]["samples"] == 1350
    assert results["snapshot"]["compact_capture"]["samples"] == 108
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
    assert (ROOT / ".agents" / "skills" / "git-commit" / "SKILL.md").is_file()
    assert (ROOT / ".agents" / "skills" / "github-pr" / "SKILL.md").is_file()
    assert (DOC_ROOT / "changelog" / "index.md").is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-review-readiness.md"
    ).is_file()

    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    assert (in_progress_root / "vllm_remote_install_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_env_artifact_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_artifact_complete.md").is_file()
    assert (in_progress_root / "vllm_remote_model_load_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_server_health_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_inference_smoke_probe.md").is_file()
    assert (
        in_progress_root / "vllm_remote_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_warmup_shape_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_request_shape_variation_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_serving_semantics_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_logprobs_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_echo_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_stop_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_64k_context_health_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_128k_context_health_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_256k_context_health_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_long_prompt_admission_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_long_prompt_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_long_prompt_warmup_followup_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_32k_long_prompt_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_64k_long_prompt_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_128k_long_prompt_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_192k_long_prompt_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_256k_long_prompt_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_256k_needle_correctness_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_256k_needle_exact_output_failure.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_256k_needle_exact_stop_sequence_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_256k_needle_exact_stop_repeat_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_256k_needle_exact_truncated_failure_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_256k_needle_position_sweep_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_chat_exact_canary_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_chat_exact_truncated_failure_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_chat_256k_needle_exact_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_chat_256k_needle_stream_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_chat_256k_needle_stream_repeat_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_chat_256k_needle_stream_position_sweep_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_chat_256k_needle_stream_usage_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root
        / "vllm_remote_chat_256k_needle_stream_truncated_failure_probe.md"
    ).is_file()
    assert (
        in_progress_root / "deepseek_v4_flash_serving_readiness.md"
    ).is_file()
    assert (
        in_progress_root / "deepseek_v4_flash_weight_manifest_preflight.md"
    ).is_file()
    assert (
        in_progress_root
        / "deepseek_v4_flash_weight_acquisition_preflight_h200.md"
    ).is_file()

    assert (ROOT / "tools" / "check_nvidia_review_ready.py").is_file()

    example_root = ROOT / "examples" / "cuda"
    assert (example_root / "README.md").is_file()
    assert (example_root / "host_schedule_vector_ops.py").is_file()
    assert (example_root / "persistent_layered_cross.py").is_file()
    assert (example_root / "persistent_moe_dispatch_combine.py").is_file()
    assert (example_root / "vllm_deepseek_v4_server_health_probe.py").is_file()
    assert (example_root / "vllm_deepseek_v4_inference_smoke_probe.py").is_file()
    assert (
        example_root / "vllm_deepseek_v4_response_contract_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_warmup_shape_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_request_shape_variation_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_serving_semantics_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_long_prompt_admission_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_long_prompt_response_contract_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_long_prompt_warmup_followup_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_needle_correctness_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_chat_exact_canary_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_chat_256k_needle_exact_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_chat_256k_needle_stream_probe.py"
    ).is_file()
    assert (
        example_root
        / "vllm_deepseek_v4_chat_256k_needle_stream_repeat_probe.py"
    ).is_file()
    assert (
        example_root
        / "vllm_deepseek_v4_chat_256k_needle_stream_position_sweep_probe.py"
    ).is_file()
    assert (
        example_root
        / "vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe.py"
    ).is_file()
    assert (
        example_root / "deepseek_v4_flash_weight_acquisition_preflight.py"
    ).is_file()


def test_nvidia_goal_status_rollup_tracks_current_boundaries():
    rollup = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "goal_status_rollup.md"
    )
    assert rollup.is_file()

    text = rollup.read_text(encoding="utf-8")
    required_phrases = [
        "`origin/main` at",
        "a6378bfbf55b15be01c334f43332ccd20c160cfa",
        "`accepted evidence`",
        "`partial evidence`",
        "DeepSeek/vLLM serving evidence",
        "Simpler-nv integration evidence",
        "`--with-uccl-ep-fused-boundary`",
        "`status: unsupported`",
        "`persistent_device_uccl_ep_runtime_fusion`",
        "`nvidia-uccl-ep-runtime-fusion-coordinator-boundary-map`",
        "guard-only blocked implementation handoff",
        "provenance-only evidence",
        "no shared payload ownership token or lifetime transition log",
        "not vLLM plugin integration",
        "not actual fused cross-GPU expert-parallel MoE execution",
    ]
    for phrase in required_phrases:
        assert phrase in text

    assert "`nvidia-moe-uccl-ep-fused-boundary-h200`" not in text


def test_uccl_in_progress_docs_omit_private_paths():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    uccl_docs = sorted(in_progress_root.glob("uccl_*.md"))
    assert uccl_docs

    offenders = []
    for path in uccl_docs:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if UCCL_PRIVATE_PATH_RE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}: {line}")

    assert offenders == []


def test_persistent_moe_two_device_baseline_is_review_safe():
    doc = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "persistent_moe_dispatch_combine_h200.md"
    ).read_text(encoding="utf-8")
    example = (ROOT / "examples" / "cuda" / "persistent_moe_dispatch_combine.py").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "examples" / "cuda" / "README.md").read_text(encoding="utf-8")

    for required in [
        "run_two_device_moe_dispatch_combine",
        "run_persistent_moe_nccl_handoff",
        "run_persistent_moe_uccl_ep_handoff",
        "--device-ids",
        "--with-nccl-handoff",
        "--with-uccl-ep-handoff",
        "same-node-two-device-baseline",
        "persistent-moe-plus-nccl-worker-control",
        "persistent-moe-plus-uccl-ep-adapter",
        "same-node two-device baseline evidence",
        "not fused cross-GPU expert-parallel MoE",
        "not CUDA host-runtime UCCL dispatch",
        "source_digests",
        "bridge_metadata_match",
    ]:
        assert required in example

    for required in [
        "--device-ids 6,7 --n 4096 --arch compute_90 --require-cuda",
        "--with-nccl-handoff",
        "--with-uccl-ep-handoff",
        "--tensor-numel 1024",
        "--build --require-cuda",
        "same-node two-device baseline evidence",
        "persistent MoE plus NCCL worker-control handoff",
        "persistent MoE plus UCCL-EP adapter handoff",
        "not fused cross-GPU",
        "expert-parallel MoE",
        "not CUDA host-runtime UCCL dispatch",
        "output error",
        "completion count",
        "scheduler error state",
        "source/bridge",
        "digests",
    ]:
        assert required in readme

    for required in [
        "Two-Device Remote H200 Result",
        "Communication-Coupled Handoff Gate",
        "REMOTE_PTO_CU=/tmp/" "pto-cu-codex-restart",
        "REMOTE_PTO_CU=/tmp/" "pto-cu-persistent-moe-nccl-handoff",
        "REMOTE_PTO_CU=/tmp/" "pto-cu-persistent-moe-uccl-ep-handoff",
        "--device-ids 6,7 --n 4096 --arch compute_90 --require-cuda",
        "--with-nccl-handoff --tensor-numel 1024 --build --require-cuda",
        "--with-uccl-ep-handoff --tensor-numel 1024 --require-cuda",
        "status`: `passed`",
        "handoff_scope`: `persistent-moe-plus-nccl-worker-control`",
        "handoff_scope`: `persistent-moe-plus-uccl-ep-adapter`",
        "evidence_scope`: `same-node-two-device-baseline`",
        "device_ids`: `[6, 7]`",
        "tensor_numel`: `1024`",
        "per_device_count`: `2`",
        "all_devices_passed`: `true`",
        "completed_count_is_5`: `true`",
        "scheduler_errors_zero`: `true`",
        "same_device_ids`: `true`",
        "nccl_worker_control_passed`: `true`",
        "uccl_ep_adapter_passed`: `true`",
        "adapter_descriptor_metadata_present`: `true`",
        "max_errors_zero`: `true`",
        "source_digests_match`: `true`",
        "bridge_metadata_match`: `true`",
        "c096ede6d4ab5e1a9a33070bc1fcf988b9fb9c405d929a770c962308b396b209",
        "7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f",
        "not fused cross-GPU expert-parallel MoE",
        "does not validate distributed expert parallelism",
        "does not validate CUDA host-runtime UCCL dispatch",
    ]:
        assert required in doc
    assert "--with-uccl-ep-handoff --tensor-numel 1024 --build --require-cuda" not in doc
    assert "--with-uccl-ep-handoff --tensor-numel 1024 --build --require-cuda" not in readme


def test_persistent_moe_uccl_ep_fused_boundary_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    doc = (in_progress_root / "persistent_moe_dispatch_combine_h200.md").read_text(
        encoding="utf-8"
    )
    boundary = (in_progress_root / "communication_runtime_boundary.md").read_text(
        encoding="utf-8"
    )
    selection = (in_progress_root / "communication_selection.md").read_text(
        encoding="utf-8"
    )
    example = (
        ROOT / "examples" / "cuda" / "persistent_moe_dispatch_combine.py"
    ).read_text(encoding="utf-8")

    for required in [
        "run_persistent_moe_uccl_ep_fused_boundary",
        "_validate_runtime_fusion_evidence",
        "--with-uccl-ep-fused-boundary",
        "reduced-fused-cross-gpu-expert-parallel-moe-boundary",
        "persistent_device_uccl_ep_runtime_fusion",
        "structured_unsupported_boundary",
        "actual_fused_cross_gpu_execution",
        "payload_provenance",
        "shared_payload_ownership",
        "payload_lifetime_transition_log",
        "failure_fields",
        "fabricated_or_untrusted_pass_evidence",
        "unsupported",
        "non-evidence",
    ]:
        assert required in example
        assert required in doc

    for text in (boundary, selection):
        assert "reduced fused cross-GPU expert-parallel MoE boundary" in text
        assert "structured unsupported boundary" in text
        assert "payload_provenance" in text
        assert "shared ownership token" in text
        assert "lifetime transition log" in text
        assert "fabricated_or_untrusted_pass_evidence" in text
        assert "failure_fields" in text
        assert "not fused evidence" in text
        assert "persistent_device_uccl_ep_runtime_fusion" in text
        assert "/" + "home/" not in text

    assert "<external-uccl-ep-bench>" in doc
    assert "<uccl-python-site-packages>" in doc
    assert "/" + "home/" not in doc


def test_persistent_device_uccl_ep_runtime_fusion_contract_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    persistent_moe = (
        in_progress_root / "persistent_moe_dispatch_combine_h200.md"
    ).read_text(encoding="utf-8")
    boundary = (in_progress_root / "communication_runtime_boundary.md").read_text(
        encoding="utf-8"
    )
    selection = (in_progress_root / "communication_selection.md").read_text(
        encoding="utf-8"
    )
    slicing = (in_progress_root / "pr_slicing_plan.md").read_text(
        encoding="utf-8"
    )
    dispatch_log = (in_progress_root / "dispatch_log.md").read_text(
        encoding="utf-8"
    )
    private_entry_header = (
        ROOT / "src" / "cuda" / "platform" / "include" / "host"
        / "pto_cuda_runtime_fusion_abi.h"
    ).read_text(encoding="utf-8")
    private_envelope_header = (
        ROOT / "src" / "cuda" / "platform" / "include" / "host"
        / "pto_cuda_private_run_envelope.h"
    ).read_text(encoding="utf-8")
    cuda_host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host"
        / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    chip_worker = (ROOT / "src" / "common" / "worker" / "chip_worker.cpp").read_text(
        encoding="utf-8"
    )
    private_entry_test = (
        ROOT / "tests" / "ut" / "py" / "test_cuda_runtime_fusion_private_entry.py"
    ).read_text(encoding="utf-8")
    normalized_persistent_moe = " ".join(persistent_moe.split())
    normalized_boundary = " ".join(boundary.split())

    for text in (persistent_moe, boundary, selection, slicing, dispatch_log):
        assert "persistent_device_uccl_ep_runtime_fusion" in text
        assert "actual fused" in text
        assert "cross-GPU expert-parallel MoE" in text

    for required in [
        "payload owner field",
        "payload lifetime state",
        "rank-to-CUDA-device mapping",
        "CUDA persistent-device runtime run context",
        "persistent-device/UCCL-EP runtime fusion coordinator",
        "Runtime Fusion Coordinator Boundary",
        "The descriptor allocation site is the CUDA persistent-device runtime run",
        "The ownership token issuer is also the coordinator",
        "The required state machine is",
        "fabricated_or_untrusted_pass_evidence",
        "PR #147 provenance is accepted input evidence only",
        "Mandatory failure states include descriptor shape mismatch",
        "Non-evidence states also include independent two-device",
        "`passed`",
        "`unsupported`",
        "`setup_failed`",
        "`failed`",
        "Unsupported and setup-failed states are non-evidence",
        "Private Runtime Entry Contract",
        "persistent_device_uccl_ep_runtime_fusion_entry",
        "`ChipWorker::run` after `ChipStorageTaskArgs` has been assembled",
        "No field is added to public `TaskArgs` or public `CallConfig`",
        "`callable_id`",
        "`rank_device_map`",
        "`persistent_graph_descriptor`",
        "`uccl_ep_capability`",
        "`descriptor_allocation_policy`",
        "`validation_policy`",
        "`output_sink`",
        "`coordinator_status`",
        "`descriptor_allocation_provenance`",
        "`ownership_token`",
        "`state_transitions`",
        "`validation_summary`",
    ]:
        assert required in boundary
    assert "Example-side JSON, adapter-only provenance" in normalized_boundary

    for required in [
        "Future Fused Execution Evidence Shape",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
        "`device_ids`, `rank_to_device`, `world_size`",
        "shared ownership token",
        "payload ownership/lifetime transition log",
        "Implementation-Readiness Map",
        "Local evidence required before a later implementation reports",
        "H200 evidence required before those fields may report passed/true",
        "persistent-moe-uccl-ep-runtime-fusion-h200.json",
        "`unsupported`, `setup_failed`, and",
        "not fused execution",
        "Runtime-Owned Descriptor Implementation Handoff",
        "No fresh H200 fused-boundary run is recorded",
        "Coordinator Boundary Map",
        "Reviewable entry point",
        "Descriptor allocation site",
        "Ownership token issuer",
        "Lifetime transition state machine",
        "Failure-field responsibilities",
        "PR #151 remains a post-PR150 status refresh",
        "Coordinator Entry Contract",
        "persistent_device_uccl_ep_runtime_fusion_entry",
        "The entry contract does not widen the public runtime API",
        "coordinator request is assembled from private runtime state",
        "The result returned to the host/runtime status artifact",
        "No fresh H200 fused-boundary run is recorded for this entry-contract",
        "Accepted Private Entry Unsupported Scaffold",
        "nvidia-uccl-ep-runtime-fusion-private-entry-unsupported",
        "private entry scaffold behind `ChipWorker::run`",
        "The expected review-safe result remains",
        "Private Entry Unsupported Scaffold",
        "pto_cuda_runtime_fusion_abi.h",
        "PtoCudaRuntimeFusionRequest",
        "PtoCudaRuntimeFusionResult",
        "explicit failure bits",
        "Closed Invalid ChipStorageTaskArgs Request Boundary Attempt",
        "slice now narrows",
        "explicitly rejects the private-envelope path",
        "nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request",
        "nvidia-uccl-ep-runtime-fusion-private-request-envelope",
        "pto_cuda_private_run_envelope.h",
        "PtoCudaPersistentDagArgs *",
        "`sizeof(ChipStorageTaskArgs)`",
        "Runtime Args Handoff Map",
        "not a `ChipStorageTaskArgs *`, `ChipStorageTaskArgs *` is not",
    ]:
        assert required in persistent_moe
    assert "same `ChipWorker::run` invocation" in normalized_persistent_moe
    assert "Example-side JSON, adapter-only provenance" in (
        normalized_persistent_moe
    )

    normalized_selection = " ".join(selection.split())
    for required in [
        "runtime fusion coordinator owns",
        "PR #147 remains accepted provenance-only input evidence",
        "actual_fused_cross_gpu_execution` remains `false`",
        "nvidia-uccl-ep-runtime-fusion-impl-runtime-owned-descriptor",
        "adds local guards only",
        "coordinator-boundary map keeps the runtime owner concrete",
        "PR #150 remains guard-only blocked implementation evidence",
        "PR #151 remains a post-PR150 status refresh",
        "persistent_device_uccl_ep_runtime_fusion_entry",
        "does not add public `TaskArgs`, public `CallConfig`, or UCCL",
        "callable id, chip-local rank/device map, persistent graph descriptor",
        "coordinator status, descriptor allocation provenance",
        "forbidden pass-evidence paths",
        "pto_cuda_runtime_fusion_abi.h",
        "PtoCudaRuntimeFusionRequest",
        "explicit failure bits",
        "PR #152 remains a coordinator-boundary map only",
        "PR #153 remains a private entry-contract only",
        "PR #155 remains a private unsupported runtime scaffold only",
        "nvidia-uccl-ep-runtime-fusion-private-entry-unsupported",
        "PR #157",
        "nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request",
        "PtoCudaPersistentDagArgs *",
        "nvidia-uccl-ep-runtime-fusion-private-request-envelope",
        "keeps the fused-boundary result `unsupported`",
        "explicitly rejects the private-envelope path",
        "PR #161 selected",
        "nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map",
        "same-invocation inputs with matching sizes and callable type",
        "Null, stale, wrong-size, wrong-callable, or cross-invocation",
        "PR #162 accepted that handoff map as docs/test dependency evidence",
        "nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff",
        "mismatched-callable cases, stale envelopes, cross-invocation",
        "adds only the private CUDA persistent DAG host-runtime association",
        "does not add a runtime-fusion coordinator",
        "PR #164's accepted implementation slice",
        "nvidia-uccl-ep-runtime-fusion-capability-metadata-map",
        "private UCCL-EP capability metadata",
        "PR #166 accepted only the private UCCL-EP capability metadata",
        "nvidia-uccl-ep-runtime-fusion-validation-policy-map",
        "private validation policy",
    ]:
        assert required in normalized_selection

    normalized_slicing = " ".join(slicing.split())
    normalized_dispatch_log = " ".join(dispatch_log.split())
    assert "Accepted Coordinator Boundary Map Slice" in slicing
    assert "Accepted Coordinator Entry Contract Slice" in slicing
    assert "Accepted Private Entry Unsupported Scaffold" in slicing
    assert "Closed Invalid ChipStorageTaskArgs Request Boundary Attempt" in slicing
    assert "Accepted Private Request Envelope Dependency Slice" in slicing
    assert "Accepted Runtime Args Handoff Map Slice" in slicing
    assert "Accepted Private Host Runtime Handoff Implementation Slice" in slicing
    assert "Accepted UCCL-EP Capability Metadata Map Slice" in slicing
    assert "Accepted Validation Policy Map Slice" in slicing
    assert "Accepted Descriptor Allocation Policy Map Slice" in slicing
    assert "Accepted Guard-Only Implementation Handoff" in slicing
    assert "Accepted Post-PR150 Status Refresh" in slicing
    assert "nvidia-uccl-ep-adapter-payload-provenance" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-readiness" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-impl-runtime-owned-descriptor" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-coordinator-boundary-map" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-coordinator-entry-contract" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-private-entry-unsupported" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-private-request-envelope" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map" in slicing
    assert "PR #157 attempted that request boundary but was closed invalid" in (
        normalized_slicing
    )
    assert "PtoCudaPersistentDagArgs *" in slicing
    assert "`sizeof(ChipStorageTaskArgs)`" in slicing
    assert "no real `ChipStorageTaskArgs` request path" in normalized_slicing
    assert "pto_cuda_private_run_envelope.h" in slicing
    assert "run_prepared_with_cuda_private_args" in slicing
    assert "does not pass `ChipStorageTaskArgs *` as runtime args" in (
        normalized_slicing
    )
    assert "PR #158 fixed the Codex monitor transcript lookup" in slicing
    assert "PR #159 recorded PR #157 as a closed invalid" in slicing
    assert "PR #160 added the private CUDA run envelope" in slicing
    assert "PR #161 recorded the post-PR160 status refresh" in slicing
    assert "PR #162 accepted that runtime-args handoff map" in slicing
    assert "PR #162: map CUDA runtime args handoff" in slicing
    assert "private request-envelope / host-runtime handoff dependency" in (
        normalized_slicing
    )
    assert "After PR #160, the private envelope and host-runtime hook" in (
        normalized_slicing
    )
    assert "After PR #162, the runtime-args handoff map is complete" in (
        normalized_slicing
    )
    assert "After PR #164, the private host-runtime handoff is accepted" in (
        normalized_slicing
    )
    assert "After PR #166, the private UCCL-EP capability metadata map" in (
        normalized_slicing
    )
    assert "After PR #168, the private validation policy map is accepted" in (
        normalized_slicing
    )
    assert "After PR #170, the private descriptor allocation policy map" in (
        normalized_slicing
    )
    assert "After PR #172, the private UCCL-EP runtime path map" in (
        normalized_slicing
    )
    assert "After PR #174, the private UCCL-EP runtime path scaffold" in (
        normalized_slicing
    )
    assert "After PR #176, the private descriptor allocation scaffold" in (
        normalized_slicing
    )
    assert (
        "Current accepted `main`: "
        "`05457b7dead2f561be22c24c72771add880f4562`"
    ) in normalized_slicing
    assert "nvidia-uccl-ep-runtime-fusion-capability-metadata-map" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-validation-policy-map" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map" in (
        slicing
    )
    assert "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map" in (
        slicing
    )
    assert "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl" in (
        slicing
    )
    assert "nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status" in (
        slicing
    )
    assert "8b5e8075000a2a3e35c4e71c5cb698224b003b44" in slicing
    assert "b58598490d37065e6c972eaaea6d4bc4900469c7" in slicing
    assert "d04732e3a5513d8172b41d0812f2d84065039526" in slicing
    assert "41a9e1e4135313a9787386fb32c21f8b85254d4b" in slicing
    assert "f1b4abb9c9544a71af70decc15bf1424837e0966" in slicing
    assert "142132a2df296ce64e4cd2c17af909d619bcad22" in slicing
    assert "6026ed7cbfa1d4724e22e109bbd75c06d0e9f9a7" in slicing
    assert "0ba8f30696132c06a3cd49b95fbd7bb46b8b9a99" in slicing
    assert "cc26283be5b3355af8148a8e4ca5421d57c2ff80" in slicing
    assert "be914b97898468033c7f834dde0c43466353ac95" in slicing
    assert "bb526ff6c3c21597cffe1acd34bf08158a947cc3" in slicing
    assert "42b996666e279024b43f490a310c490a591a897d" in slicing
    assert "bd0b59ee8d5afc969020d3aea047aafc9f3152be" in slicing
    assert "21b2b32a475dc04e19700115af74510daef70859" in slicing
    assert "3b4b19a04855d27289fb9cdad802fee0c47d8265" in slicing
    assert "Runtime Args Handoff Map Slice" in slicing
    assert "Private Host Runtime Handoff Implementation Slice" in slicing
    assert "Capability Metadata Map Slice" in slicing
    assert "Validation Policy Map Slice" in slicing
    assert "Descriptor Allocation Policy Map Slice" in slicing
    assert "Accepted UCCL-EP Runtime Path Map Slice" in slicing
    assert "Accepted UCCL-EP Runtime Path Implementation Slice" in slicing
    assert "Descriptor Allocation Implementation Slice" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl" in (
        slicing
    )
    assert "PtoCudaPrivateRunArgsEnvelope" in slicing
    assert "same-invocation" in normalized_slicing
    assert "mismatched-callable" in normalized_slicing
    assert "cross-invocation envelopes" in normalized_slicing
    assert "forbidden public/API evidence" in normalized_slicing
    assert "real same-invocation `ChipStorageTaskArgs *` and" in slicing
    assert "Implemented surface in this branch" in slicing
    assert "per-`ChipWorker` private invocation id" in slicing
    assert "UCCL-EP runtime fusion coordinator boundary map" in slicing
    assert "persistent_device_uccl_ep_runtime_fusion_entry" in slicing
    assert "callable id, chip-local rank/device map" in normalized_slicing
    assert "public `TaskArgs`, and public `CallConfig`" in normalized_slicing
    assert "fabricated or untrusted pass evidence stay explicit" in (
        normalized_slicing
    )
    assert "accepted provenance-only input fields" in normalized_slicing
    assert "no shared payload ownership token" in normalized_slicing
    assert "a6378bfbf55b15be01c334f43332ccd20c160cfa" in slicing
    assert "guard UCCL-EP runtime fusion evidence" in slicing
    assert "guard-only blocked implementation handoff" in slicing
    assert "3548a5761c2785bc855d68ec53469651d2227096" in slicing
    assert "Refresh NVIDIA status after runtime fusion guard" in slicing
    assert "nvidia-uccl-ep-runtime-fusion-impl-h200" in dispatch_log
    assert "8c7b3715" in dispatch_log
    assert "Synthetic pass evidence derived from handoff metadata is invalid" in (
        normalized_dispatch_log
    )
    assert "https://github.com/uv-xiao/pto-cu/pull/147" in dispatch_log
    assert "6405dfbd8b403b8d6a0e82813e185c209d4d7e08" in dispatch_log
    assert "https://github.com/uv-xiao/pto-cu/pull/148" in dispatch_log
    assert "2e9b01450efb709ed4e42f80a5128a01e8f9ad21" in dispatch_log
    assert "Refresh NVIDIA status after payload provenance" in dispatch_log
    assert "d7d1679d84ef08202e3a61a821613e031edd49bd" in dispatch_log
    readiness_entry = dispatch_log.split(
        "### 2026-06-22 - UCCL-EP Runtime Fusion Readiness Worker", 1
    )[1].split("\n### ", 1)[0]
    normalized_readiness_entry = " ".join(readiness_entry.split())
    assert "https://github.com/uv-xiao/pto-cu/pull/149" in readiness_entry
    assert "d7d1679d84ef08202e3a61a821613e031edd49bd" in readiness_entry
    assert "accepted as a design/readiness map only by PR #149" in (
        normalized_readiness_entry
    )
    assert "did not accept fused execution evidence" in normalized_readiness_entry
    assert "pending PR creation" not in readiness_entry
    assert "https://github.com/uv-xiao/pto-cu/pull/150" in dispatch_log
    assert "a6378bfbf55b15be01c334f43332ccd20c160cfa" in dispatch_log
    assert "Guard UCCL EP runtime fusion evidence" in dispatch_log
    assert "UCCL-EP Runtime Fusion Guard-Only Worker" in dispatch_log
    assert "UCCL-EP Runtime Fusion Coordinator Boundary Map Worker" in dispatch_log
    assert "UCCL-EP Runtime Fusion Coordinator Entry Contract Worker" in (
        dispatch_log
    )
    assert "Post-Coordinator-Entry-Contract Status Refresh Worker" in (
        dispatch_log
    )
    coordinator_boundary_entry = dispatch_log.split(
        "### 2026-06-22 - UCCL-EP Runtime Fusion Coordinator Boundary Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    assert "pending PR creation" not in coordinator_boundary_entry
    assert "pending dispatcher review and merge decision" not in (
        coordinator_boundary_entry
    )
    assert "accepted as a coordinator-boundary map only" in (
        " ".join(coordinator_boundary_entry.split())
    )
    assert "nvidia-uccl-ep-runtime-fusion-coordinator-boundary-map" in dispatch_log
    assert "nvidia-uccl-ep-runtime-fusion-coordinator-entry-contract" in (
        dispatch_log
    )
    assert "3548a5761c2785bc855d68ec53469651d2227096" in dispatch_log
    entry_contract_entry = dispatch_log.split(
        "### 2026-06-22 - UCCL-EP Runtime Fusion Coordinator Entry Contract Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_entry_contract_entry = " ".join(entry_contract_entry.split())
    assert "8b5e8075000a2a3e35c4e71c5cb698224b003b44" in (
        entry_contract_entry
    )
    assert "persistent_device_uccl_ep_runtime_fusion_entry" in (
        entry_contract_entry
    )
    assert "`ChipWorker::run` / `ChipStorageTaskArgs` request path" in (
        entry_contract_entry
    )
    assert "https://github.com/uv-xiao/pto-cu/pull/153" in (
        entry_contract_entry
    )
    assert "Opened as a non-draft PR" in normalized_entry_contract_entry
    assert "pending dispatcher review and merge decision" not in (
        entry_contract_entry
    )
    assert "accepted as a private entry-contract slice only" in (
        normalized_entry_contract_entry
    )
    assert "b58598490d37065e6c972eaaea6d4bc4900469c7" in (
        entry_contract_entry
    )
    assert "nvidia-uccl-ep-runtime-fusion-private-entry-unsupported" in (
        entry_contract_entry
    )
    assert "does not implement CUDA runtime behavior" in (
        normalized_entry_contract_entry
    )
    assert "PR #151 remains a post-PR150 status refresh" in (
        normalized_dispatch_log
    )
    assert "not actual fused cross-GPU expert-parallel MoE execution" in (
        normalized_slicing
    )
    assert "https://github.com/uv-xiao/pto-cu/pull/145" in dispatch_log

    for required in [
        "PtoCudaRuntimeFusionRequest",
        "PtoCudaRuntimeFusionResult",
        "PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_DESCRIPTOR_ALLOCATOR",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_UCCL_EP_RUNTIME",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_VALIDATION_POLICY",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_FABRICATED_OR_UNTRUSTED_PASS_EVIDENCE",
        "persistent_device_uccl_ep_runtime_fusion_entry",
    ]:
        assert required in private_entry_header

    assert "persistent_device_uccl_ep_runtime_fusion_entry" in cuda_host_runtime
    assert "record_runtime_fusion_unsupported" in cuda_host_runtime
    assert "PtoCudaRuntimeFusionRequest" in cuda_host_runtime
    assert "PtoCudaRuntimeFusionResult" in cuda_host_runtime
    assert "pto_cuda_private_run_envelope_validate" in cuda_host_runtime
    assert "PTO_CUDA_PRIVATE_RUN_ENVELOPE_CALLABLE_TYPE_MISMATCH" in (
        cuda_host_runtime
    )
    assert "envelope.chip_storage_task_args = args;" in chip_worker
    assert "envelope.runtime_task_args = args;" not in chip_worker
    assert "PTO_CUDA_PRIVATE_RUN_ENVELOPE_CROSS_INVOCATION" in (
        private_envelope_header
    )
    assert "PTO_CUDA_PRIVATE_RUN_ENVELOPE_WRONG_CHIP_STORAGE_SIZE" in (
        private_envelope_header
    )

    for required in [
        "test_private_runtime_fusion_entry_reports_missing_runtime_surfaces",
        "test_private_runtime_fusion_entry_rejects_forbidden_pass_evidence",
        "test_private_runtime_fusion_entry_keeps_pass_unreachable_without_evidence",
        "test_cuda_host_runtime_hooks_private_entry_without_public_api_expansion",
        "test_private_runtime_fusion_envelope_validates_same_invocation_dag_only",
        "test_private_runtime_fusion_envelope_rejects_null_wrong_size_and_stale",
    ]:
        assert required in private_entry_test

    private_entry_worker = dispatch_log.split(
        "### 2026-06-22 - UCCL-EP Runtime Fusion Private Entry Unsupported Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_private_entry_worker = " ".join(private_entry_worker.split())
    assert "29da72a171b25deeeb53db399f9cdf54d38c647a" in private_entry_worker
    assert "d04732e3a5513d8172b41d0812f2d84065039526" in private_entry_worker
    assert "https://github.com/uv-xiao/pto-cu/pull/155" in private_entry_worker
    assert "Opened as a non-draft PR" in normalized_private_entry_worker
    assert "unsupported private scaffold" in normalized_private_entry_worker
    assert "branch remains unsupported" in normalized_private_entry_worker
    assert "accepted as a private unsupported runtime scaffold only" in (
        normalized_private_entry_worker
    )
    assert "pending PR creation" not in private_entry_worker
    assert "pending dispatcher review" not in private_entry_worker
    assert "nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request" in (
        private_entry_worker
    )
    assert "`persistent_device_uccl_ep_runtime_fusion.status: passed`" in (
        private_entry_worker
    )
    assert "`actual_fused_cross_gpu_execution: true`" in private_entry_worker

    blocked_handoff_entry = dispatch_log.split(
        "### 2026-06-22 - UCCL-EP Runtime Fusion ChipStorage Blocked Handoff",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_blocked_handoff_entry = " ".join(blocked_handoff_entry.split())
    for required in [
        "nvidia-uccl-ep-runtime-fusion-chip-storage-blocked-handoff",
        "https://github.com/uv-xiao/pto-cu/pull/157",
        "PR #157 is closed invalid, not accepted",
        "PtoCudaRuntimeFusionRequest::chip_storage_task_args",
        "`sizeof(ChipStorageTaskArgs)`",
        "PtoCudaPersistentDagArgs *",
        "not a `ChipStorageTaskArgs *`",
        "Current `main` intentionally does not contain PR #157",
        "no real `ChipStorageTaskArgs` request path",
        "nvidia-uccl-ep-runtime-fusion-private-request-envelope",
        "without expanding public `TaskArgs`, public `CallConfig`,",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_blocked_handoff_entry
    assert "PR #158 already fixed monitor transcript lookup" in (
        normalized_blocked_handoff_entry
    )
    assert "pending PR creation and review" in normalized_blocked_handoff_entry

    private_request_entry = dispatch_log.split(
        "### 2026-06-22 - UCCL-EP Runtime Fusion Private Request Envelope",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_private_request_entry = " ".join(private_request_entry.split())
    for required in [
        "https://github.com/uv-xiao/pto-cu/pull/160",
        "142132a2df296ce64e4cd2c17af909d619bcad22",
        "accepted as a private request-envelope and host-runtime handoff",
        "dependency only",
        "did not accept runtime-fusion success",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_private_request_entry
    assert "pending PR creation and review" not in private_request_entry

    post_private_envelope_entry = dispatch_log.split(
        "### 2026-06-22 - Post-Private-Envelope Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_private_envelope_entry = " ".join(
        post_private_envelope_entry.split()
    )
    assert "nvidia-goal-status-post-private-envelope" in (
        post_private_envelope_entry
    )
    assert "nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map" in (
        post_private_envelope_entry
    )
    assert "selected exactly one next PR-sized slice" in (
        normalized_post_private_envelope_entry
    )
    assert "not runtime-fusion success" in normalized_post_private_envelope_entry
    assert "https://github.com/uv-xiao/pto-cu/pull/161" in (
        post_private_envelope_entry
    )
    assert "completed before PR creation" in normalized_post_private_envelope_entry
    assert "61 passed" in normalized_post_private_envelope_entry
    assert "planned before PR creation" not in post_private_envelope_entry

    runtime_args_entry = dispatch_log.split(
        "### 2026-06-22 - Runtime Args Handoff Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_runtime_args_entry = " ".join(runtime_args_entry.split())
    for required in [
        "6026ed7cbfa1d4724e22e109bbd75c06d0e9f9a7",
        "0ba8f30696132c06a3cd49b95fbd7bb46b8b9a99",
        "nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map",
        "tmp/worker-prompts/run-nvidia-runtime-args-handoff-map.sh",
        "019eefd0-948d-7941-a911-22b627ba15ba",
        "rollout-2026-06-22T22-51-24-019eefd0-948d-7941-a911-22b627ba15ba.jsonl",
        "pto-worker-nvidia-runtime-args-handoff-map:0.0",
        "tmp/codex-goal-monitor/nvidia-runtime-args-handoff-map/",
        "dirty_count: 0",
        "https://github.com/uv-xiao/pto-cu/pull/162",
        "Opened as a non-draft PR with `gh pr create --repo uv-xiao/pto-cu",
        "review-facing docs/tests only",
        "real `ChipStorageTaskArgs *` owned by `ChipWorker::run`",
        "real `PtoCudaPersistentDagArgs *` owned by the CUDA persistent DAG",
        "PtoCudaPrivateRunArgsEnvelope",
        "same-invocation pointers",
        "mismatched callable types",
        "cross-invocation envelopes",
        "accepted as a docs/test dependency map only",
        "Map CUDA runtime args handoff",
        "nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_runtime_args_entry

    post_runtime_args_entry = dispatch_log.split(
        "### 2026-06-22 - Post-Runtime-Args-Handoff-Map Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_runtime_args_entry = " ".join(
        post_runtime_args_entry.split()
    )
    for required in [
        "nvidia-goal-status-post-runtime-args-handoff-map",
        "0ba8f30696132c06a3cd49b95fbd7bb46b8b9a99",
        "tmp/worker-prompts/run-nvidia-post-runtime-args-status-refresh.sh",
        "019eefe5-d11e-73e1-b91f-1da94553b711",
        "rollout-2026-06-22T23-14-36-019eefe5-d11e-73e1-b91f-1da94553b711.jsonl",
        "pto-worker-nvidia-post-runtime-args-status:0.0",
        "tmp/codex-goal-monitor/nvidia-post-runtime-args-status-refresh/",
        "dirty_count: 0",
        "https://github.com/uv-xiao/pto-cu/pull/163",
        "Opened as a non-draft PR with `gh pr create --repo uv-xiao/pto-cu",
        "PR #162 is accepted only as a runtime-args handoff map",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff",
        "selected exactly one next PR-sized implementation slice",
        "null pointers, wrong sizes, mismatched callable types",
        "stale envelopes, cross-invocation envelopes",
        "forbidden public/API evidence paths",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_post_runtime_args_entry
    assert "No nested workers were launched" in normalized_post_runtime_args_entry

    private_host_handoff_entry = dispatch_log.split(
        "### 2026-06-22 - Private Host Runtime Handoff Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_private_host_handoff_entry = " ".join(
        private_host_handoff_entry.split()
    )
    for required in [
        "nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff",
        "cc26283be5b3355af8148a8e4ca5421d57c2ff80",
        "tmp/worker-prompts/run-nvidia-private-host-runtime-handoff.sh",
        "019eeffa-3757-7611-9a9e-e288b1a1258b",
        "rollout-2026-06-22T23-36-53-019eeffa-3757-7611-9a9e-e288b1a1258b.jsonl",
        "pto-worker-nvidia-private-host-runtime-handoff:0.0",
        "tmp/codex-goal-monitor/nvidia-private-host-runtime-handoff/",
        "dirty_count: 0",
        "https://github.com/uv-xiao/pto-cu/pull/164",
        "be914b97898468033c7f834dde0c43466353ac95",
        "Opened as a non-draft PR with `gh pr create --repo uv-xiao/pto-cu",
        "private CUDA persistent DAG host-runtime handoff",
        "accepted as a private host-runtime handoff implementation only",
        "Add CUDA private runtime handoff",
        "4 failed, 5 passed",
        "70 passed",
        "17 passed",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_private_host_handoff_entry
    assert "No nested workers were launched" in normalized_private_host_handoff_entry

    post_private_host_handoff_entry = dispatch_log.split(
        "### 2026-06-23 - Post-Private-Host-Runtime-Handoff Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_private_host_handoff_entry = " ".join(
        post_private_host_handoff_entry.split()
    )
    for required in [
        "nvidia-goal-status-post-private-host-runtime-handoff",
        "run-nvidia-post-private-host-runtime-handoff-status-refresh.sh",
        "nvidia-post-private-host-runtime-handoff-status-refresh.md",
        "019ef01e-0a6f-7e62-a40a-dc7f4fb954f0",
        "rollout-2026-06-23T00-16-01-019ef01e-0a6f-7e62-a40a-dc7f4fb954f0.jsonl",
        "pto-worker-nvidia-post-private-host-runtime-handoff:0.0",
        "tmp/codex-goal-monitor/nvidia-post-private-host-runtime-handoff-status-refresh/",
        "20260622T163142Z",
        "`dirty_count: 0`",
        "fb9c7e65",
        "be914b97898468033c7f834dde0c43466353ac95",
        "PR #164 is accepted only as a private CUDA persistent DAG",
        "https://github.com/uv-xiao/pto-cu/pull/165",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "No runtime code changes",
        "completed before initial PR creation",
        "87 passed",
        "selected exactly one next PR-sized dependency slice",
        "nvidia-uccl-ep-runtime-fusion-capability-metadata-map",
        "private UCCL-EP capability metadata",
        "No nested workers were launched",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_post_private_host_handoff_entry

    capability_metadata_entry = dispatch_log.split(
        "### 2026-06-23 - UCCL-EP Runtime Fusion Capability Metadata Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_capability_metadata_entry = " ".join(
        capability_metadata_entry.split()
    )
    for required in [
        "pto-worker-nvidia-uccl-ep-runtime-fusion-capability-metadata-map",
        "tmp/worker-prompts/run-nvidia-capability-metadata-map.sh",
        "tmp/worker-prompts/nvidia-capability-metadata-map.md",
        "019ef035-0d37-7132-9ace-acc82b2da5b7",
        "rollout-2026-06-23T00-41-09-019ef035-0d37-7132-9ace-acc82b2da5b7.jsonl",
        "pto-worker-nvidia-capability-metadata-map:0.0",
        "tmp/codex-goal-monitor/nvidia-capability-metadata-map/",
        "20260622T165808Z",
        "`dirty_count: 0`",
        "3dfafd61",
        "nvidia-uccl-ep-runtime-fusion-capability-metadata-map",
        "bb526ff6c3c21597cffe1acd34bf08158a947cc3",
        "Planned PR URL slot",
        "https://github.com/uv-xiao/pto-cu/pull/166",
        "Opened as a non-draft PR",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "PR #164 association between real same-invocation",
        "PR #165 selected this dependency slice",
        "capability id, world size, rank-to-device map",
        "descriptor vocabulary, transport mode",
        "adapter provenance handles",
        "setup/validation failure ownership",
        "missing, stale, mismatched-rank, mismatched-world-size",
        "public/API-sourced capability metadata",
        "unsupported or failed",
        "public `TaskArgs`, public `CallConfig`",
        "common runtime C API",
        "UCCL host-runtime ABI",
        "example JSON, adapter provenance, and handoff metadata",
        "No runtime code changes",
        "No fresh H200 command is planned",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_capability_metadata_entry
    assert "No nested workers were launched" in normalized_capability_metadata_entry

    post_capability_metadata_entry = dispatch_log.split(
        "### 2026-06-23 - Post-Capability-Metadata-Map Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_capability_metadata_entry = " ".join(
        post_capability_metadata_entry.split()
    )
    for required in [
        "pto-worker-nvidia-goal-status-post-capability-metadata-map",
        "tmp/worker-prompts/run-nvidia-post-capability-metadata-map-status-refresh.sh",
        "tmp/worker-prompts/nvidia-post-capability-metadata-map-status-refresh.md",
        "019ef048-030a-7333-9374-9c6d2f528ad5",
        "rollout-2026-06-23T01-01-51-019ef048-030a-7333-9374-9c6d2f528ad5.jsonl",
        "pto-worker-nvidia-post-capability-metadata-map:0.0",
        "tmp/codex-goal-monitor/nvidia-post-capability-metadata-map-status-refresh/",
        "20260622T171837Z",
        "`dirty_count: 0`",
        "35aaae64",
        "nvidia-goal-status-post-capability-metadata-map",
        "42b996666e279024b43f490a310c490a591a897d",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "PR #164 is accepted only for the private CUDA persistent DAG host-runtime",
        "PR #165 is accepted only as the post-PR164 docs/test status refresh",
        "PR #166 is accepted only as a private UCCL-EP capability metadata",
        "capability id, world size, rank-to-device map, descriptor vocabulary",
        "transport mode, adapter provenance handles",
        "setup/validation failure ownership",
        "did not implement a runtime-fusion coordinator",
        "descriptor allocator",
        "UCCL-EP runtime path",
        "validation policy",
        "CUDA runtime behavior",
        "pass evidence",
        "H200 fused-success evidence",
        "selected exactly one next PR-sized dependency slice",
        "nvidia-uccl-ep-runtime-fusion-validation-policy-map",
        "private validation policy",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_post_capability_metadata_entry
    assert "No nested workers were launched" in normalized_post_capability_metadata_entry

    validation_policy_entry = dispatch_log.split(
        "### 2026-06-23 - UCCL-EP Runtime Fusion Validation Policy Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_validation_policy_entry = " ".join(validation_policy_entry.split())
    for required in [
        "pto-worker-nvidia-uccl-ep-runtime-fusion-validation-policy-map",
        "tmp/worker-prompts/run-nvidia-validation-policy-map.sh",
        "tmp/worker-prompts/nvidia-validation-policy-map.md",
        "019ef05b-0a43-7d33-9cc4-4cbb058a1f9f",
        "rollout-2026-06-23T01-22-38-019ef05b-0a43-7d33-9cc4-4cbb058a1f9f.jsonl",
        "pto-worker-nvidia-validation-policy-map:0.0",
        "tmp/codex-goal-monitor/nvidia-validation-policy-map/",
        "20260622T173909Z",
        "`dirty_count: 0`",
        "bad748c6",
        "nvidia-uccl-ep-runtime-fusion-validation-policy-map",
        "20b3e625ea8c9d6e4f06bb3992779b807f65acf9",
        "https://github.com/uv-xiao/pto-cu/pull/168",
        "Opened as a non-draft PR",
        "review-facing docs/tests only",
        "private validation policy",
        "PR #164 same-invocation request args",
        "PR #166 UCCL-EP capability metadata",
        "missing metadata",
        "stale metadata",
        "mismatched-rank",
        "mismatched-world-size",
        "descriptor-vocabulary mismatch",
        "transport-mode mismatch",
        "adapter-provenance mismatch",
        "public/API-sourced metadata",
        "No CUDA runtime behavior change",
        "No fresh H200 command is planned",
        "No nested workers were launched",
        "accepted as a private validation policy dependency map only",
        "e33d232deccdf947b9c382a3605191d0d5ae0004",
        "Map UCCL EP validation policy",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_validation_policy_entry

    post_validation_policy_entry = dispatch_log.split(
        "### 2026-06-23 - Post-Validation-Policy-Map Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_validation_policy_entry = " ".join(
        post_validation_policy_entry.split()
    )
    for required in [
        "pto-worker-nvidia-goal-status-post-validation-policy-map",
        "tmp/worker-prompts/run-nvidia-post-validation-policy-map-status-refresh.sh",
        "tmp/worker-prompts/nvidia-post-validation-policy-map-status-refresh.md",
        "019ef06d-a68c-7be0-ad09-b5a288e4a867",
        "rollout-2026-06-23T01-42-58-019ef06d-a68c-7be0-ad09-b5a288e4a867.jsonl",
        "pto-worker-nvidia-post-validation-policy-map:0.0",
        "tmp/codex-goal-monitor/nvidia-post-validation-policy-map-status-refresh/",
        "20260622T175938Z",
        "`dirty_count: 0`",
        "b076e344",
        "nvidia-goal-status-post-validation-policy-map",
        "e33d232deccdf947b9c382a3605191d0d5ae0004",
        "https://github.com/uv-xiao/pto-cu/pull/169",
        "Opened as a non-draft PR",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "PR #164 is accepted only for the private CUDA persistent DAG",
        "PR #166 is accepted only as a private UCCL-EP capability metadata",
        "PR #167 is accepted only as the post-PR166 docs/test status refresh",
        "PR #168 is accepted only as a private validation policy",
        "did not implement CUDA runtime behavior",
        "descriptor allocation policy",
        "UCCL-EP runtime dispatch",
        "a coordinator",
        "pass evidence",
        "H200 fused-success evidence",
        "selected exactly one next PR-sized dependency slice",
        "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map",
        "No nested workers were launched",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_post_validation_policy_entry

    descriptor_policy_entry = dispatch_log.split(
        "### 2026-06-23 - UCCL-EP Runtime Fusion Descriptor "
        "Allocation Policy Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_descriptor_policy_entry = " ".join(
        descriptor_policy_entry.split()
    )
    for required in [
        "pto-worker-nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map",
        "tmp/worker-prompts/run-nvidia-descriptor-allocation-policy-map.sh",
        "tmp/worker-prompts/nvidia-descriptor-allocation-policy-map.md",
        "019ef080-a2ce-7b12-b4ea-b262014674f1",
        "rollout-2026-06-23T02-03-42-019ef080-a2ce-7b12-b4ea-b262014674f1.jsonl",
        "pto-worker-nvidia-descriptor-allocation-policy-map:0.0",
        "tmp/codex-goal-monitor/nvidia-descriptor-allocation-policy-map/",
        "20260622T182026Z",
        "dirty_count: 0",
        "af0cafdf",
        "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map",
        "c0bff19b3f5571da34ea030d81c9de184a9ec230",
        "https://github.com/uv-xiao/pto-cu/pull/170",
        "Opened as a non-draft PR",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "PR #164 is accepted only for the private CUDA persistent DAG",
        "PR #166 is accepted only as a private UCCL-EP capability metadata",
        "PR #168 is accepted only as a private validation policy",
        "prerequisites, not pass evidence",
        "private descriptor allocation policy",
        "allocator owner is the future private",
        "host-control record policy",
        "device-visible descriptor buffer policy",
        "dispatch descriptor identity",
        "combine descriptor identity",
        "shared-token requirement",
        "allocation lifetime failure ownership",
        "missing policy is unsupported",
        "stale policy is failed",
        "non-runtime-owned allocation is failed",
        "descriptor-vocabulary mismatch is failed",
        "token-sharing mismatch is failed",
        "rank/device mismatch is failed",
        "public/API-sourced policy fields are failed",
        "No descriptor allocation implementation",
        "No CUDA runtime behavior change",
        "No fresh H200 command is planned",
        "No public `TaskArgs`, public `CallConfig`, common runtime C API",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_descriptor_policy_entry
    assert "No nested workers were launched" in normalized_descriptor_policy_entry

    post_descriptor_policy_entry = dispatch_log.split(
        "### 2026-06-23 - Post-Descriptor-Allocation-Policy-Map "
        "Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_descriptor_policy_entry = " ".join(
        post_descriptor_policy_entry.split()
    )
    for required in [
        "pto-worker-nvidia-goal-status-post-descriptor-allocation-policy-map",
        "run-nvidia-post-descriptor-allocation-policy-map-status-refresh.sh",
        "nvidia-post-descriptor-allocation-policy-map-status-refresh.md",
        "019ef097-163d-7d73-8086-aa1b83fe9dc2",
        "rollout-2026-06-23T02-28-14-019ef097-163d-7d73-8086-aa1b83fe9dc2.jsonl",
        "pto-worker-nvidia-post-descriptor-allocation-policy-map:0.0",
        "tmp/codex-goal-monitor/nvidia-post-descriptor-allocation-policy-map/",
        "20260622T184343Z",
        "pane_status: missing",
        "dirty_count: 0",
        "87c8b976",
        "nvidia-goal-status-post-descriptor-allocation-policy-map",
        "bd0b59ee8d5afc969020d3aea047aafc9f3152be",
        "https://github.com/uv-xiao/pto-cu/pull/171",
        "Opened as a non-draft PR",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "PR #164 is accepted only for the private CUDA persistent DAG",
        "PR #166 is accepted only as a private UCCL-EP capability metadata",
        "PR #168 is accepted only as a private validation policy",
        "PR #170 accepted only the private descriptor allocation policy",
        "allocator owner",
        "host-control record policy",
        "device-visible descriptor buffer policy",
        "dispatch/combine descriptor identity",
        "shared-token requirement",
        "allocation lifetime failure ownership",
        "did not implement CUDA runtime behavior",
        "descriptor allocation",
        "UCCL-EP runtime dispatch",
        "a coordinator",
        "pass evidence",
        "H200 fused-success evidence",
        "selected exactly one next PR-sized dependency slice",
        "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map",
        "private UCCL-EP runtime path",
        "runtime-path owner",
        "dispatch descriptor handoff",
        "combine descriptor handoff",
        "descriptor-token checks",
        "transport-mode checks",
        "No nested workers were launched",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_post_descriptor_policy_entry

    for text in (
        persistent_moe,
        boundary,
        selection,
        slicing,
        dispatch_log,
    ):
        normalized_text = " ".join(text.split())
        for required in [
            "Capability Metadata Map Slice",
            "private UCCL-EP capability metadata",
            "capability id",
            "world size",
            "rank-to-device map",
            "descriptor vocabulary",
            "transport mode",
            "adapter provenance handles",
            "setup/validation failure ownership",
            "missing, stale, mismatched-rank, mismatched-world-size",
            "public/API-sourced capability metadata",
            "unsupported or failed",
            "PR #164 association",
            "real same-invocation `ChipStorageTaskArgs *` and",
            "PtoCudaPersistentDagArgs *",
            "forbidden pass-evidence paths",
            "public `TaskArgs`, public `CallConfig`",
            "common runtime C API",
            "UCCL host-runtime ABI",
            "example JSON, adapter provenance, and handoff metadata",
            "no CUDA runtime behavior change",
            "no runtime-fusion coordinator implementation",
            "no descriptor allocator implementation",
            "no UCCL-EP runtime path implementation",
            "no validation policy implementation",
            "no fresh H200 fused-success evidence",
            "42b996666e279024b43f490a310c490a591a897d",
            "Validation Policy Map Slice",
            "nvidia-uccl-ep-runtime-fusion-validation-policy-map",
            "private validation policy",
            "validates PR #164 same-invocation request args and PR #166",
            "capability metadata together",
            "missing metadata is unsupported",
            "stale metadata is failed",
            "mismatched-rank metadata is failed",
            "mismatched-world-size metadata is failed",
            "descriptor-vocabulary mismatch",
            "descriptor vocabulary must match dispatch/combine payload terms",
            "transport-mode mismatch",
            "transport mode must be `ep`",
            "adapter-provenance mismatch",
            "adapter provenance handles must match",
            "public/API-sourced metadata is failed",
            "validation policy remains private to the CUDA persistent-device runtime path",
            "no descriptor allocation policy implementation",
            "e33d232deccdf947b9c382a3605191d0d5ae0004",
            "Descriptor Allocation Policy Map Slice",
            "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map",
            "private descriptor allocation policy",
            "host-control record",
            "host-control record policy",
            "device-visible descriptor buffer",
            "device-visible descriptor buffer policy",
            "dispatch descriptor identity",
            "combine descriptor identity",
            "shared-token requirement",
            "allocator owner is the future private",
            "allocation lifetime failure ownership",
            "missing policy is unsupported",
            "stale policy is failed",
            "non-runtime-owned allocation is failed",
            "descriptor-vocabulary mismatch is failed",
            "token-sharing mismatch is failed",
            "rank/device mismatch is failed",
            "public/API-sourced policy fields are failed",
            "coordinator-issued shared token",
            "same shared token as dispatch",
            "PR #170 accepted only the private descriptor allocation policy",
            "bd0b59ee8d5afc969020d3aea047aafc9f3152be",
            "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map",
            "UCCL-EP Runtime Path Map Slice",
            "private UCCL-EP runtime path",
            "runtime-path owner",
            "dispatch descriptor handoff",
            "combine descriptor handoff",
            "descriptor-token checks",
            "transport-mode checks",
            "missing runtime path is unsupported",
            "stale descriptor",
            "public/API-sourced runtime-path fields",
            "PR #172 accepted only",
            "21b2b32a475dc04e19700115af74510daef70859",
            "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl",
            "UCCL-EP Runtime Path Implementation Slice",
        ]:
            assert required in normalized_text


def test_uccl_ep_runtime_path_map_slice_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }

    required_path_map_terms = [
        "uccl-ep runtime path map slice",
        "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map",
        "private uccl-ep runtime path",
        "cuda persistent-device runtime path",
        "runtime-path owner",
        "dispatch descriptor handoff",
        "combine descriptor handoff",
        "descriptor-token checks",
        "rank/device checks",
        "transport-mode checks",
        "runtime-path failure ownership",
        "missing runtime path is unsupported",
        "stale descriptor views are failed",
        "descriptor-token mismatch is failed",
        "rank/device mismatch is failed",
        "transport-mode mismatch is failed",
        "descriptor-vocabulary mismatch is failed",
        "public/api-sourced runtime-path fields are failed",
        "pr #164",
        "same-invocation request args",
        "pr #166",
        "uccl-ep capability metadata",
        "pr #168",
        "validation policy",
        "pr #170",
        "descriptor allocation policy",
        "prerequisites",
        "pass evidence",
        "coordinator-issued shared token",
        "token",
        "transport mode: ep",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_path_map_terms:
            assert required in normalized, f"{name} missing {required!r}"

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-23 - UCCL-EP Runtime Fusion Runtime Path Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "pto-worker-nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map",
        "tmp/worker-prompts/run-nvidia-uccl-ep-runtime-path-map.sh",
        "tmp/worker-prompts/nvidia-uccl-ep-runtime-path-map.md",
        "019ef0a8-ee9a-7550-ac78-f244dd6f84cb",
        "rollout-2026-06-23T02-47-43-019ef0a8-ee9a-7550-ac78-f244dd6f84cb.jsonl",
        "pto-worker-nvidia-uccl-ep-runtime-path-map:0.0",
        "tmp/codex-goal-monitor/nvidia-uccl-ep-runtime-path-map/",
        "20260622T190307Z",
        "pane_status: missing",
        "dirty_count: 0",
        "cd800f20",
        "planned PR URL slot",
        "https://github.com/uv-xiao/pto-cu/pull/172",
        "Opened as a non-draft PR",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "75cf6045b4042ef592bb6962a592f0f658fc4d29",
        "review-facing docs/tests only",
        "No nested workers were launched",
        "No pass evidence",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
        "No RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency",
        "No fresh H200 command is planned",
        "completed before initial PR creation",
        "0 error(s)",
        "NVIDIA review guard passed",
        "62 passed",
    ]:
        assert required in normalized_dispatch

    post_runtime_path_entry = texts["dispatch"].split(
        "### 2026-06-23 - Post-UCCL-EP-Runtime-Path-Map "
        "Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_post_runtime_path_entry = " ".join(
        post_runtime_path_entry.split()
    )
    for required in [
        "pto-worker-nvidia-goal-status-post-uccl-ep-runtime-path-map",
        "run-nvidia-post-uccl-ep-runtime-path-map-status-refresh.sh",
        "nvidia-post-uccl-ep-runtime-path-map-status-refresh.md",
        "019ef0bb-3bdf-72f2-a655-aea215e1bbc6",
        "rollout-2026-06-23T03-07-42-019ef0bb-3bdf-72f2-a655-aea215e1bbc6.jsonl",
        "pto-worker-nvidia-post-uccl-ep-runtime-path-map:0.0",
        "tmp/codex-goal-monitor/nvidia-post-uccl-ep-runtime-path-map/",
        "20260622T192303Z",
        "pane_status: missing",
        "dirty_count: 0",
        "c412c78e",
        "nvidia-goal-status-post-uccl-ep-runtime-path-map",
        "21b2b32a475dc04e19700115af74510daef70859",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "review-facing docs/tests only",
        "PR #164 is accepted only for the private CUDA persistent DAG",
        "PR #166 is accepted only as a private UCCL-EP capability metadata",
        "PR #168 is accepted only as a private validation policy",
        "PR #170 is accepted only as a private descriptor allocation policy",
        "PR #172 accepted only a private UCCL-EP runtime path",
        "runtime-path owner",
        "dispatch descriptor handoff",
        "combine descriptor handoff",
        "descriptor-token checks",
        "rank/device checks",
        "transport-mode checks",
        "runtime-path failure ownership",
        "did not implement CUDA runtime behavior",
        "UCCL-EP runtime dispatch",
        "a coordinator",
        "descriptor allocation",
        "pass evidence",
        "H200 fused-success evidence",
        "selected exactly one next PR-sized implementation slice",
        "UCCL-EP Runtime Path Implementation Slice",
        "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl",
        "missing descriptor allocation and missing coordinator behavior",
        "No nested workers were launched",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
        "No fresh H200 command is planned",
    ]:
        assert required in normalized_post_runtime_path_entry


def test_uccl_ep_runtime_path_impl_slice_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }
    runtime_fusion_abi = (
        ROOT / "src" / "cuda" / "platform" / "include" / "host"
        / "pto_cuda_runtime_fusion_abi.h"
    ).read_text(encoding="utf-8")
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host"
        / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    common_abi = (
        ROOT / "src" / "common" / "worker" / "pto_runtime_c_api.h"
    ).read_text(encoding="utf-8")

    required_impl_terms = [
        "uccl-ep runtime path implementation slice",
        "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl",
        "private runtime-path",
        "cuda persistent-device runtime",
        "same-invocation id",
        "descriptor-token mismatch",
        "rank/device mismatch",
        "transport-mode mismatch",
        "descriptor-vocabulary mismatch",
        "stale descriptor views",
        "public/api-sourced runtime-path fields",
        "missing descriptor allocation",
        "missing coordinator",
        "pr #164",
        "pr #166",
        "pr #168",
        "pr #170",
        "pr #172",
        "prerequisites",
        "pass evidence",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_impl_terms:
            assert required in normalized, f"{name} missing {required!r}"

    for name in ["boundary", "slicing", "dispatch"]:
        normalized = " ".join(texts[name].split()).lower()
        for required in [
            "ptocudaucclepruntimepath",
            "ptocudaucclepruntimedescriptorview",
        ]:
            assert required in normalized, f"{name} missing {required!r}"

    for required in [
        "PtoCudaUcclEpRuntimePath",
        "PtoCudaUcclEpRuntimeDescriptorView",
        "PtoCudaRuntimeFusionRequest",
        "invocation_id",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_STALE_DESCRIPTOR_VIEW",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_TOKEN_MISMATCH",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_RANK_DEVICE_MISMATCH",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_TRANSPORT_MODE_MISMATCH",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_DESCRIPTOR_VOCABULARY_MISMATCH",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_PUBLIC_API_RUNTIME_PATH",
        "pto_cuda_runtime_fusion_validate_uccl_ep_runtime_path",
        "PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_COORDINATOR_OWNED",
        "PTO_CUDA_UCCL_EP_RUNTIME_PATH_SOURCE_PUBLIC_API",
    ]:
        assert required in runtime_fusion_abi

    assert "request.invocation_id = expected_invocation_id" in host_runtime
    assert "persistent_device_uccl_ep_runtime_fusion_entry" in host_runtime
    assert "PtoCudaUcclEpRuntimePath" not in common_abi
    assert "PtoCudaUcclEpRuntimeDescriptorView" not in common_abi
    assert "invocation_id" not in common_abi

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-23 - UCCL-EP Runtime Fusion Runtime Path "
        "Implementation Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "pto-worker-nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl",
        "tmp/worker-prompts/run-nvidia-uccl-ep-runtime-path-impl.sh",
        "tmp/worker-prompts/nvidia-uccl-ep-runtime-path-impl.md",
        "019ef0cd-b772-72b0-b8c2-838341262729",
        "rollout-2026-06-23T03-27-54-019ef0cd-b772-72b0-b8c2-838341262729.jsonl",
        "pto-worker-nvidia-uccl-ep-runtime-path-impl:0.0",
        "tmp/codex-goal-monitor/nvidia-uccl-ep-runtime-path-impl/",
        "20260622T195828Z",
        "pane_status: missing",
        "dirty_count: 0",
        "a39dba21",
        "nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl",
        "a37913b1cf5e3e501863253a789833289e918e15",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "No nested workers were launched",
        "PtoCudaUcclEpRuntimePath",
        "PtoCudaUcclEpRuntimeDescriptorView",
        "same-invocation id",
        "descriptor token",
        "transport mode `ep`",
        "public/API-sourced runtime-path",
        "missing descriptor allocator and missing coordinator",
        "No pass evidence",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
        "No fresh H200 command is planned",
        "PR #174 merged as `3b4b19a04855d27289fb9cdad802fee0c47d8265`",
    ]:
        assert required in normalized_dispatch


def test_post_uccl_ep_runtime_path_impl_status_refresh_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }

    required_refresh_terms = [
        "pr #174",
        "3b4b19a04855d27289fb9cdad802fee0c47d8265",
        "ptocudaucclepruntimepath",
        "ptocudaucclepruntimedescriptorview",
        "private descriptor-view validation",
        "invocation-id propagation",
        "descriptor allocation implementation",
        "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl",
        "narrower than",
        "coordinator",
        "uccl-ep runtime dispatch",
        "pass evidence",
        "fresh h200 fused-success evidence",
        "serving",
        "vllm",
        "deepseek",
        "throughput",
        "latency",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_refresh_terms:
            assert required in normalized, f"{name} missing {required!r}"

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-23 - Post-UCCL-EP-Runtime-Path-Impl "
        "Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "pto-worker-nvidia-goal-status-post-uccl-ep-runtime-path-impl",
        "run-nvidia-post-uccl-ep-runtime-path-impl-status-refresh.sh",
        "nvidia-post-uccl-ep-runtime-path-impl-status-refresh.md",
        "019ef0f2-c2d3-7ae2-91dd-ba81b660fd1b",
        "rollout-2026-06-23T04-08-21-019ef0f2-c2d3-7ae2-91dd-ba81b660fd1b.jsonl",
        "pto-worker-nvidia-post-uccl-ep-runtime-path-impl-status-refresh:0.0",
        "tmp/codex-goal-monitor/nvidia-post-uccl-ep-runtime-path-impl-status-refresh/",
        "20260622T201758Z",
        "pane_status: missing",
        "dirty_count: 0",
        "2c2256e8",
        "nvidia-goal-status-post-uccl-ep-runtime-path-impl",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "uv-xiao/pto-cu",
        "base branch `main`",
        "starting commit `3b4b19a04855d27289fb9cdad802fee0c47d8265`",
        "review-facing docs/tests only",
        "docs/in_progress/nvidia_backend/dispatch_log.md",
        "docs/in_progress/nvidia_backend/pr_slicing_plan.md",
        "docs/in_progress/nvidia_backend/communication_runtime_boundary.md",
        "docs/in_progress/nvidia_backend/communication_selection.md",
        "docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md",
        "tests/ut/py/test_nvidia_review_artifacts.py",
        "No nested workers were launched",
        "PR #174 accepted only the private UCCL-EP runtime path scaffold",
        "PtoCudaUcclEpRuntimePath",
        "PtoCudaUcclEpRuntimeDescriptorView",
        "private descriptor-view validation",
        "invocation-id propagation",
        "did not implement the runtime-fusion coordinator",
        "descriptor allocation",
        "UCCL-EP runtime dispatch",
        "fresh H200 fused-success evidence",
        "public `TaskArgs`",
        "public `CallConfig`",
        "common runtime C API",
        "UCCL host-runtime ABI",
        "serving, vLLM, DeepSeek, throughput, or latency",
        "selected exactly one next PR-sized implementation slice",
        "Descriptor Allocation Implementation Slice",
        "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl",
        "narrower than runtime-fusion coordinator construction",
        "UCCL-EP runtime dispatch",
        "git diff --check",
        "markdownlint-cli2",
        "NVIDIA review guard",
        "tests/ut/py/test_nvidia_review_artifacts.py",
        "No fresh H200 command is planned",
    ]:
        assert required in normalized_dispatch


def test_uccl_ep_descriptor_allocation_impl_slice_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }
    runtime_fusion_abi = (
        ROOT / "src" / "cuda" / "platform" / "include" / "host"
        / "pto_cuda_runtime_fusion_abi.h"
    ).read_text(encoding="utf-8")
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host"
        / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    common_abi = (
        ROOT / "src" / "common" / "worker" / "pto_runtime_c_api.h"
    ).read_text(encoding="utf-8")
    private_entry_test = (
        ROOT / "tests" / "ut" / "py" / "test_cuda_runtime_fusion_private_entry.py"
    ).read_text(encoding="utf-8")

    required_impl_terms = [
        "descriptor allocation implementation slice",
        "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl",
        "private descriptor allocation",
        "host-control record",
        "device-visible dispatch/combine descriptor buffer",
        "same invocation id",
        "pr #170",
        "pr #174",
        "runtime-path scaffold",
        "missing coordinator",
        "missing uccl-ep runtime dispatch",
        "unsupported or failed states",
        "public `taskargs`",
        "public `callconfig`",
        "common runtime c api",
        "uccl host-runtime abi",
        "pass evidence",
        "fresh h200 fused-success evidence",
        "actual_fused_cross_gpu_execution: true",
        "throughput",
        "latency",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_impl_terms:
            assert required in normalized, f"{name} missing {required!r}"

    for name in ["persistent_moe", "boundary", "selection", "slicing"]:
        normalized = " ".join(texts[name].split()).lower()
        for required in [
            "ptocudaucclepdescriptorhostcontrol",
            "ptocudaucclepdevicedescriptorbuffer",
            "ptocudaucclepdescriptorallocation",
            "pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors",
        ]:
            assert required in normalized, f"{name} missing {required!r}"

    for required in [
        "PTO_CUDA_UCCL_EP_DESCRIPTOR_ALLOCATION_VERSION",
        "PtoCudaUcclEpDescriptorHostControl",
        "PtoCudaUcclEpDeviceDescriptorBuffer",
        "PtoCudaUcclEpDescriptorAllocation",
        "pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors",
        "PtoCudaUcclEpRuntimePath",
        "PtoCudaRuntimeFusionRequest",
        "invocation_id",
        "shared_token",
        "PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_DISPATCH",
        "PTO_CUDA_RUNTIME_FUSION_DESCRIPTOR_VOCABULARY_COMBINE",
    ]:
        assert required in runtime_fusion_abi

    for required in [
        "runtime_fusion_device_descriptor_buffer_",
        "runtime_fusion_coordinator_",
        "pto_cuda_runtime_fusion_prepare_private_coordinator",
        "request.coordinator = &runtime_fusion_coordinator_",
        "request.descriptor_allocator = &runtime_fusion_coordinator_.descriptor_allocation",
        "request.uccl_ep_runtime = &runtime_fusion_coordinator_.descriptor_allocation.runtime_path",
        "request.invocation_id = expected_invocation_id",
    ]:
        assert required in host_runtime

    for forbidden in [
        "PtoCudaUcclEpDescriptorAllocation",
        "PtoCudaUcclEpDeviceDescriptorBuffer",
        "PTO_CUDA_UCCL_EP_DESCRIPTOR_ALLOCATION_VERSION",
    ]:
        assert forbidden not in common_abi

    assert (
        "test_private_descriptor_allocation_builds_runtime_path_but_stays_unsupported"
        in private_entry_test
    )

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-23 - UCCL-EP Runtime Fusion Descriptor "
        "Allocation Implementation Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "pto-worker-nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl",
        "tmp/worker-prompts/run-nvidia-uccl-ep-descriptor-allocation-impl.sh",
        "tmp/worker-prompts/nvidia-uccl-ep-descriptor-allocation-impl.md",
        "No nested workers were launched",
        "dispatcher review for PR #176",
        "019ef0ff-7281-7f00-a994-258c4b20cbc4",
        "pto-worker-nvidia-uccl-ep-descriptor-allocation-impl:0.0",
        "tmp/codex-goal-monitor/nvidia-uccl-ep-descriptor-allocation-impl/",
        "pane_status: missing",
        "dirty_count: 0",
        "https://github.com/uv-xiao/pto-cu/pull/176",
        "uv-xiao/pto-cu",
        "base branch `main`",
        "starting commit `f03ec5b5f77b786b69e53408796d04def05ced5f`",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl",
        "PtoCudaUcclEpDeviceDescriptorBuffer",
        "PtoCudaUcclEpDescriptorAllocation",
        "PTO_CUDA_UCCL_EP_DESCRIPTOR_ALLOCATION_VERSION",
        "pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors",
        "1 failed, 12 passed",
        "private host-control record",
        "device-visible dispatch/combine descriptor buffer",
        "same invocation id",
        "missing UCCL-EP runtime dispatch remain unsupported or failed states",
        "No fresh H200 command is planned",
        "PR #176 accepted and merged as `6e0cecc174ae9db47573c4c0f1698be7accb295c`",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_dispatch
    assert "dispatcher to fill after monitor setup" not in normalized_dispatch
    assert "pending dispatcher review" not in normalized_dispatch


def test_post_descriptor_allocation_impl_status_refresh_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }

    required_refresh_terms = [
        "pr #176",
        "6e0cecc174ae9db47573c4c0f1698be7accb295c",
        "private descriptor allocation scaffold",
        "ptocudaucclepdescriptorhostcontrol",
        "ptocudaucclepdevicedescriptorbuffer",
        "ptocudaucclepdescriptorallocation",
        "private host-control record",
        "device-visible dispatch/combine descriptor buffer",
        "coordinator construction",
        "uccl-ep runtime dispatch",
        "pass evidence",
        "fresh h200 fused-success evidence",
        "public `taskargs`",
        "public `callconfig`",
        "common runtime c api",
        "uccl host-runtime abi",
        "serving",
        "vllm",
        "deepseek",
        "throughput",
        "latency",
        "nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_refresh_terms:
            assert required in normalized, f"{name} missing {required!r}"

    slicing = texts["slicing"]
    assert (
        "## Descriptor Allocation Implementation Slice\n\nSelected branch:"
        not in slicing
    )
    assert (
        "`3b4b19a04855d27289fb9cdad802fee0c47d8265`, and the next "
        "slice is exactly one narrow implementation slice"
        not in " ".join(slicing.split())
    )
    assert "Accepted Descriptor Allocation Implementation Slice" in slicing
    assert "Runtime Fusion Coordinator Scaffold Status Slice" in slicing

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-23 - Post-Descriptor-Allocation-Impl "
        "Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "pto-worker-nvidia-post-descriptor-allocation-impl-status-refresh",
        "tmp/worker-prompts/nvidia-post-descriptor-allocation-impl-status-refresh.md",
        "tmp/worker-prompts/run-nvidia-post-descriptor-allocation-impl-status-refresh.sh",
        "019ef118-0c3f-7da2-84fb-a55be183e287",
        "rollout-2026-06-23T04-49-05-019ef118-0c3f-7da2-84fb-a55be183e287.jsonl",
        "pto-worker-nvidia-post-descriptor-allocation-impl-status-refresh:0.0",
        "tmp/codex-goal-monitor/nvidia-post-descriptor-allocation-impl-status-refresh/",
        "pane_status: missing",
        "dirty_count: 0",
        "nvidia-goal-status-post-descriptor-allocation-impl",
        "https://github.com/uv-xiao/pto-cu/pull/177",
        "uv-xiao/pto-cu",
        "base branch `main`",
        "starting commit `6e0cecc174ae9db47573c4c0f1698be7accb295c`",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "docs/in_progress/nvidia_backend/dispatch_log.md",
        "docs/in_progress/nvidia_backend/pr_slicing_plan.md",
        "docs/in_progress/nvidia_backend/communication_runtime_boundary.md",
        "docs/in_progress/nvidia_backend/communication_selection.md",
        "docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md",
        "tests/ut/py/test_nvidia_review_artifacts.py",
        "No nested workers were launched",
        "PR #176 accepted only the private descriptor allocation scaffold",
        "selected exactly one next PR-sized coordinator-construction scaffold/status slice",
        "nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status",
        "narrower than UCCL-EP runtime dispatch",
        "narrower than pass evidence",
        "cannot claim fused success until UCCL-EP runtime dispatch and fresh H200",
        "No fresh H200 command is planned",
        "no `persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "no `actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_dispatch
    assert "pending dispatcher review" not in normalized_dispatch
    assert "parent dispatcher to fill after PR creation" not in normalized_dispatch
    assert "persistent_device_uccl_ep_runtime_fusion.status: passed`" in (
        normalized_dispatch
    )
    assert "no `persistent_device_uccl_ep_runtime_fusion.status: passed`" in (
        normalized_dispatch
    )


def test_coordinator_scaffold_status_slice_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }
    runtime_fusion_abi = (
        ROOT / "src" / "cuda" / "platform" / "include" / "host"
        / "pto_cuda_runtime_fusion_abi.h"
    ).read_text(encoding="utf-8")
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host"
        / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    common_abi = (
        ROOT / "src" / "common" / "worker" / "pto_runtime_c_api.h"
    ).read_text(encoding="utf-8")
    private_entry_test = (
        ROOT / "tests" / "ut" / "py" / "test_cuda_runtime_fusion_private_entry.py"
    ).read_text(encoding="utf-8")

    required_terms = [
        "nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status",
        "pr #178",
        "aea89cc9dea8560602c72f84e5ff6e78ca526434",
        "private coordinator",
        "descriptor allocation",
        "runtime path",
        "same invocation id",
        "unsupported/failure status",
        "output sink",
        "missing_coordinator",
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status",
        "runtime-dispatch scaffold/status gate",
        "uccl-ep runtime dispatch",
        "pass evidence",
        "fresh h200 fused-success evidence",
        "persistent_device_uccl_ep_runtime_fusion.status: passed",
        "actual_fused_cross_gpu_execution: true",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_terms:
            assert required in normalized, f"{name} missing {required!r}"

    for required in [
        "PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION",
        "PtoCudaRuntimeFusionCoordinator",
        "pto_cuda_runtime_fusion_prepare_private_coordinator",
        "pto_cuda_runtime_fusion_request_has_private_coordinator_shape",
        "pto_cuda_runtime_fusion_validate_private_coordinator",
        "PtoCudaUcclEpDescriptorAllocation descriptor_allocation",
        "PtoCudaRuntimeFusionResult *output_sink",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR",
    ]:
        assert required in runtime_fusion_abi

    for required in [
        "runtime_fusion_coordinator_",
        "pto_cuda_runtime_fusion_prepare_private_coordinator",
        "request.coordinator = &runtime_fusion_coordinator_",
        "request.descriptor_allocator = &runtime_fusion_coordinator_.descriptor_allocation",
        "request.output_sink = runtime_fusion_coordinator_.output_sink",
    ]:
        assert required in host_runtime

    for forbidden in [
        "PtoCudaRuntimeFusionCoordinator",
        "PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION",
        "pto_cuda_runtime_fusion_prepare_private_coordinator",
    ]:
        assert forbidden not in common_abi

    assert (
        "test_private_coordinator_scaffold_owns_runtime_path_for_one_invocation"
        in private_entry_test
    )
    assert "output_sink.status == result.status" in private_entry_test
    assert (
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_COORDINATOR) == 0U"
        in private_entry_test
    )
    assert "PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED" in private_entry_test

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-25 - UCCL-EP Runtime Fusion Coordinator "
        "Scaffold Status Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "multi-agent-worker-coordinator-scaffold-status",
        "019ef12d-1c34-7523-86bc-756db68f3b68",
        "019efea2-8e09-70c1-bce5-9f02250f27f3",
        "abandoned launcher attempt",
        "active worker",
        "No nested workers were launched",
        "https://github.com/uv-xiao/pto-cu/pull/178",
        "uv-xiao/pto-cu",
        "base branch `main`",
        "starting commit `0ee279a21a7341e7113ac353849b543899d6742a`",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status",
        "PtoCudaRuntimeFusionCoordinator",
        "pto_cuda_runtime_fusion_prepare_private_coordinator",
        "runtime_fusion_coordinator_",
        "clears `missing_coordinator` only when",
        "1 failed in 0.32s",
        "PtoCudaRuntimeFusionCoordinator` was not declared",
        "No CUDA/H200 command was run or planned",
        "no real UCCL-EP runtime dispatch",
        "no pass evidence",
        "no fresh H200 fused success",
        "no `persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "no `actual_fused_cross_gpu_execution: true`",
        "PR #178 merged as `aea89cc9dea8560602c72f84e5ff6e78ca526434`",
        "accepts only the private UCCL-EP runtime-fusion coordinator",
        "It does not accept runtime dispatch, pass evidence, or H200",
    ]:
        assert required in normalized_dispatch
    assert "pending dispatcher review" not in normalized_dispatch

    post_refresh_entry = texts["dispatch"].split(
        "### 2026-06-25 - Post-Coordinator-Scaffold Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_refresh = " ".join(post_refresh_entry.split())
    for required in [
        "nvidia-goal-status-post-coordinator-scaffold-status",
        "019efeb7-1957-74d1-926c-462419958916",
        "No tmux pane is used by this worker",
        "No nested workers were launched",
        "post-PR #178 NVIDIA backend status refresh",
        "PR #178 merged as `aea89cc9dea8560602c72f84e5ff6e78ca526434`",
        "accepted only for the private UCCL-EP runtime-fusion coordinator",
        "accepted descriptor allocation, runtime path, same invocation id",
        "unsupported/failure status, and output sink",
        "It remains unsupported and provides no runtime dispatch",
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status",
        "private UCCL-EP runtime-dispatch scaffold/status gate",
        "must not run real UCCL-EP dispatch/combine work",
        "scheduler/runtime pass evidence",
        "fresh H200 fused success",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_refresh


def test_runtime_dispatch_scaffold_status_slice_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }
    runtime_fusion_abi = (
        ROOT / "src" / "cuda" / "platform" / "include" / "host"
        / "pto_cuda_runtime_fusion_abi.h"
    ).read_text(encoding="utf-8")
    host_runtime = (
        ROOT / "src" / "cuda" / "platform" / "onboard" / "host"
        / "pto_runtime_c_api.cpp"
    ).read_text(encoding="utf-8")
    common_abi = (
        ROOT / "src" / "common" / "worker" / "pto_runtime_c_api.h"
    ).read_text(encoding="utf-8")
    private_entry_test = (
        ROOT / "tests" / "ut" / "py" / "test_cuda_runtime_fusion_private_entry.py"
    ).read_text(encoding="utf-8")

    required_terms = [
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status",
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map",
        "runtime-dispatch scaffold/status gate",
        "runtime dispatch request/driver handoff",
        "coordinator-owned",
        "pr #180",
        "dc32c52dfccfd7838f865a11c3d4837e8ee568ba",
        "missing_runtime_dispatch_scaffold",
        "unsupported",
        "failed",
        "persistent_device_uccl_ep_runtime_fusion.status: passed",
        "actual_fused_cross_gpu_execution: true",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_terms:
            assert required in normalized, f"{name} missing {required!r}"

    for required in [
        "PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD",
        "PtoCudaUcclEpRuntimeDispatchScaffoldStatus",
        "runtime_dispatch_scaffold_status",
        "pto_cuda_runtime_fusion_prepare_runtime_dispatch_scaffold_status",
        "pto_cuda_runtime_fusion_failure_is_runtime_dispatch_scaffold_failed",
        "missing_runtime_dispatch_scaffold",
        "private UCCL-EP runtime dispatch scaffold/status gate validation failed",
    ]:
        assert required in runtime_fusion_abi

    for required in [
        "pto_cuda_runtime_fusion_prepare_runtime_dispatch_scaffold_status",
        "runtime_fusion_coordinator_",
        "request.uccl_ep_runtime = &runtime_fusion_coordinator_",
    ]:
        assert required in host_runtime

    for forbidden in [
        "PtoCudaUcclEpRuntimeDispatchScaffoldStatus",
        "PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION",
        "PTO_CUDA_RUNTIME_FUSION_FAILURE_MISSING_RUNTIME_DISPATCH_SCAFFOLD",
        "pto_cuda_runtime_fusion_prepare_runtime_dispatch_scaffold_status",
    ]:
        assert forbidden not in common_abi

    assert (
        "test_private_runtime_dispatch_scaffold_status_gate_is_coordinator_owned"
        in private_entry_test
    )
    assert "missing_gate_result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_FAILED" in (
        private_entry_test
    )
    assert "result.status == PTO_CUDA_RUNTIME_FUSION_STATUS_UNSUPPORTED" in (
        private_entry_test
    )
    assert "PTO_CUDA_RUNTIME_FUSION_STATUS_PASSED" in private_entry_test

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-25 - UCCL-EP Runtime Fusion Runtime Dispatch "
        "Scaffold Status Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "multi-agent-worker-runtime-dispatch-scaffold-status",
        "019efec4-746e-7503-994d-38557ed64c8e",
        "No tmux pane is used for this worker",
        "No nested workers were launched",
        "uv-xiao/pto-cu",
        "base branch `main`",
        "starting commit `0f562e7fb475ef042d1b97d6261d25b503d2eb2f`",
        "PR #180 <https://github.com/uv-xiao/pto-cu/pull/180>",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "1 failed in 0.32s",
        "runtime_dispatch_scaffold_status",
        "PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_SCAFFOLD_STATUS_VERSION",
        "PtoCudaUcclEpRuntimeDispatchScaffoldStatus",
        "missing_runtime_dispatch_scaffold",
        "No fresh H200 command is planned or run",
        "no H200 fused-success",
        "does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "does not set `actual_fused_cross_gpu_execution: true`",
        "merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba` by PR #180",
        "Accepted only for the private coordinator-owned runtime-dispatch",
        "eligible prepared gate remains `unsupported`",
        "mirrored to the runtime-owned sink",
    ]:
        assert required in normalized_dispatch
    assert "/home/" not in normalized_dispatch
    assert "pending PR #180 dispatcher review" not in normalized_dispatch
    assert "persistent_device_uccl_ep_runtime_fusion.status: passed`" in (
        normalized_dispatch
    )
    assert "does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`" in (
        normalized_dispatch
    )

    post_refresh_entry = texts["dispatch"].split(
        "### 2026-06-25 - Post-Runtime-Dispatch-Scaffold "
        "Status Refresh Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_refresh = " ".join(post_refresh_entry.split())
    for required in [
        "nvidia-goal-status-post-runtime-dispatch-scaffold-status",
        "019efed6-f837-7360-b6f6-c435917ba20f",
        "No tmux pane is used for this worker",
        "No nested workers were launched",
        "post-PR #180 NVIDIA backend status refresh",
        "PR #180 merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba`",
        "Add private UCCL EP runtime dispatch scaffold gate (#180)",
        "accepted only for the private coordinator-owned runtime-dispatch",
        "missing gate yields `missing_runtime_dispatch_scaffold`",
        "failed private result",
        "eligible prepared gate remains `unsupported`",
        "mirrored to the runtime-owned sink",
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map",
        "selected exactly one next PR-sized dependency slice",
        "runtime dispatch request/driver handoff",
        "must not run UCCL-EP dispatch/combine work",
        "scheduler/runtime pass evidence",
        "fresh H200 fused success",
        "`persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "`actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_refresh
    assert "/home/" not in normalized_refresh


def test_runtime_dispatch_request_handoff_map_slice_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "persistent_moe": in_progress_root
        / "persistent_moe_dispatch_combine_h200.md",
        "boundary": in_progress_root / "communication_runtime_boundary.md",
        "selection": in_progress_root / "communication_selection.md",
        "slicing": in_progress_root / "pr_slicing_plan.md",
        "dispatch": in_progress_root / "dispatch_log.md",
    }
    texts = {
        name: path.read_text(encoding="utf-8") for name, path in docs.items()
    }

    required_terms = [
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map",
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status",
        "runtime dispatch request/driver handoff map",
        "request owner",
        "driver owner",
        "status dependency",
        "failure ownership",
        "unsupported handoff state",
        "failed handoff state",
        "pr #180",
        "dc32c52dfccfd7838f865a11c3d4837e8ee568ba",
        "pr #181",
        "05457b7dead2f561be22c24c72771add880f4562",
        "missing_runtime_dispatch_scaffold",
        "runtime-dispatch scaffold/status gate",
        "no uccl-ep dispatch/combine work",
        "no scheduler/runtime pass evidence",
        "no fresh h200 fused success",
        "no public taskargs",
        "no public callconfig",
        "no common runtime c api",
        "no uccl host-runtime abi",
        "no examples",
        "no stable docs",
        "no performance claims",
        "persistent_device_uccl_ep_runtime_fusion.status: passed",
        "actual_fused_cross_gpu_execution: true",
    ]
    for name, text in texts.items():
        normalized = " ".join(text.split()).lower()
        for required in required_terms:
            assert required in normalized, f"{name} missing {required!r}"

    dispatch_entry = texts["dispatch"].split(
        "### 2026-06-25 - UCCL-EP Runtime Fusion Runtime Dispatch "
        "Request Handoff Map Worker",
        1,
    )[1].split("\n### ", 1)[0]
    normalized_dispatch = " ".join(dispatch_entry.split())
    for required in [
        "multi-agent-worker-runtime-dispatch-request-handoff-map",
        "019efee7-3530-7ed2-a4d1-a48a105e4a42",
        "No tmux pane is used for this worker",
        "No nested workers were launched",
        "uv-xiao/pto-cu",
        "base branch `main`",
        "starting commit `05457b7dead2f561be22c24c72771add880f4562`",
        "PR #182 <https://github.com/uv-xiao/pto-cu/pull/182>",
        "gh pr create --repo uv-xiao/pto-cu --base main --head",
        "docs/in_progress/nvidia_backend/dispatch_log.md",
        "docs/in_progress/nvidia_backend/pr_slicing_plan.md",
        "docs/in_progress/nvidia_backend/communication_runtime_boundary.md",
        "docs/in_progress/nvidia_backend/communication_selection.md",
        "docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md",
        "tests/ut/py/test_nvidia_review_artifacts.py",
        "request owner",
        "driver owner",
        "status dependency",
        "failure ownership",
        "unsupported handoff state",
        "failed handoff state",
        "No fresh H200 command is planned or run",
        "selected exactly one next PR-sized implementation slice",
        "nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status",
        "narrower than pass evidence",
        "no real UCCL-EP dispatch/combine work",
        "no scheduler/runtime pass evidence",
        "no fresh H200 fused success",
        "no public `TaskArgs`",
        "no public `CallConfig`",
        "no common runtime C API",
        "no UCCL host-runtime ABI",
        "no examples, stable docs, or performance claims",
        "no `persistent_device_uccl_ep_runtime_fusion.status: passed`",
        "no `actual_fused_cross_gpu_execution: true`",
    ]:
        assert required in normalized_dispatch
    assert "/home/" not in normalized_dispatch
    assert "pending dispatcher review" not in normalized_dispatch


def test_chat_256k_needle_stream_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_stream_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "PROBE_EXIT_STATUS=" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28153" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "stream: true" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=64" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert "expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_OK_28153" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "stream_events_received:" in evidence
    assert "event_count:" in evidence
    assert "content_chunk_count:" in evidence
    assert "normalized_output_equals_expected:" in evidence
    assert "normalized_output_length_chars:" in evidence
    assert "expected_answer_exact:" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "raw streaming chunk content is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "private absolute paths are not recorded" in evidence
    assert "local_only_vllm_chat_256k_needle_stream" in readiness
    assert "vllm_remote_chat_256k_needle_stream_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_256k_needle_stream_repeat_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_stream_repeat_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "PROBE_EXIT_STATUS=" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28154" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "stream: true" in evidence
    assert "repeat_count: 2" in evidence
    assert "attempts_completed:" in evidence
    assert "passed_attempts:" in evidence
    assert "failed_attempts:" in evidence
    assert "same streaming request payload" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=64" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert (
        "expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_REPEAT_OK_28154"
        in evidence
    )
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "stream_events_received:" in evidence
    assert "done_seen:" in evidence
    assert "finish_reason:" in evidence
    assert "normalized_output_equals_expected:" in evidence
    assert "expected_answer_exact:" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "raw streaming chunk content is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "private absolute paths are not recorded" in evidence
    assert "local_only_vllm_chat_256k_needle_stream_repeat" in readiness
    assert "vllm_remote_chat_256k_needle_stream_repeat_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_256k_needle_stream_position_sweep_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_stream_position_sweep_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "PROBE_EXIT_STATUS=" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28156" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "stream: true" in evidence
    assert "needle_position_sweep: early,middle,late" in evidence
    assert "positions_completed:" in evidence
    assert "passed_positions:" in evidence
    assert "failed_positions:" in evidence
    assert "position: early" in evidence
    assert "position: middle" in evidence
    assert "position: late" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=64" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert (
        "expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156"
        in evidence
    )
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "stream_events_received:" in evidence
    assert "done_seen:" in evidence
    assert "finish_reason:" in evidence
    assert "normalized_output_equals_expected:" in evidence
    assert "expected_answer_exact:" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "raw streaming chunk content is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "private absolute paths are not recorded" in evidence
    assert "local_only_vllm_chat_256k_needle_stream_position_sweep" in readiness
    assert (
        "vllm_remote_chat_256k_needle_stream_position_sweep_probe.md"
        in readiness
    )
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_256k_needle_stream_usage_contract_docs_are_guarded():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_stream_usage_contract_probe.md"
    ).read_text(encoding="utf-8")
    example = (
        ROOT
        / "examples"
        / "cuda"
        / "vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe.py"
    )
    readme = (ROOT / "examples" / "cuda" / "README.md").read_text(
        encoding="utf-8"
    )
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")
    required_text = [
        "vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe.py",
        "28159",
        "stream_options.include_usage=true",
        "PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28159",
    ]

    assert example.is_file()
    for text in required_text:
        assert text in readme
        assert text in readiness
        assert text in evidence
    assert "returned streaming usage" in readme
    assert "strict exact output matching" in readme
    assert "PROBE_EXIT_STATUS=0" in evidence
    assert "status: passed" in evidence
    assert "server_port: 28159" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "tokenize_endpoint: /tokenize" in evidence
    assert "stream: true" in evidence
    assert "actual_prompt_tokens: 255797" in evidence
    assert (
        "prompt_token_measurement_source: "
        "vllm_server_tokenize_chat_count"
        in evidence
    )
    assert "prompt_token_measurement_status: passed" in evidence
    assert "prompt_token_measurement_http_status: 200" in evidence
    assert "tokenizer_accounting: vLLM server /tokenize chat count" in evidence
    assert "event_count: 22" in evidence
    assert "content_chunk_count: 18" in evidence
    assert "done_seen: true" in evidence
    assert "finish_reason: stop" in evidence
    assert "normalized_output_equals_expected: true" in evidence
    assert "expected_answer_exact: passed" in evidence
    assert "usage_presence: passed" in evidence
    assert "usage_prompt_tokens_match: passed" in evidence
    assert "usage.prompt_tokens: 255797" in evidence
    assert "usage.completion_tokens: 20" in evidence
    assert "usage.total_tokens: 255817" in evidence
    assert "choice_count: 0" in evidence
    assert "usage_keys: completion_tokens,prompt_tokens,total_tokens" in evidence
    assert "cleanup.status: passed" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "raw streaming chunk content is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "private absolute paths are not recorded" in evidence
    assert "PROBE_EXIT_STATUS=0" in readme
    assert (
        "local_only_vllm_chat_256k_needle_stream_usage_contract: passed"
        in readiness
    )
    status_block_match = re.search(r"```text\n(.*?)\n```", readiness, re.S)
    assert status_block_match is not None
    status_block = status_block_match.group(1)
    assert (
        "local_only_vllm_chat_256k_needle_stream_usage_contract: passed "
        "under recorded\n"
        "  262144-token boundary with stream_options.include_usage=true "
        "and\n"
        "  server-side /tokenize prompt-token accounting"
        in status_block
    )
    assert "chat_needle_stream_choice_shape" not in status_block
    assert "chat_needle_stream_prompt_token_mismatch" not in status_block
    assert (
        "vllm_remote_chat_256k_needle_stream_usage_contract_probe.md"
        in readiness
    )
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence
    assert "/" + "tmp/pto-cu" not in evidence


def test_chat_256k_needle_stream_truncated_failure_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_stream_truncated_failure_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: failed" in evidence
    assert "PROBE_EXIT_STATUS=2" in evidence
    assert (
        "failure_category: chat_needle_stream_expected_answer_not_exact"
        in evidence
    )
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28155" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "stream: true" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=1" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert (
        "expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_TRUNCATED_OK_28155"
        in evidence
    )
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "HTTP status: 200" in evidence
    assert "stream_events_received: true" in evidence
    assert "event_count: 2" in evidence
    assert "content_chunk_count: 1" in evidence
    assert "done_seen: true" in evidence
    assert "finish_reason: length" in evidence
    assert "normalized_output_equals_expected: false" in evidence
    assert "normalized_output_length_chars: 1" in evidence
    assert "expected_answer_exact: failed" in evidence
    assert "usage: not_returned" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "strict exact-comparator failure" in evidence
    assert "not a transport/server failure" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "raw streaming chunk content is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "private absolute paths are not recorded" in evidence
    assert (
        "local_only_vllm_chat_256k_needle_stream_truncated_failure: failed"
        in readiness
    )
    assert (
        "vllm_remote_chat_256k_needle_stream_truncated_failure_probe.md"
        in readiness
    )
    assert "generated_text_length_chars" not in evidence
    assert "normalized_generated_text" not in evidence
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "logprobs" not in evidence
    assert "/" + "home/" not in evidence


def test_deepseek_v4_weight_manifest_preflight_omits_local_free_bytes():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_weight_manifest_preflight.md"
    ).read_text(encoding="utf-8")

    assert not re.search(r"storage_free_bytes: [0-9]+", evidence)


def test_deepseek_v4_weight_acquisition_preflight_artifacts_are_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = (
        in_progress_root
        / "deepseek_v4_flash_weight_acquisition_preflight_h200.md"
    ).read_text(encoding="utf-8")
    readiness = (
        in_progress_root / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")
    work_prep = (in_progress_root / "work_preparation.md").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "examples" / "cuda" / "README.md").read_text(
        encoding="utf-8"
    )

    for text in (evidence, readiness, work_prep, readme):
        assert "deepseek_v4_flash_weight_acquisition_preflight.py" in text
        assert "--require-capacity" in text
        assert "can_attempt_download" in text
        assert "can_attempt_model_load" in text
        assert "not serving evidence" in text
        assert "not model-load evidence" in text
        assert "not DeepSeek correctness" in text

    for required in [
        "PROBE_EXIT_STATUS=",
        "model_id: deepseek-ai/DeepSeek-V4-Flash",
        "artifact_dir: tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash",
        "indexed_shard_count:",
        "present_shard_count:",
        "missing_shard_count:",
        "indexed_bytes:",
        "metadata_storage_bytes:",
        "estimated_required_bytes_remaining:",
        "filesystem_free_bytes:",
        "required_capacity_bytes:",
        "has_required_capacity:",
        "no shard download was attempted",
        "no model load was attempted",
        "no vLLM server was started",
        "no generated text was produced",
    ]:
        assert required in evidence

    assert "/" + "home/" not in evidence
    assert "/" + "tmp/pto-cu" not in evidence


def test_64k_long_prompt_response_contract_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_64k_long_prompt_response_contract_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28139" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=64000" in evidence
    assert "actual_prompt_tokens=63999" in evidence
    assert "prompt_chars: 418877" in evidence
    assert "usage.prompt_tokens: 63999" in evidence
    assert "usage.completion_tokens: 4" in evidence
    assert "usage.total_tokens: 64003" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "local_only_vllm_64k_long_prompt_response_contract" in readiness
    assert "vllm_remote_64k_long_prompt_response_contract_probe.md" in readiness
    assert "text_" + "sha256" not in evidence


def test_128k_long_prompt_response_contract_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_128k_long_prompt_response_contract_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28140" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=128000" in evidence
    assert "actual_prompt_tokens=127997" in evidence
    assert "prompt_chars: 837773" in evidence
    assert "usage.prompt_tokens: 127997" in evidence
    assert "usage.completion_tokens: 4" in evidence
    assert "usage.total_tokens: 128001" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "local_only_vllm_128k_long_prompt_response_contract" in readiness
    assert "vllm_remote_128k_long_prompt_response_contract_probe.md" in readiness
    assert "text_" + "sha256" not in evidence


def test_192k_long_prompt_response_contract_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_192k_long_prompt_response_contract_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28141" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=192000" in evidence
    assert "actual_prompt_tokens=191995" in evidence
    assert "prompt_chars: 1256669" in evidence
    assert "usage.prompt_tokens: 191995" in evidence
    assert "usage.completion_tokens: 4" in evidence
    assert "usage.total_tokens: 191999" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "local_only_vllm_192k_long_prompt_response_contract" in readiness
    assert "vllm_remote_192k_long_prompt_response_contract_probe.md" in readiness
    assert "text_" + "sha256" not in evidence


def test_256k_long_prompt_response_contract_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_long_prompt_response_contract_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28142" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=256000" in evidence
    assert "actual_prompt_tokens=256004" in evidence
    assert "prompt_chars: 1675637" in evidence
    assert "usage.prompt_tokens: 256004" in evidence
    assert "usage.completion_tokens: 4" in evidence
    assert "usage.total_tokens: 256008" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "TIME-WAIT" in evidence
    assert "local_only_vllm_256k_long_prompt_response_contract" in readiness
    assert "vllm_remote_256k_long_prompt_response_contract_probe.md" in readiness
    assert "text_" + "sha256" not in evidence


def test_256k_needle_correctness_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_needle_correctness_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "PROBE_EXIT_STATUS=0" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28143" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "actual_prompt_tokens=255799" in evidence
    assert "prompt_chars: 1230965" in evidence
    assert "expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "needle_occurrences: 1" in evidence
    assert "usage.prompt_tokens: 255799" in evidence
    assert "usage.completion_tokens: 64" in evidence
    assert "usage.total_tokens: 255863" in evidence
    assert "expected_answer_contained: passed" in evidence
    assert " PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "local_only_vllm_256k_needle_correctness" in readiness
    assert "vllm_remote_256k_needle_correctness_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_256k_needle_exact_output_failure_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_needle_exact_output_failure.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: failed" in evidence
    assert "PROBE_EXIT_STATUS=2" in evidence
    assert "failure_category: needle_expected_answer_not_exact" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28144" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "actual_prompt_tokens=255799" in evidence
    assert "prompt_chars: 1230965" in evidence
    assert "expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "needle_occurrences: 1" in evidence
    assert "match_mode: exact" in evidence
    assert "finish_reason: stop" in evidence
    assert "generated_text_length_chars: 37" in evidence
    assert "expected_answer_exact: failed" in evidence
    assert " PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "local_only_vllm_256k_needle_exact_output: failed" in readiness
    assert "vllm_remote_256k_needle_exact_output_failure.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_256k_needle_exact_stop_sequence_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_needle_exact_stop_sequence_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "PROBE_EXIT_STATUS=0" in evidence
    assert "failure_category" not in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28145" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "actual_prompt_tokens=255799" in evidence
    assert "prompt_chars: 1230965" in evidence
    assert "max_tokens=64" in evidence
    assert "expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "needle_occurrences: 1" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "finish_reason: stop" in evidence
    assert "generated_text_length_chars: 33" in evidence
    assert "expected_answer_exact: passed" in evidence
    assert "usage.prompt_tokens: 255799" in evidence
    assert "usage.completion_tokens: 17" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "private absolute paths are not recorded" in evidence
    assert "local_only_vllm_256k_needle_exact_stop_sequence: passed" in readiness
    assert "vllm_remote_256k_needle_exact_stop_sequence_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_256k_needle_exact_stop_repeat_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_needle_exact_stop_repeat_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "PROBE_EXIT_STATUS=0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28146" in evidence
    assert "repeat_count: 3" in evidence
    assert "passed_attempts: 3" in evidence
    assert "failed_attempts: 0" in evidence
    assert "max_model_len=262144" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "actual_prompt_tokens=255799" in evidence
    assert "max_tokens=64" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    for attempt_index in (1, 2, 3):
        assert f"attempt_index: {attempt_index}" in evidence
    assert evidence.count("finish_reason: stop") == 3
    assert evidence.count("generated_text_length_chars: 33") == 3
    assert evidence.count("exact_check: passed") == 3
    assert evidence.count("usage.prompt_tokens: 255799") == 3
    assert evidence.count("usage.completion_tokens: 17") == 3
    assert evidence.count("usage.total_tokens: 255816") == 3
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "local_only_vllm_256k_needle_exact_stop_repeat: passed" in readiness
    assert "vllm_remote_256k_needle_exact_stop_repeat_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_256k_needle_exact_truncated_failure_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_needle_exact_truncated_failure_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: failed" in evidence
    assert "PROBE_EXIT_STATUS=2" in evidence
    assert "failure_category: needle_expected_answer_not_exact" in evidence
    assert "expected failure-mode characterization" in evidence
    assert "vllm: 0.23.0" in evidence
    assert "torch: 2.11.0+cu130" in evidence
    assert "torch CUDA: 13.0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28147" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "actual_prompt_tokens=255799" in evidence
    assert "prompt_chars: 1230965" in evidence
    assert "max_tokens=1" in evidence
    assert "expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143" in evidence
    assert "needle_occurrences: 1" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "finish_reason: length" in evidence
    assert "generated_text_length_chars: 2" in evidence
    assert "normalized_generated_text: P" in evidence
    assert "expected_answer_exact: failed" in evidence
    assert "usage.prompt_tokens: 255799" in evidence
    assert "usage.completion_tokens: 1" in evidence
    assert "usage.total_tokens: 255800" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert (
        "local_only_vllm_256k_needle_exact_truncated_failure: failed"
        in readiness
    )
    assert (
        "vllm_remote_256k_needle_exact_truncated_failure_probe.md"
        in readiness
    )
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_256k_needle_position_sweep_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_256k_needle_position_sweep_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "PROBE_EXIT_STATUS=0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28148" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=64" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequences_configured: true" in evidence
    assert 'stop: ["\\n```"]' in evidence
    assert "positions_requested: early,middle,late" in evidence
    assert "positions_completed: early,middle,late" in evidence
    assert "passed_attempts: 3" in evidence
    assert "failed_attempts: 0" in evidence
    for position in ("early", "middle", "late"):
        assert f"needle_position: {position}" in evidence
    assert evidence.count("finish_reason: stop") == 3
    assert evidence.count("exact_check: passed") == 3
    assert evidence.count("usage.prompt_tokens: 255799") == 3
    assert evidence.count("usage.completion_tokens: 17") == 2
    assert evidence.count("usage.total_tokens: 255816") == 2
    assert "usage.completion_tokens: 16" in evidence
    assert "usage.total_tokens: 255815" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "local_only_vllm_256k_needle_position_sweep: passed" in readiness
    assert "vllm_remote_256k_needle_position_sweep_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_exact_canary_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_exact_canary_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: passed" in evidence
    assert "PROBE_EXIT_STATUS=0" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28149" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "max_tokens=16" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert "message_count: 2" in evidence
    assert "message_roles: system,user" in evidence
    assert "expected_answer: PTO_CHAT_EXACT_CANARY_28149" in evidence
    assert "match_mode: exact" in evidence
    assert "HTTP status: 200" in evidence
    assert "finish_reason: stop" in evidence
    assert "normalized_output_equals_expected: true" in evidence
    assert "normalized_output_length_chars: 27" in evidence
    assert "expected_answer_exact: passed" in evidence
    assert "usage.prompt_tokens: 33" in evidence
    assert "usage.completion_tokens: 13" in evidence
    assert "usage.total_tokens: 46" in evidence
    assert "POST /v1/chat/completions HTTP/1.1" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "local_only_vllm_chat_exact_canary: passed" in readiness
    assert "vllm_remote_chat_exact_canary_probe.md" in readiness
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_exact_truncated_failure_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_exact_truncated_failure_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "status: failed" in evidence
    assert "PROBE_EXIT_STATUS=2" in evidence
    assert "failure_category: chat_canary_expected_answer_not_exact" in evidence
    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28150" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "max_tokens=1" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert "message_count: 2" in evidence
    assert "message_roles: system,user" in evidence
    assert "expected_answer: PTO_CHAT_EXACT_CANARY_28149" in evidence
    assert "match_mode: exact" in evidence
    assert "HTTP status: 200" in evidence
    assert "finish_reason: length" in evidence
    assert "normalized_output_equals_expected: false" in evidence
    assert "expected_answer_exact: failed" in evidence
    assert "usage.completion_tokens: 1" in evidence
    assert "POST /v1/chat/completions HTTP/1.1" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "strict exact-comparator failure" in evidence
    assert "transport/server failure" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "local_only_vllm_chat_exact_truncated_failure: failed" in readiness
    assert "262144-token boundary with max_tokens=1 as expected" in readiness
    assert "vllm_remote_chat_exact_truncated_failure_probe.md" in readiness
    assert "generated_text_length_chars" not in evidence
    assert "normalized_generated_text" not in evidence
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_256k_needle_exact_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_exact_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28151" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=64" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert "message_count: 2" in evidence
    assert "message_roles: system,user" in evidence
    assert "expected_answer: PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequence: \\n```" in evidence
    assert "needle_occurrences: 1" in evidence
    assert "HTTP status: 200" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "vllm_remote_chat_256k_needle_exact_probe.md" in readiness

    if "status: passed" in evidence:
        assert "PROBE_EXIT_STATUS=0" in evidence
        assert "finish_reason: stop" in evidence
        assert "normalized_output_equals_expected: true" in evidence
        assert "expected_answer_exact: passed" in evidence
        assert "local_only_vllm_chat_256k_needle_exact: passed" in readiness
    else:
        assert "status: failed" in evidence
        assert "PROBE_EXIT_STATUS=2" in evidence
        assert "failure_category: chat_needle_expected_answer_not_exact" in evidence
        assert "normalized_output_equals_expected: false" in evidence
        assert "expected_answer_exact: failed" in evidence
        assert "strict exact-comparator failure" in evidence
        assert "local_only_vllm_chat_256k_needle_exact: failed" in readiness

    assert "messages" not in evidence
    assert "NEEDLE_ANSWER" not in evidence
    assert "Synthetic filler" not in evidence
    assert "generated_text" not in evidence
    assert "normalized_generated_text" not in evidence
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "logprobs" not in evidence
    assert "/" + "home/" not in evidence


def test_chat_256k_needle_repeat_evidence_is_review_safe():
    evidence = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "vllm_remote_chat_256k_needle_repeat_probe.md"
    ).read_text(encoding="utf-8")
    readiness = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "deepseek_v4_flash_serving_readiness.md"
    ).read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=1,7" in evidence
    assert "server_port: 28152" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "max_model_len=262144" in evidence
    assert "tensor_parallel_size=2" in evidence
    assert "target_prompt_tokens=255800" in evidence
    assert "max_tokens=64" in evidence
    assert "temperature=0.0" in evidence
    assert "top_p=1.0" in evidence
    assert "seed=0" in evidence
    assert "repeat_count: 2" in evidence
    assert "message_count: 2" in evidence
    assert "message_roles: system,user" in evidence
    assert "expected_answer: PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152" in evidence
    assert "match_mode: exact" in evidence
    assert "stop_sequence: \\n```" in evidence
    assert "needle_occurrences: 1" in evidence
    assert "HTTP status: 200" in evidence
    assert "attempt_index: 1" in evidence
    assert "attempt_index: 2" in evidence
    assert "remaining_process_group_pids: []" in evidence
    assert "raw prompt text is not recorded" in evidence
    assert "raw request payload is not recorded" in evidence
    assert "raw generated text is not recorded" in evidence
    assert "token ID arrays are not recorded" in evidence
    assert "logprob values are not recorded" in evidence
    assert "generated-text digests are not recorded" in evidence
    assert "vllm_remote_chat_256k_needle_repeat_probe.md" in readiness

    if "status: passed" in evidence:
        assert "PROBE_EXIT_STATUS=0" in evidence
        assert "passed_attempts: 2" in evidence
        assert "failed_attempts: 0" in evidence
        assert evidence.count("finish_reason: stop") == 2
        assert evidence.count("exact_check: passed") == 2
        assert "local_only_vllm_chat_256k_needle_repeat: passed" in readiness
    else:
        assert "status: failed" in evidence
        assert "PROBE_EXIT_STATUS=2" in evidence
        assert "failure_category:" in evidence
        assert "local_only_vllm_chat_256k_needle_repeat: failed" in readiness

    assert "messages" not in evidence
    assert "NEEDLE_ANSWER" not in evidence
    assert "Synthetic filler" not in evidence
    assert "generated_text" not in evidence
    assert "normalized_generated_text" not in evidence
    assert "text_" + "sha256" not in evidence
    assert "token_ids" not in evidence
    assert "logprobs" not in evidence
    assert "/" + "home/" not in evidence


def test_long_prompt_admission_probe_dry_run_contract():
    script = (
        ROOT
        / "examples"
        / "cuda"
        / "vllm_deepseek_v4_long_prompt_admission_probe.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--port",
            "28135",
            "--target-prompt-tokens",
            "16000",
            "--max-model-len",
            "262144",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28135
    assert payload["request"]["endpoint"] == "/v1/completions"
    assert payload["request"]["limits"]["target_prompt_tokens"] == 16000
    assert payload["request"]["limits"]["max_tokens"] == 1
    assert payload["request"]["limits"]["stream"] is False
    assert payload["request"]["limits"]["echo"] is False
    assert payload["request"]["limits"]["logprobs"] is False
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert "prompt" not in payload["request"]
    assert "payload" not in payload["request"]
    assert not any("generated text" in claim for claim in payload["non_claims"])
    assert payload["contract_checks"] == [
        "HTTP 200 from /health",
        "HTTP 200 from /v1/models",
        "HTTP 200 from one non-streaming /v1/completions request",
        "exactly one response choice when HTTP 200 returns",
        "usage fields recorded when returned",
        "raw prompt text is not recorded",
        "raw generated text is not recorded",
        "server process group cleanup leaves no remaining PIDs",
    ]


def test_long_prompt_response_contract_probe_dry_run_contract():
    script = (
        ROOT
        / "examples"
        / "cuda"
        / "vllm_deepseek_v4_long_prompt_response_contract_probe.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--port",
            "28136",
            "--target-prompt-tokens",
            "16000",
            "--max-model-len",
            "262144",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28136
    assert payload["request"]["endpoint"] == "/v1/completions"
    assert payload["request"]["limits"]["target_prompt_tokens"] == 16000
    assert payload["request"]["limits"]["max_tokens"] == 4
    assert payload["request"]["limits"]["stream"] is False
    assert payload["request"]["limits"]["echo"] is False
    assert payload["request"]["limits"]["logprobs"] is False
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert "prompt" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "text_" + "sha256" not in result.stdout
    assert not any("generated text" in claim for claim in payload["non_claims"])
    assert payload["contract_checks"] == [
        "HTTP 200 from /health",
        "HTTP 200 from /v1/models",
        "model list includes served model and max_model_len=262144",
        "HTTP 200 from one non-streaming /v1/completions request",
        "top-level completion response is a JSON object",
        "response model field matches served model when returned",
        "exactly one response choice object",
        "first choice exposes text and finish_reason fields",
        "generated text length is recorded without generated text contents",
        "usage prompt/completion/total token fields are internally consistent when returned",
        "usage.prompt_tokens matches measured prompt tokens when available",
        "usage.completion_tokens within request max_tokens",
        "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
        "raw prompt text is not recorded",
        "raw generated text is not recorded",
        "server process group cleanup leaves no remaining PIDs",
    ]


def test_long_prompt_warmup_followup_probe_dry_run_contract():
    script = (
        ROOT
        / "examples"
        / "cuda"
        / "vllm_deepseek_v4_long_prompt_warmup_followup_probe.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--port",
            "28137",
            "--target-prompt-tokens",
            "16000",
            "--max-model-len",
            "262144",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28137
    assert payload["endpoints"] == [
        "/health",
        "/v1/models",
        "/v1/completions",
        "/v1/completions",
    ]
    for request_key, label in [
        ("warmup_request", "warmup"),
        ("followup_request", "followup"),
    ]:
        request = payload[request_key]
        assert request["label"] == label
        assert request["endpoint"] == "/v1/completions"
        assert request["limits"]["target_prompt_tokens"] == 16000
        assert request["limits"]["max_tokens"] == 4
        assert request["limits"]["stream"] is False
        assert request["limits"]["echo"] is False
        assert request["limits"]["logprobs"] is False
        assert request["prompt_text_recorded"] is False
        assert request["payload_recorded"] is False
        assert "prompt" not in request
        assert "payload" not in request
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert "text_" + "sha256" not in result.stdout
    assert not any("generated text" in claim for claim in payload["non_claims"])
    assert payload["contract_checks"] == [
        "HTTP 200 from /health",
        "HTTP 200 from /v1/models",
        "model list includes served model and max_model_len=262144",
        "HTTP 200 from warmup non-streaming /v1/completions request",
        "HTTP 200 from followup non-streaming /v1/completions request",
        "top-level completion responses are JSON objects",
        "response model fields match served model when returned",
        "exactly one response choice object per response",
        "first choices expose text and finish_reason fields",
        "generated text lengths are recorded without generated text contents",
        "usage prompt/completion/total token fields are internally consistent when returned",
        "usage.prompt_tokens matches measured prompt tokens when available",
        "usage.completion_tokens within request max_tokens",
        "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
        "raw prompt text is not recorded",
        "raw generated text is not recorded",
        "server process group cleanup leaves no remaining PIDs",
    ]


def test_needle_correctness_probe_dry_run_contract():
    script = (
        ROOT
        / "examples"
        / "cuda"
        / "vllm_deepseek_v4_needle_correctness_probe.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--port",
            "28143",
            "--target-prompt-tokens",
            "255800",
            "--max-model-len",
            "262144",
            "--max-tokens",
            "64",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28143
    assert payload["request"]["endpoint"] == "/v1/completions"
    assert payload["request"]["limits"]["target_prompt_tokens"] == 255800
    assert payload["request"]["limits"]["max_tokens"] == 64
    assert payload["request"]["limits"]["match_mode"] == "contains"
    assert payload["request"]["limits"]["normalization"] == (
        "strip leading/trailing whitespace, then strip one surrounding "
        "Markdown code fence when the whole output is fenced"
    )
    assert payload["request"]["limits"]["expected_answer"] == (
        "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    )
    assert payload["request"]["limits"]["needle_occurrences"] == 1
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "prompt" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "text_" + "sha256" not in result.stdout
    assert "token_ids" not in result.stdout
    assert payload["contract_checks"] == [
        "HTTP 200 from /health",
        "HTTP 200 from /v1/models",
        "model list includes served model and max_model_len=262144",
        "HTTP 200 from one non-streaming /v1/completions request",
        "top-level completion response is a JSON object",
        "response model field matches served model when returned",
        "exactly one response choice object",
        "first choice exposes text and finish_reason fields",
        "generated output contains the exact expected needle answer in contains mode",
        "normalized generated output equals the expected needle answer in exact mode",
        "short synthetic generated output is recorded when within review-safe bound",
        "usage prompt/completion/total token fields are internally consistent when returned",
        "usage.prompt_tokens matches measured prompt tokens when available",
        "usage.completion_tokens within request max_tokens",
        "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
        "raw prompt text is not recorded",
        "raw request payload is not recorded",
        "token ID arrays are not recorded",
        "logprob values are not recorded",
        "server process group cleanup leaves no remaining PIDs",
    ]


def test_chat_256k_needle_exact_probe_dry_run_contract():
    script = (
        ROOT
        / "examples"
        / "cuda"
        / "vllm_deepseek_v4_chat_256k_needle_exact_probe.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--port",
            "28151",
            "--target-prompt-tokens",
            "255800",
            "--max-model-len",
            "262144",
            "--max-tokens",
            "64",
            "--temperature",
            "0.0",
            "--top-p",
            "1.0",
            "--seed",
            "0",
            "--expected-answer",
            "PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151",
            "--stop-sequence",
            "\n```",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28151
    assert payload["request"]["endpoint"] == "/v1/chat/completions"
    assert payload["request"]["limits"]["target_prompt_tokens"] == 255800
    assert payload["request"]["limits"]["max_tokens"] == 64
    assert payload["request"]["limits"]["temperature"] == 0.0
    assert payload["request"]["limits"]["top_p"] == 1.0
    assert payload["request"]["limits"]["seed"] == 0
    assert payload["request"]["limits"]["expected_answer"] == (
        "PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151"
    )
    assert payload["request"]["limits"]["match_mode"] == "exact"
    assert payload["request"]["limits"]["stop"] == ["\n```"]
    assert payload["request"]["limits"]["needle_occurrences"] == 1
    assert payload["request"]["limits"]["message_count"] == 2
    assert payload["request"]["limits"]["message_roles"] == ["system", "user"]
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "messages" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "prompt" not in payload["request"]
    assert "generated_text" not in result.stdout
    assert "generated_text_digest" not in result.stdout
    assert "text_" + "sha256" not in result.stdout
    assert "token_ids" not in result.stdout
    assert "logprobs" not in result.stdout
    assert payload["contract_checks"] == [
        "HTTP 200 from /health",
        "HTTP 200 from /v1/models",
        "model list includes served model and max_model_len=262144",
        "HTTP 200 from one non-streaming /v1/chat/completions request",
        "top-level chat completion response is a JSON object",
        "response model field matches served model when returned",
        "exactly one response choice object",
        "first choice exposes assistant message content and finish_reason fields",
        "normalized assistant content equals the expected 256K needle answer",
        "usage prompt/completion/total token fields are internally consistent when returned",
        "usage.prompt_tokens matches measured chat prompt tokens when available",
        "usage.completion_tokens within request max_tokens",
        "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
        "raw prompt text is not recorded",
        "raw request payload is not recorded",
        "raw generated text is not recorded",
        "token ID arrays are not recorded",
        "logprob values are not recorded",
        "generated-text digests are not recorded",
        "server process group cleanup leaves no remaining PIDs",
    ]


def test_active_review_surfaces_do_not_reference_removed_eval_scripts():
    for path in ACTIVE_REVIEW_SURFACES:
        text = path.read_text(encoding="utf-8")
        assert REMOVED_EVAL_SCRIPT_RE.search(text) is None, path


def test_gluon_gemm_review_docs_use_placeholders_for_remote_paths():
    for path in GLUON_GEMM_REVIEW_DOCS:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert UCCL_PRIVATE_PATH_RE.search(text) is None, path
        assert "tmp/gluon-" in text, path


def test_gluon_tensor_core_gemm_records_bf16_h200_correctness():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_tensor_core_gemm.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"

    evidence_text = evidence.read_text(encoding="utf-8")
    evidence_search_text = re.sub(r"\s+", " ", evidence_text)
    for required in [
        "fresh project-local `.venv` preflight failed because Torch and Triton",
        "preserved Gluon environment preflight passed",
        "Triton `3.7.1`",
        "`gemm_tensor_core_tiled_bf16_f32`",
        "\"status\": \"passed\"",
        "\"passed_cases\": 2",
        "\"skipped_cases\": 0",
        "\"max_abs_error\": 0.002899169921875",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not generated-kernel performance evidence",
        "not production-readiness evidence",
    ]:
        assert required in evidence_search_text
    assert (
        "The BF16 serving-shape fixture is generated and skip-safe"
        not in evidence_text
    )
    assert (
        "This is unsupported-API evidence, not BF16 runtime correctness evidence"
        not in evidence_text
    )

    checklist_text = checklist.read_text(encoding="utf-8")
    checklist_search_text = re.sub(r"\s+", " ", checklist_text)
    assert "BF16 tensor-core GEMM correctness on H200" in checklist_search_text
    assert "case statuses: passed, passed" in checklist_search_text
    assert "make the BF16 tensor-core fixture pass" not in checklist_search_text


def test_gluon_tensor_core_gemm_records_fp8_unsupported_boundary():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_tensor_core_gemm.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"

    evidence_text = evidence.read_text(encoding="utf-8")
    evidence_search_text = re.sub(r"\s+", " ", evidence_text)
    opener_claim = (
        "It is correctness and unsupported-boundary evidence only, "
        "not performance evidence."
    )
    assert opener_claim in evidence_search_text
    assert "It is correctness evidence only, not performance evidence." not in (
        evidence_search_text
    )
    for required in [
        "`gemm_tensor_core_tiled_fp8e4nv_f32`",
        "torch.float8_e4m3fn",
        "gl.float8e4nv",
        "float8e4b15",
        "float8e5b16",
        "\"kind\": \"gluon_fp8_wgmma_compile\"",
        "\"error\": \"PassManager::run failed\"",
        "WGMMA type or shape is not supported",
        "not FP8 GEMM correctness evidence",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "<remote-gluon-venv>/bin/python",
        "not FlashInfer integration evidence",
        "not serving integration evidence",
        "not generated-kernel performance evidence",
        "not production-readiness evidence",
        "not BF16/FP4/grouped GEMM/MoE/FlashAttention/vLLM integration evidence",
    ]:
        assert required in evidence_search_text
    assert "\"status\": \"passed\"" in evidence_search_text
    assert "This is unsupported-boundary evidence only" not in evidence_text

    checklist_text = checklist.read_text(encoding="utf-8")
    checklist_search_text = re.sub(r"\s+", " ", checklist_text)
    assert "not FP8 GEMM correctness evidence" in checklist_search_text
    assert "make FP8 WGMMA lowering pass" in checklist_search_text
    assert "PassManager::run failed" in checklist_search_text


def test_gluon_tensor_core_gemm_records_fp4_unsupported_boundary():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_tensor_core_gemm.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"

    evidence_text = evidence.read_text(encoding="utf-8")
    evidence_search_text = re.sub(r"\s+", " ", evidence_text)
    for required in [
        "H200 FP4 Boundary Evidence",
        (
            "This section tracks source-generating tensor-core GEMM artifacts "
            "from `KernelCompiler(platform=\"cuda\").generate_gluon_kernel(...)` "
            "plus the FP4 dtype/API and grouped GEMM source/API boundary "
            "harnesses"
        ),
        "BF16 and FP8 source-generating harnesses emit source and manifest",
        "FP4 remains a dtype/API boundary harness",
        (
            "does not generate source or manifest artifacts unless a confirmed "
            "Gluon FP4 WGMMA operand dtype path exists"
        ),
        "torch.float4_e2m1fn_x2",
        "`fp4_to_fp`",
        "\"kind\": \"gluon_fp4_dtype_api_unavailable\"",
        "\"status\": \"skipped\"",
        "\"artifact\": null",
        "missing Gluon FP4 WGMMA dtype API",
        "not FP4 GEMM correctness evidence",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "<remote-gluon-venv>/bin/python",
        "not FlashInfer integration evidence",
        "not serving integration evidence",
        "not generated-kernel performance evidence",
        "not production-readiness evidence",
    ]:
        assert required in evidence_search_text
    assert "tensor-core harnesses always generate source and manifest artifacts" not in (
        evidence_search_text
    )
    assert "now supports three Gluon tensor-core GEMM artifacts" not in (
        evidence_search_text
    )
    assert "FP4 GEMM correctness evidence exists" not in evidence_search_text

    checklist_text = checklist.read_text(encoding="utf-8")
    checklist_search_text = re.sub(r"\s+", " ", checklist_text)
    assert "FP4 API/lowering boundary is explicitly recorded" in checklist_search_text
    assert "not FP4 GEMM correctness evidence" in checklist_search_text
    assert "missing Gluon FP4 WGMMA dtype API" in checklist_search_text


def test_gluon_tensor_core_gemm_records_grouped_gemm_boundary():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_tensor_core_gemm.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    example = ROOT / "examples" / "cuda" / "gluon_gemm_grouped_tensor_core.py"

    assert example.is_file()
    example_text = example.read_text(encoding="utf-8")
    for required in [
        "run_grouped_tensor_core_boundary",
        "probe_grouped_gemm_api",
        "gluon_grouped_gemm_source_path_unavailable",
        "--require-cuda",
    ]:
        assert required in example_text

    evidence_text = evidence.read_text(encoding="utf-8")
    evidence_search_text = re.sub(r"\s+", " ", evidence_text)
    for required in [
        "H200 Grouped GEMM Boundary Evidence",
        "`examples/cuda/gluon_gemm_grouped_tensor_core.py`",
        "`gemm_grouped_tensor_core_f16_f32`",
        "\"kind\": \"gluon_grouped_gemm_source_path_unavailable\"",
        "\"status\": \"skipped\"",
        "\"artifact\": null",
        "fresh remote `.venv` lacked Torch and Triton/Gluon",
        "two_group_smoke",
        "linear_style_grouped",
        "missing grouped GEMM WGMMA source path",
        "No module named 'triton'",
        "Generic Hopper WGMMA warpgroup primitives",
        "`hopper_warpgroup_attrs`",
        "\"gluon_language_grouped_gemm_attrs\": []",
        "\"hopper_grouped_gemm_attrs\": []",
        "\"hopper_warpgroup_attrs\": []",
        "not grouped GEMM correctness evidence",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "No usable preserved Gluon-capable Python environment was found",
        "not FlashInfer integration evidence",
        "not serving integration evidence",
        "not generated-kernel performance evidence",
        "not production-readiness evidence",
    ]:
        assert required in evidence_search_text
    assert "grouped GEMM correctness evidence exists" not in evidence_search_text

    checklist_text = checklist.read_text(encoding="utf-8")
    checklist_search_text = re.sub(r"\s+", " ", checklist_text)
    assert "Grouped GEMM boundary harness" in checklist_search_text
    assert "records generic Hopper `warpgroup_mma*` primitives separately" in (
        checklist_search_text
    )
    assert "not grouped GEMM correctness evidence" in checklist_search_text
    assert "missing grouped GEMM WGMMA source path" in checklist_search_text


def test_nvidia_branch_ci_avoids_ascend_jobs():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "nvidia-review:" in workflow
    assert "runs-on: [self-hosted, a2a3]" not in workflow
    assert "runs-on: [self-hosted, a5]" not in workflow
    assert "--platform a2a3" not in workflow
    assert "--platform a5" not in workflow


def test_cuda_comm_descriptor_and_nccl_worker_control_artifacts_are_recorded():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    boundary = in_progress_root / "communication_runtime_boundary.md"
    selection = in_progress_root / "communication_selection.md"
    baseline = in_progress_root / "nccl_two_h200_baseline.md"
    worker_control = in_progress_root / "nccl_worker_control_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"

    for path in (
        boundary,
        selection,
        baseline,
        worker_control,
        ROOT / "simpler_setup" / "cuda_comm.py",
        ROOT / "src" / "cuda" / "platform" / "include" / "host" / "pto_cuda_comm_descriptor_abi.h",
        ROOT / "examples" / "cuda" / "nccl_two_gpu_baseline.py",
        ROOT / "examples" / "cuda" / "nccl_worker_control_ops.py",
        ROOT / "tests" / "ut" / "py" / "test_cuda_comm.py",
    ):
        assert path.is_file(), path

    boundary_text = boundary.read_text(encoding="utf-8")
    assert "CudaCommDeviceDescriptor" in boundary_text
    assert "configure_cuda_comm_descriptor" in boundary_text
    assert "CTRL_COMM_OP" in boundary_text
    assert "TaskArgs" in boundary_text
    assert "CallConfig" in boundary_text
    assert "does not claim UCCL host-runtime dispatch" in boundary_text
    assert "DeepSeek model correctness" in boundary_text

    selection_text = selection.read_text(encoding="utf-8")
    assert "NCCL" in selection_text
    assert "UCCL adapter execution" in selection_text
    assert "multi-node evidence" in selection_text

    baseline_text = baseline.read_text(encoding="utf-8")
    assert "examples/cuda/nccl_two_gpu_baseline.py" in baseline_text
    assert "all_reduce" in baseline_text
    assert "reduce_scatter" in baseline_text
    assert "all_gather" in baseline_text
    assert "send_recv" in baseline_text

    worker_control_text = worker_control.read_text(encoding="utf-8")
    assert "examples/cuda/nccl_worker_control_ops.py" in worker_control_text
    assert "--device-ids 6,7" in worker_control_text
    assert "--tensor-numel 1024" in worker_control_text
    assert "max_abs_error: 0.0" in worker_control_text
    assert "CTRL_COMM_OP" in worker_control_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "nccl_two_gpu_baseline.py" in readme_text
    assert "nccl_worker_control_ops.py" in readme_text


def test_uccl_adapter_artifacts_are_recorded_without_host_runtime_abi_claims():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "boundary": in_progress_root / "uccl_adapter_boundary.md",
        "plan": in_progress_root / "uccl_ep_p2p_probe_plan.md",
        "probe": in_progress_root / "uccl_ep_p2p_h200.md",
        "p2p": in_progress_root / "uccl_p2p_adapter_h200.md",
        "ep": in_progress_root / "uccl_ep_adapter_h200.md",
        "comparison": in_progress_root / "uccl_ep_nccl_worker_control_comparison.md",
    }
    readme = ROOT / "examples" / "cuda" / "README.md"

    for path in (
        *docs.values(),
        ROOT / "examples" / "cuda" / "uccl_p2p_ipc_adapter.py",
        ROOT / "examples" / "cuda" / "uccl_ep_dispatch_combine_adapter.py",
    ):
        assert path.is_file(), path

    boundary_text = docs["boundary"].read_text(encoding="utf-8")
    for required in [
        "adapter/probe evidence",
        "simpler_setup/cuda_comm.py",
        "UcclP2PWriteIpcDescriptor",
        "UcclEpDispatchCombineDescriptor",
        "UcclP2PCudaCommRuntime",
        "TaskArgs",
        "CallConfig",
        "No CUDA host-runtime UCCL ABI is added",
        "RDMA is not proven",
        "multi-node is not proven",
        "serving integration is not proven",
    ]:
        assert required in boundary_text

    for key in ("probe", "p2p", "ep", "comparison"):
        text = docs[key].read_text(encoding="utf-8")
        assert "historical restart context" in text
        assert "Fresh PR evidence" in text
        assert "not RDMA evidence" in text
        assert "not multi-node evidence" in text
        assert "not serving evidence" in text
        assert "/" + "home/" not in text

    plan_text = docs["plan"].read_text(encoding="utf-8")
    assert "descriptor/adapter-preparation" in plan_text
    assert "examples/cuda/uccl_ep_dispatch_combine_adapter.py --require-cuda" in plan_text
    assert "no CUDA host-runtime UCCL ABI" in plan_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "uccl_p2p_ipc_adapter.py" in readme_text
    assert "uccl_ep_dispatch_combine_adapter.py" in readme_text


def test_deepseek_v4_model_load_probe_review_artifacts_are_recorded():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    review_doc = in_progress_root / "vllm_deepseek_v4_model_load_probe.md"
    readiness_doc = in_progress_root / "deepseek_v4_flash_serving_readiness.md"
    work_prep = in_progress_root / "work_preparation.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    probe = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_model_load_probe.py"
    probe_test = ROOT / "tests" / "ut" / "py" / "test_vllm_deepseek_v4_model_load_probe.py"

    for path in (review_doc, readiness_doc, work_prep, readme, probe, probe_test):
        assert path.is_file(), path

    review_text = review_doc.read_text(encoding="utf-8")
    for required in [
        "examples/cuda/vllm_deepseek_v4_model_load_probe.py",
        "--require-artifacts --require-vllm --require-cuda",
        "--tensor-parallel-size 2",
        "--max-model-len 4096",
        "--kv-cache-dtype fp8",
        "does not start an HTTP server",
        "does not run generation",
        "missing-shard or missing-vLLM run is not model-load evidence",
        "H200 was not rerun for this child slice",
        "not simpler-nv/vLLM kernel integration evidence",
    ]:
        assert required in review_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "vllm_deepseek_v4_model_load_probe.py" in readme_text
    assert "--require-artifacts --require-vllm --require-cuda" in readme_text
    assert "Missing artifacts, missing vLLM, or missing CUDA report structured skips" in readme_text

    readiness_text = readiness_doc.read_text(encoding="utf-8")
    assert "vllm_deepseek_v4_model_load_probe.py" in readiness_text
    assert "--require-cuda" in readiness_text
    assert "A missing-shard or missing-vLLM run is not model-load evidence" in readiness_text

    work_prep_text = work_prep.read_text(encoding="utf-8")
    assert "run_model_load_probe" in work_prep_text
    assert "vllm_deepseek_v4_model_load_probe.py" in work_prep_text
    assert "--require-artifacts --require-vllm --require-cuda" in work_prep_text


def test_status_rollup_records_current_deepseek_serving_boundary():
    status_text = (DOC_ROOT / "status.md").read_text(encoding="utf-8")

    for required in [
        "docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_repeat_probe.md",
        "docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_truncated_failure_probe.md",
        "docs/in_progress/nvidia_backend/pypto_serving_source_contract_h200.md",
        "docs/in_progress/nvidia_backend/flashinfer_serving_operator_checklist.md",
        "docs/in_progress/nvidia_backend/gluon_rmsnorm_h200.md",
        "docs/in_progress/nvidia_backend/gluon_layernorm_h200.md",
        "docs/in_progress/nvidia_backend/gluon_silu_h200.md",
        "docs/in_progress/nvidia_backend/gluon_gelu_h200.md",
        "docs/in_progress/nvidia_backend/gluon_gated_silu_h200.md",
        "FlashInfer-derived operator checklist",
        "generated Gluon FP32 RMSNorm shape sweep",
        "generated Gluon FP32 LayerNorm shape sweep",
        "generated Gluon FP32 SiLU fixture",
        "generated Gluon FP32 GELU fixture",
        "generated Gluon FP32 gated SiLU fixture",
        "not FlashInfer integration evidence",
        "not a transport/server failure",
        "not simpler-nv/vLLM kernel integration evidence",
        "source-route `stream=true`",
        "bounded H200 source-route matrix",
        "streaming `/v1/chat/completions`",
        "cloned source still omits non-streaming usage fields",
    ]:
        assert required in status_text


def test_flashinfer_serving_operator_checklist_is_recorded():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    target = in_progress_root / "serving_target_selection.md"
    readme = ROOT / "examples" / "cuda" / "README.md"

    for path in (checklist, target, readme):
        assert path.is_file(), f"missing {path.relative_to(ROOT)}"

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "| FlashInfer reference family |" not in checklist_text
    assert "| PTO current evidence |" not in checklist_text
    assert "| Gap / next PTO milestone |" not in checklist_text
    assert "| Explicit non-claim |" not in checklist_text
    for required in [
        "# FlashInfer Serving Operator Checklist",
        "tmp/sources/repos/external/flashinfer/README.md",
        "FlashInfer reference family",
        "PTO current evidence",
        "Gap / next PTO milestone",
        "Explicit non-claim",
        "attention kernels",
        "Paged and Ragged KV-Cache",
        "Decode, Prefill, and Append",
        "MLA Attention",
        "Cascade Attention",
        "Sparse Attention",
        "POD-Attention",
        "BF16 GEMM",
        "FP8 GEMM",
        "FP4 GEMM",
        "Grouped GEMM",
        "Fused MoE Kernels",
        "Multiple Routing Methods",
        "Quantized MoE",
        "Sorting-Free Sampling",
        "Speculative Decoding",
        "AllReduce",
        "Multi-Node NVLink",
        "NVSHMEM Integration",
        "RoPE",
        "Normalization",
        "gluon_rmsnorm_h200.md",
        "gluon_layernorm_h200.md",
        "gluon_silu_h200.md",
        "gluon_gelu_h200.md",
        "gluon_gated_silu_h200.md",
        "RMSNorm",
        "rmsnorm_f32",
        "LayerNorm",
        "layernorm_f32",
        "SiLU",
        "silu_f32",
        "GELU",
        "gelu_f32",
        "gated SiLU",
        "gated_silu_f32",
        "Gemma-style fused norm",
        "Activations",
        "Hopper SM 9.0 includes H100 and H200",
        "not FlashInfer integration evidence",
        "not DeepSeek serving through pypto-serving",
        "not generated-kernel performance evidence",
        "not production readiness evidence",
    ]:
        assert required in checklist_text

    target_text = target.read_text(encoding="utf-8")
    assert "flashinfer_serving_operator_checklist.md" in target_text
    assert "Use the FlashInfer checklist" in target_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "FlashInfer serving operator checklist" in readme_text
    assert "flashinfer_serving_operator_checklist.md" in readme_text


def test_gluon_rmsnorm_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_rmsnorm_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_rmsnorm_f32.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon RMSNorm FP32 H200 Correctness",
        "rmsnorm_f32",
        "x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight",
        "--sweep",
        "--rows 2 --hidden 16 --eps 1e-5",
        "rows=1, hidden=7168, eps=1e-5",
        "DeepSeek-V4-Flash config hidden_size",
        "tests/ut/py/test_vllm_deepseek_v4_artifact_probe.py",
        "examples/cuda/vllm_deepseek_v4_artifact_probe.py",
        "--require-cuda --device 0 --arch compute_90",
        "status: passed",
        "case statuses: passed, passed",
        "max absolute error:",
        "machine class: H200",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not performance, throughput, or latency evidence",
        "not fused normalization or fused activation evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_rmsnorm_f32.py" in readme_text
    assert "rmsnorm_f32" in readme_text
    assert "--sweep" in readme_text
    assert "hidden=7168" in readme_text
    assert "DeepSeek-V4-Flash config hidden_size" in readme_text
    assert "gluon_rmsnorm_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_rmsnorm_h200.md" in checklist_text
    assert "RMSNorm" in checklist_text
    assert "hidden=7168" in checklist_text
    assert "DeepSeek-V4-Flash config hidden_size" in checklist_text
    assert "LayerNorm" in checklist_text
    assert "Gemma-style fused norm" in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_rmsnorm_h200.md" in status_text
    assert "generated Gluon FP32 RMSNorm shape sweep" in status_text
    assert "hidden=7168" in status_text
    assert "DeepSeek-V4-Flash config hidden_size" in status_text


def test_gluon_flashattention_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_flashattention_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_flashattention_fwd.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon FlashAttention H200 Correctness",
        "flashattention_fwd_f32",
        "softmax((q @ k.T) * scale) @ v",
        "--output-dir tmp/gluon-flashattention-shape-coverage-h200",
        "--output-dir tmp/gluon-flashattention-prefill-sweep-h200",
        "--output-dir tmp/gluon-flashattention-decode-coverage-h200",
        "--output-dir tmp/gluon-flashattention-append-sweep-h200",
        "--output-dir tmp/gluon-flashattention-causal-boundary-h200",
        "--output-dir tmp/gluon-flashattention-decode-boundary-h200",
        "--output-dir tmp/gluon-flashattention-append-boundary-h200",
        "--output-dir tmp/gluon-flashattention-prefill-boundary-h200",
        "--output-dir tmp/gluon-flashattention-kvcache-paged-unsupported-h200",
        "--output-dir tmp/gluon-flashattention-kvcache-ragged-unsupported-h200",
        "--output-dir tmp/gluon-flashattention-varlen-unsupported-h200",
        "--output-dir tmp/gluon-flashattention-mla-unsupported-h200",
        "--output-dir tmp/gluon-flashattention-cascade-unsupported-h200",
        "--output-dir tmp/gluon-flashattention-sparse-unsupported-h200",
        "--output-dir tmp/gluon-flashattention-pod-unsupported-h200",
        "--require-cuda --arch compute_90",
        "--sweep --causal --require-cuda",
        "--sweep --causal --causal-sweep-phase decode --require-cuda",
        "--sweep --causal --causal-sweep-phase append --require-cuda",
        "--tile-shape 32x32x64 --causal --require-cuda",
        "--tile-shape 1x32x64 --causal --require-cuda",
        "--tile-shape 4x32x64 --causal --require-cuda",
        "--kv-cache-boundary paged --require-cuda",
        "--kv-cache-boundary ragged --require-cuda",
        "--sequence-boundary varlen --require-cuda",
        "--attention-variant mla --require-cuda",
        "--attention-variant cascade --require-cuda",
        "--attention-variant sparse --require-cuda",
        "--attention-variant pod --require-cuda",
        "schema_version: 1",
        "phase: prefill",
        "phase: decode",
        "phase: append",
        "causal: true",
        "status: skipped",
        "sequence_boundary: varlen",
        "attention_variant: mla",
        "attention_variant: cascade",
        "attention_variant: sparse",
        "attention_variant: pod",
        "unsupported_boundary.kind: paged_kv_cache",
        "unsupported_boundary.kind: ragged_kv_cache",
        "unsupported_boundary.kind: varlen_attention",
        "unsupported_boundary.kind: mla_attention",
        "unsupported_boundary.kind: cascade_attention",
        "unsupported_boundary.kind: sparse_attention",
        "unsupported_boundary.kind: pod_attention",
        "unsupported-boundary evidence only",
        "softmax(masked_fill((q @ k.T) * scale, key_index > query_index, -inf)) @ v",
        (
            "softmax(masked_fill((q @ k.T) * scale, "
            "key_index > query_index + (seqlen_k - seqlen_q), -inf)) @ v"
        ),
        "artifact paths are repo-relative",
        "private absolute paths are not recorded",
        "status: passed",
        "same-length multi-query prefill-shaped",
        "bounded causal prefill sweep correctness evidence",
        "only causal prefill cases",
        "case_name: prefill_64x64x64",
        "bounded causal decode sweep correctness evidence",
        "only causal decode cases",
        "bounded causal append sweep correctness evidence",
        "only causal append cases",
        "case_count: 3",
        "case_name: append_8x64x64",
        "shape: seqlen_q=8, seqlen_k=64, head_dim=64",
        "broader bounded multi-query causal append H200 gate",
        "phase: prefill",
        "single-query decode-shaped",
        "small multi-query append-shaped",
        "shape: seqlen_q=32, seqlen_k=32, head_dim=32",
        "shape: seqlen_q=32, seqlen_k=32, head_dim=64",
        "shape: seqlen_q=64, seqlen_k=64, head_dim=64",
        "shape: seqlen_q=16, seqlen_k=64, head_dim=64",
        "shape: seqlen_q=1, seqlen_k=32, head_dim=64",
        "shape: seqlen_q=4, seqlen_k=32, head_dim=64",
        "shape: seqlen_q=8, seqlen_k=32, head_dim=64",
        "causal_query_offset: 28",
        "common serving attention head dimension",
        "32x32x64 failed H200 correctness",
        "--tile-shape 32x32x64",
        "max_abs_error: 3.5762786865234375e-07",
        "max_abs_error: 4.172325134277344e-07",
        "source_sha256: c611666d3b527e615f2d8e4658b57f10865f1547fd370e8bb45639353682a06e",
        "source_sha256: f9f0ff900d33023c462579063be9aa8560a82c63d43aae2bd851369cfcfb58a4",
        "remains separate from the two-case promoted sweep",
        "case_count: 4",
        "case_name: decode_1x16x64",
        "shape: seqlen_q=1, seqlen_k=16, head_dim=64",
        "case_name: decode_1x32x64",
        "case_name: decode_1x64x64",
        "shape: seqlen_q=1, seqlen_k=64, head_dim=64",
        "case_name: decode_1x128x32",
        "shape: seqlen_q=1, seqlen_k=128, head_dim=32",
        "1x128x64 hit a Triton CUDA out-of-memory boundary",
        "per-case artifact paths are repo-relative",
        "max_abs_error: 2.384185791015625e-07",
        "source_sha256:",
        "machine class: H200",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not production serving readiness",
        "not FlashInfer integration evidence",
        "not DeepSeek semantic correctness",
        "not full prefill coverage",
        "not bounded append KV-cache coverage",
        "not paged/ragged KV-cache correctness",
        "not full decode",
        "not full append",
        "not attention-variant correctness",
        "not performance, throughput, or latency evidence",
        "not paged/ragged KV-cache correctness",
        "not varlen attention correctness",
        "not MLA attention correctness",
        "not Cascade Attention correctness",
        "not Sparse Attention correctness",
        "not POD-Attention correctness",
        "not full prefill, full decode, full append, or append coverage",
        "not multi-tile attention coverage",
        "not fused attention integration",
        "not KV-cache integration",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_flashattention_fwd.py" in readme_text
    assert "flashattention_fwd_f32" in readme_text
    assert "--sweep" in readme_text
    assert "--causal" in readme_text
    assert "--sweep --causal --require-cuda" in readme_text
    assert "--sweep --causal --causal-sweep-phase decode --require-cuda" in readme_text
    assert "--sweep --causal --causal-sweep-phase append --require-cuda" in readme_text
    assert "head_dim=64" in readme_text
    assert "32x32x64 failed H200 correctness" in readme_text
    assert "causal: true" in readme_text
    assert "phase: prefill" in readme_text
    assert "phase: decode" in readme_text
    assert "phase: append" in readme_text
    assert "--tile-shape 32x32x64 --causal" in readme_text
    assert "--tile-shape 1x32x64 --causal" in readme_text
    assert "--tile-shape 4x32x64 --causal" in readme_text
    assert "--kv-cache-boundary paged" in readme_text
    assert "--kv-cache-boundary ragged" in readme_text
    assert "--sequence-boundary varlen" in readme_text
    assert "--attention-variant mla" in readme_text
    assert "--attention-variant cascade" in readme_text
    assert "--attention-variant sparse" in readme_text
    assert "--attention-variant pod" in readme_text
    assert "unsupported_boundary" in readme_text
    assert "paged_kv_cache" in readme_text
    assert "ragged_kv_cache" in readme_text
    assert "varlen_attention" in readme_text
    assert "mla_attention" in readme_text
    assert "cascade_attention" in readme_text
    assert "sparse_attention" in readme_text
    assert "pod_attention" in readme_text
    assert "sequence_boundary" in readme_text
    assert "attention_variant" in readme_text
    assert "unsupported-boundary evidence only" in readme_text
    assert "not varlen attention correctness" in readme_text
    assert "not MLA attention correctness" in readme_text
    assert "not Cascade Attention correctness" in readme_text
    assert "not Sparse Attention correctness" in readme_text
    assert "not POD-Attention correctness" in readme_text
    assert "same-length multi-query prefill-shaped" in readme_text
    assert "bounded causal prefill sweep correctness evidence" in readme_text
    assert "only causal prefill cases" in readme_text
    assert "prefill_64x64x64" in readme_text
    assert "bounded causal decode sweep correctness evidence" in readme_text
    assert "only causal decode cases" in readme_text
    assert "decode_1x64x64" in readme_text
    assert "decode_1x128x32" in readme_text
    assert "1x128x64 hit a Triton CUDA out-of-memory boundary" in readme_text
    assert "bounded causal append sweep correctness evidence" in readme_text
    assert "only causal append cases" in readme_text
    assert "append_8x64x64" in readme_text
    assert "not full prefill coverage" in readme_text
    assert "not attention-variant correctness" in readme_text
    assert "single-query decode-shaped" in readme_text
    assert "small multi-query append-shaped" in readme_text
    assert "aggregate structured JSON" in readme_text
    assert "schema_version" in readme_text
    assert "repo-relative artifact paths" in readme_text
    assert "gluon_flashattention_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    normalized_checklist_text = " ".join(checklist_text.split())
    assert "gluon_flashattention_h200.md" in checklist_text
    assert "small FP32 FlashAttention shape sweep" in normalized_checklist_text
    assert "bounded causal prefill sweep correctness evidence" in checklist_text
    assert "bounded causal decode sweep correctness evidence" in checklist_text
    assert "bounded causal append sweep correctness evidence" in checklist_text
    assert "--sweep --causal --require-cuda" in checklist_text
    assert "--sweep --causal --causal-sweep-phase decode --require-cuda" in checklist_text
    assert "--sweep --causal --causal-sweep-phase append --require-cuda" in checklist_text
    assert "same-length multi-query prefill-shaped" in normalized_checklist_text
    assert "prefill_64x64x64" in checklist_text
    assert "only causal decode cases" in checklist_text
    assert "decode_1x64x64" in checklist_text
    assert "decode_1x128x32" in checklist_text
    assert "1x128x64 hit a Triton CUDA out-of-memory boundary" in checklist_text
    assert "only causal append cases" in checklist_text
    assert "append_8x64x64" in checklist_text
    assert "single-query decode-shaped" in normalized_checklist_text
    assert "small multi-query append-shaped" in normalized_checklist_text
    assert "causal: true" in checklist_text
    assert "phase: prefill" in checklist_text
    assert "phase: decode" in checklist_text
    assert "phase: append" in checklist_text
    assert "head_dim=64" in checklist_text
    assert "--tile-shape 1x32x64" in checklist_text
    assert "--tile-shape 4x32x64" in checklist_text
    assert "--tile-shape 32x32x64" in checklist_text
    assert "--kv-cache-boundary paged" in checklist_text
    assert "--kv-cache-boundary ragged" in checklist_text
    assert "--sequence-boundary varlen" in checklist_text
    assert "--attention-variant mla" in checklist_text
    assert "--attention-variant cascade" in checklist_text
    assert "--attention-variant sparse" in checklist_text
    assert "--attention-variant pod" in checklist_text
    assert "unsupported_boundary.kind: paged_kv_cache" in checklist_text
    assert "unsupported_boundary.kind: ragged_kv_cache" in checklist_text
    assert "unsupported_boundary.kind: varlen_attention" in checklist_text
    assert "unsupported_boundary.kind: mla_attention" in checklist_text
    assert "unsupported_boundary.kind: cascade_attention" in checklist_text
    assert "unsupported_boundary.kind: sparse_attention" in checklist_text
    assert "unsupported_boundary.kind: pod_attention" in checklist_text
    assert "sequence_boundary: varlen" in checklist_text
    assert "attention_variant: mla" in checklist_text
    assert "attention_variant: cascade" in checklist_text
    assert "attention_variant: sparse" in checklist_text
    assert "attention_variant: pod" in checklist_text
    assert "unsupported-boundary evidence only" in checklist_text
    assert "--causal" in checklist_text
    assert "32x32x64 failed H200 correctness" in normalized_checklist_text
    assert "now passes with structured JSON" in normalized_checklist_text
    assert "remains separate from the promoted sweep" in checklist_text
    assert "repo-relative artifact paths" in checklist_text
    assert "schema_version" in checklist_text
    assert "not FlashInfer integration evidence" in checklist_text
    assert "not full prefill coverage" in checklist_text
    assert "not attention-variant correctness" in checklist_text
    assert "not paged/ragged KV-cache correctness" in checklist_text
    assert "not varlen attention correctness" in checklist_text
    assert "not MLA attention correctness" in checklist_text
    assert "not Cascade Attention correctness" in checklist_text
    assert "not Sparse Attention correctness" in checklist_text
    assert "not POD-Attention correctness" in checklist_text
    assert (
        "not full prefill, full decode, full append, or append KV-cache coverage"
        in normalized_checklist_text
    )

    status_text = status.read_text(encoding="utf-8")
    normalized_status_text = " ".join(status_text.split())
    assert "gluon_flashattention_h200.md" in status_text
    assert "generated Gluon FlashAttention shape sweep" in status_text
    assert "head_dim=64" in status_text
    assert "--tile-shape 32x32x64" in status_text
    assert "32x32x64 failed H200 correctness" in normalized_status_text
    assert "now passes with structured JSON" in normalized_status_text
    assert "remains separate from the promoted sweep" in normalized_status_text
    assert "schema_version" in status_text
    assert "repo-relative artifact paths" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_layernorm_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_layernorm_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_layernorm_f32.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon LayerNorm FP32 H200 Correctness",
        "layernorm_f32",
        "mean = average(x)",
        "var = average((x - mean) ** 2)",
        "out = (x - mean) * rsqrt(var + eps) * weight + bias",
        "--sweep",
        "--rows 2 --hidden 16 --eps 1e-5",
        "rows=1, hidden=7168, eps=1e-5",
        "DeepSeek-V4-Flash config hidden_size",
        "tests/ut/py/test_vllm_deepseek_v4_artifact_probe.py",
        "examples/cuda/vllm_deepseek_v4_artifact_probe.py",
        "--require-cuda --device 0 --arch compute_90",
        "status: passed",
        "case statuses: passed, passed",
        "max absolute error:",
        "machine class: H200",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not Gemma-style fused norm coverage",
        "not activation coverage",
        "not fused attention evidence",
        "not KV-cache integration evidence",
        "not throughput or latency evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_layernorm_f32.py" in readme_text
    assert "layernorm_f32" in readme_text
    assert "--sweep" in readme_text
    assert "hidden=7168" in readme_text
    assert "DeepSeek-V4-Flash config hidden_size" in readme_text
    assert "gluon_layernorm_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_layernorm_h200.md" in checklist_text
    assert "LayerNorm" in checklist_text
    assert "hidden=7168" in checklist_text
    assert "DeepSeek-V4-Flash config hidden_size" in checklist_text
    assert "Gemma-style fused norm" in checklist_text
    assert "SiLU" in checklist_text
    assert "GELU" in checklist_text
    assert "There is still no" not in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_layernorm_h200.md" in status_text
    assert "generated Gluon FP32 LayerNorm shape sweep" in status_text
    assert "hidden=7168" in status_text
    assert "DeepSeek-V4-Flash config hidden_size" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_rope_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_rope_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_rope_f32.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon RoPE FP32 H200 Correctness",
        "rope_f32",
        "out_even = x_even * cos - x_odd * sin",
        "out_odd = x_even * sin + x_odd * cos",
        "--output-dir tmp/gluon-rope-shape-coverage-h200",
        "--sweep",
        "--batch 1 --seq 2 --head-dim 8",
        "batch=1, seq=4, head_dim=64",
        "tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json",
        "rope_head_dim: 64",
        "--require-cuda --device 0 --arch compute_90",
        "schema_version: 1",
        "status: passed",
        "case_count: 2",
        "case statuses: passed, passed",
        "artifact paths are repo-relative",
        "private absolute paths are not recorded",
        "max absolute error:",
        "machine class: H200",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not fused attention evidence",
        "not KV-cache integration evidence",
        "not throughput or latency evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_rope_f32.py" in readme_text
    assert "rope_f32" in readme_text
    assert "--sweep" in readme_text
    assert "head_dim=64" in readme_text
    assert "rope_head_dim: 64" in readme_text
    assert "aggregate structured JSON" in readme_text
    assert "gluon_rope_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_rope_h200.md" in checklist_text
    assert "RoPE" in checklist_text
    assert "shape sweep" in checklist_text
    assert "head_dim=64" in checklist_text
    assert "rope_head_dim: 64" in checklist_text
    assert "gluon_layernorm_h200.md" in checklist_text
    assert "LayerNorm" in checklist_text
    assert "SiLU" in checklist_text
    assert "GELU" in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_rope_h200.md" in status_text
    assert "generated Gluon FP32 RoPE shape sweep" in status_text
    assert "head_dim=64" in status_text
    assert "rope_head_dim: 64" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_silu_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_silu_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_silu_f32.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon SiLU FP32 H200 Correctness",
        "silu_f32",
        "out = x / (1.0 + exp(-x))",
        "out = x * sigmoid(x)",
        "--sweep --require-cuda --device 0 --arch compute_90",
        "--n 32",
        "n=2048",
        "moe_inter_dim: 2048",
        "swiglu_limit: 10.0",
        "--require-cuda --device 0 --arch compute_90",
        "status: passed",
        "passed cases: `2`",
        "failed cases: `0`",
        "skipped cases: `0`",
        "max absolute error:",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not GELU coverage",
        "not gated activation coverage",
        "not broader activation coverage",
        "not Gemma-style fused norm coverage",
        "not fused attention evidence",
        "not KV-cache integration evidence",
        "not throughput or latency evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_silu_f32.py" in readme_text
    assert "silu_f32" in readme_text
    assert "gluon_silu_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_silu_h200.md" in checklist_text
    assert "SiLU" in checklist_text
    assert "silu_f32` FP32 SiLU correctness sweep" in checklist_text
    assert "moe_inter_dim: 2048" in checklist_text
    assert "swiglu_limit: 10.0" in checklist_text
    assert "standalone SiLU gate-activation-width evidence" in checklist_text
    assert "GELU" in checklist_text
    assert "gated activation" in checklist_text
    assert "gluon_gated_silu_h200.md" in checklist_text
    assert "gated_silu_f32" in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_silu_h200.md" in status_text
    assert "generated Gluon FP32 SiLU fixture" in status_text
    assert "SiLU fixture shape sweep" in status_text
    assert "moe_inter_dim: 2048" in status_text
    assert "swiglu_limit: 10.0" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_gelu_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_gelu_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_gelu_f32.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon GELU FP32 H200 Correctness",
        "gelu_f32",
        "0.5 * x * (1.0 + erf(x / sqrt(2.0)))",
        "--sweep",
        "--n 32",
        "n=2048",
        "moe_inter_dim: 2048",
        "swiglu_limit: 10.0",
        "--require-cuda --device 0 --arch compute_90",
        "status: passed",
        "max absolute error:",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not SiLU coverage",
        "not gated activation coverage",
        "not broader activation coverage",
        "not Gemma-style fused norm coverage",
        "not fused attention evidence",
        "not KV-cache integration evidence",
        "not throughput or latency evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_gelu_f32.py" in readme_text
    assert "gelu_f32" in readme_text
    assert "gluon_gelu_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_gelu_h200.md" in checklist_text
    assert "GELU" in checklist_text
    assert "gelu_f32" in checklist_text
    assert "gated activation" in checklist_text
    assert "gluon_gated_silu_h200.md" in checklist_text
    assert "gated_silu_f32" in checklist_text
    assert "Remaining activation gaps include GELU" not in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_gelu_h200.md" in status_text
    assert "generated Gluon FP32 GELU fixture shape sweep" in status_text
    assert "moe_inter_dim: 2048" in status_text
    assert "swiglu_limit: 10.0" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_gated_silu_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_gated_silu_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_gated_silu_f32.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon Gated SiLU FP32 H200 Correctness",
        "gated_silu_f32",
        "out = value * gate / (1.0 + exp(-gate))",
        "--n 32",
        "--sweep",
        "case count: `2`",
        "shape: `n=2048`",
        (
            "tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/"
            "inference/config.json"
        ),
        "`moe_inter_dim: 2048`",
        "`swiglu_limit: 10.0`",
        "--require-cuda --device 0 --arch compute_90",
        "status: passed",
        "max absolute error:",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not Gemma-style fused norm coverage",
        "not fused attention evidence",
        "not KV-cache integration evidence",
        "not throughput or latency evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_gated_silu_f32.py" in readme_text
    assert "gated_silu_f32" in readme_text
    assert "--sweep" in readme_text
    assert "moe_inter_dim: 2048" in readme_text
    assert "gluon_gated_silu_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_gated_silu_h200.md" in checklist_text
    assert "gated_silu_f32" in checklist_text
    assert "gated activation" in checklist_text
    assert "moe_inter_dim: 2048" in checklist_text
    assert "Remaining activation gaps include gated activation" not in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_gated_silu_h200.md" in status_text
    assert "generated Gluon FP32 gated SiLU fixture" in status_text
    assert "moe_inter_dim: 2048" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_gemma_fused_rmsnorm_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_gemma_fused_rmsnorm_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    status = DOC_ROOT / "status.md"

    for path in (
        evidence,
        readme,
        checklist,
        status,
        ROOT / "examples" / "cuda" / "gluon_gemma_fused_rmsnorm_f32.py",
    ):
        assert path.is_file(), path

    reference = (
        "out[row, col] = x[row, col] * rsqrt(mean(x[row, :]^2) + eps) "
        "* (1.0 + weight[col])"
    )
    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon Gemma Fused RMSNorm FP32 H200 Correctness",
        "gemma_fused_rmsnorm_f32",
        reference,
        "--rows 2 --hidden 16 --eps 1e-5",
        "--require-cuda --device 0 --arch compute_90",
        "status: passed",
        "max absolute error:",
        "python environment: <remote-gluon-venv>",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not production serving readiness",
        "not DeepSeek semantic correctness",
        "not broader normalization coverage",
        "not activation coverage",
        "not fused attention evidence",
        "not KV-cache integration evidence",
        "not throughput or latency evidence",
        "not vLLM/simpler-nv integration evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_gemma_fused_rmsnorm_f32.py" in readme_text
    assert "gemma_fused_rmsnorm_f32" in readme_text
    assert "gluon_gemma_fused_rmsnorm_h200.md" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_gemma_fused_rmsnorm_h200.md" in checklist_text
    assert "gemma_fused_rmsnorm_f32" in checklist_text
    assert "Gemma-style fused norm" in checklist_text
    assert "Remaining normalization gaps include Gemma-style fused norm" not in checklist_text
    assert "plus Gemma-style fused norm" not in checklist_text
    assert "broader LayerNorm shape coverage" in checklist_text
    assert "additional non-RMSNorm normalization variants" in checklist_text

    status_text = status.read_text(encoding="utf-8")
    assert "gluon_gemma_fused_rmsnorm_h200.md" in status_text
    assert "generated Gluon FP32 Gemma-style fused RMSNorm fixture" in status_text
    assert "not FlashInfer integration evidence" in status_text


def test_gluon_topk_sampling_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_topk_sampling_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"

    for path in (
        evidence,
        readme,
        checklist,
        ROOT / "examples" / "cuda" / "gluon_topk_sampling.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon Top-K Sampling H200 Correctness",
        "topk_sampling_f32",
        "values",
        "indices",
        "lower token id first",
        "default fixture: `rows=2, vocab=8, k=3`",
        "broader fixture: `rows=3, vocab=16, k=5`",
        "--output-dir tmp/gluon-topk-shape-coverage-h200",
        "--require-cuda",
        "--arch compute_90",
        "status: passed",
        "rows=3, vocab=16, k=5",
        "values shape, indices shape, values, and indices match",
        "machine class: H200",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not vLLM or simpler-nv kernel integration evidence",
        "not DeepSeek serving correctness evidence",
        "not generated-text or tokenizer-semantics evidence",
        "not throughput or latency evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_topk_sampling.py" in readme_text
    assert "topk_sampling_f32" in readme_text
    assert "gluon_topk_sampling_h200.md" in readme_text
    assert "rows=3, vocab=16, k=5" in readme_text
    assert "payload shape checks" in readme_text
    assert "not FlashInfer integration evidence" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_topk_sampling_h200.md" in checklist_text
    assert "pypto_serving_topk_sampling_launcher_h200.md" in checklist_text
    assert "topk_sampling_f32" in checklist_text
    assert "Top-K" in checklist_text
    assert "rows=3, vocab=16, k=5" in checklist_text
    assert "serving-route launcher/probe" in checklist_text
    assert "checks result payload shapes before" in checklist_text
    assert "top-p, min-p" not in checklist_text
    assert "min-p" in checklist_text
    assert "not FlashInfer integration evidence" in checklist_text


def test_gluon_topp_sampling_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_topp_sampling_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"

    for path in (
        evidence,
        readme,
        checklist,
        ROOT / "examples" / "cuda" / "gluon_topp_sampling.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon Top-P Sampling H200 Correctness",
        "topp_sampling_f32",
        "values",
        "indices",
        "selected_counts",
        "cumulative_probabilities",
        "probabilities already sum to one",
        "lower token id first",
        "default review gate remains",
        "rows=2, vocab=8, max_k=5, p=0.75",
        "rows=3, vocab=16, max_k=6, p=0.80",
        "strict validation flags for result payload shape",
        "--output-dir tmp/gluon-topp-shape-coverage-h200",
        "--require-cuda",
        "--arch compute_90",
        "status: passed",
        "CPU golden selected counts: `[4, 5, 6]`",
        "GPU result cumulative probabilities: `[0.8000001, 0.8, 0.8]`",
        "values shape, indices shape, selected counts shape",
        "max cumulative probability error: `9.999999994736442e-08`",
        "machine class: H200",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not vLLM or simpler-nv kernel integration evidence",
        "not DeepSeek serving correctness evidence",
        "not generated-text or tokenizer-semantics evidence",
        "not throughput or latency evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_topp_sampling.py" in readme_text
    assert "topp_sampling_f32" in readme_text
    assert "gluon_topp_sampling_h200.md" in readme_text
    assert "cumulative probability boundary" in readme_text
    assert "rows=3, vocab=16, max_k=6, p=0.80" in readme_text
    assert "payload shape checks" in readme_text
    assert "not FlashInfer integration evidence" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_topp_sampling_h200.md" in checklist_text
    assert "topp_sampling_f32" in checklist_text
    assert "Top-P" in checklist_text
    assert "rows=3, vocab=16, max_k=6, p=0.80" in checklist_text
    assert "checks result" in checklist_text
    assert "payload shapes before comparisons" in checklist_text
    assert "top-p, min-p" not in checklist_text
    assert "min-p" in checklist_text
    assert "not FlashInfer integration evidence" in checklist_text


def test_gluon_minp_sampling_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_minp_sampling_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    guard = ROOT / ".agents" / "checks" / "check_nvidia_review_ready.py"

    for path in (
        evidence,
        readme,
        checklist,
        guard,
        ROOT / "examples" / "cuda" / "gluon_minp_sampling.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon Min-P Sampling H200 Correctness",
        "minp_sampling_f32",
        "values",
        "indices",
        "selected_counts",
        "probabilities already sum to one",
        "probability >= min_p * row_max_probability",
        "lower token id first",
        "--output-dir tmp/gluon-minp-shape-coverage-h200",
        "--require-cuda",
        "--arch compute_90",
        "status: passed",
        "rows=2, vocab=8, max_k=5, min_p=0.5",
        "rows=3, vocab=16, max_k=6, min_p=0.5",
        "values_shape_match",
        "indices_shape_match",
        "selected_counts_shape_match",
        "machine class: H200",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not vLLM or simpler-nv kernel integration evidence",
        "not DeepSeek serving correctness evidence",
        "not generated-text or tokenizer-semantics evidence",
        "not throughput or latency evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_minp_sampling.py" in readme_text
    assert "minp_sampling_f32" in readme_text
    assert "gluon_minp_sampling_h200.md" in readme_text
    assert "row maximum" in readme_text
    assert "rows=3, vocab=16, max_k=6, min_p=0.5" in readme_text
    assert "payload shape checks" in readme_text
    assert "not FlashInfer integration evidence" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_minp_sampling_h200.md" in checklist_text
    assert "minp_sampling_f32" in checklist_text
    assert "Min-P" in checklist_text
    assert "rows=3, vocab=16, max_k=6, min_p=0.5" in checklist_text
    assert "payload shapes before comparisons" in checklist_text
    assert "Remaining sampling gaps include min-p" not in checklist_text
    assert "Speculative Decoding" in checklist_text
    assert "not FlashInfer integration evidence" in checklist_text

    guard_text = guard.read_text(encoding="utf-8")
    assert "gluon_minp_sampling_h200.md" in guard_text
    assert "gluon_minp_sampling.py" in guard_text


def test_gluon_speculative_decoding_h200_evidence_is_review_safe():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    evidence = in_progress_root / "gluon_speculative_decoding_h200.md"
    readme = ROOT / "examples" / "cuda" / "README.md"
    checklist = in_progress_root / "flashinfer_serving_operator_checklist.md"
    guard = ROOT / ".agents" / "checks" / "check_nvidia_review_ready.py"

    for path in (
        evidence,
        readme,
        checklist,
        guard,
        ROOT / "examples" / "cuda" / "gluon_speculative_decoding.py",
    ):
        assert path.is_file(), path

    evidence_text = evidence.read_text(encoding="utf-8")
    for required in [
        "# Gluon Speculative Decoding H200 Correctness",
        "speculative_accept_f32",
        "accepted_token_ids",
        "accept_mask",
        "accepted_counts",
        "threshold <= min(1.0, target_probability / draft_probability)",
        "stop at first reject per row",
        "--output-dir tmp/gluon-speculative-shape-coverage-h200",
        "--require-cuda",
        "--arch compute_90",
        "status: passed",
        "rows=2, max_draft=4",
        "rows=3, max_draft=6",
        "payload shape checks",
        "validation shape checks",
        "accepted_token_ids_shape_match",
        "accept_mask_shape_match",
        "accepted_counts_shape_match",
        "machine class: H200",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "not FlashInfer integration evidence",
        "not vLLM or simpler-nv kernel integration evidence",
        "not DeepSeek serving correctness evidence",
        "not generated-text or tokenizer-semantics evidence",
        "not throughput or latency evidence",
    ]:
        assert required in evidence_text
    assert UCCL_PRIVATE_PATH_RE.search(evidence_text) is None
    assert "/" + "home/" not in evidence_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "gluon_speculative_decoding.py" in readme_text
    assert "speculative_accept_f32" in readme_text
    assert "gluon_speculative_decoding_h200.md" in readme_text
    assert "stop at first reject" in readme_text
    assert "rows=3, max_draft=6" in readme_text
    assert "payload shape checks" in readme_text
    assert "not FlashInfer integration evidence" in readme_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert "gluon_speculative_decoding_h200.md" in checklist_text
    assert "speculative_accept_f32" in checklist_text
    assert "Speculative Decoding" in checklist_text
    assert "rows=3, max_draft=6" in checklist_text
    assert "payload shapes before comparisons" in checklist_text
    assert "broaden shape coverage for speculative decoding" not in checklist_text
    assert "Remaining sampling gaps include speculative decoding" not in checklist_text
    assert "serving-stack integration" in checklist_text
    assert "not FlashInfer integration evidence" in checklist_text

    guard_text = guard.read_text(encoding="utf-8")
    assert "gluon_speculative_decoding_h200.md" in guard_text
    assert "gluon_speculative_decoding.py" in guard_text


def test_host_runtime_comm_operation_symbols_are_exported_by_all_producers():
    required_symbols = [
        "comm_all_reduce_f32",
        "comm_reduce_scatter_f32",
        "comm_all_gather_f32",
        "comm_send_recv_f32",
    ]
    producer_sources = {
        "cuda-onboard": [
            ROOT / "src" / "cuda" / "platform" / "onboard" / "host" / "pto_runtime_c_api.cpp",
        ],
        "common-sim": [
            ROOT / "src" / "common" / "platform_comm" / "comm_sim.cpp",
        ],
        "a2a3-onboard": [
            ROOT / "src" / "a2a3" / "platform" / "onboard" / "host" / "comm_hccl.cpp",
        ],
        "a5-onboard": [
            ROOT / "src" / "a5" / "platform" / "onboard" / "host" / "pto_runtime_c_api.cpp",
        ],
    }

    for producer, paths in producer_sources.items():
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for symbol in required_symbols:
            assert re.search(
                rf'(?:extern "C"\s+)?int\s+{symbol}\s*\(', text
            ), f"{producer} is missing {symbol}"


def test_pypto_serving_fixture_review_artifacts_are_recorded():
    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    docs = {
        "target": in_progress_root / "serving_target_selection.md",
        "design": in_progress_root / "pypto_serving_nv_shim_design.md",
        "local": in_progress_root / "pypto_serving_nv_shim_local.md",
        "completion": in_progress_root / "pypto_serving_openai_completion_fixture.md",
        "engine": in_progress_root / "pypto_serving_engine_fixture.md",
        "http": in_progress_root / "pypto_serving_http_fixture.md",
        "source_contract": in_progress_root / "pypto_serving_source_contract_h200.md",
        "topk_launcher": in_progress_root / "pypto_serving_topk_sampling_launcher_h200.md",
        "topp_launcher": in_progress_root / "pypto_serving_topp_sampling_launcher_h200.md",
        "minp_launcher": in_progress_root / "pypto_serving_minp_sampling_launcher_h200.md",
        "speculative_launcher": in_progress_root
        / "pypto_serving_speculative_decoding_launcher_h200.md",
    }
    example = ROOT / "examples" / "cuda" / "pypto_serving_nv_shim.py"
    tests = ROOT / "tests" / "ut" / "py" / "test_pypto_serving_nv_shim.py"
    readme = ROOT / "examples" / "cuda" / "README.md"

    for path in (*docs.values(), example, tests, readme):
        assert path.is_file(), f"missing {path.relative_to(ROOT)}"

    target_text = docs["target"].read_text(encoding="utf-8")
    for required in [
        "Serving Target Selection",
        "first PTO-owned integration target",
        "DeepSeek-V4-Flash compatibility baseline",
        "ModelExecutor",
        "PyptoExecutor",
        "OpenAI-compatible API",
        "not serving evidence",
        "not DeepSeek correctness",
        "not a vLLM plugin implementation",
    ]:
        assert required in target_text

    design_text = docs["design"].read_text(encoding="utf-8")
    for required in [
        "pypto-serving simpler-nv Shim Design",
        "SimplerNvExecutor",
        "SimplerNvModelRunner",
        "RuntimeConfig",
        "PrefillBatch",
        "DecodeBatch",
        "synthetic model fixture",
        "no DeepSeek-V4-Flash claim",
        "no vLLM plugin claim",
        "REMOTE_PTO_CU=<remote-pto-cu>",
    ]:
        assert required in design_text

    for key in ("local", "completion", "engine", "http", "source_contract"):
        text = docs[key].read_text(encoding="utf-8")
        assert "not DeepSeek-V4-Flash correctness" in text
        assert "not vLLM plugin evidence" in text
        assert "REMOTE_PTO_CU=<remote-pto-cu>" in text
        assert "/" + "home/" not in text
        assert "/" + "tmp/pto-cu" not in text
        assert "git" + "@" not in text
        assert "s" + "sh://" not in text
        assert "s" + "sh " not in text

    source_contract = docs["source_contract"].read_text(encoding="utf-8")
    for required in [
        "pypto-serving Source Contract H200 Evidence",
        "tmp/sources/repos/hw-native-sys/pypto-serving/python/core/server.py",
        "create_serving_app",
        "ServingServer",
        "PyptoServingSourceAsyncEngineAdapter",
        "--pypto-serving-source",
        "server: pypto-serving-source",
        "H200 Completion Evidence",
        "route: /v1/completions",
        "object: text_completion",
        "text: NV",
        "H200 Chat Evidence",
        "route: /v1/chat/completions",
        "object: chat.completion",
        "pto_status: passed",
        "pto_launch_count: 2",
        "<remote-pto-cu>/tmp/sources/repos/hw-native-sys/pypto-serving/",
        "Source Chat Contract",
        "run_pypto_serving_source_chat_completion_fixture",
        "run_pypto_serving_source_stream_completion_fixture",
        "run_pypto_serving_source_stream_chat_completion_fixture",
        "--pypto-serving-source-chat",
        "--pypto-serving-source-stream",
        "--pypto-serving-source-chat-stream",
        "--pypto-serving-vllm-compat",
        "--kernel-launcher gluon-moe-expert",
        "--kernel-launcher persistent-moe-dispatch-combine",
        "Generated Gluon MoE Expert Launch Contract",
        "Persistent MoE Dispatch/Combine Launch Contract",
        "H200 Persistent MoE Source-Route Matrix",
        "H200 Generated Gluon MoE Source-Route Matrix",
        "pto-cu: b66e1ece",
        "pypto-serving source clone: 0b0d8a0",
        "remote Git refresh: not used",
        "python environment: <remote-gluon-venv>",
        "torch: 2.8.0+cu128",
        "torch CUDA: 12.8",
        "triton: 3.7.1",
        "--pypto-serving-vllm-compat --kernel-launcher gluon-moe-expert",
        "comparison_baseline: vllm-openai-compatible-deepseek",
        "fixture: completions",
        "fixture: chat_completions",
        "fixture: stream_completions",
        "fixture: stream_chat_completions",
        "usage_keys: []",
        "stderr caveat: no Torch/NumPy compatibility warning was printed",
        "object: text_completion",
        "text: N",
        "pto_token_ids: [1]",
        "kernel_name: moe_expert_affine_f32",
        "launch_kind: gluon-moe-expert",
        "launch_kind: persistent-moe-dispatch-combine",
        "phase: prefill",
        "dag_shape: graph_descriptor_moe_dispatch_combine",
        "shape.n: 16",
        "completed_count: 5",
        "max_abs_error: 0.0",
        "scheduler_error_summary: {count: 0, code: 0, task_id: 0}",
        "device_scheduler_errors: {count: 0, code: 0, task_id: 0}",
        "fanin_remaining: [0, 0, 0, 0, 0]",
        "task_body_digest.source_sha256",
        "7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f",
        "source_sha256",
        "max_abs_error: 1.1920928955078125e-07",
        "H200 Persistent MoE Aggregate Compatibility Evidence",
        "same-process repeated `ctypes.CDLL(...)` runtime binding crash",
        "exit status: 0",
        "status: passed",
        "not fused MoE dispatch/combine serving readiness",
        "assistant_message: {role: assistant, content: N}",
        "assistant_message: {role: assistant, content: NV}",
        "Source Streaming Contract",
        "stream: true",
        "event_count: 3",
        "chunk_count: 2",
        "event_count: 2",
        "chunk_count: 1",
        "done_seen: true",
        "assembled_text: NV",
        "assembled_text: N",
        "assistant_deltas: [N, V]",
        "assistant_deltas: [N]",
        "assembled_assistant_text: N",
        "terminal `[DONE]`",
        "H200 Streaming Completion Evidence",
        "--pypto-serving-source-stream --require-cuda",
        "H200 Streaming Chat Evidence",
        "--pypto-serving-source-chat-stream --require-cuda",
        "vLLM Compatibility Contract",
        "route, HTTP 200, object/model shape, text or assistant delta",
        "usage presence for non-streaming responses",
        "terminal `[DONE]` presence for streaming responses",
        "not tokenizer semantics",
        "not real DeepSeek weights",
        "not simpler-nv/vLLM kernel integration evidence",
    ]:
        assert required in source_contract

    topk_launcher = docs["topk_launcher"].read_text(encoding="utf-8")
    for required in [
        "pypto-serving Top-K Sampling Launcher H200 Evidence",
        "serving-route launcher/probe for a generated Top-K sampling correctness gate",
        "examples/cuda/pypto_serving_nv_shim.py",
        "examples/cuda/gluon_topk_sampling.py",
        "run_topk_sampling_correctness",
        "create_generated_gluon_topk_sampling_launcher",
        "--kernel-launcher gluon-topk-sampling",
        "--pypto-serving-source",
        "--require-cuda",
        "kernel_name: topk_sampling_f32",
        "launch_kind: gluon-topk-sampling",
        "shape: {rows: 3, vocab: 16, k: 5}",
        "validation.max_abs_error: 0.0",
        "validation.values_match: true",
        "validation.indices_match: true",
        "source_sha256",
        "artifact.source_path",
        "artifact.manifest_path",
        "H200 Source-Route Evidence",
        "server: pypto-serving-source",
        "route: /v1/completions",
        "status: passed",
        "pto_status: passed",
        "pto_launch_count: 1",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "remote Git refresh: not required",
        "not FlashInfer integration",
        "not tokenizer semantics",
        "not generated text correctness",
        "not DeepSeek serving readiness",
        "not production serving evidence",
    ]:
        assert required in topk_launcher
    assert UCCL_PRIVATE_PATH_RE.search(topk_launcher) is None
    assert "/" + "home/" not in topk_launcher

    topp_launcher = docs["topp_launcher"].read_text(encoding="utf-8")
    for required in [
        "pypto-serving Top-P Sampling Launcher H200 Evidence",
        "serving-route launcher/probe for a generated Top-P sampling correctness gate",
        "examples/cuda/pypto_serving_nv_shim.py",
        "examples/cuda/gluon_topp_sampling.py",
        "run_topp_sampling_correctness",
        "create_generated_gluon_topp_sampling_launcher",
        "--kernel-launcher gluon-topp-sampling",
        "--pypto-serving-source",
        "--require-cuda",
        "kernel_name: topp_sampling_f32",
        "launch_kind: gluon-topp-sampling",
        "shape: {rows: 3, vocab: 16, max_k: 6}",
        "p: 0.80",
        "validation.max_abs_error",
        "validation.max_cumulative_probability_error",
        "validation.values_match: true",
        "validation.indices_match: true",
        "validation.selected_counts_match: true",
        "validation.cumulative_probabilities_match: true",
        "source_sha256",
        "artifact.source_path",
        "artifact.manifest_path",
        "H200 Source-Route Evidence",
        "server: pypto-serving-source",
        "route: /v1/completions",
        "status: passed",
        "pto_status: passed",
        "pto_launch_count: 1",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "remote Git refresh: not required",
        "not FlashInfer integration",
        "not tokenizer semantics",
        "not generated text correctness",
        "not DeepSeek serving readiness",
        "not production serving evidence",
    ]:
        assert required in topp_launcher
    assert UCCL_PRIVATE_PATH_RE.search(topp_launcher) is None
    assert "/" + "home/" not in topp_launcher

    minp_launcher = docs["minp_launcher"].read_text(encoding="utf-8")
    for required in [
        "pypto-serving Min-P Sampling Launcher H200 Evidence",
        "serving-route launcher/probe for a generated Min-P sampling correctness gate",
        "examples/cuda/pypto_serving_nv_shim.py",
        "examples/cuda/gluon_minp_sampling.py",
        "run_minp_sampling_correctness",
        "create_generated_gluon_minp_sampling_launcher",
        "--kernel-launcher gluon-minp-sampling",
        "--pypto-serving-source",
        "--require-cuda",
        "kernel_name: minp_sampling_f32",
        "launch_kind: gluon-minp-sampling",
        "shape: {rows: 3, vocab: 16, max_k: 6}",
        "min_p: 0.5",
        "validation.max_abs_error",
        "validation.values_match: true",
        "validation.indices_match: true",
        "validation.selected_counts_match: true",
        "source_sha256",
        "artifact.source_path",
        "artifact.manifest_path",
        "H200 Source-Route Evidence",
        "server: pypto-serving-source",
        "route: /v1/completions",
        "status: passed",
        "pto_status: passed",
        "pto_launch_count: 1",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "remote Git refresh: not required",
        "not FlashInfer integration",
        "not tokenizer semantics",
        "not generated text correctness",
        "not DeepSeek serving readiness",
        "not production serving evidence",
    ]:
        assert required in minp_launcher
    assert UCCL_PRIVATE_PATH_RE.search(minp_launcher) is None
    assert "/" + "home/" not in minp_launcher

    speculative_launcher = docs["speculative_launcher"].read_text(encoding="utf-8")
    for required in [
        "pypto-serving Speculative Decoding Launcher H200 Evidence",
        "serving-route launcher/probe for a generated speculative decoding accept/reject correctness gate",
        "examples/cuda/pypto_serving_nv_shim.py",
        "examples/cuda/gluon_speculative_decoding.py",
        "run_speculative_decoding_correctness",
        "create_generated_gluon_speculative_decoding_launcher",
        "--kernel-launcher gluon-speculative-decoding",
        "--pypto-serving-source",
        "--require-cuda",
        "kernel_name: speculative_accept_f32",
        "launch_kind: gluon-speculative-decoding",
        "shape: {rows: 3, max_draft: 6}",
        "validation.accepted_token_ids_match: true",
        "validation.accept_mask_match: true",
        "validation.accepted_counts_match: true",
        "source_sha256",
        "artifact.source_path",
        "artifact.manifest_path",
        "H200 Source-Route Evidence",
        "server: pypto-serving-source",
        "route: /v1/completions",
        "status: passed",
        "pto_status: passed",
        "pto_launch_count: 1",
        "REMOTE_PTO_CU=<remote-pto-cu>",
        "remote Git refresh: not required",
        "not FlashInfer integration",
        "not tokenizer semantics",
        "not generated text correctness",
        "not DeepSeek serving readiness",
        "not production serving evidence",
        "not throughput/latency evidence",
    ]:
        assert required in speculative_launcher
    assert UCCL_PRIVATE_PATH_RE.search(speculative_launcher) is None
    assert "/" + "home/" not in speculative_launcher

    example_text = example.read_text(encoding="utf-8")
    for required in [
        "class SimplerNvExecutor",
        "class SimplerNvModelRunner",
        "class SyntheticPyptoServingEngine",
        "PyptoServingSourceAsyncEngineAdapter",
        "run_synthetic_serving_request",
        "run_synthetic_openai_completion",
        "run_synthetic_openai_chat_completion",
        "run_synthetic_http_completion_fixture",
        "run_pypto_serving_source_completion_fixture",
        "run_pypto_serving_source_chat_completion_fixture",
        "run_pypto_serving_source_stream_completion_fixture",
        "run_pypto_serving_source_stream_chat_completion_fixture",
        "create_openai_chat_completion",
        "--openai-chat-completion",
        "/v1/chat/completions",
        "--pypto-serving-source",
        "--pypto-serving-source-chat",
        "--pypto-serving-source-stream",
        "--pypto-serving-source-chat-stream",
        "--pypto-serving-vllm-compat",
        "run_pypto_serving_vllm_compat_fixture",
        "create_generated_gluon_moe_launcher",
        "create_generated_gluon_topk_sampling_launcher",
        "create_generated_gluon_topp_sampling_launcher",
        "create_generated_gluon_minp_sampling_launcher",
        "create_generated_gluon_speculative_decoding_launcher",
        "create_persistent_moe_dispatch_combine_launcher",
        "run_topk_sampling_correctness",
        "run_topp_sampling_correctness",
        "run_minp_sampling_correctness",
        "run_speculative_decoding_correctness",
        "run_moe_dispatch_combine",
        "--kernel-launcher",
        "gluon-moe-expert",
        "gluon-topk-sampling",
        "gluon-topp-sampling",
        "gluon-minp-sampling",
        "gluon-speculative-decoding",
        "persistent-moe-dispatch-combine",
        "moe_expert_affine_f32",
        "topk_sampling_f32",
        "topp_sampling_f32",
        "minp_sampling_f32",
        "speculative_accept_f32",
    ]:
        assert required in example_text

    test_text = tests.read_text(encoding="utf-8")
    for required in [
        "test_synthetic_serving_request_uses_simpler_nv_executor_boundary",
        "test_engine_chat_completion_uses_messages_and_openai_shape",
        "test_synthetic_fastapi_app_serves_health_models_and_completions",
        "test_synthetic_fastapi_app_serves_chat_completions",
        "test_openai_chat_completion_cli_mode_outputs_chat_json",
        "test_pypto_serving_source_chat_fixture_uses_real_chat_route",
        "test_pypto_serving_source_chat_cli_mode_outputs_contract_json",
        "test_pypto_serving_source_stream_fixture_uses_real_completion_route",
        "test_pypto_serving_source_stream_completion_cli_outputs_summary_json",
        "test_pypto_serving_source_stream_chat_fixture_uses_real_chat_route",
        "test_pypto_serving_source_stream_chat_cli_outputs_summary_json",
        "test_pypto_serving_source_chat_route_fixture_is_documented",
        "test_pypto_serving_vllm_compat_summary_records_structural_fields",
        "test_pypto_serving_vllm_compat_cli_outputs_summary_json",
        "test_pypto_serving_source_server_contract_uses_real_routes",
        "test_pypto_serving_source_cli_mode_outputs_contract_json",
        "test_generated_gluon_moe_launcher_records_review_safe_metadata",
        "test_generated_gluon_topk_sampling_launcher_records_review_safe_metadata",
        "test_generated_gluon_topp_sampling_launcher_records_review_safe_metadata",
        "test_generated_gluon_minp_sampling_launcher_records_review_safe_metadata",
        "test_generated_gluon_speculative_decoding_launcher_records_review_safe_metadata",
        "test_generated_launcher_can_run_through_source_route_fixtures",
        "test_topk_sampling_launcher_can_run_through_source_route_fixtures",
        "test_topp_sampling_launcher_can_run_through_source_route_fixtures",
        "test_minp_sampling_launcher_can_run_through_source_route_fixtures",
        "test_speculative_decoding_launcher_can_run_through_source_route_fixtures",
        "test_persistent_moe_dispatch_combine_launcher_records_review_safe_metadata",
        "test_persistent_launcher_can_run_through_source_route_fixtures",
        "test_generated_kernel_cli_mode_outputs_launch_metadata",
        "test_topk_sampling_cli_mode_outputs_launch_metadata",
        "test_topp_sampling_cli_mode_outputs_launch_metadata",
        "test_minp_sampling_cli_mode_outputs_launch_metadata",
        "test_speculative_decoding_cli_mode_outputs_launch_metadata",
        "test_persistent_moe_cli_mode_outputs_launch_metadata",
        "test_generated_launcher_can_run_through_vllm_compat_summary",
        "test_topk_sampling_launcher_can_run_through_vllm_compat_summary",
        "test_topp_sampling_launcher_can_run_through_vllm_compat_summary",
        "test_minp_sampling_launcher_can_run_through_vllm_compat_summary",
        "test_speculative_decoding_launcher_can_run_through_vllm_compat_summary",
        "test_default_launcher_selection_uses_cuda_seed_and_add_op",
    ]:
        assert required in test_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "pypto_serving_nv_shim.py" in readme_text
    assert "pypto_serving_source_contract_h200.md" in readme_text
    assert "--openai-chat-completion" in readme_text
    assert "--pypto-serving-source-chat" in readme_text
    assert "--pypto-serving-source-stream" in readme_text
    assert "--pypto-serving-source-chat-stream" in readme_text
    assert "--pypto-serving-vllm-compat" in readme_text
    assert "--kernel-launcher gluon-moe-expert" in readme_text
    assert "--kernel-launcher gluon-topk-sampling" in readme_text
    assert "--kernel-launcher gluon-topp-sampling" in readme_text
    assert "--kernel-launcher gluon-minp-sampling" in readme_text
    assert "--kernel-launcher gluon-speculative-decoding" in readme_text
    assert "--kernel-launcher persistent-moe-dispatch-combine" in readme_text
    assert "moe_expert_affine_f32" in readme_text
    assert "topk_sampling_f32" in readme_text
    assert "topp_sampling_f32" in readme_text
    assert "minp_sampling_f32" in readme_text
    assert "speculative_accept_f32" in readme_text
    assert "not production fused MoE dispatch/combine serving" in readme_text
    assert "OpenAI-compatible structural fields" in readme_text
    assert "/v1/chat/completions" in readme_text
