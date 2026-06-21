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
]
UCCL_PRIVATE_PATH_RE = re.compile(
    r"/" + "home/"
    r"|/" + "Users/"
    r"|/" + "tmp/pto-cu"
    r"|/" + "tmp/uccl-"
    r"|" + "uv" + "xiao"
    r"|" + "bizhao" + "h200"
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
        "28158",
        "stream_options.include_usage=true",
        "PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28158",
    ]

    assert example.is_file()
    for text in required_text:
        assert text in readme
        assert text in readiness
        assert text in evidence
    assert "returned streaming usage" in readme
    assert "strict exact output matching" in readme
    assert "PROBE_EXIT_STATUS=2" in evidence
    assert (
        "failure_category: chat_needle_stream_prompt_token_mismatch"
        in evidence
    )
    assert "server_port: 28158" in evidence
    assert "endpoint: /v1/chat/completions" in evidence
    assert "stream: true" in evidence
    assert "event_count: 22" in evidence
    assert "content_chunk_count: 18" in evidence
    assert "done_seen: true" in evidence
    assert "finish_reason: stop" in evidence
    assert "normalized_output_equals_expected: true" in evidence
    assert "expected_answer_exact: passed" in evidence
    assert "usage_presence: passed" in evidence
    assert "usage_prompt_tokens_match: not_available" in evidence
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
    assert "chat_needle_stream_prompt_token_mismatch" in readme
    assert (
        "local_only_vllm_chat_256k_needle_stream_usage_contract: failed"
        in readiness
    )
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


def test_status_rollup_records_current_deepseek_serving_boundary():
    status_text = (DOC_ROOT / "status.md").read_text(encoding="utf-8")

    for required in [
        "docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_repeat_probe.md",
        "docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_truncated_failure_probe.md",
        "docs/in_progress/nvidia_backend/pypto_serving_source_contract_h200.md",
        "not a transport/server failure",
        "not simpler-nv/vLLM kernel integration evidence",
    ]:
        assert required in status_text


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
        "pto_status: passed",
        "pto_launch_count: 2",
        "<remote-pto-cu>/tmp/sources/repos/hw-native-sys/pypto-serving/",
        "Current Chat Source Limitation",
        "tmp/sources/repos/hw-native-sys/pypto-serving is absent",
        "/v1/chat/completions",
        "source-route chat fixture",
    ]:
        assert required in source_contract

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
        "create_openai_chat_completion",
        "--openai-chat-completion",
        "/v1/chat/completions",
        "--pypto-serving-source",
    ]:
        assert required in example_text

    test_text = tests.read_text(encoding="utf-8")
    for required in [
        "test_synthetic_serving_request_uses_simpler_nv_executor_boundary",
        "test_engine_chat_completion_uses_messages_and_openai_shape",
        "test_synthetic_fastapi_app_serves_health_models_and_completions",
        "test_synthetic_fastapi_app_serves_chat_completions",
        "test_openai_chat_completion_cli_mode_outputs_chat_json",
        "test_pypto_serving_source_chat_route_limitation_is_documented",
        "test_pypto_serving_source_server_contract_uses_real_routes",
        "test_pypto_serving_source_cli_mode_outputs_contract_json",
    ]:
        assert required in test_text

    readme_text = readme.read_text(encoding="utf-8")
    assert "pypto_serving_nv_shim.py" in readme_text
    assert "pypto_serving_source_contract_h200.md" in readme_text
    assert "--openai-chat-completion" in readme_text
    assert "/v1/chat/completions" in readme_text
