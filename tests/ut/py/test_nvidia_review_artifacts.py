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
        in_progress_root / "deepseek_v4_flash_serving_readiness.md"
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
    assert "text_sha256" not in evidence


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
    assert "text_sha256" not in evidence


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
    assert "text_sha256" not in evidence


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
    assert "text_sha256" not in evidence


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
    assert "text_sha256" not in result.stdout
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
    assert "text_sha256" not in result.stdout
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


def test_active_review_surfaces_do_not_reference_removed_eval_scripts():
    for path in ACTIVE_REVIEW_SURFACES:
        text = path.read_text(encoding="utf-8")
        assert REMOVED_EVAL_SCRIPT_RE.search(text) is None, path


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
