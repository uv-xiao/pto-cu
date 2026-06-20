import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_needle_correctness_probe.py"
)


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_needle_correctness_probe",
        PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def completion_payload(text, *, prompt_tokens=255800, completion_tokens=8):
    return {
        "id": "cmpl-needle-test",
        "object": "text_completion",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_planned_needle_request_is_review_safe():
    probe = load_probe_module()

    request = probe.build_planned_needle_request(
        target_prompt_tokens=255800,
        max_tokens=64,
        expected_answer="PTO_NEEDLE_256K_CONTEXT_OK_28143",
    )
    safe = probe.review_safe_request(request)

    assert request["endpoint"] == "/v1/completions"
    assert request["limits"]["target_prompt_tokens"] == 255800
    assert request["limits"]["actual_prompt_tokens"] is None
    assert request["limits"]["max_tokens"] == 64
    assert request["limits"]["expected_answer"] == "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    assert request["limits"]["needle_occurrences"] == 1
    assert request["limits"]["stream"] is False
    assert request["limits"]["echo"] is False
    assert request["limits"]["logprobs"] is False
    assert safe["prompt_text_recorded"] is False
    assert safe["payload_recorded"] is False
    assert "prompt" not in safe
    assert "payload" not in safe


def test_planned_needle_request_rejects_unbounded_generation():
    probe = load_probe_module()

    try:
        probe.build_planned_needle_request(max_tokens=65)
    except ValueError as exc:
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError("expected max_tokens validation failure")


def test_validate_needle_correctness_passes_on_exact_expected_answer():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(expected_answer=expected)

    result = probe.validate_needle_correctness_response(
        completion_payload(f"\n{expected}\n"),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
    )

    assert result["status"] == "passed"
    assert result["checks"]["expected_answer_contained"] == "passed"
    assert result["expected_answer"] == expected
    assert result["generated_text"] == f"\n{expected}\n"
    assert result["generated_text_recorded"] == "short_synthetic_output"
    assert result["usage"]["prompt_tokens"] == 255800


def test_validate_needle_correctness_fails_without_weakening_assertion():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(expected_answer=expected)

    result = probe.validate_needle_correctness_response(
        completion_payload("I could not find the requested marker."),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
    )

    assert result["status"] == "failed"
    assert result["checks"]["expected_answer_contained"] == "failed"
    assert result["failure"]["category"] == "needle_expected_answer_missing"
    assert result["expected_answer"] == expected
    assert result["generated_text"] == "I could not find the requested marker."


def test_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
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
    assert payload["request"]["limits"]["expected_answer"] == (
        "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    )
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "prompt" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "text_" + "sha256" not in result.stdout
    assert "token_ids" not in result.stdout
    assert "raw " + "filler prompt" not in result.stdout
    assert "generated output contains the exact expected needle answer" in (
        payload["contract_checks"]
    )
    assert not any(
        claim == "not generated-text correctness evidence"
        for claim in payload["non_claims"]
    )
