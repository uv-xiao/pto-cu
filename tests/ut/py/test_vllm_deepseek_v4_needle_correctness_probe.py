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
    assert "stop" not in request
    assert "stop" not in safe["limits"]
    assert "prompt" not in safe
    assert "payload" not in safe


def test_needle_request_omits_stop_control_by_default(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(
        probe,
        "build_synthetic_needle_prompt",
        lambda **_: {
            "prompt": "NEEDLE_ANSWER: PTO_NEEDLE_256K_CONTEXT_OK_28143",
            "accounting": {
                "target_prompt_tokens": 255800,
                "actual_prompt_tokens": 255799,
                "tokenizer_accounting": "test",
                "prompt_chars": 44,
                "prompt_unit_chars": 1,
                "needle_occurrences": 1,
            },
        },
    )

    request = probe.build_needle_request()

    assert "stop" not in request["payload"]
    assert "stop" not in request["limits"]


def test_stop_sequence_is_carried_in_request_and_review_safe_limits(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(
        probe,
        "build_synthetic_needle_prompt",
        lambda **_: {
            "prompt": "NEEDLE_ANSWER: PTO_NEEDLE_256K_CONTEXT_OK_28143",
            "accounting": {
                "target_prompt_tokens": 255800,
                "actual_prompt_tokens": 255799,
                "tokenizer_accounting": "test",
                "prompt_chars": 44,
                "prompt_unit_chars": 1,
                "needle_occurrences": 1,
            },
        },
    )

    request = probe.build_needle_request(stop_sequences=["\n```"])
    safe = probe.review_safe_request(request)

    assert request["payload"]["stop"] == ["\n```"]
    assert request["limits"]["stop"] == ["\n```"]
    assert safe["limits"]["stop"] == ["\n```"]
    assert safe["prompt_text_recorded"] is False
    assert safe["payload_recorded"] is False
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
        match_mode="contains",
    )

    assert result["status"] == "passed"
    assert result["checks"]["expected_answer_contained"] == "passed"
    assert result["expected_answer"] == expected
    assert result["generated_text"] == f"\n{expected}\n"
    assert result["generated_text_recorded"] == "short_synthetic_output"
    assert result["usage"]["prompt_tokens"] == 255800


def test_validate_needle_correctness_exact_mode_normalizes_narrowly():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(
        expected_answer=expected,
        match_mode="exact",
    )

    result = probe.validate_needle_correctness_response(
        completion_payload(f"\n```text\n{expected}\n```\n"),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
        match_mode="exact",
    )

    assert result["status"] == "passed"
    assert result["match_mode"] == "exact"
    assert result["checks"]["expected_answer_exact"] == "passed"
    assert result["normalization"] == (
        "strip leading/trailing whitespace, then strip one surrounding "
        "Markdown code fence when the whole output is fenced"
    )
    assert result["normalized_generated_text"] == expected


def test_validate_needle_correctness_exact_mode_rejects_extra_wording():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(
        expected_answer=expected,
        match_mode="exact",
    )

    result = probe.validate_needle_correctness_response(
        completion_payload(
            "\n"
            f"{expected}\n\n"
            "The model correctly extracts the needle value from the context."
        ),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
        match_mode="exact",
    )

    assert result["status"] == "failed"
    assert result["checks"]["expected_answer_exact"] == "failed"
    assert result["failure"]["category"] == "needle_expected_answer_not_exact"
    assert result["normalized_generated_text"].startswith(expected)
    assert "correctly extracts" in result["normalized_generated_text"]


def test_validate_needle_correctness_exact_mode_rejects_unmatched_trailing_fence():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(
        expected_answer=expected,
        match_mode="exact",
    )

    result = probe.validate_needle_correctness_response(
        completion_payload(f"\n{expected}\n```\n"),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
        match_mode="exact",
    )

    assert result["status"] == "failed"
    assert result["checks"]["expected_answer_exact"] == "failed"
    assert result["failure"]["category"] == "needle_expected_answer_not_exact"
    assert result["normalized_generated_text"] == f"{expected}\n```"


def test_validate_needle_correctness_contains_mode_still_allows_extra_wording():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(
        expected_answer=expected,
        match_mode="contains",
    )

    result = probe.validate_needle_correctness_response(
        completion_payload(f"\n{expected}\n\nExtra wording remains allowed here."),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
        match_mode="contains",
    )

    assert result["status"] == "passed"
    assert result["match_mode"] == "contains"
    assert result["checks"]["expected_answer_contained"] == "passed"


def test_validate_needle_correctness_fails_without_weakening_assertion():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(expected_answer=expected)

    result = probe.validate_needle_correctness_response(
        completion_payload("I could not find the requested marker."),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=expected,
        match_mode="contains",
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
            "--match-mode",
            "exact",
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
    assert payload["request"]["limits"]["match_mode"] == "exact"
    assert payload["request"]["limits"]["normalization"] == (
        "strip leading/trailing whitespace, then strip one surrounding "
        "Markdown code fence when the whole output is fenced"
    )
    assert payload["request"]["limits"]["expected_answer"] == (
        "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    )
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "stop" not in payload["request"]["limits"]
    assert "prompt" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "text_" + "sha256" not in result.stdout
    assert "token_ids" not in result.stdout
    assert "raw " + "filler prompt" not in result.stdout
    assert "normalized generated output equals the expected needle answer in exact mode" in (
        payload["contract_checks"]
    )
    assert not any(
        claim == "not generated-text correctness evidence"
        for claim in payload["non_claims"]
    )


def test_dry_run_cli_output_records_configured_stop_sequence_safely():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28145",
            "--target-prompt-tokens",
            "255800",
            "--max-model-len",
            "262144",
            "--max-tokens",
            "64",
            "--match-mode",
            "exact",
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
    assert payload["server_port"] == 28145
    assert payload["request"]["limits"]["match_mode"] == "exact"
    assert payload["request"]["limits"]["stop"] == ["\n```"]
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "payload" not in payload["request"]
    assert "raw request payload" in result.stdout
