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


def test_synthetic_prompt_records_requested_needle_position(monkeypatch):
    probe = load_probe_module()

    class FakeTokenizer:
        pass

    monkeypatch.setattr(probe, "_load_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(
        probe,
        "_count_tokens",
        lambda _tokenizer, text: text.count(probe.DEFAULT_PROMPT_UNIT) + 12,
    )

    early = probe.build_synthetic_needle_prompt(
        artifact_dir=Path("unused"),
        target_prompt_tokens=112,
        expected_answer="NEEDLE",
        needle_position="early",
    )
    middle = probe.build_synthetic_needle_prompt(
        artifact_dir=Path("unused"),
        target_prompt_tokens=112,
        expected_answer="NEEDLE",
        needle_position="middle",
    )
    late = probe.build_synthetic_needle_prompt(
        artifact_dir=Path("unused"),
        target_prompt_tokens=112,
        expected_answer="NEEDLE",
        needle_position="late",
    )

    assert early["accounting"]["needle_position"] == "early"
    assert middle["accounting"]["needle_position"] == "middle"
    assert late["accounting"]["needle_position"] == "late"
    assert early["accounting"]["filler_units_before_needle"] < middle["accounting"][
        "filler_units_before_needle"
    ]
    assert middle["accounting"]["filler_units_before_needle"] < late["accounting"][
        "filler_units_before_needle"
    ]
    assert early["accounting"]["filler_units_after_needle"] > middle["accounting"][
        "filler_units_after_needle"
    ]
    assert middle["accounting"]["filler_units_after_needle"] > late["accounting"][
        "filler_units_after_needle"
    ]


def test_planned_needle_request_records_requested_position():
    probe = load_probe_module()

    request = probe.build_planned_needle_request(needle_position="late")

    assert request["limits"]["needle_position"] == "late"


def test_build_needle_request_records_position_limits(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(
        probe,
        "build_synthetic_needle_prompt",
        lambda **kwargs: {
            "prompt": "NEEDLE_ANSWER: PTO_NEEDLE_256K_CONTEXT_OK_28143",
            "accounting": {
                "target_prompt_tokens": kwargs["target_prompt_tokens"],
                "actual_prompt_tokens": 255799,
                "tokenizer_accounting": "test",
                "prompt_chars": 44,
                "prompt_unit_chars": 1,
                "needle_occurrences": 1,
                "needle_position": kwargs["needle_position"],
                "filler_units_before_needle": 9,
                "filler_units_after_needle": 91,
            },
        },
    )

    request = probe.build_needle_request(needle_position="early")

    assert request["limits"]["needle_position"] == "early"
    assert request["limits"]["filler_units_before_needle"] == 9
    assert request["limits"]["filler_units_after_needle"] == 91
    assert "prompt" not in probe.review_safe_request(request)
    assert "payload" not in probe.review_safe_request(request)


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


def test_planned_needle_request_rejects_invalid_position():
    probe = load_probe_module()

    try:
        probe.build_planned_needle_request(needle_position="near")
    except ValueError as exc:
        assert "needle_position" in str(exc)
    else:
        raise AssertionError("expected needle position validation failure")


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
    assert payload["request"]["limits"]["needle_position"] == "middle"
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


def test_dry_run_cli_output_records_requested_needle_position_safely():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28148",
            "--target-prompt-tokens",
            "255800",
            "--max-model-len",
            "262144",
            "--max-tokens",
            "64",
            "--match-mode",
            "exact",
            "--needle-position",
            "late",
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
    assert payload["request"]["limits"]["needle_position"] == "late"
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "prompt" not in payload["request"]
    assert "payload" not in payload["request"]


def test_dry_run_cli_output_records_position_sweep_safely():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28148",
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
            "--needle-position-sweep",
            "early,middle,late",
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
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["sweep"]["positions_planned"] == ["early", "middle", "late"]
    assert payload["sweep"]["attempts_planned"] == 3
    assert payload["sweep"]["attempt_summaries_record"] == "review_safe_only"
    assert [request["limits"]["needle_position"] for request in payload["sweep"]["requests"]] == [
        "early",
        "middle",
        "late",
    ]
    assert all(
        request["prompt_text_recorded"] is False
        and request["payload_recorded"] is False
        and "prompt" not in request
        and "payload" not in request
        for request in payload["sweep"]["requests"]
    )
    assert "completion" not in payload


def test_parse_args_rejects_duplicate_sweep_positions():
    probe = load_probe_module()

    try:
        probe.parse_args(["--needle-position-sweep", "early,middle,early"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected duplicate sweep validation failure")


def test_parse_args_rejects_missing_sweep_position():
    probe = load_probe_module()

    try:
        probe.parse_args(["--needle-position-sweep", "early,,late"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected missing sweep position validation failure")


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


def test_dry_run_cli_output_preserves_single_request_shape_by_default():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28146",
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
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert "repeat" not in payload
    assert "repeat_count" not in payload
    assert "attempts" not in payload
    assert "completion" not in payload


def test_parse_args_rejects_invalid_repeat_count():
    probe = load_probe_module()

    try:
        probe.parse_args(["--repeat-count", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected repeat count validation failure")


def test_repeat_aggregation_fails_if_any_attempt_fails():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    request = probe.build_planned_needle_request(
        expected_answer=expected,
        match_mode="exact",
        stop_sequences=["\n```"],
    )
    attempts = [
        {
            "status": "passed",
            "http_status": 200,
            "validation": {
                "checks": {"expected_answer_exact": "passed"},
            },
            "observation": {
                "finish_reason": "stop",
                "text_length_chars": 33,
                "usage": {
                    "prompt_tokens": 255799,
                    "completion_tokens": 17,
                    "total_tokens": 255816,
                },
                "generated_text": expected,
                "normalized_generated_text": expected,
            },
        },
        {
            "status": "failed",
            "http_status": 200,
            "failure": {
                "category": "needle_expected_answer_not_exact",
                "message": "normalized generated output did not exactly equal expected",
            },
            "validation": {
                "checks": {"expected_answer_exact": "failed"},
            },
            "observation": {
                "finish_reason": "stop",
                "text_length_chars": 37,
                "usage": {
                    "prompt_tokens": 255799,
                    "completion_tokens": 18,
                    "total_tokens": 255817,
                },
                "generated_text": f"{expected}\n```",
                "normalized_generated_text": f"{expected}\n```",
            },
        },
    ]

    aggregate = probe.aggregate_repeat_attempts(attempts, request=request)

    assert aggregate["status"] == "failed"
    assert aggregate["repeat_count"] == 2
    assert aggregate["passed_attempts"] == 1
    assert aggregate["failed_attempts"] == 1
    assert aggregate["failure"]["category"] == "needle_expected_answer_not_exact"
    assert aggregate["attempts"][0]["attempt_index"] == 1
    assert aggregate["attempts"][1]["attempt_index"] == 2
    assert aggregate["attempts"][1]["exact_check"] == "failed"
    assert "generated_text" not in aggregate["attempts"][0]
    assert "normalized_generated_text" not in aggregate["attempts"][0]
    assert "payload" not in aggregate["attempts"][0]
    assert "prompt" not in aggregate["attempts"][0]
    assert expected not in json.dumps(aggregate["attempts"])


def test_repeat_aggregation_fails_when_attempt_count_is_incomplete():
    probe = load_probe_module()
    request = probe.build_planned_needle_request(match_mode="exact")
    attempts = [
        {
            "status": "passed",
            "http_status": 200,
            "validation": {
                "checks": {"expected_answer_exact": "passed"},
                "finish_reason": "stop",
                "text_length_chars": 33,
                "usage": {
                    "prompt_tokens": 255799,
                    "completion_tokens": 17,
                    "total_tokens": 255816,
                },
            },
        },
    ]

    aggregate = probe.aggregate_repeat_attempts(
        attempts,
        request=request,
        expected_count=3,
    )

    assert aggregate["status"] == "failed"
    assert aggregate["repeat_count"] == 3
    assert aggregate["attempts_completed"] == 1
    assert aggregate["failure"]["category"] == "needle_repeat_incomplete"


def test_repeat_summary_uses_review_safe_fields_only():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    completion = {
        "status": "passed",
        "endpoint": "/v1/completions",
        "url": "http://127.0.0.1:28146/v1/completions",
        "http_status": 200,
        "request_limits": {
            "target_prompt_tokens": 255800,
            "actual_prompt_tokens": 255799,
            "max_tokens": 64,
            "match_mode": "exact",
        },
        "validation": {
            "checks": {
                "expected_answer_exact": "passed",
                "usage_prompt_tokens_match": "passed",
                "usage_completion_bound": "passed",
            },
        },
        "observation": {
            "expected_answer": expected,
            "match_mode": "exact",
            "finish_reason": "stop",
            "text_length_chars": 33,
            "generated_text": expected,
            "normalized_generated_text": expected,
            "usage": {
                "prompt_tokens": 255799,
                "completion_tokens": 17,
                "total_tokens": 255816,
            },
        },
    }

    summary = probe.review_safe_repeat_attempt_summary(completion, attempt_index=3)

    assert summary == {
        "attempt_index": 3,
        "status": "passed",
        "http_status": 200,
        "finish_reason": "stop",
        "generated_text_length_chars": 33,
        "exact_check": "passed",
        "usage": {
            "prompt_tokens": 255799,
            "completion_tokens": 17,
            "total_tokens": 255816,
        },
    }
    assert expected not in json.dumps(summary)


def test_sweep_aggregation_fails_if_any_position_fails():
    probe = load_probe_module()
    expected = "PTO_NEEDLE_256K_CONTEXT_OK_28143"
    requests = {
        position: probe.build_planned_needle_request(
            expected_answer=expected,
            match_mode="exact",
            stop_sequences=["\n```"],
            needle_position=position,
        )
        for position in ("early", "middle", "late")
    }
    completions = [
        {
            "status": "passed",
            "http_status": 200,
            "request_limits": requests["early"]["limits"],
            "validation": {"checks": {"expected_answer_exact": "passed"}},
            "observation": {
                "finish_reason": "stop",
                "text_length_chars": 33,
                "usage": {
                    "prompt_tokens": 255799,
                    "completion_tokens": 17,
                    "total_tokens": 255816,
                },
                "generated_text": expected,
                "normalized_generated_text": expected,
            },
        },
        {
            "status": "failed",
            "http_status": 200,
            "request_limits": requests["middle"]["limits"],
            "failure": {
                "category": "needle_expected_answer_not_exact",
                "message": "normalized generated output did not exactly equal expected",
            },
            "validation": {"checks": {"expected_answer_exact": "failed"}},
            "observation": {
                "finish_reason": "stop",
                "text_length_chars": 37,
                "usage": {
                    "prompt_tokens": 255799,
                    "completion_tokens": 18,
                    "total_tokens": 255817,
                },
                "generated_text": f"{expected}\n```",
                "normalized_generated_text": f"{expected}\n```",
            },
        },
    ]

    aggregate = probe.aggregate_position_sweep_attempts(
        completions,
        requests=requests,
        expected_positions=["early", "middle", "late"],
    )

    assert aggregate["status"] == "failed"
    assert aggregate["positions_requested"] == ["early", "middle", "late"]
    assert aggregate["positions_completed"] == ["early", "middle"]
    assert aggregate["passed_attempts"] == 1
    assert aggregate["failed_attempts"] == 1
    assert aggregate["failure"]["category"] == "needle_expected_answer_not_exact"
    assert aggregate["attempts"][0]["needle_position"] == "early"
    assert aggregate["attempts"][1]["needle_position"] == "middle"
    assert aggregate["attempts"][1]["exact_check"] == "failed"
    assert "generated_text" not in aggregate["attempts"][0]
    assert "normalized_generated_text" not in aggregate["attempts"][0]
    assert expected not in json.dumps(aggregate["attempts"])


def test_sweep_aggregation_fails_when_position_is_incomplete():
    probe = load_probe_module()
    requests = {
        position: probe.build_planned_needle_request(
            match_mode="exact",
            needle_position=position,
        )
        for position in ("early", "middle", "late")
    }
    completions = [
        {
            "status": "passed",
            "http_status": 200,
            "request_limits": requests["early"]["limits"],
            "validation": {
                "checks": {"expected_answer_exact": "passed"},
                "finish_reason": "stop",
                "text_length_chars": 33,
                "usage": {
                    "prompt_tokens": 255799,
                    "completion_tokens": 17,
                    "total_tokens": 255816,
                },
            },
        },
    ]

    aggregate = probe.aggregate_position_sweep_attempts(
        completions,
        requests=requests,
        expected_positions=["early", "middle", "late"],
    )

    assert aggregate["status"] == "failed"
    assert aggregate["positions_completed"] == ["early"]
    assert aggregate["failure"]["category"] == "needle_position_sweep_incomplete"


def test_sweep_aggregation_fails_on_duplicate_completed_position():
    probe = load_probe_module()
    requests = {
        position: probe.build_planned_needle_request(
            match_mode="exact",
            needle_position=position,
        )
        for position in ("early", "middle", "late")
    }
    completions = [
        {
            "status": "passed",
            "http_status": 200,
            "request_limits": requests["early"]["limits"],
            "validation": {"checks": {"expected_answer_exact": "passed"}},
        },
        {
            "status": "passed",
            "http_status": 200,
            "request_limits": requests["early"]["limits"],
            "validation": {"checks": {"expected_answer_exact": "passed"}},
        },
    ]

    aggregate = probe.aggregate_position_sweep_attempts(
        completions,
        requests=requests,
        expected_positions=["early", "middle", "late"],
    )

    assert aggregate["status"] == "failed"
    assert aggregate["failure"]["category"] == "needle_position_sweep_duplicate"
