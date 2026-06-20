import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "examples"
    / "cuda"
    / "vllm_deepseek_v4_chat_256k_needle_stream_probe.py"
)
EXPECTED = "PTO_CHAT_NEEDLE_256K_STREAM_OK_28153"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_stream_probe",
        PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeStreamingResponse:
    def __init__(self, status, events):
        self.status = status
        self._lines = []
        for event in events:
            if isinstance(event, bytes):
                self._lines.append(event)
            elif event == "[DONE]":
                self._lines.append(b"data: [DONE]\n\n")
            else:
                self._lines.append(f"data: {json.dumps(event)}\n\n".encode())
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def readline(self):
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line


class CapturingHttpPost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def chunk(content=None, *, finish_reason=None, usage=None):
    choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if content is not None:
        choice["delta"]["content"] = content
    payload = {
        "id": "chatcmpl-stream-needle-test",
        "object": "chat.completion.chunk",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_stream_dry_run_contract_is_review_safe():
    probe = load_probe_module()

    result = probe.run_probe(dry_run=True)
    serialized = json.dumps(result)

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["server_port"] == 28153
    assert result["request"]["endpoint"] == "/v1/chat/completions"
    assert result["request"]["limits"]["target_prompt_tokens"] == 255800
    assert result["request"]["limits"]["max_tokens"] == 64
    assert result["request"]["limits"]["temperature"] == 0.0
    assert result["request"]["limits"]["top_p"] == 1.0
    assert result["request"]["limits"]["seed"] == 0
    assert result["request"]["limits"]["expected_answer"] == EXPECTED
    assert result["request"]["limits"]["match_mode"] == "exact"
    assert result["request"]["limits"]["stream"] is True
    assert result["request"]["limits"]["stop"] == ["\n```"]
    assert result["request"]["limits"]["message_count"] == 2
    assert result["request"]["limits"]["message_roles"] == ["system", "user"]
    assert result["request"]["limits"]["needle_occurrences"] == 1
    assert result["generation_attempted"] is False
    assert result["prompt_sent"] is False
    assert result["stream_events_received"] is False
    assert result["request"]["prompt_text_recorded"] is False
    assert result["request"]["payload_recorded"] is False
    assert "payload" not in result["request"]
    assert "messages" not in result["request"]
    assert "NEEDLE_ANSWER" not in serialized
    assert "Synthetic filler" not in serialized
    assert "generated_text" not in serialized
    assert "text_sha256" not in serialized
    assert "token_ids" not in serialized
    assert "logprobs" not in serialized
    assert "/" + "home/" not in serialized


def test_streaming_request_shape_targets_chat_stream(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(
        probe.base_probe,
        "build_synthetic_chat_needle_messages",
        lambda **kwargs: {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "NEEDLE_ANSWER: " + EXPECTED},
            ],
            "accounting": {
                "target_prompt_tokens": kwargs["target_prompt_tokens"],
                "actual_prompt_tokens": 255800,
                "tokenizer_accounting": "test",
                "prompt_chars": 123,
                "prompt_unit_chars": 1,
                "needle_occurrences": 1,
                "needle_position": kwargs["needle_position"],
                "filler_units_before_needle": 9,
                "filler_units_after_needle": 91,
            },
        },
    )

    request = probe.build_chat_needle_stream_request(
        max_tokens=64,
        expected_answer=EXPECTED,
    )

    assert request["endpoint"] == "/v1/chat/completions"
    assert request["payload"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert request["payload"]["max_tokens"] == 64
    assert request["payload"]["temperature"] == 0.0
    assert request["payload"]["top_p"] == 1.0
    assert request["payload"]["seed"] == 0
    assert request["payload"]["n"] == 1
    assert request["payload"]["stream"] is True
    assert request["payload"]["stop"] == ["\n```"]
    assert [message["role"] for message in request["payload"]["messages"]] == [
        "system",
        "user",
    ]
    assert EXPECTED in request["payload"]["messages"][1]["content"]
    assert "prompt" not in request["payload"]
    assert request["limits"]["actual_prompt_tokens"] == 255800
    assert request["limits"]["stream"] is True
    assert "payload" not in probe.review_safe_request(request)


def test_sse_parser_assembles_review_safe_deltas_and_usage():
    probe = load_probe_module()
    response = FakeStreamingResponse(
        200,
        [
            chunk(),
            chunk("PTO_CHAT_"),
            chunk("NEEDLE_256K_STREAM_OK_28153"),
            chunk(
                finish_reason="stop",
                usage={
                    "prompt_tokens": 255800,
                    "completion_tokens": 17,
                    "total_tokens": 255817,
                },
            ),
            "[DONE]",
        ],
    )

    parsed = probe.parse_streaming_chat_completion_response(response)
    serialized = json.dumps(parsed)

    assert parsed["status"] == "passed"
    assert parsed["event_count"] == 4
    assert parsed["content_chunk_count"] == 2
    assert parsed["done_seen"] is True
    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"] == {
        "prompt_tokens": 255800,
        "completion_tokens": 17,
        "total_tokens": 255817,
    }
    assert parsed["assembled_content"] == EXPECTED
    assert "raw prompt" not in serialized
    assert "token_ids" not in serialized
    assert "logprobs" not in serialized


def test_streaming_exact_match_passes_with_assembled_content():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_stream_request(expected_answer=EXPECTED)
    parsed = {
        "status": "passed",
        "event_count": 3,
        "content_chunk_count": 2,
        "done_seen": True,
        "finish_reason": "stop",
        "assembled_content": f"\n```text\n{EXPECTED}\n```\n",
        "usage": "not_returned",
    }

    result = probe.validate_streaming_chat_needle_response(
        parsed,
        request=request,
        expected_answer=EXPECTED,
    )

    assert result["status"] == "passed"
    assert result["checks"]["stream_events_received"] == "passed"
    assert result["checks"]["expected_answer_exact"] == "passed"
    assert result["normalized_output_equals_expected"] is True
    assert result["normalized_output_length_chars"] == len(EXPECTED)
    assert result["usage"] == "not_returned"
    assert "assembled_content" not in result
    assert "generated_text" not in result


def test_streaming_exact_match_fails_without_done_event():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_stream_request(expected_answer=EXPECTED)
    parsed = {
        "status": "passed",
        "event_count": 3,
        "content_chunk_count": 2,
        "done_seen": False,
        "finish_reason": "stop",
        "assembled_content": EXPECTED,
        "usage": "not_returned",
    }

    result = probe.validate_streaming_chat_needle_response(
        parsed,
        request=request,
        expected_answer=EXPECTED,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "chat_needle_stream_done_missing"
    assert result["checks"]["stream_done_seen"] == "failed"
    assert "assembled_content" not in result
    assert "generated_text" not in result


def test_streaming_exact_match_fails_without_finish_reason():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_stream_request(expected_answer=EXPECTED)
    parsed = {
        "status": "passed",
        "event_count": 3,
        "content_chunk_count": 2,
        "done_seen": True,
        "finish_reason": None,
        "assembled_content": EXPECTED,
        "usage": "not_returned",
    }

    result = probe.validate_streaming_chat_needle_response(
        parsed,
        request=request,
        expected_answer=EXPECTED,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == (
        "chat_needle_stream_finish_reason_missing"
    )
    assert result["checks"]["stream_done_seen"] == "passed"
    assert result["checks"]["stream_finish_reason_present"] == "failed"
    assert "assembled_content" not in result
    assert "generated_text" not in result


def test_streaming_failure_does_not_record_raw_assembled_output():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_stream_request(expected_answer=EXPECTED)
    parsed = {
        "status": "passed",
        "event_count": 2,
        "content_chunk_count": 1,
        "done_seen": True,
        "finish_reason": "stop",
        "assembled_content": EXPECTED + "\nextra text",
        "usage": "not_returned",
    }

    result = probe.validate_streaming_chat_needle_response(
        parsed,
        request=request,
        expected_answer=EXPECTED,
    )
    serialized = json.dumps(result)

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "chat_needle_stream_expected_answer_not_exact"
    assert result["checks"]["expected_answer_exact"] == "failed"
    assert result["normalized_output_equals_expected"] is False
    assert result["normalized_output_length_chars"] > len(EXPECTED)
    assert "extra text" not in serialized
    assert "assembled_content" not in serialized
    assert "generated_text" not in serialized
    assert "normalized_generated_text" not in serialized
    assert "text_sha256" not in serialized
    assert "token_ids" not in serialized
    assert "logprobs" not in serialized


def test_streaming_request_reports_http_error_without_nonstreaming_fallback():
    probe = load_probe_module()
    request = {
        **probe.build_planned_chat_needle_stream_request(expected_answer=EXPECTED),
        "payload": {"messages": [], "model": "deepseek-ai/DeepSeek-V4-Flash"},
    }
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:28153/v1/chat/completions",
        code=500,
        msg="server error",
        hdrs=None,
        fp=None,
    )
    sender = CapturingHttpPost(error)

    result = probe.send_chat_needle_stream_request(
        port=28153,
        request=request,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["endpoint"] == "/v1/chat/completions"
    assert result["http_status"] == 500
    assert result["stream_events_received"] is False
    assert result["failure"]["category"] == "chat_needle_stream_http_error"
    assert "/v1/completions" not in json.dumps(result)


def test_stream_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28153",
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
            EXPECTED,
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
    assert payload["server_port"] == 28153
    assert payload["request"]["limits"]["expected_answer"] == EXPECTED
    assert payload["request"]["limits"]["stream"] is True
    assert payload["request"]["limits"]["stop"] == ["\n```"]
    assert payload["stream_events_received"] is False
    assert "NEEDLE_ANSWER" not in result.stdout
    assert "Synthetic filler" not in result.stdout
    assert "generated_text" not in result.stdout
    assert "text_sha256" not in result.stdout
    assert "token_ids" not in result.stdout
    assert "logprobs" not in result.stdout
    assert "tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash" not in result.stdout
    assert "/" + "home/" not in result.stdout
