import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT
    / "examples"
    / "cuda"
    / "vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe.py"
)
EXPECTED = "PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28157"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe",
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
            if event == "[DONE]":
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
        return self.response


def chunk(content=None, *, finish_reason=None, usage=None):
    choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if content is not None:
        choice["delta"]["content"] = content
    payload = {
        "id": "chatcmpl-stream-usage-contract-needle-test",
        "object": "chat.completion.chunk",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def streaming_response(*, usage):
    return FakeStreamingResponse(
        200,
        [
            chunk(),
            chunk(EXPECTED[:18]),
            chunk(EXPECTED[18:]),
            chunk(finish_reason="stop", usage=usage),
            "[DONE]",
        ],
    )


def streaming_response_with_final_zero_choice_usage(*, usage):
    return FakeStreamingResponse(
        200,
        [
            chunk(),
            chunk(EXPECTED[:18]),
            chunk(EXPECTED[18:], finish_reason="stop"),
            {
                "id": "chatcmpl-stream-usage-contract-needle-test",
                "object": "chat.completion.chunk",
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [],
                "usage": usage,
            },
            "[DONE]",
        ],
    )


def planned_request_with_payload(probe):
    return {
        **probe.build_planned_chat_needle_stream_usage_contract_request(
            expected_answer=EXPECTED,
        ),
        "payload": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "NEEDLE_ANSWER: " + EXPECTED}],
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "n": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
            "stop": ["\n```"],
        },
    }


def test_stream_usage_contract_dry_run_is_review_safe():
    probe = load_probe_module()

    result = probe.run_probe(dry_run=True)
    serialized = json.dumps(result)

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["server_port"] == 28157
    assert result["request"]["endpoint"] == "/v1/chat/completions"
    assert result["request"]["limits"]["target_prompt_tokens"] == 255800
    assert result["request"]["limits"]["max_tokens"] == 64
    assert result["request"]["limits"]["temperature"] == 0.0
    assert result["request"]["limits"]["top_p"] == 1.0
    assert result["request"]["limits"]["seed"] == 0
    assert result["request"]["limits"]["expected_answer"] == EXPECTED
    assert result["request"]["limits"]["match_mode"] == "exact"
    assert result["request"]["limits"]["stream"] is True
    assert result["request"]["limits"]["stream_options_include_usage"] is True
    assert result["request"]["limits"]["stop"] == ["\n```"]
    assert result["generation_attempted"] is False
    assert result["prompt_sent"] is False
    assert result["stream_events_received"] is False
    assert any("stream_options.include_usage=true" in check for check in result["contract_checks"])
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


def test_usage_contract_request_includes_openai_stream_usage_option(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(
        probe.stream_probe.base_probe,
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

    request = probe.build_chat_needle_stream_usage_contract_request(
        max_tokens=64,
        expected_answer=EXPECTED,
    )

    assert request["endpoint"] == "/v1/chat/completions"
    assert request["payload"]["stream"] is True
    assert request["payload"]["stream_options"] == {"include_usage": True}
    assert request["payload"]["stop"] == ["\n```"]
    assert request["limits"]["stream"] is True
    assert request["limits"]["stream_options_include_usage"] is True
    assert request["limits"]["actual_prompt_tokens"] == 255800
    assert "payload" not in probe.review_safe_request(request)


def test_usage_contract_fails_when_usage_is_not_returned():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    sender = CapturingHttpPost(streaming_response(usage=None))

    result = probe.send_chat_needle_stream_usage_contract_request(
        port=28157,
        request=request,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "chat_needle_stream_usage_not_returned"
    assert result["stream_events_received"] is True
    assert result["validation"]["checks"]["usage_presence"] == "failed"
    assert sender.calls[0][1]["stream_options"] == {"include_usage": True}


def test_usage_contract_passes_with_consistent_usage_accounting():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    request["limits"]["actual_prompt_tokens"] = 255800
    sender = CapturingHttpPost(
        streaming_response(
            usage={
                "prompt_tokens": 255800,
                "completion_tokens": 18,
                "total_tokens": 255818,
            }
        )
    )

    result = probe.send_chat_needle_stream_usage_contract_request(
        port=28157,
        request=request,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "passed"
    assert result["observation"]["usage"] == {
        "prompt_tokens": 255800,
        "completion_tokens": 18,
        "total_tokens": 255818,
    }
    assert result["observation"]["usage_shape"] == "passed"
    assert result["observation"]["usage_prompt_tokens_match"] == "passed"
    assert result["observation"]["usage_completion_bound"] == "passed"
    assert result["observation"]["usage_total_tokens"] == "passed"
    assert result["observation"]["exact_match"] is True
    assert "assembled_content" not in json.dumps(result)


def test_usage_contract_accepts_final_zero_choice_usage_chunk():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    request["limits"]["actual_prompt_tokens"] = 255800
    sender = CapturingHttpPost(
        streaming_response_with_final_zero_choice_usage(
            usage={
                "prompt_tokens": 255800,
                "completion_tokens": 18,
                "total_tokens": 255818,
            }
        )
    )

    result = probe.send_chat_needle_stream_usage_contract_request(
        port=28157,
        request=request,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "passed"
    assert result["observation"]["usage"] == {
        "prompt_tokens": 255800,
        "completion_tokens": 18,
        "total_tokens": 255818,
    }
    assert result["validation"]["chunk_shapes"][-1]["choice_count"] == 0
    assert result["observation"]["usage_shape"] == "passed"
    assert result["observation"]["exact_match"] is True
    assert "assembled_content" not in json.dumps(result)


def test_usage_contract_rejects_zero_choice_chunk_without_usage():
    probe = load_probe_module()
    response = FakeStreamingResponse(
        200,
        [
            chunk(EXPECTED, finish_reason="stop"),
            {
                "id": "chatcmpl-stream-usage-contract-needle-test",
                "object": "chat.completion.chunk",
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [],
            },
            "[DONE]",
        ],
    )

    parsed = probe.parse_streaming_chat_completion_response(response)

    assert parsed["status"] == "failed"
    assert parsed["failure"]["category"] == "chat_needle_stream_choice_shape"


def test_usage_contract_rejects_non_object_choice():
    probe = load_probe_module()
    response = FakeStreamingResponse(
        200,
        [
            {
                "id": "chatcmpl-stream-usage-contract-needle-test",
                "object": "chat.completion.chunk",
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": ["not-an-object"],
            },
            "[DONE]",
        ],
    )

    parsed = probe.parse_streaming_chat_completion_response(response)

    assert parsed["status"] == "failed"
    assert parsed["failure"]["category"] == "chat_needle_stream_choice_shape"


def test_stream_usage_contract_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28157",
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
    assert payload["server_port"] == 28157
    assert payload["request"]["limits"]["expected_answer"] == EXPECTED
    assert payload["request"]["limits"]["stream"] is True
    assert payload["request"]["limits"]["stream_options_include_usage"] is True
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
