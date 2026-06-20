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
    / "vllm_deepseek_v4_chat_256k_needle_stream_repeat_probe.py"
)
EXPECTED = "PTO_CHAT_NEEDLE_256K_STREAM_REPEAT_OK_28154"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_stream_repeat_probe",
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
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def chunk(content=None, *, finish_reason=None, usage=None):
    choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if content is not None:
        choice["delta"]["content"] = content
    payload = {
        "id": "chatcmpl-stream-repeat-needle-test",
        "object": "chat.completion.chunk",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def streaming_response(answer=EXPECTED, *, done=True, finish_reason="stop"):
    events = [
        chunk(),
        chunk(answer[:18]),
        chunk(answer[18:]),
        chunk(finish_reason=finish_reason),
    ]
    if done:
        events.append("[DONE]")
    return FakeStreamingResponse(200, events)


def planned_request_with_payload(probe):
    return {
        **probe.build_planned_chat_needle_stream_request(expected_answer=EXPECTED),
        "payload": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "NEEDLE_ANSWER: " + EXPECTED}],
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "n": 1,
            "stream": True,
            "stop": ["\n```"],
        },
    }


def test_stream_repeat_dry_run_contract_is_review_safe():
    probe = load_probe_module()

    result = probe.run_probe(dry_run=True)
    serialized = json.dumps(result)

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["server_port"] == 28154
    assert result["repeat_count"] == 2
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
    assert any("exactly 2 identical streaming" in check for check in result["contract_checks"])
    assert "NEEDLE_ANSWER" not in serialized
    assert "Synthetic filler" not in serialized
    assert "generated_text" not in serialized
    assert "text_sha256" not in serialized
    assert "token_ids" not in serialized
    assert "logprobs" not in serialized
    assert "/" + "home/" not in serialized


def test_stream_repeat_planned_builder_defaults_to_repeat_expected_answer():
    probe = load_probe_module()

    request = probe.build_planned_chat_needle_stream_request()

    assert request["limits"]["expected_answer"] == EXPECTED
    assert request["limits"]["stream"] is True
    assert request["limits"]["stop"] == ["\n```"]


def test_stream_repeat_sender_posts_two_identical_streaming_chat_requests():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    sender = CapturingHttpPost([streaming_response(), streaming_response()])

    result = probe.send_chat_needle_stream_repeat_requests(
        port=28154,
        request=request,
        repeat_count=2,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "passed"
    assert result["repeat_count"] == 2
    assert result["attempts_completed"] == 2
    assert result["passed_attempts"] == 2
    assert result["failed_attempts"] == 0
    assert result["stream_events_received"] is True
    assert len(sender.calls) == 2
    assert sender.calls[0][0].endswith("/v1/chat/completions")
    assert sender.calls[0][1] == sender.calls[1][1]
    assert sender.calls[0][1]["stream"] is True
    assert result["attempts"][0]["attempt_index"] == 1
    assert result["attempts"][1]["attempt_index"] == 2
    assert result["attempts"][0]["stream_done_seen"] is True
    assert result["attempts"][1]["stream_done_seen"] is True
    assert result["attempts"][0]["finish_reason"] == "stop"
    assert result["attempts"][1]["finish_reason"] == "stop"
    assert result["attempts"][0]["exact_check"] == "passed"
    assert result["attempts"][1]["exact_check"] == "passed"


def test_stream_repeat_aggregate_fails_when_attempt_count_is_incomplete():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    passed = probe.send_chat_needle_stream_request(
        port=28154,
        request=request,
        expected_answer=EXPECTED,
        http_post=CapturingHttpPost([streaming_response()]),
    )

    result = probe.aggregate_chat_stream_repeat_attempts(
        [passed],
        request=request,
        expected_count=2,
    )

    assert result["status"] == "failed"
    assert result["repeat_count"] == 2
    assert result["attempts_completed"] == 1
    assert result["passed_attempts"] == 1
    assert result["failed_attempts"] == 0
    assert result["failure"]["category"] == "chat_needle_stream_repeat_incomplete"


def test_stream_repeat_fails_without_done_even_when_content_matches():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    sender = CapturingHttpPost([streaming_response(done=False), streaming_response()])

    result = probe.send_chat_needle_stream_repeat_requests(
        port=28154,
        request=request,
        repeat_count=2,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["attempts_completed"] == 2
    assert result["passed_attempts"] == 1
    assert result["failed_attempts"] == 1
    assert result["failure"]["category"] == "chat_needle_stream_done_missing"
    assert result["attempts"][0]["stream_done_seen"] is False
    assert result["attempts"][0]["exact_check"] is None


def test_stream_repeat_fails_without_finish_reason_even_when_content_matches():
    probe = load_probe_module()
    request = planned_request_with_payload(probe)
    sender = CapturingHttpPost(
        [streaming_response(finish_reason=None), streaming_response()]
    )

    result = probe.send_chat_needle_stream_repeat_requests(
        port=28154,
        request=request,
        repeat_count=2,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["attempts_completed"] == 2
    assert result["passed_attempts"] == 1
    assert result["failed_attempts"] == 1
    assert result["failure"]["category"] == "chat_needle_stream_finish_reason_missing"
    assert result["attempts"][0]["stream_finish_reason_present"] == "failed"
    assert result["attempts"][0]["exact_check"] is None


def test_stream_repeat_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28154",
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
    assert payload["server_port"] == 28154
    assert payload["repeat_count"] == 2
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
