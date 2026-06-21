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
    / "vllm_deepseek_v4_chat_256k_needle_stream_position_sweep_probe.py"
)
EXPECTED = "PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_stream_position_sweep_probe",
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
        "id": "chatcmpl-stream-position-sweep-needle-test",
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
        chunk(
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": 255800,
                "completion_tokens": 12,
                "total_tokens": 255812,
            },
        ),
    ]
    if done:
        events.append("[DONE]")
    return FakeStreamingResponse(200, events)


def planned_request_with_payload(probe, position):
    return {
        **probe.build_planned_chat_needle_stream_request(
            expected_answer=EXPECTED,
            needle_position=position,
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
            "stop": ["\n```"],
        },
    }


def test_stream_position_sweep_dry_run_contract_is_review_safe():
    probe = load_probe_module()

    result = probe.run_probe(dry_run=True)
    serialized = json.dumps(result)

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["server_port"] == 28156
    assert result["needle_position_sweep"] == ["early", "middle", "late"]
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
    assert result["request"]["limits"]["needle_position"] == "early"
    assert result["generation_attempted"] is False
    assert result["prompt_sent"] is False
    assert result["stream_events_received"] is False
    assert any("early,middle,late" in check for check in result["contract_checks"])
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


def test_stream_position_sweep_sender_posts_one_request_per_position():
    probe = load_probe_module()
    sender = CapturingHttpPost(
        [streaming_response(), streaming_response(), streaming_response()]
    )
    requests = [
        planned_request_with_payload(probe, "early"),
        planned_request_with_payload(probe, "middle"),
        planned_request_with_payload(probe, "late"),
    ]

    result = probe.send_chat_needle_stream_position_sweep_requests(
        port=28156,
        requests=requests,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "passed"
    assert result["needle_position_sweep"] == ["early", "middle", "late"]
    assert result["positions_completed"] == 3
    assert result["passed_positions"] == 3
    assert result["failed_positions"] == 0
    assert result["stream_events_received"] is True
    assert len(sender.calls) == 3
    assert all(call[0].endswith("/v1/chat/completions") for call in sender.calls)
    assert [summary["position"] for summary in result["positions"]] == [
        "early",
        "middle",
        "late",
    ]
    assert [summary["status"] for summary in result["positions"]] == [
        "passed",
        "passed",
        "passed",
    ]
    assert all(summary["stream_done_seen"] is True for summary in result["positions"])
    assert all(summary["finish_reason"] == "stop" for summary in result["positions"])
    assert all(summary["exact_check"] == "passed" for summary in result["positions"])
    assert all(
        summary["normalized_output_length_chars"] == len(EXPECTED)
        for summary in result["positions"]
    )


def test_stream_position_sweep_fails_when_any_position_fails():
    probe = load_probe_module()
    sender = CapturingHttpPost(
        [
            streaming_response(),
            streaming_response(done=False),
            streaming_response(),
        ]
    )
    requests = [
        planned_request_with_payload(probe, "early"),
        planned_request_with_payload(probe, "middle"),
        planned_request_with_payload(probe, "late"),
    ]

    result = probe.send_chat_needle_stream_position_sweep_requests(
        port=28156,
        requests=requests,
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["positions_completed"] == 3
    assert result["passed_positions"] == 2
    assert result["failed_positions"] == 1
    assert result["failure"]["category"] == "chat_needle_stream_done_missing"
    assert result["positions"][1]["position"] == "middle"
    assert result["positions"][1]["stream_done_seen"] is False
    assert result["positions"][1]["exact_check"] is None


def test_stream_position_sweep_rejects_duplicate_positions():
    probe = load_probe_module()

    try:
        probe.parse_needle_position_sweep("early,middle,early")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate position validation failure")


def test_stream_position_sweep_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28156",
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
    assert payload["server_port"] == 28156
    assert payload["needle_position_sweep"] == ["early", "middle", "late"]
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
