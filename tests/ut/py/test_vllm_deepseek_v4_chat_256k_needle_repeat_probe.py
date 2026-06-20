import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_256k_needle_repeat_probe.py"
)
EXPECTED = "PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_repeat_probe",
        PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status, body=b"{}"):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


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


def chat_payload(content, *, prompt_tokens=255800, completion_tokens=17):
    return {
        "id": "chatcmpl-needle-repeat-test",
        "object": "chat.completion",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_planned_repeat_request_is_review_safe():
    probe = load_probe_module()

    result = probe.run_probe(dry_run=True)

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["server_port"] == 28152
    assert result["repeat_count"] == 2
    assert result["request"]["endpoint"] == "/v1/chat/completions"
    assert result["request"]["limits"]["target_prompt_tokens"] == 255800
    assert result["request"]["limits"]["max_tokens"] == 64
    assert result["request"]["limits"]["temperature"] == 0.0
    assert result["request"]["limits"]["top_p"] == 1.0
    assert result["request"]["limits"]["seed"] == 0
    assert result["request"]["limits"]["expected_answer"] == EXPECTED
    assert result["request"]["limits"]["match_mode"] == "exact"
    assert result["request"]["limits"]["stop"] == ["\n```"]
    assert result["request"]["limits"]["message_count"] == 2
    assert result["request"]["limits"]["message_roles"] == ["system", "user"]
    assert result["request"]["limits"]["needle_occurrences"] == 1
    assert result["generation_attempted"] is False
    assert result["prompt_sent"] is False
    assert result["request"]["prompt_text_recorded"] is False
    assert result["request"]["payload_recorded"] is False
    assert "payload" not in result["request"]
    assert "messages" not in result["request"]
    assert any("exactly 2 identical" in check for check in result["contract_checks"])


def test_repeat_sender_posts_two_identical_chat_requests():
    probe = load_probe_module()
    request = {
        **probe.build_planned_chat_needle_request(expected_answer=EXPECTED),
        "payload": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "NEEDLE_ANSWER: " + EXPECTED}],
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "n": 1,
            "stream": False,
            "stop": ["\n```"],
        },
    }
    sender = CapturingHttpPost(
        [
            FakeResponse(200, json.dumps(chat_payload(EXPECTED)).encode()),
            FakeResponse(200, json.dumps(chat_payload(f"```text\n{EXPECTED}\n```")).encode()),
        ]
    )

    result = probe.send_chat_needle_repeat_requests(
        port=28152,
        request=request,
        repeat_count=2,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "passed"
    assert result["repeat_count"] == 2
    assert result["attempts_completed"] == 2
    assert result["passed_attempts"] == 2
    assert result["failed_attempts"] == 0
    assert len(sender.calls) == 2
    assert sender.calls[0][0].endswith("/v1/chat/completions")
    assert sender.calls[0][1] == sender.calls[1][1]
    assert result["attempts"][0]["attempt_index"] == 1
    assert result["attempts"][1]["attempt_index"] == 2
    assert result["attempts"][0]["exact_check"] == "passed"
    assert result["attempts"][1]["exact_check"] == "passed"


def test_repeat_aggregate_fails_when_second_attempt_fails_without_raw_output():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_request(expected_answer=EXPECTED)
    passed = probe.send_chat_needle_request(
        port=28152,
        request={**request, "payload": {"model": "deepseek-ai/DeepSeek-V4-Flash"}},
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
        http_post=CapturingHttpPost(
            [FakeResponse(200, json.dumps(chat_payload(EXPECTED)).encode())]
        ),
    )
    failed = probe.send_chat_needle_request(
        port=28152,
        request={**request, "payload": {"model": "deepseek-ai/DeepSeek-V4-Flash"}},
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
        http_post=CapturingHttpPost(
            [
                FakeResponse(
                    200,
                    json.dumps(chat_payload(f"{EXPECTED}\nextra text")).encode(),
                )
            ]
        ),
    )

    result = probe.aggregate_chat_repeat_attempts(
        [passed, failed],
        request=request,
        expected_count=2,
    )
    serialized = json.dumps(result)

    assert result["status"] == "failed"
    assert result["repeat_count"] == 2
    assert result["attempts_completed"] == 2
    assert result["passed_attempts"] == 1
    assert result["failed_attempts"] == 1
    assert result["failure"]["category"] == "chat_needle_expected_answer_not_exact"
    assert result["attempts"][1]["exact_check"] == "failed"
    assert "extra text" not in serialized
    assert "generated_text" not in serialized
    assert "normalized_generated_text" not in serialized
    assert "text_" + "sha256" not in serialized
    assert "token_ids" not in serialized
    assert "logprobs" not in serialized


def test_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
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
    assert payload["server_port"] == 28152
    assert payload["repeat_count"] == 2
    assert payload["request"]["limits"]["expected_answer"] == EXPECTED
    assert payload["request"]["limits"]["stop"] == ["\n```"]
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "NEEDLE_ANSWER" not in result.stdout
    assert "Synthetic filler" not in result.stdout
    assert "generated_text" not in result.stdout
    assert "generated_text_digest" not in result.stdout
    assert "text_sha256" not in result.stdout
    assert "token_ids" not in result.stdout
    assert "logprobs" not in result.stdout
    assert "/" + "home/" not in result.stdout
    assert "uv" + "xiao" not in result.stdout
