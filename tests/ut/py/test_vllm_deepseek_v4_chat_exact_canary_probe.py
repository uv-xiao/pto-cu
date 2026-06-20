import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_exact_canary_probe.py"
)
EXPECTED = "PTO_CHAT_EXACT_CANARY_28149"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_exact_canary_probe",
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
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def chat_payload(content, *, prompt_tokens=19, completion_tokens=7):
    return {
        "id": "chatcmpl-test",
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


def test_planned_chat_canary_request_is_review_safe():
    probe = load_probe_module()

    request = probe.build_planned_chat_canary_request(
        max_tokens=16,
        expected_answer=EXPECTED,
    )
    safe = probe.review_safe_request(request)

    assert request["endpoint"] == "/v1/chat/completions"
    assert request["limits"]["max_tokens"] == 16
    assert request["limits"]["expected_answer"] == EXPECTED
    assert request["limits"]["temperature"] == 0.0
    assert request["limits"]["top_p"] == 1.0
    assert request["limits"]["seed"] == 0
    assert request["limits"]["stream"] is False
    assert request["limits"]["n"] == 1
    assert request["limits"]["message_count"] == 2
    assert request["limits"]["prompt_text_recording"] is False
    assert request["limits"]["payload_recording"] is False
    assert safe["prompt_text_recorded"] is False
    assert safe["payload_recorded"] is False
    assert "messages" not in safe
    assert "payload" not in safe


def test_build_chat_canary_payload_targets_chat_endpoint():
    probe = load_probe_module()

    request = probe.build_chat_canary_request(max_tokens=16, expected_answer=EXPECTED)

    assert request["endpoint"] == "/v1/chat/completions"
    assert request["payload"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert request["payload"]["max_tokens"] == 16
    assert request["payload"]["temperature"] == 0.0
    assert request["payload"]["top_p"] == 1.0
    assert request["payload"]["seed"] == 0
    assert request["payload"]["n"] == 1
    assert request["payload"]["stream"] is False
    assert [message["role"] for message in request["payload"]["messages"]] == [
        "system",
        "user",
    ]
    assert EXPECTED in request["payload"]["messages"][1]["content"]
    assert "prompt" not in request["payload"]


def test_validate_chat_canary_accepts_exact_expected_output():
    probe = load_probe_module()
    request = probe.build_planned_chat_canary_request(expected_answer=EXPECTED)

    result = probe.validate_chat_canary_response(
        chat_payload(f"\n```text\n{EXPECTED}\n```\n"),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
    )

    assert result["status"] == "passed"
    assert result["checks"]["expected_answer_exact"] == "passed"
    assert result["normalized_output_equals_expected"] is True
    assert result["normalized_output_length_chars"] == len(EXPECTED)
    assert "generated_text" not in result
    assert "text_sha256" not in json.dumps(result)


def test_validate_chat_canary_rejects_extra_text_without_weakening_exact_match():
    probe = load_probe_module()
    request = probe.build_planned_chat_canary_request(expected_answer=EXPECTED)

    result = probe.validate_chat_canary_response(
        chat_payload(f"{EXPECTED}\nAdditional explanation."),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
    )

    assert result["status"] == "failed"
    assert result["checks"]["expected_answer_exact"] == "failed"
    assert result["failure"]["category"] == "chat_canary_expected_answer_not_exact"
    assert result["normalized_output_equals_expected"] is False
    assert result["normalized_output_length_chars"] > len(EXPECTED)
    assert "generated_text" not in result
    assert "Additional explanation" not in json.dumps(result)


def test_response_shape_omits_prompt_text_token_id_and_logprob_keys():
    probe = load_probe_module()
    request = probe.build_planned_chat_canary_request(expected_answer=EXPECTED)
    payload = chat_payload(EXPECTED)
    payload["prompt_text"] = ["raw prompt"]
    payload["prompt_token_ids"] = [[1, 2, 3]]
    payload["prompt_logprobs"] = [None]

    result = probe.validate_chat_canary_response(
        payload,
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
    )
    serialized = json.dumps(result)

    assert result["status"] == "passed"
    assert "prompt_text" not in serialized
    assert "prompt_token_ids" not in serialized
    assert "prompt_logprobs" not in serialized
    assert "raw prompt" not in serialized


def test_send_chat_canary_request_reports_non_200_as_endpoint_failure():
    probe = load_probe_module()
    request = probe.build_chat_canary_request(expected_answer=EXPECTED)
    sender = CapturingHttpPost(FakeResponse(404, b'{"error": "missing"}'))

    result = probe.send_chat_canary_request(
        port=28149,
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["endpoint"] == "/v1/chat/completions"
    assert result["http_status"] == 404
    assert result["failure"]["category"] == "chat_completion_http_status"
    assert sender.calls[0][0] == "http://127.0.0.1:28149/v1/chat/completions"


def test_send_chat_canary_request_reports_http_error_without_fallback():
    probe = load_probe_module()
    request = probe.build_chat_canary_request(expected_answer=EXPECTED)
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:28149/v1/chat/completions",
        code=500,
        msg="server error",
        hdrs=None,
        fp=None,
    )
    sender = CapturingHttpPost(error)

    result = probe.send_chat_canary_request(
        port=28149,
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["endpoint"] == "/v1/chat/completions"
    assert result["http_status"] == 500
    assert result["failure"]["category"] == "chat_completion_http_error"
    assert "/v1/completions" not in json.dumps(result)


def test_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28149",
            "--max-model-len",
            "262144",
            "--max-tokens",
            "16",
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
    output = result.stdout

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28149
    assert payload["request"]["endpoint"] == "/v1/chat/completions"
    assert payload["request"]["limits"]["expected_answer"] == EXPECTED
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "messages" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "Return exactly" not in output
    assert "text_sha256" not in output
    assert "token_ids" not in output
    assert "logprobs" not in output
    assert "tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash" not in output
    assert "/" + "home/" not in output
    assert "uv" + "xiao" not in output
