import importlib.util
import json
import subprocess
import sys
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_256k_needle_exact_probe.py"
)
EXPECTED = "PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_exact_probe",
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


def chat_payload(content, *, prompt_tokens=255800, completion_tokens=17):
    return {
        "id": "chatcmpl-needle-test",
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


def test_planned_chat_256k_needle_request_is_review_safe():
    probe = load_probe_module()

    request = probe.build_planned_chat_needle_request(
        target_prompt_tokens=255800,
        max_tokens=64,
        expected_answer=EXPECTED,
    )
    safe = probe.review_safe_request(request)

    assert request["endpoint"] == "/v1/chat/completions"
    assert request["limits"]["target_prompt_tokens"] == 255800
    assert request["limits"]["actual_prompt_tokens"] is None
    assert request["limits"]["max_tokens"] == 64
    assert request["limits"]["temperature"] == 0.0
    assert request["limits"]["top_p"] == 1.0
    assert request["limits"]["seed"] == 0
    assert request["limits"]["expected_answer"] == EXPECTED
    assert request["limits"]["match_mode"] == "exact"
    assert request["limits"]["stream"] is False
    assert request["limits"]["n"] == 1
    assert request["limits"]["message_count"] == 2
    assert request["limits"]["message_roles"] == ["system", "user"]
    assert request["limits"]["needle_occurrences"] == 1
    assert request["limits"]["assistant_content_recording"] is False
    assert safe["prompt_text_recorded"] is False
    assert safe["payload_recorded"] is False
    assert "messages" not in safe
    assert "payload" not in safe
    assert "prompt" not in safe


def test_chat_needle_request_targets_chat_endpoint(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(
        probe,
        "build_synthetic_chat_needle_messages",
        lambda **kwargs: {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "NEEDLE_ANSWER: " + EXPECTED},
            ],
            "accounting": {
                "target_prompt_tokens": kwargs["target_prompt_tokens"],
                "actual_prompt_tokens": 255799,
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

    request = probe.build_chat_needle_request(
        max_tokens=64,
        expected_answer=EXPECTED,
        stop_sequences=["\n```"],
    )

    assert request["endpoint"] == "/v1/chat/completions"
    assert request["payload"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert request["payload"]["max_tokens"] == 64
    assert request["payload"]["temperature"] == 0.0
    assert request["payload"]["top_p"] == 1.0
    assert request["payload"]["seed"] == 0
    assert request["payload"]["n"] == 1
    assert request["payload"]["stream"] is False
    assert request["payload"]["stop"] == ["\n```"]
    assert [message["role"] for message in request["payload"]["messages"]] == [
        "system",
        "user",
    ]
    assert EXPECTED in request["payload"]["messages"][1]["content"]
    assert "prompt" not in request["payload"]
    assert request["limits"]["actual_prompt_tokens"] == 255799
    assert request["limits"]["stop"] == ["\n```"]
    assert "payload" not in probe.review_safe_request(request)


def test_chat_prompt_builder_falls_back_when_tokenizer_has_no_chat_template(monkeypatch):
    probe = load_probe_module()

    class NoChatTemplateTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("tokenizer.chat_template is not set")

        def encode(self, text, add_special_tokens=False):
            return text.split()

    monkeypatch.setattr(probe, "_load_tokenizer", lambda _path: NoChatTemplateTokenizer())

    result = probe.build_synthetic_chat_needle_messages(
        artifact_dir=Path("unused"),
        target_prompt_tokens=128,
        expected_answer=EXPECTED,
    )

    assert result["accounting"]["target_prompt_tokens"] == 128
    assert result["accounting"]["actual_prompt_tokens"] is None
    assert result["accounting"]["tokenizer_accounting"] == (
        "transformers.AutoTokenizer fallback encode estimate"
    )
    assert result["accounting"]["needle_occurrences"] == 1
    assert [message["role"] for message in result["messages"]] == ["system", "user"]
    assert EXPECTED in result["messages"][1]["content"]


def test_validate_chat_needle_accepts_one_surrounding_code_fence_only():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_request(expected_answer=EXPECTED)

    result = probe.validate_chat_needle_response(
        chat_payload(f"\n```text\n{EXPECTED}\n```\n"),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
    )

    assert result["status"] == "passed"
    assert result["match_mode"] == "exact"
    assert result["checks"]["expected_answer_exact"] == "passed"
    assert result["normalized_output_equals_expected"] is True
    assert result["normalized_output_length_chars"] == len(EXPECTED)
    assert result["usage"]["prompt_tokens"] == 255800
    assert "generated_text" not in result
    assert "normalized_generated_text" not in result
    assert "text_sha256" not in json.dumps(result)


def test_validate_chat_needle_rejects_extra_text_without_recording_output():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_request(expected_answer=EXPECTED)

    result = probe.validate_chat_needle_response(
        chat_payload(f"{EXPECTED}\nAdditional explanation."),
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
    )
    serialized = json.dumps(result)

    assert result["status"] == "failed"
    assert result["checks"]["expected_answer_exact"] == "failed"
    assert result["failure"]["category"] == "chat_needle_expected_answer_not_exact"
    assert result["normalized_output_equals_expected"] is False
    assert result["normalized_output_length_chars"] > len(EXPECTED)
    assert "generated_text" not in result
    assert "normalized_generated_text" not in result
    assert "Additional explanation" not in serialized
    assert EXPECTED in serialized


def test_response_shape_omits_raw_payload_text_token_ids_logprobs_and_digests():
    probe = load_probe_module()
    request = probe.build_planned_chat_needle_request(expected_answer=EXPECTED)
    payload = chat_payload(EXPECTED)
    payload["messages"] = [{"role": "user", "content": "raw prompt"}]
    payload["prompt"] = "raw prompt"
    payload["prompt_text"] = ["raw prompt"]
    payload["prompt_token_ids"] = [[1, 2, 3]]
    payload["token_ids"] = [1, 2, 3]
    payload["prompt_logprobs"] = [None]
    payload["logprobs"] = [0.0]
    payload["generated_text_digest"] = "abc"

    result = probe.validate_chat_needle_response(
        payload,
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
    )
    serialized = json.dumps(result)

    assert result["status"] == "passed"
    assert "messages" not in serialized
    assert "prompt_text" not in serialized
    assert "raw prompt" not in serialized
    assert "prompt_token_ids" not in serialized
    assert "token_ids" not in serialized
    assert "prompt_logprobs" not in serialized
    assert "logprobs" not in serialized
    assert "generated_text_digest" not in serialized
    assert "text_sha256" not in serialized


def test_send_chat_needle_request_reports_http_error_without_completion_fallback():
    probe = load_probe_module()
    request = {
        **probe.build_planned_chat_needle_request(expected_answer=EXPECTED),
        "payload": {"messages": [], "model": "deepseek-ai/DeepSeek-V4-Flash"},
    }
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:28151/v1/chat/completions",
        code=500,
        msg="server error",
        hdrs=None,
        fp=None,
    )
    sender = CapturingHttpPost(error)

    result = probe.send_chat_needle_request(
        port=28151,
        request=request,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        expected_answer=EXPECTED,
        http_post=sender,
    )

    assert result["status"] == "failed"
    assert result["endpoint"] == "/v1/chat/completions"
    assert result["http_status"] == 500
    assert result["failure"]["category"] == "chat_needle_http_error"
    assert "/v1/completions" not in json.dumps(result)


def test_dry_run_cli_output_is_review_safe():
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--dry-run",
            "--port",
            "28151",
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
    output = result.stdout

    assert payload["status"] == "planned"
    assert payload["server_host"] == "127.0.0.1"
    assert payload["server_port"] == 28151
    assert payload["request"]["endpoint"] == "/v1/chat/completions"
    assert payload["request"]["limits"]["target_prompt_tokens"] == 255800
    assert payload["request"]["limits"]["max_tokens"] == 64
    assert payload["request"]["limits"]["temperature"] == 0.0
    assert payload["request"]["limits"]["top_p"] == 1.0
    assert payload["request"]["limits"]["seed"] == 0
    assert payload["request"]["limits"]["expected_answer"] == EXPECTED
    assert payload["request"]["limits"]["match_mode"] == "exact"
    assert payload["request"]["limits"]["stop"] == ["\n```"]
    assert payload["generation_attempted"] is False
    assert payload["prompt_sent"] is False
    assert payload["request"]["prompt_text_recorded"] is False
    assert payload["request"]["payload_recorded"] is False
    assert "messages" not in payload["request"]
    assert "payload" not in payload["request"]
    assert "prompt" not in payload["request"]
    assert "NEEDLE_ANSWER" not in output
    assert "Synthetic filler" not in output
    assert "generated_text" not in output
    assert "generated_text_digest" not in output
    assert "text_sha256" not in output
    assert "token_ids" not in output
    assert "logprobs" not in output
    assert "tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash" not in output
    assert "/" + "home/" not in output
    assert "uv" + "xiao" not in output
