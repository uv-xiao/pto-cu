#!/usr/bin/env python3
"""Synthetic pypto-serving-style shim for simpler NVIDIA execution."""

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYPTO_SERVING_SOURCE = ROOT / "tmp" / "sources" / "repos" / "hw-native-sys" / "pypto-serving"
DEFAULT_GENERATED_GLUON_OUTPUT_DIR = Path("tmp/pypto-serving-gluon-moe-expert")
PERSISTENT_MOE_DISPATCH_COMBINE_LAUNCH_KIND = "persistent-moe-dispatch-combine"


@dataclass(frozen=True)
class RuntimeConfig:
    max_batch_size: int = 1
    max_seq_len: int = 16
    device: str = "cpu"
    max_new_tokens: int = 2


@dataclass(frozen=True)
class RuntimeModel:
    model_id: str
    runtime: RuntimeConfig
    vocab_size: int = 3
    hidden_size: int = 4


@dataclass(frozen=True)
class PrefillBatch:
    request_ids: list[str]
    token_ids: list[int]


@dataclass(frozen=True)
class PrefillResult:
    last_hidden: list[list[float]]
    logits: list[list[float]]


@dataclass(frozen=True)
class DecodeBatch:
    request_ids: list[str]
    token_ids: list[int]
    hidden_states: list[list[float]]


@dataclass(frozen=True)
class DecodeResult:
    hidden_states: list[list[float]]
    logits: list[list[float]]


@dataclass(frozen=True)
class KernelLaunchRequest:
    phase: str
    platform: str
    runtime: str
    device_id: int
    op: str
    n: int
    block_dim: int
    arch: str


@dataclass
class SimplerNvModelRunner:
    """Minimal pypto-serving-style runner for one synthetic CUDA model."""

    platform: str = "cuda"
    runtime: str = "host_schedule"
    device_id: int = 0
    arch: str = "compute_90"
    n: int = 16
    block_dim: int = 16
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None
    launch_results: list[dict[str, Any]] = field(default_factory=list)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        self._launch("prefill")
        return PrefillResult(
            last_hidden=[[1.0, 0.0, 0.0, 0.0] for _ in batch.request_ids],
            logits=[[0.0, 10.0, 0.0] for _ in batch.request_ids],
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        self._launch("decode")
        return DecodeResult(
            hidden_states=[[0.0, 1.0, 0.0, 0.0] for _ in batch.request_ids],
            logits=[[0.0, 0.0, 10.0] for _ in batch.request_ids],
        )

    def _launch(self, phase: str) -> None:
        request = KernelLaunchRequest(
            phase=phase,
            platform=self.platform,
            runtime=self.runtime,
            device_id=self.device_id,
            op="add",
            n=self.n,
            block_dim=self.block_dim,
            arch=self.arch,
        )
        launcher = default_cuda_seed_launcher if self.kernel_launcher is None else self.kernel_launcher
        result = launcher(request)
        self.launch_results.append(result)


class SimplerNvExecutor:
    """Small ModelExecutor-shaped adapter for the synthetic serving smoke."""

    def __init__(
        self,
        *,
        platform: str = "cuda",
        runtime: str = "host_schedule",
        device_id: int = 0,
        arch: str = "compute_90",
        kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
    ) -> None:
        self.platform = platform
        self.runtime = runtime
        self.device_id = int(device_id)
        self.arch = arch
        self.kernel_launcher = kernel_launcher
        self._runners: dict[str, SimplerNvModelRunner] = {}

    def register_model(self, model_id: str, model: RuntimeModel) -> None:
        self._runners[model_id] = SimplerNvModelRunner(
            platform=self.platform,
            runtime=self.runtime,
            device_id=self.device_id,
            arch=self.arch,
            kernel_launcher=self.kernel_launcher,
        )

    def runner_for(self, model_id: str) -> SimplerNvModelRunner:
        return self._runners[model_id]

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        return self.runner_for(model.model_id).run_prefill(model, batch)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        return self.runner_for(model.model_id).run_decode(model, batch)


@dataclass(frozen=True)
class SyntheticModelRecord:
    model: RuntimeModel
    model_dir: str


class SyntheticPyptoServingEngine:
    """Small LLMEngine-shaped fixture for synthetic simpler-nv generation."""

    def __init__(
        self,
        *,
        platform: str = "cuda",
        runtime: str = "host_schedule",
        device_id: int = 0,
        arch: str = "compute_90",
        kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
    ) -> None:
        self.executor = SimplerNvExecutor(
            platform=platform,
            runtime=runtime,
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
        self._records: dict[str, SyntheticModelRecord] = {}

    def init_model(
        self,
        model_id: str = "synthetic-simpler-nv",
        model_dir: str = "synthetic://simpler-nv",
    ) -> None:
        if model_id != "synthetic-simpler-nv":
            raise ValueError("only the synthetic-simpler-nv model is available")
        model = create_synthetic_runtime_model()
        self.executor.register_model(model_id, model)
        self._records[model_id] = SyntheticModelRecord(model=model, model_dir=model_dir)

    def model_ids(self) -> list[str]:
        return sorted(self._records)

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 2,
    ) -> dict[str, Any]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if model_id not in self._records:
            raise KeyError(f"Model {model_id} is not initialized.")

        record = self._records[model_id]
        model = record.model
        prompt_tokens = _encode_prompt(prompt)
        prefill = self.executor.run_prefill(
            model,
            PrefillBatch(request_ids=["req-0"], token_ids=prompt_tokens),
        )
        token_ids = [_argmax(prefill.logits[0])]
        hidden = prefill.last_hidden
        while len(token_ids) < max_new_tokens:
            decode = self.executor.run_decode(
                model,
                DecodeBatch(
                    request_ids=["req-0"],
                    token_ids=[token_ids[-1]],
                    hidden_states=hidden,
                ),
            )
            hidden = decode.hidden_states
            token_ids.append(_argmax(decode.logits[0]))

        runner = self.executor.runner_for(model_id)
        launch_statuses = [item.get("status") for item in runner.launch_results]
        if "failed" in launch_statuses:
            status = "failed"
        elif all(item == "passed" for item in launch_statuses):
            status = "passed"
        else:
            status = "skipped"
        return {
            "engine": type(self).__name__,
            "status": status,
            "backend": "simpler-nv",
            "model_id": model_id,
            "model_dir": record.model_dir,
            "prompt": prompt,
            "text": _decode_tokens(token_ids),
            "token_ids": token_ids,
            "finish_reason": "length",
            "launch_count": len(runner.launch_results),
            "launch_results": runner.launch_results,
        }

    def create_openai_completion(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int = 2,
    ) -> dict[str, Any]:
        result = self.generate(
            model_id=model,
            prompt=prompt,
            max_new_tokens=max_tokens,
        )
        prompt_tokens = _encode_prompt(prompt)
        completion_tokens = result["token_ids"]
        return _openai_completion_from_result(
            model=model,
            result=result,
            prompt_token_count=len(prompt_tokens),
            extra={"pto_engine": result["engine"]},
        )

    def create_openai_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2,
    ) -> dict[str, Any]:
        prompt = _prompt_from_chat_messages(messages)
        result = self.generate(
            model_id=model,
            prompt=prompt,
            max_new_tokens=max_tokens,
        )
        prompt_tokens = _encode_prompt(prompt)
        return _openai_chat_completion_from_result(
            model=model,
            result=result,
            prompt_token_count=len(prompt_tokens),
            extra={"pto_engine": result["engine"]},
        )


@dataclass(frozen=True)
class PyptoServingSourceTokenOutput:
    text: str
    finished: bool
    finish_reason: str


class PyptoServingSourceAsyncEngineAdapter:
    """AsyncLLMEngine-shaped adapter for the actual pypto-serving server."""

    def __init__(self, engine: SyntheticPyptoServingEngine) -> None:
        self.engine = engine
        self.last_pto_status = ""
        self.last_token_ids: list[int] = []
        self.last_launch_count = 0
        self.last_launch_results: list[dict[str, Any]] = []

    async def add_request(self, request_id: str, prompt: str, config):
        result = self.engine.generate(
            model_id="synthetic-simpler-nv",
            prompt=prompt,
            max_new_tokens=int(config.max_new_tokens),
        )
        self.last_pto_status = result["status"]
        self.last_token_ids = list(result["token_ids"])
        self.last_launch_count = int(result["launch_count"])
        self.last_launch_results = list(result["launch_results"])
        for token_index in range(len(self.last_token_ids)):
            finished = token_index == len(self.last_token_ids) - 1
            yield PyptoServingSourceTokenOutput(
                text=_decode_tokens(self.last_token_ids[: token_index + 1]),
                finished=finished,
                finish_reason="FINISHED_LENGTH" if finished else "",
            )


PYPTO_SERVING_VLLM_COMPAT_CHECKED_FIELDS = [
    "route",
    "http_status_200",
    "model_or_stream_object_shape",
    "choice_text_or_message_delta_presence",
    "finish_reason",
    "usage_presence_when_non_streaming",
    "sse_done_presence_when_streaming",
]

PYPTO_SERVING_VLLM_COMPAT_NON_CLAIMS = [
    "tokenizer semantics",
    "logprob values",
    "stop-token semantics",
    "production readiness",
    "throughput",
    "latency",
    "real DeepSeek weights",
    "simpler-nv/vLLM kernel integration",
]


@contextlib.contextmanager
def _pypto_serving_source_import_path():
    if not PYPTO_SERVING_SOURCE.is_dir():
        raise FileNotFoundError(f"missing {PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    source_path = str(PYPTO_SERVING_SOURCE)
    sys.path.insert(0, source_path)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(source_path)


def create_pypto_serving_source_app(
    *,
    model: str = "synthetic-simpler-nv",
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
):
    with _pypto_serving_source_import_path():
        from python.core.server import create_serving_app

    engine = SyntheticPyptoServingEngine(
        device_id=device_id,
        arch=arch,
        kernel_launcher=kernel_launcher,
    )
    engine.init_model(model)
    adapter = PyptoServingSourceAsyncEngineAdapter(engine)
    return create_serving_app(adapter, model), adapter


def run_pypto_serving_source_completion_fixture(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as exc:
        return {
            "status": "skipped",
            "reason": f"missing FastAPI TestClient: {exc}",
            "server": "pypto-serving-source",
            "route": "/v1/completions",
        }
    try:
        app, adapter = create_pypto_serving_source_app(
            model=model,
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "server": "pypto-serving-source",
            "route": "/v1/completions",
        }

    response = TestClient(app).post(
        "/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
    )
    return {
        "status": "passed" if response.status_code == 200 else "failed",
        "server": "pypto-serving-source",
        "route": "/v1/completions",
        "status_code": response.status_code,
        "response": response.json(),
        "pto_status": adapter.last_pto_status,
        "pto_token_ids": adapter.last_token_ids,
        "pto_launch_count": adapter.last_launch_count,
        "pto_launch_results": adapter.last_launch_results,
    }


def run_pypto_serving_source_chat_completion_fixture(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as exc:
        return {
            "status": "skipped",
            "reason": f"missing FastAPI TestClient: {exc}",
            "server": "pypto-serving-source",
            "route": "/v1/chat/completions",
        }
    try:
        app, adapter = create_pypto_serving_source_app(
            model=model,
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "server": "pypto-serving-source",
            "route": "/v1/chat/completions",
        }

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    )
    return {
        "status": "passed" if response.status_code == 200 else "failed",
        "server": "pypto-serving-source",
        "route": "/v1/chat/completions",
        "status_code": response.status_code,
        "response": response.json(),
        "pto_status": adapter.last_pto_status,
        "pto_token_ids": adapter.last_token_ids,
        "pto_launch_count": adapter.last_launch_count,
        "pto_launch_results": adapter.last_launch_results,
    }


def run_pypto_serving_source_stream_completion_fixture(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as exc:
        return {
            "status": "skipped",
            "reason": f"missing FastAPI TestClient: {exc}",
            "server": "pypto-serving-source",
            "route": "/v1/completions",
            "stream": True,
        }
    try:
        app, adapter = create_pypto_serving_source_app(
            model=model,
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "server": "pypto-serving-source",
            "route": "/v1/completions",
            "stream": True,
        }

    response = TestClient(app).post(
        "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        },
    )
    summary = _summarize_pypto_serving_source_stream(response, chat=False)
    return {
        "status": "passed"
        if response.status_code == 200 and summary["done_seen"]
        else "failed",
        "server": "pypto-serving-source",
        "route": "/v1/completions",
        "stream": True,
        "status_code": response.status_code,
        **summary,
        "pto_status": adapter.last_pto_status,
        "pto_token_ids": adapter.last_token_ids,
        "pto_launch_count": adapter.last_launch_count,
        "pto_launch_results": adapter.last_launch_results,
    }


def run_pypto_serving_source_stream_chat_completion_fixture(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as exc:
        return {
            "status": "skipped",
            "reason": f"missing FastAPI TestClient: {exc}",
            "server": "pypto-serving-source",
            "route": "/v1/chat/completions",
            "stream": True,
        }
    try:
        app, adapter = create_pypto_serving_source_app(
            model=model,
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "server": "pypto-serving-source",
            "route": "/v1/chat/completions",
            "stream": True,
        }

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        },
    )
    summary = _summarize_pypto_serving_source_stream(response, chat=True)
    return {
        "status": "passed"
        if response.status_code == 200 and summary["done_seen"]
        else "failed",
        "server": "pypto-serving-source",
        "route": "/v1/chat/completions",
        "stream": True,
        "status_code": response.status_code,
        **summary,
        "pto_status": adapter.last_pto_status,
        "pto_token_ids": adapter.last_token_ids,
        "pto_launch_count": adapter.last_launch_count,
        "pto_launch_results": adapter.last_launch_results,
    }


def run_pypto_serving_vllm_compat_fixture(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fixture_results = [
        _compat_non_stream_fixture(
            name="completions",
            result=run_pypto_serving_source_completion_fixture(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                device_id=device_id,
                arch=arch,
                kernel_launcher=kernel_launcher,
            ),
            expected_route="/v1/completions",
            expected_object="text_completion",
            model=model,
            chat=False,
        ),
        _compat_non_stream_fixture(
            name="chat_completions",
            result=run_pypto_serving_source_chat_completion_fixture(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                device_id=device_id,
                arch=arch,
                kernel_launcher=kernel_launcher,
            ),
            expected_route="/v1/chat/completions",
            expected_object="chat.completion",
            model=model,
            chat=True,
        ),
        _compat_stream_fixture(
            name="stream_completions",
            result=run_pypto_serving_source_stream_completion_fixture(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                device_id=device_id,
                arch=arch,
                kernel_launcher=kernel_launcher,
            ),
            expected_route="/v1/completions",
            chat=False,
        ),
        _compat_stream_fixture(
            name="stream_chat_completions",
            result=run_pypto_serving_source_stream_chat_completion_fixture(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                device_id=device_id,
                arch=arch,
                kernel_launcher=kernel_launcher,
            ),
            expected_route="/v1/chat/completions",
            chat=True,
        ),
    ]
    statuses = [item["status"] for item in fixture_results]
    if "failed" in statuses:
        status = "failed"
    elif "skipped" in statuses:
        status = "skipped"
    else:
        status = "passed"

    return {
        "status": status,
        "server": "pypto-serving-source",
        "comparison_baseline": "vllm-openai-compatible-deepseek",
        "model": model,
        "checked_fields": PYPTO_SERVING_VLLM_COMPAT_CHECKED_FIELDS,
        "fixtures": fixture_results,
        "non_claims": PYPTO_SERVING_VLLM_COMPAT_NON_CLAIMS,
    }


def create_synthetic_runtime_model() -> RuntimeModel:
    return RuntimeModel(
        model_id="synthetic-simpler-nv",
        runtime=RuntimeConfig(),
    )


def run_synthetic_serving_request(
    *,
    prompt: str,
    max_new_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    engine = SyntheticPyptoServingEngine(
        device_id=device_id,
        arch=arch,
        kernel_launcher=kernel_launcher,
    )
    engine.init_model("synthetic-simpler-nv")
    result = engine.generate(
        model_id="synthetic-simpler-nv",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    result.pop("engine", None)
    result.pop("model_dir", None)
    return result


def run_synthetic_openai_completion(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model != "synthetic-simpler-nv":
        raise ValueError("only the synthetic-simpler-nv model is available")

    result = run_synthetic_serving_request(
        prompt=prompt,
        max_new_tokens=max_tokens,
        device_id=device_id,
        arch=arch,
        kernel_launcher=kernel_launcher,
    )
    prompt_tokens = _encode_prompt(prompt)
    return _openai_completion_from_result(
        model=model,
        result=result,
        prompt_token_count=len(prompt_tokens),
    )


def run_synthetic_openai_chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model != "synthetic-simpler-nv":
        raise ValueError("only the synthetic-simpler-nv model is available")

    prompt = _prompt_from_chat_messages(messages)
    result = run_synthetic_serving_request(
        prompt=prompt,
        max_new_tokens=max_tokens,
        device_id=device_id,
        arch=arch,
        kernel_launcher=kernel_launcher,
    )
    prompt_tokens = _encode_prompt(prompt)
    return _openai_chat_completion_from_result(
        model=model,
        result=result,
        prompt_token_count=len(prompt_tokens),
    )


def create_synthetic_openai_app(
    *,
    model: str = "synthetic-simpler-nv",
    device_id: int = 0,
    arch: str = "compute_90",
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("FastAPI and Pydantic are required for the HTTP fixture") from exc

    class CompletionRequest(BaseModel):
        model: str = ""
        prompt: str = ""
        max_tokens: int = 2

    class ChatCompletionRequest(BaseModel):
        model: str = ""
        messages: list[dict[str, Any]] = Field(default_factory=list)
        max_tokens: int = 2

    app = FastAPI(title="Synthetic PyPTO Serving")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "pypto"}],
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> dict[str, Any]:
        engine = SyntheticPyptoServingEngine(
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
        engine.init_model(model)
        return engine.create_openai_completion(
            model=request.model or model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
        )

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        engine = SyntheticPyptoServingEngine(
            device_id=device_id,
            arch=arch,
            kernel_launcher=kernel_launcher,
        )
        engine.init_model(model)
        return engine.create_openai_chat_completion(
            model=request.model or model,
            messages=request.messages,
            max_tokens=request.max_tokens,
        )

    return app


def run_synthetic_http_completion_fixture(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 2,
    device_id: int = 0,
    arch: str = "compute_90",
    chat: bool = False,
    kernel_launcher: Callable[[KernelLaunchRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError) as exc:
        return {
            "status": "skipped",
            "reason": f"missing FastAPI TestClient: {exc}",
            "route": "/v1/chat/completions" if chat else "/v1/completions",
        }

    app = create_synthetic_openai_app(
        model=model,
        device_id=device_id,
        arch=arch,
        kernel_launcher=kernel_launcher,
    )
    client = TestClient(app)
    if chat:
        route = "/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    else:
        route = "/v1/completions"
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    response = client.post(route, json=payload)
    return {
        "status": "passed" if response.status_code == 200 else "failed",
        "route": route,
        "status_code": response.status_code,
        "response": response.json(),
    }


def _openai_completion_from_result(
    *,
    model: str,
    result: dict[str, Any],
    prompt_token_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion_tokens = result["token_ids"]
    response = {
        "id": "cmpl-synthetic-0",
        "object": "text_completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": result["text"],
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": len(completion_tokens),
            "total_tokens": prompt_token_count + len(completion_tokens),
        },
        "pto_backend": result["backend"],
        "pto_status": result["status"],
        "pto_launch_count": result["launch_count"],
    }
    if extra:
        response.update(extra)
    return response


def _openai_chat_completion_from_result(
    *,
    model: str,
    result: dict[str, Any],
    prompt_token_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion_tokens = result["token_ids"]
    response = {
        "id": "chatcmpl-synthetic-0",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": len(completion_tokens),
            "total_tokens": prompt_token_count + len(completion_tokens),
        },
        "pto_backend": result["backend"],
        "pto_status": result["status"],
        "pto_launch_count": result["launch_count"],
    }
    if extra:
        response.update(extra)
    return response


def create_generated_gluon_moe_launcher(
    *,
    output_dir: Path = DEFAULT_GENERATED_GLUON_OUTPUT_DIR,
    scale_a: float = 1.25,
    scale_b: float = 0.5,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    seed: int = 0,
) -> Callable[[KernelLaunchRequest], dict[str, Any]]:
    def launch(request: KernelLaunchRequest) -> dict[str, Any]:
        result = run_moe_expert_correctness(
            output_dir=output_dir / request.phase,
            arch=request.arch,
            n=request.n,
            scale_a=scale_a,
            scale_b=scale_b,
            atol=atol,
            rtol=rtol,
            seed=seed,
        )
        artifact = result.get("artifact", {})
        if not isinstance(artifact, dict):
            artifact = {}
        return {
            "status": result.get("status", "failed"),
            "phase": request.phase,
            "op": request.op,
            "launch_kind": "gluon-moe-expert",
            "kernel_name": result.get("kernel_name", "moe_expert_affine_f32"),
            "shape": result.get("shape", {"n": request.n}),
            "artifact": artifact,
            "source_sha256": artifact.get("source_sha256", ""),
            "generated_kernel": result,
        }

    return launch


def create_persistent_moe_dispatch_combine_launcher(
    *,
    scheduler_blocks: int = 1,
    worker_blocks: int = 4,
    queue_capacity: int = 5,
    stream_id: int = 0,
) -> Callable[[KernelLaunchRequest], dict[str, Any]]:
    def launch(request: KernelLaunchRequest) -> dict[str, Any]:
        try:
            result = run_moe_dispatch_combine(
                device=request.device_id,
                n=request.n,
                arch=request.arch,
                block_dim=request.block_dim,
                scheduler_blocks=scheduler_blocks,
                worker_blocks=worker_blocks,
                queue_capacity=queue_capacity,
                stream_id=stream_id,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "dag_shape": "graph_descriptor_moe_dispatch_combine",
                "n": request.n,
                "error_type": type(exc).__name__,
                "reason": _clean_error_text(str(exc)),
            }
        return _persistent_moe_dispatch_combine_launch_result(request, result)

    return launch


def _persistent_moe_dispatch_combine_launch_result(
    request: KernelLaunchRequest,
    result: dict[str, Any],
) -> dict[str, Any]:
    launch = {
        "status": result.get("status", "failed"),
        "phase": request.phase,
        "op": request.op,
        "launch_kind": PERSISTENT_MOE_DISPATCH_COMBINE_LAUNCH_KIND,
        "dag_shape": result.get("dag_shape", ""),
        "shape": {"n": result.get("n", request.n)},
        "completed_count": result.get("completed_count"),
        "max_abs_error": result.get("max_abs_error"),
        "scheduler_error_summary": result.get("device_scheduler_errors"),
        "gluon_expert_bridge": result.get("gluon_expert_bridge"),
        "task_body_digest": _persistent_moe_task_body_digest(result),
        "error_type": result.get("error_type", ""),
        "persistent_moe": result,
    }
    if "reason" in result:
        launch["reason"] = result["reason"]
    return launch


def _persistent_moe_task_body_digest(result: dict[str, Any]) -> dict[str, Any]:
    for task_body in result.get("task_bodies", []):
        if isinstance(task_body, dict) and task_body.get("func_id") == 12:
            return {
                "func_id": task_body.get("func_id"),
                "source_sha256": task_body.get("source_sha256", ""),
            }
    return {}


def _clean_error_text(text: str) -> str:
    return text.replace(ROOT.as_posix(), ".").replace(Path.home().as_posix(), "~")


def run_moe_expert_correctness(**kwargs) -> dict[str, Any]:
    from examples.cuda.gluon_moe_expert_affine import run_moe_expert_correctness

    return run_moe_expert_correctness(**kwargs)


def run_moe_dispatch_combine(**kwargs) -> dict[str, Any]:
    from examples.cuda.persistent_moe_dispatch_combine import run_moe_dispatch_combine

    return run_moe_dispatch_combine(**kwargs)


def default_cuda_seed_launcher(request: KernelLaunchRequest) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {"status": "skipped", "reason": "nvcc is required for CUDA seed launch"}
    if request.op != "add":
        return {"status": "skipped", "reason": f"unsupported CUDA seed op: {request.op}"}

    with tempfile.TemporaryDirectory(prefix="pypto-serving-cuda-seed-") as temp_dir:
        temp_path = Path(temp_dir)
        source = temp_path / "seed.cu"
        executable = temp_path / "seed"
        source.write_text(_cuda_seed_source(request), encoding="utf-8")
        compile_result = subprocess.run(
            [
                nvcc,
                f"-arch={request.arch}",
                str(source),
                "-o",
                str(executable),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if compile_result.returncode != 0:
            return {
                "status": "skipped",
                "phase": request.phase,
                "op": request.op,
                "reason": compile_result.stdout.strip(),
                "returncode": compile_result.returncode,
            }
        result = subprocess.run(
            [str(executable), str(request.device_id)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "failed" if result.returncode else "passed",
            "stdout": result.stdout,
        }
    if result.returncode != 0 and payload.get("status") != "skipped":
        payload = {
            **payload,
            "status": "skipped",
            "reason": payload.get("stdout", result.stdout).strip(),
            "returncode": result.returncode,
        }
    launch_status = payload.get("status", "passed")
    if launch_status == "pass":
        launch_status = "passed"
    return {
        "status": launch_status,
        "phase": request.phase,
        "op": request.op,
        "reason": payload.get("reason", ""),
        "cuda_seed": payload,
    }


def _cuda_seed_source(request: KernelLaunchRequest) -> str:
    return f"""\
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

__global__ void seed_add(float *out, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        out[idx] = static_cast<float>(idx) + 1.0f;
    }}
}}

int main(int argc, char **argv) {{
    int device = argc > 1 ? std::atoi(argv[1]) : {request.device_id};
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {{
        std::printf("{{\\"status\\":\\"skipped\\",\\"reason\\":\\"cudaSetDevice failed: %s\\"}}\\n", cudaGetErrorString(err));
        return 2;
    }}
    constexpr int n = {request.n};
    constexpr int block_dim = {request.block_dim};
    float *out = nullptr;
    err = cudaMalloc(&out, n * sizeof(float));
    if (err != cudaSuccess) {{
        std::printf("{{\\"status\\":\\"skipped\\",\\"reason\\":\\"cudaMalloc failed: %s\\"}}\\n", cudaGetErrorString(err));
        return 2;
    }}
    int grid = (n + block_dim - 1) / block_dim;
    seed_add<<<grid, block_dim>>>(out, n);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {{
        cudaFree(out);
        std::printf("{{\\"status\\":\\"skipped\\",\\"reason\\":\\"kernel failed: %s\\"}}\\n", cudaGetErrorString(err));
        return 2;
    }}
    float last = 0.0f;
    err = cudaMemcpy(&last, out + n - 1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(out);
    if (err != cudaSuccess) {{
        std::printf("{{\\"status\\":\\"skipped\\",\\"reason\\":\\"cudaMemcpy failed: %s\\"}}\\n", cudaGetErrorString(err));
        return 2;
    }}
    if (last != static_cast<float>(n)) {{
        std::printf("{{\\"status\\":\\"failed\\",\\"reason\\":\\"unexpected CUDA seed output\\"}}\\n");
        return 1;
    }}
    std::printf("{{\\"status\\":\\"pass\\",\\"runtime\\":\\"{request.runtime}\\",\\"ptx_arch\\":\\"{request.arch}\\",\\"op\\":\\"{request.op}\\",\\"n\\":%d,\\"block_dim\\":%d}}\\n", n, block_dim);
    return 0;
}}
"""


def _summarize_pypto_serving_source_stream(response, *, chat: bool) -> dict[str, Any]:
    data_items = _pypto_serving_source_sse_data(response.text)
    chunks: list[dict[str, Any]] = []
    done_seen = False
    finish_reason = None

    for item in data_items:
        if item == "[DONE]":
            done_seen = True
            continue
        chunk = json.loads(item)
        chunks.append(chunk)
        choices = chunk.get("choices", [])
        if choices:
            choice_finish_reason = choices[0].get("finish_reason")
            if choice_finish_reason is not None:
                finish_reason = choice_finish_reason

    if chat:
        deltas = [
            str((chunk["choices"][0].get("delta") or {}).get("content", ""))
            for chunk in chunks
            if chunk.get("choices")
        ]
        return {
            "event_count": len(data_items),
            "chunk_count": len(chunks),
            "done_seen": done_seen,
            "assistant_deltas": deltas,
            "assembled_assistant_text": "".join(deltas),
            "finish_reason": finish_reason,
        }

    text_chunks = [
        str(chunk["choices"][0].get("text", ""))
        for chunk in chunks
        if chunk.get("choices")
    ]
    return {
        "event_count": len(data_items),
        "chunk_count": len(chunks),
        "done_seen": done_seen,
        "assembled_text": "".join(text_chunks),
        "finish_reason": finish_reason,
    }


def _compat_non_stream_fixture(
    *,
    name: str,
    result: dict[str, Any],
    expected_route: str,
    expected_object: str,
    model: str,
    chat: bool,
) -> dict[str, Any]:
    response = result.get("response")
    if not isinstance(response, dict):
        return _compat_unavailable_fixture(name=name, result=result)

    choices = response.get("choices", [])
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    usage = response.get("usage")
    matches = {
        "route": result.get("route") == expected_route,
        "http_status_200": result.get("status_code") == 200,
        "object": response.get("object") == expected_object,
        "model": response.get("model") == model,
    }
    if chat:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            message = {}
        matches.update(
            {
                "message_role": message.get("role") == "assistant",
                "message_content": bool(message.get("content")),
            }
        )
        text_present = bool(message.get("content"))
    else:
        matches["choice_text"] = bool(choice.get("text"))
        text_present = bool(choice.get("text"))

    matches.update(
        {
            "finish_reason": choice.get("finish_reason") is not None,
            "usage": _has_openai_usage_shape(usage),
        }
    )
    return {
        "name": name,
        "status": _compat_status(result=result, matches=matches),
        "route": result.get("route"),
        "stream": False,
        "matches": matches,
        "observed": {
            "status_code": result.get("status_code"),
            "object": response.get("object"),
            "model": response.get("model"),
            "choice_count": len(choices) if isinstance(choices, list) else 0,
            "text_present": text_present,
            "finish_reason": choice.get("finish_reason"),
            "usage_keys": sorted(usage) if isinstance(usage, dict) else [],
            "pto_status": result.get("pto_status"),
            "pto_launch_count": result.get("pto_launch_count"),
        },
    }


def _compat_stream_fixture(
    *,
    name: str,
    result: dict[str, Any],
    expected_route: str,
    chat: bool,
) -> dict[str, Any]:
    matches = {
        "route": result.get("route") == expected_route,
        "http_status_200": result.get("status_code") == 200,
        "stream": result.get("stream") is True,
    }
    if chat:
        matches["assistant_delta"] = bool(result.get("assembled_assistant_text"))
        observed_text_key = "assembled_assistant_text"
    else:
        matches["choice_text_delta"] = bool(result.get("assembled_text"))
        observed_text_key = "assembled_text"

    matches.update(
        {
            "finish_reason": result.get("finish_reason") is not None,
            "sse_done": result.get("done_seen") is True,
        }
    )
    return {
        "name": name,
        "status": _compat_status(result=result, matches=matches),
        "route": result.get("route"),
        "stream": True,
        "matches": matches,
        "observed": {
            "status_code": result.get("status_code"),
            "event_count": result.get("event_count"),
            "chunk_count": result.get("chunk_count"),
            "done_seen": result.get("done_seen"),
            "finish_reason": result.get("finish_reason"),
            observed_text_key: result.get(observed_text_key, ""),
            "pto_status": result.get("pto_status"),
            "pto_launch_count": result.get("pto_launch_count"),
        },
    }


def _compat_unavailable_fixture(
    *,
    name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    status = result.get("status")
    if status not in {"passed", "failed", "skipped"}:
        status = "failed"
    return {
        "name": name,
        "status": status,
        "route": result.get("route"),
        "stream": bool(result.get("stream", False)),
        "matches": {},
        "observed": {
            "reason": result.get("reason", ""),
            "status_code": result.get("status_code"),
        },
    }


def _compat_status(*, result: dict[str, Any], matches: dict[str, bool]) -> str:
    result_status = result.get("status")
    if result_status == "skipped":
        return "skipped"
    if result_status == "failed":
        return "failed"
    return "passed"


def _has_openai_usage_shape(usage: Any) -> bool:
    return isinstance(usage, dict) and {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    } <= usage.keys()


def _pypto_serving_source_sse_data(text: str) -> list[str]:
    data_items = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_items.append(line.removeprefix("data:").strip())
    return data_items


def _encode_prompt(prompt: str) -> list[int]:
    return [0] if prompt else [0]


def _prompt_from_chat_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    has_user_content = False
    for message in messages:
        role = str(message.get("role", ""))
        content = _chat_message_content_text(message.get("content"))
        if role == "user" and content.strip():
            has_user_content = True
        if content:
            lines.append(f"{role}: {content}" if role else content)
    if not has_user_content:
        raise ValueError("chat completion requires at least one user message with content")
    return "\n".join(lines)


def _chat_message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _decode_tokens(token_ids: list[int]) -> str:
    vocab = {1: "N", 2: "V"}
    return "".join(vocab.get(token_id, "") for token_id in token_ids)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--model", default="synthetic-simpler-nv")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="compute_90")
    parser.add_argument(
        "--openai-completion",
        action="store_true",
        help="emit a synthetic OpenAI-compatible /v1/completions response",
    )
    parser.add_argument(
        "--openai-chat-completion",
        action="store_true",
        help="emit a synthetic OpenAI-compatible /v1/chat/completions response",
    )
    parser.add_argument(
        "--engine",
        action="store_true",
        help="route through the synthetic pypto-serving engine fixture",
    )
    parser.add_argument(
        "--http-fixture",
        action="store_true",
        help="exercise the synthetic FastAPI /v1/completions fixture in process",
    )
    parser.add_argument(
        "--pypto-serving-source",
        action="store_true",
        help="exercise the actual pypto-serving source server contract in process",
    )
    parser.add_argument(
        "--pypto-serving-source-stream",
        action="store_true",
        help="exercise the actual pypto-serving source /v1/completions stream route in process",
    )
    parser.add_argument(
        "--pypto-serving-source-chat",
        action="store_true",
        help="exercise the actual pypto-serving source /v1/chat/completions route in process",
    )
    parser.add_argument(
        "--pypto-serving-source-chat-stream",
        action="store_true",
        help="exercise the actual pypto-serving source /v1/chat/completions stream route in process",
    )
    parser.add_argument(
        "--pypto-serving-vllm-compat",
        action="store_true",
        help="emit source-route structural compatibility against vLLM OpenAI fields",
    )
    parser.add_argument(
        "--kernel-launcher",
        choices=(
            "cuda-seed",
            "gluon-moe-expert",
            PERSISTENT_MOE_DISPATCH_COMBINE_LAUNCH_KIND,
        ),
        default="cuda-seed",
        help="select the simpler-nv kernel launch path used by synthetic and source fixtures",
    )
    parser.add_argument(
        "--generated-output-dir",
        type=Path,
        default=DEFAULT_GENERATED_GLUON_OUTPUT_DIR,
        help="repo-relative output directory for generated Gluon kernel artifacts",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return non-zero when the selected kernel launch is skipped",
    )
    args = parser.parse_args(argv)
    kernel_launcher = _create_kernel_launcher_from_args(args)

    if args.pypto_serving_vllm_compat:
        result = run_pypto_serving_vllm_compat_fixture(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
        if status == "passed":
            status = _compat_pto_status(result)
    elif args.pypto_serving_source_chat_stream:
        result = run_pypto_serving_source_stream_chat_completion_fixture(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
        if status == "passed":
            status = result.get("pto_status", status)
    elif args.pypto_serving_source_stream:
        result = run_pypto_serving_source_stream_completion_fixture(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
        if status == "passed":
            status = result.get("pto_status", status)
    elif args.pypto_serving_source_chat:
        result = run_pypto_serving_source_chat_completion_fixture(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
        if status == "passed":
            status = result.get("pto_status", status)
    elif args.pypto_serving_source:
        result = run_pypto_serving_source_completion_fixture(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
        if status == "passed":
            status = result.get("pto_status", status)
    elif args.http_fixture:
        result = run_synthetic_http_completion_fixture(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            chat=args.openai_chat_completion,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
        if status == "passed" and isinstance(result.get("response"), dict):
            status = result["response"].get("pto_status", status)
    elif args.engine:
        engine = SyntheticPyptoServingEngine(
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        engine.init_model(args.model)
        if args.openai_chat_completion:
            result = engine.create_openai_chat_completion(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=args.max_new_tokens,
            )
            status = result["pto_status"]
        elif args.openai_completion:
            result = engine.create_openai_completion(
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_new_tokens,
            )
            status = result["pto_status"]
        else:
            result = engine.generate(
                model_id=args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            status = result["status"]
    elif args.openai_chat_completion:
        result = run_synthetic_openai_chat_completion(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["pto_status"]
    elif args.openai_completion:
        result = run_synthetic_openai_completion(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["pto_status"]
    else:
        result = run_synthetic_serving_request(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device_id=args.device,
            arch=args.arch,
            kernel_launcher=kernel_launcher,
        )
        status = result["status"]
    print(json.dumps(result, indent=2, sort_keys=True))
    if status == "failed":
        return 1
    if status == "skipped" and args.require_cuda:
        return 2
    return 0


def _create_kernel_launcher_from_args(args) -> Callable[[KernelLaunchRequest], dict[str, Any]] | None:
    if args.kernel_launcher == "cuda-seed":
        return None
    if args.kernel_launcher == "gluon-moe-expert":
        return create_generated_gluon_moe_launcher(output_dir=args.generated_output_dir)
    if args.kernel_launcher == PERSISTENT_MOE_DISPATCH_COMBINE_LAUNCH_KIND:
        return create_persistent_moe_dispatch_combine_launcher()
    raise ValueError(f"unsupported kernel launcher: {args.kernel_launcher}")


def _compat_pto_status(result: dict[str, Any]) -> str:
    for fixture in result.get("fixtures", []):
        if not isinstance(fixture, dict):
            continue
        observed = fixture.get("observed", {})
        if not isinstance(observed, dict):
            continue
        pto_status = observed.get("pto_status")
        if pto_status in {"failed", "skipped"}:
            return str(pto_status)
    return "passed"


if __name__ == "__main__":
    raise SystemExit(main())
