"""Static serving-baseline environment specifications."""

from __future__ import annotations

from typing import Any


ENVIRONMENT_SPECS: dict[str, dict[str, Any]] = {
    "vllm": {
        "title": "vLLM Isolated Runtime Environment",
        "dependency_sources": [
            "pyproject.toml",
            "requirements/build/cuda.txt",
            "requirements/common.txt",
            "requirements/cuda.txt",
        ],
        "critical_packages": [
            "cmake",
            "ninja",
            "setuptools-rust",
            "setuptools-scm",
            "torch",
            "torchvision",
            "pydantic",
            "cbor2",
            "flashinfer-python",
            "tilelang",
        ],
        "manual_packages": [
            {
                "name": "uvloop",
                "why": (
                    "The pinned api_server.py imports uvloop, but uvloop is "
                    "not declared in the inspected runtime requirement files."
                ),
            },
            {
                "name": "scipy",
                "why": (
                    "The --system-site-packages environment can otherwise "
                    "import an old system SciPy that is incompatible with the "
                    "NumPy version installed by the pinned vLLM dependency "
                    "set during Transformers/OpenAI entrypoint imports."
                ),
            },
            {
                "name": "pandas",
                "why": (
                    "H200 serving model inspection imports vllm._aiter_ops, "
                    "which imports pandas before the server is ready; an old "
                    "system pandas can be binary-incompatible with the env "
                    "NumPy under --system-site-packages."
                ),
            },
            {
                "name": "numexpr",
                "why": (
                    "Env-local pandas can still import a binary-incompatible "
                    "system numexpr unless the package is installed inside "
                    "the isolated vLLM environment."
                ),
            },
            {
                "name": "bottleneck",
                "why": (
                    "Env-local pandas can still import a binary-incompatible "
                    "system bottleneck unless the package is installed inside "
                    "the isolated vLLM environment."
                ),
            },
        ],
        "install_steps": [
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install --upgrade pip setuptools wheel",
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install uvloop",
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install "
            "-r requirements/common.txt -r requirements/cuda.txt",
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install "
            "-r requirements/build/cuda.txt",
        ],
        "preflight_steps": [
            "{env_python} "
            ".agents/skills/cuda-backend-eval/scripts/vllm_spinloop_preflight.py "
            "--source {build_source_path} --env-python {env_python}",
        ],
        "install_after_preflight_steps": [
            "REPO_ROOT=$PWD && "
            "VLLM_VERSION_OVERRIDE=$($REPO_ROOT/{env_python} -c "
            "\"import setuptools_scm; print(setuptools_scm.get_version(root='$REPO_ROOT/{source_path}'))\") && "
            "cd {build_source_path} && "
            "env VLLM_VERSION_OVERRIDE=$VLLM_VERSION_OVERRIDE "
            "PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install --no-build-isolation -e .",
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install 'scipy>=1.15.0' "
            "'pandas>=2.2.0' numexpr bottleneck",
        ],
        "source_overlay_steps": [
            "{env_python} "
            ".agents/skills/cuda-backend-eval/scripts/vllm_spinloop_source_overlay.py "
            "--source {source_path} --overlay {build_source_path}",
        ],
        "validation_modules": [
            "vllm",
            "vllm.entrypoints.cli.main",
            "vllm.entrypoints.openai.api_server",
            "vllm.engine.arg_utils",
            "vllm.model_executor.models.qwen3",
        ],
        "notes": [
            "The vLLM source declares torch==2.11.0 while the project venv "
            "may carry a different CUDA/PyTorch stack, so the serving "
            "baseline must use a dedicated environment under tmp/.",
            "Editable installation is kept in the isolated environment because "
            "the installed vllm module and console script are required before "
            "server and throughput runs.",
            "The Python 3.10 evaluation host builds vLLM from a copied source "
            "overlay that unsets Py_LIMITED_API for the spinloop CXX compile; "
            "the pinned upstream checkout under tmp/baselines is not modified.",
            "Editable install derives VLLM_VERSION_OVERRIDE from the pinned "
            "upstream checkout before changing into the overlay, because the "
            "overlay intentionally omits Git metadata.",
        ],
    },
    "sglang": {
        "title": "SGLang Isolated Runtime Environment",
        "dependency_sources": [
            "python/pyproject.toml",
        ],
        "critical_packages": [
            "torch",
            "torchvision",
            "orjson",
            "uvloop",
            "flashinfer_python",
            "tilelang",
        ],
        "manual_packages": [],
        "install_steps": [
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install --upgrade pip setuptools wheel",
        ],
        "preflight_steps": [],
        "install_after_preflight_steps": [
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install --no-build-isolation -e \"python[all]\"",
        ],
        "validation_modules": [
            "sglang",
            "orjson",
            "torchvision",
            "sglang.bench_serving",
            "sglang.bench_offline_throughput",
            "sglang.bench_one_batch",
        ],
        "notes": [
            "SGLang imports orjson and torchvision during benchmark module "
            "initialization, so dependency validation must run with "
            "PYTHONNOUSERSITE=1 to avoid user-site leakage.",
            "The pinned source declares torch==2.11.0 and CUDA 13 packages; "
            "installing it into the project venv would make PTO tests "
            "non-reproducible.",
        ],
    },
}
