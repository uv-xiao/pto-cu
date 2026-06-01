from __future__ import annotations

import importlib.util
import sys

from .common import *  # noqa: F403


def check_viewer_schema_contract() -> None:
    validator_path = (
        ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    )
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load benchmark viewer data validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_viewer_data(ROOT)


def check_changelog_contract() -> None:
    validator_path = (
        ROOT / ".agents" / "checks" / "validate_nvidia_changelog.py"
    )
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_nvidia_changelog", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load NVIDIA changelog validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_changelog(ROOT)


def check_cuda_example_contract() -> None:
    validator_path = ROOT / ".agents" / "checks" / "validate_cuda_examples.py"
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_cuda_examples", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load CUDA example validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_examples(ROOT)


def check_remote_evaluation_contract() -> None:
    validator_path = ROOT / ".agents" / "checks" / "validate_remote_evaluation.py"
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_remote_evaluation", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load remote evaluation validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_remote_evaluation(ROOT)

