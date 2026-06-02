from __future__ import annotations

from .contracts import (
    check_changelog_contract,
    check_cuda_example_contract,
    check_remote_evaluation_contract,
    check_viewer_schema_contract,
)
from .docs_goal import check_evaluation_docs, check_ultimate_goal_contract
from .policy import check_examples_and_rules, check_manual_ci_policy
from .status_gaps import check_remaining_gap_contract
from .viewer_data import check_viewer_data


def main() -> None:
    check_evaluation_docs()
    check_ultimate_goal_contract()
    check_viewer_data()
    check_remaining_gap_contract()
    check_viewer_schema_contract()
    check_changelog_contract()
    check_cuda_example_contract()
    check_remote_evaluation_contract()
    check_examples_and_rules()
    check_manual_ci_policy()
    print("nvidia review guard passed")
