# CUDA Backend Evaluation History

This stable archive entry now points to focused history pages rather than
carrying every legacy capture in one long document. The active review entry
points remain [evaluation.md](../../evaluation.md) and
[evaluation-current.md](../../evaluation-current.md).

The linked pages preserve early CUDA backend microbenchmark evidence. They are
not end-to-end LLM serving results. The stable `legacy-captures.md` path is
kept because review guards and tests use it as the legacy-capture contract.

## Archive Map

- [Sources and baselines](legacy-captures/source-artifacts-and-baselines.md)
- [Launch and vector sweeps](legacy-captures/launch-and-vector-sweeps.md)
- [DAG shape results](legacy-captures/dag-shapes.md)
- [Legacy reproduction commands](legacy-captures/reproduction-commands.md)
- [Legacy next evaluation gaps](legacy-captures/next-evaluation-gaps.md)

## Review Contract

Use these pages as historical context for how CUDA launch overhead, persistent
executor shape, stream concurrency, and descriptor coverage were explored.
Current reviewable status should come from the active evaluation index and the
latest current-capture summary.
