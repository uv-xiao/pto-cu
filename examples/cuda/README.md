# CUDA Examples

These examples are the representative CUDA entry points for NVIDIA backend
review. They are intentionally limited to the paths a reviewer should run
first: host-schedule launch, persistent-device scheduling, generated Qwen task
bodies, and the advanced Qwen decode-loop runner.

Development probes and narrow lifecycle checks may remain in this directory as
support code, but they are not part of the public example catalog unless they
become an essential end-to-end case.

## Review Map

- [Host-Schedule Vector Ops](docs/host-schedule-vector-ops.md) (21 lines)
- [Persistent Layered-Cross Graph](docs/persistent-layered-cross-graph.md) (20 lines)
- [Qwen Persistent Task Bodies](docs/qwen-persistent-task-bodies.md) (61 lines)
- [Qwen Decode Loop Runner](docs/qwen-decode-loop-runner.md) (99 lines)

## Contract

`manifest.json` is the source of truth for the review-facing examples. Each
entry records the benchmark id, runtime id, method id, command, expected
output, and implementation symbols that tie the example back to the CUDA
backend and evaluation artifacts.
