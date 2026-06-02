# CUDA Examples

These examples are thin wrappers around the CUDA smoke paths used by the
NVIDIA backend evaluation. They are intentionally close to the benchmark
commands so reviewers can connect examples, docs, and artifacts directly.

## Review Map

- [Host-Schedule Vector Ops](docs/host-schedule-vector-ops.md) (21 lines)
- [Persistent Layered-Cross Graph](docs/persistent-layered-cross-graph.md) (20 lines)
- [Persistent Qwen Serving Scaffold](docs/persistent-qwen-serving-scaffold.md) (21 lines)
- [Qwen Serving Lifecycle Plan](docs/qwen-serving-lifecycle-plan.md) (22 lines)
- [Qwen KV-Cache Binding](docs/qwen-kv-cache-binding.md) (24 lines)
- [Qwen Decode Loop Runner](docs/qwen-decode-loop-runner.md) (88 lines)
- [Qwen Persistent Task Bodies](docs/qwen-persistent-task-bodies.md) (32 lines)
- [Qwen Unit Math Live](docs/qwen-unit-math-live.md) (30 lines)
- [Qwen Persistent Proxy Live](docs/qwen-persistent-proxy-live.md) (25 lines)
- [Qwen Persistent Microdecode Live](docs/qwen-persistent-microdecode-live.md) (28 lines)
- [Qwen Prompt Accounting](docs/qwen-prompt-accounting.md) (26 lines)
- [Qwen Runtime Input Binding](docs/qwen-runtime-input-binding.md) (25 lines)
- [Qwen CUDA Token Buffer Binding](docs/qwen-cuda-token-buffer-binding.md) (23 lines)
- [Qwen Persistent Decode Arguments](docs/qwen-persistent-decode-arguments.md) (22 lines)
- [Qwen Token Pointer Table](docs/qwen-token-pointer-table.md) (24 lines)
- [Qwen Weight Inventory](docs/qwen-weight-inventory.md) (24 lines)
- [Qwen Safetensors Shard Status](docs/qwen-safetensors-shard-status.md) (23 lines)
- [Qwen Safetensors Metadata Probe](docs/qwen-safetensors-metadata-probe.md) (24 lines)
- [Qwen CUDA Weight Binding](docs/qwen-cuda-weight-binding.md) (36 lines)
- [Qwen Persistent Weight Arguments](docs/qwen-persistent-weight-arguments.md) (23 lines)
- [Qwen Persistent Weight Materialization](docs/qwen-persistent-weight-materialization.md) (25 lines)
- [Qwen Resident Weight Table](docs/qwen-resident-weight-table.md) (23 lines)

## Contract

Each focused example page records the benchmark id, runtime id, method id,
command, expected output, and the review caveat for that executable artifact.
The executable examples remain in this directory; the split docs only keep the
review text short enough to audit.
