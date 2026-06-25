# Persistent MoE Dispatch/Combine H200 Note

This note tracks the first PR-sized persistent-device MoE dispatch/combine
slice after the Gluon MoE expert primitive.

## Scope

- Example: `examples/cuda/persistent_moe_dispatch_combine.py`
- Runtime shape: `graph_descriptor_moe_dispatch_combine`
- Tasks: four expert transforms and one weighted combine
- Default expert 0 path: `gluon_gen` persistent task-body bridge for
  `moe_expert_affine_f32` as func id `12`
- Dependency behavior: device-side fan-in releases the combine task after all
  four expert tasks complete
- Output: structured JSON for local skip/pass/fail and remote pass/fail
  evidence, including top-level `gluon_expert_bridge` metadata and the
  matching `task_bodies` entry

This is not distributed expert parallelism, a serving integration, DeepSeek
inference, direct Triton/Gluon JIT linking into the persistent kernel, or a
performance claim.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --output-json tmp/persistent-moe-dispatch-combine-local.json
```

The local command is skip-safe when CUDA tooling or a visible NVIDIA GPU is not
available. Add `--require-cuda` when a skip should fail the command.

## Remote H200 Command Shape

Use a fresh synced remote checkout or an intentionally refreshed remote branch,
then run:

```bash
REMOTE_PTO_CU=<remote-checkout> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  <python> examples/cuda/persistent_moe_dispatch_combine.py \
    --device 0 --n 4096 --arch compute_90 --require-cuda
```

Record the exact remote checkout path, Python environment, command output,
pass/fail result, and these JSON fields in the PR body and worker handoff:
`status`, `dag_shape`, `completed_count`, `max_abs_error`,
`device_scheduler_errors`, `fanin_remaining`, `gluon_expert_bridge`, the
`task_bodies` entry for func id `12`, and `artifact.source_kind`. Do not treat
a skipped or setup-failed remote command as H200 correctness evidence.

For a bounded same-node two-device baseline, run the same example with
`--device-ids`:

```bash
REMOTE_PTO_CU=<remote-checkout> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    python3 examples/cuda/persistent_moe_dispatch_combine.py \
    --device-ids 6,7 --n 4096 --arch compute_90 --require-cuda'
```

That mode runs the existing graph independently on each requested CUDA device
in isolated child processes and aggregates output, scheduler, completion,
fan-in, and source/bridge metadata validation. It is same-node two-device
baseline evidence, not fused cross-GPU expert-parallel MoE.

## Single-Device Remote H200 Result

Run method: generic remote runner with `--sync` into a temporary remote
checkout, using plain system `python3`.

```bash
REMOTE_PTO_CU=/tmp/pto-cu-persistent-moe-gluon-wrapper \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    python3 examples/cuda/persistent_moe_dispatch_combine.py \
    --device 0 --n 4096 --arch compute_90 --require-cuda \
    --output-json tmp/persistent-moe-gluon-wrapper-h200.json'
```

Result: pass.

- `status`: `passed`
- `dag_shape`: `graph_descriptor_moe_dispatch_combine`
- `arch`: `compute_90`
- `completed_count`: `5`
- `fanin_remaining`: `[0, 0, 0, 0, 0]`
- `device_scheduler_errors`: `{"count": 0, "code": 0, "task_id": 0}`
- `max_abs_error`: `0.0`
- `gluon_expert_bridge`: func id `12`, kernel
  `moe_expert_affine_f32`, task `gluon_moe_expert_affine_f32`, source kind
  `gluon-persistent-task-body-bridge`, source sha256
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `task_bodies` func id `12`: name `gluon_moe_expert_affine_f32`, source
  kind `gluon-persistent-task-body-bridge`, source sha256
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `artifact.source_kind`: `generated-dispatch`

This validates only the synthetic single-process persistent-device graph shape
above. It does not validate distributed expert parallelism, serving, DeepSeek
inference, direct Triton/Gluon JIT linking into the persistent kernel, or
performance.

## Two-Device Remote H200 Result

Run method: generic remote runner with `--sync` into a temporary remote
checkout, using plain system `python3`. The remote checkout was a synced
working tree; remote Git metadata was not used for this evidence capture.

Environment:

```text
machine class: NVIDIA H200 host
devices: 6,7
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
nvcc: Build cuda_12.8.r12.8/compiler.35404655_0
python: 3.12.3
source sync: --sync into /tmp/pto-cu-codex-restart
```

Command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    python3 examples/cuda/persistent_moe_dispatch_combine.py \
    --device-ids 6,7 --n 4096 --arch compute_90 --require-cuda \
    --output-json tmp/persistent-moe-two-h200-baseline.json'
```

Result: pass.

- `status`: `passed`
- `evidence_scope`: `same-node-two-device-baseline`
- `evidence_statement`: same-node two-device baseline evidence; not fused
  cross-GPU expert-parallel MoE
- `device_ids`: `[6, 7]`
- `per_device_count`: `2`
- validation:
  - `all_devices_passed`: `true`
  - `completed_count_is_5`: `true`
  - `scheduler_errors_zero`: `true`
  - `fanin_remaining_zero`: `true`
  - `source_digests_match`: `true`
  - `bridge_metadata_match`: `true`
- source digests:
  - `dispatch_source_sha256`:
    `c096ede6d4ab5e1a9a33070bc1fcf988b9fb9c405d929a770c962308b396b209`
  - `gluon_expert_bridge_sha256`:
    `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
  - `task_body_func12_sha256`:
    `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`

Per-device validation:

- Device `6`: status `passed`, max error `0.0`, completed `5`,
  scheduler errors `{count: 0, code: 0, task_id: 0}`, fan-in
  `[0, 0, 0, 0, 0]`, host `2446295` ns, device `46144` ns.
- Device `7`: status `passed`, max error `0.0`, completed `5`,
  scheduler errors `{count: 0, code: 0, task_id: 0}`, fan-in
  `[0, 0, 0, 0, 0]`, host `60360` ns, device `49888` ns.

The aggregate command records the same `gluon_expert_bridge` metadata and the
matching func id `12` `task_bodies` digest on both devices. This remains a
same-node independent two-device baseline. It does not validate distributed
expert parallelism, fused cross-GPU MoE dispatch/combine, NCCL or UCCL
transport, serving, DeepSeek inference, or performance.

## Communication-Coupled Handoff Gate

Run method: generic remote runner with `--sync` into a temporary remote
checkout. The synced checkout did not initially have `.venv`, so the command
creates the project-local venv and installs the build dependencies before the
editable package install.

Environment:

```text
machine class: NVIDIA H200 host
devices: 6,7
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
nvcc: Build cuda_12.8.r12.8/compiler.35404655_0
python: 3.12.3
source sync: --sync into /tmp/pto-cu-persistent-moe-nccl-handoff
```

Command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-persistent-moe-nccl-handoff \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    source .venv/bin/activate && \
    pip install scikit-build-core nanobind cmake ninja nvidia-nccl-cu12 \
      >/tmp/pto-cu-build-deps-install.log && \
    pip install --no-build-isolation -e . >/tmp/pto-cu-pip-install.log && \
    NCCL_DEBUG=WARN PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
      --device-ids 6,7 --n 4096 --arch compute_90 \
      --with-nccl-handoff --tensor-numel 1024 --build --require-cuda'
```

Result: pass. The command exited with status `0`.

- `status`: `passed`
- `handoff_scope`: `persistent-moe-plus-nccl-worker-control`
- `device_ids`: `[6, 7]`
- `tensor_numel`: `1024`
- `persistent_moe.status`: `passed`
- `persistent_moe.evidence_scope`: `same-node-two-device-baseline`
- `persistent_moe.per_device_count`: `2`
- `persistent_moe_max_abs_error`: `0.0`
- `persistent_moe_validation.all_devices_passed`: `true`
- `persistent_moe_validation.completed_count_is_5`: `true`
- `persistent_moe_validation.scheduler_errors_zero`: `true`
- `persistent_moe_validation.fanin_remaining_zero`: `true`
- `persistent_moe_validation.source_digests_match`: `true`
- `persistent_moe_validation.bridge_metadata_match`: `true`
- `persistent_moe_source_digests.dispatch_source_sha256`:
  `c096ede6d4ab5e1a9a33070bc1fcf988b9fb9c405d929a770c962308b396b209`
- `persistent_moe_source_digests.gluon_expert_bridge_sha256`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `persistent_moe_source_digests.task_body_func12_sha256`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `nccl_worker_control.status`: `passed`
- `nccl_worker_control.transport`: `worker_control`
- `nccl_worker_control.backend`: `nccl`
- `nccl_worker_control.world_size`: `2`
- `nccl_worker_control.operations`:
  `[all_reduce, reduce_scatter, all_gather, send_recv]`
- `nccl_worker_control_max_abs_error`: `0.0`
- `nccl_worker_control_validation.all_reduce_passed`: `true`
- `nccl_worker_control_validation.reduce_scatter_passed`: `true`
- `nccl_worker_control_validation.all_gather_passed`: `true`
- `nccl_worker_control_validation.send_recv_passed`: `true`
- `nccl_worker_control_validation.max_abs_error_zero`: `true`
- `handoff_validation.same_device_ids`: `true`
- `handoff_validation.persistent_moe_passed`: `true`
- `handoff_validation.nccl_worker_control_passed`: `true`
- `handoff_validation.persistent_moe_validation_passed`: `true`
- `handoff_validation.nccl_worker_control_validation_passed`: `true`
- `handoff_validation.source_digests_present`: `true`
- `handoff_validation.bridge_digests_match`: `true`
- `handoff_boundary.nccl_capability_id`:
  `nccl:rank0->cuda6,rank1->cuda7`

This is a communication-coupled review gate that composes two existing paths
on the same H200 device pair. It proves that one command validates the
persistent MoE graph on both devices, validates descriptor-backed NCCL
worker-control operations on the same devices, and reports an explicit
handoff boundary tying the results together. It is not fused cross-GPU
expert-parallel MoE, serving, DeepSeek/vLLM integration, UCCL/RDMA,
multi-node evidence, or a performance claim.

## UCCL-EP Adapter Handoff Gate

Run method: generic remote runner with `--sync` into a temporary remote
checkout. The synced checkout creates a project-local venv and installs the
editable package. The UCCL-EP adapter is then launched through
`torch.distributed.run` from the handoff helper. The remote UCCL-capable
Python site packages and external UCCL EP bench helper were provided as
dependency paths for this run.

Environment:

```text
machine class: NVIDIA H200 host
devices: 6,7
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
nvcc: Build cuda_12.8.r12.8/compiler.35404655_0
python: 3.12.3
source sync: --sync into /tmp/pto-cu-persistent-moe-uccl-ep-handoff
UCCL_EP_BENCH_DIR: external UCCL EP bench helper
```

Command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-persistent-moe-uccl-ep-handoff \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    source .venv/bin/activate && \
    pip install scikit-build-core nanobind cmake ninja nvidia-nccl-cu12 \
      >/tmp/pto-cu-uccl-ep-build-deps-install.log && \
    pip install --no-build-isolation -e . \
      >/tmp/pto-cu-uccl-ep-pip-install.log && \
    UCCL_EP_BENCH_DIR=<external-uccl-ep-bench>/ep/bench \
    NCCL_DEBUG=WARN \
    PYTHONPATH=$PWD:$PWD/python:<uccl-python-site-packages> \
    .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
      --device-ids 6,7 --n 4096 --arch compute_90 \
      --with-uccl-ep-handoff --tensor-numel 1024 --require-cuda'
```

Result: pass. The command exited with status `0`.

- `status`: `passed`
- `handoff_scope`: `persistent-moe-plus-uccl-ep-adapter`
- `device_ids`: `[6, 7]`
- `tensor_numel`: `1024`
- `persistent_moe.status`: `passed`
- `persistent_moe.evidence_scope`: `same-node-two-device-baseline`
- `persistent_moe.per_device_count`: `2`
- `persistent_moe_max_abs_error`: `0.0`
- `persistent_moe_validation.all_devices_passed`: `true`
- `persistent_moe_validation.completed_count_is_5`: `true`
- `persistent_moe_validation.scheduler_errors_zero`: `true`
- `persistent_moe_validation.fanin_remaining_zero`: `true`
- `persistent_moe_validation.source_digests_match`: `true`
- `persistent_moe_validation.bridge_metadata_match`: `true`
- `persistent_moe_source_digests.dispatch_source_sha256`:
  `c096ede6d4ab5e1a9a33070bc1fcf988b9fb9c405d929a770c962308b396b209`
- `persistent_moe_source_digests.gluon_expert_bridge_sha256`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `persistent_moe_source_digests.task_body_func12_sha256`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `uccl_ep_adapter.status`: `passed`
- `uccl_ep_adapter.transport`: `ep`
- `uccl_ep_adapter.operation`: `ep_dispatch_combine`
- `uccl_ep_adapter.world_size`: `2`
- `uccl_ep_adapter.descriptor.hidden`: `1024`
- `uccl_ep_adapter.descriptor.num_tokens`: `64`
- `uccl_ep_adapter.descriptor.num_topk`: `4`
- `uccl_ep_adapter.descriptor.num_experts`: `16`
- `uccl_ep_adapter.descriptor.input_dtype`: `bf16`
- `uccl_ep_adapter.descriptor.metadata_shapes.topk_idx`: `[64, 4]`
- `uccl_ep_adapter_max_abs_error`: `0.0`
- `uccl_ep_adapter_topk_weight_error`: `0.0`
- `uccl_ep_adapter_validation.adapter_passed`: `true`
- `uccl_ep_adapter_validation.transport_is_ep`: `true`
- `uccl_ep_adapter_validation.operation_is_dispatch_combine`: `true`
- `uccl_ep_adapter_validation.descriptor_metadata_present`: `true`
- `uccl_ep_adapter_validation.all_ranks_passed`: `true`
- `uccl_ep_adapter_validation.max_abs_error_zero`: `true`
- `uccl_ep_adapter_validation.topk_weight_error_zero`: `true`
- `handoff_validation.same_device_ids`: `true`
- `handoff_validation.persistent_moe_passed`: `true`
- `handoff_validation.uccl_ep_adapter_passed`: `true`
- `handoff_validation.persistent_moe_validation_passed`: `true`
- `handoff_validation.uccl_ep_adapter_validation_passed`: `true`
- `handoff_validation.source_digests_present`: `true`
- `handoff_validation.bridge_digests_match`: `true`
- `handoff_validation.adapter_descriptor_metadata_present`: `true`
- `handoff_validation.max_errors_zero`: `true`
- `payload_provenance.uccl_ep_adapter.producer`:
  `uccl_ep_dispatch_combine_adapter`
- `payload_provenance.uccl_ep_adapter.capability_id`:
  `uccl:rank0->cuda6,rank1->cuda7`
- `payload_provenance.uccl_ep_adapter.descriptor.num_tokens`: `64`
- `payload_provenance.uccl_ep_adapter.descriptor.hidden`: `1024`
- `payload_provenance.uccl_ep_adapter.descriptor.num_topk`: `4`
- `payload_provenance.uccl_ep_adapter.descriptor.num_experts`: `16`
- `payload_provenance.uccl_ep_adapter.descriptor.input_dtype`: `bf16`
- `payload_provenance.uccl_ep_adapter.descriptor.metadata_shapes.topk_idx`:
  `[64, 4]`
- `payload_provenance.uccl_ep_adapter.rank_results[0].recv_tokens`:
  `[88]`
- `payload_provenance.uccl_ep_adapter.rank_results[1].recv_tokens`:
  `[88]`
- `payload_provenance.persistent_device_graph.graph_descriptor_id`:
  `graph_descriptor_moe_dispatch_combine`
- `payload_provenance.persistent_device_graph.device_ids`: `[6, 7]`
- `payload_provenance.persistent_device_graph.rank_to_device`:
  `{"0": 6, "1": 7}`
- `payload_provenance.persistent_device_graph.source_digests`:
  recorded from the persistent-device aggregate result
- `payload_provenance.persistent_device_graph.bridge_digest`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `payload_provenance.shared_payload_ownership.exists`: `false`
- `payload_provenance.shared_payload_ownership.ownership_token`: `null`
- `payload_provenance.shared_payload_ownership.lifetime_transition_log`:
  `[]`
- `handoff_boundary.uccl_capability_id`:
  `uccl:rank0->cuda6,rank1->cuda7`
- `uccl_ep_adapter.rank_results[0].recv_tokens`: `[88]`
- `uccl_ep_adapter.rank_results[1].recv_tokens`: `[88]`

This is a communication-coupled review gate that composes two existing paths
on the same H200 device pair. It proves that one command validates the
persistent MoE graph on both devices, validates the Python-side UCCL-EP
dispatch/combine adapter on the same devices, records descriptor metadata,
and reports an explicit handoff boundary tying the results together. It does
not validate distributed expert parallelism, fused cross-GPU MoE
dispatch/combine, CUDA host-runtime UCCL dispatch, RDMA, multi-node transport,
serving, DeepSeek/vLLM integration, or performance.
It does not validate CUDA host-runtime UCCL dispatch.

## Reduced Fused Boundary Gate

This slice adds an explicit fused-boundary mode without relabeling the
accepted UCCL-EP handoff as fused evidence. The command routes the same
bounded payload through the persistent MoE graph and UCCL-EP adapter path in
one H200 command, then records the reduced fused cross-GPU expert-parallel MoE
boundary as a structured unsupported boundary. The implementation anchor is
`run_persistent_moe_uccl_ep_fused_boundary` in
`examples/cuda/persistent_moe_dispatch_combine.py`.

Command shape:

```bash
REMOTE_PTO_CU=<remote-checkout> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    source .venv/bin/activate && \
    pip install scikit-build-core nanobind cmake ninja nvidia-nccl-cu12 \
      >/tmp/pto-cu-uccl-ep-fused-boundary-build-deps-install.log && \
    pip install --no-build-isolation -e . \
      >/tmp/pto-cu-uccl-ep-fused-boundary-pip-install.log && \
    UCCL_SITE=$PWD/.venv/lib/python3.12/site-packages && \
    mkdir -p "$UCCL_SITE/uccl" && \
    cp <external-uccl-ep-bench>/uccl/__init__.py \
      "$UCCL_SITE/uccl/__init__.py" && \
    cp <external-uccl-ep-bench>/ep/build/lib.linux-x86_64-cpython-312/ep*.so \
      "$UCCL_SITE/uccl/" && \
    UCCL_EP_BENCH_DIR=<external-uccl-ep-bench>/ep/bench \
    NCCL_DEBUG=WARN \
    PYTHONPATH=$PWD:$PWD/python:$UCCL_SITE:<uccl-python-site-packages> \
    .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
      --device-ids 6,7 --n 4096 --arch compute_90 \
      --with-uccl-ep-fused-boundary --tensor-numel 1024 \
      --require-cuda \
      --output-json tmp/persistent-moe-uccl-ep-payload-provenance-h200.json'
```

Fresh payload-provenance evidence for this slice used
`REMOTE_PTO_CU=/tmp/pto-cu-uccl-ep-adapter-payload-provenance` with
`--sync`. The external UCCL checkout and external Torch site-packages path are
intentionally represented by the placeholders in the command above. The JSON
artifact path in the synced checkout was
`tmp/persistent-moe-uccl-ep-payload-provenance-h200.json`.

Run method: generic remote runner with `--sync` into a temporary remote
checkout. The synced checkout creates a project-local venv, installs this repo
editable, copies the prebuilt UCCL-EP extension from
`<external-uccl-ep-bench>` into that venv, and uses
`<uccl-python-site-packages>` only as an external Python dependency path.

Environment:

```text
machine class: NVIDIA H200 host
devices: 6,7
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
python: 3.12.3
source sync: --sync into /tmp/pto-cu-uccl-ep-adapter-payload-provenance
UCCL dependency path policy: sanitized external UCCL checkout plus
  project-local venv copy of uccl.ep; external Torch site-packages via
  PYTHONPATH
artifact: tmp/persistent-moe-uccl-ep-payload-provenance-h200.json
```

Result: structured unsupported boundary. The command exited with status `3`.

- `status`: `unsupported`
- `fused_boundary_scope`:
  `reduced-fused-cross-gpu-expert-parallel-moe-boundary`
- `handoff_scope`: `persistent-moe-plus-uccl-ep-adapter`
- `persistent_device_uccl_ep_runtime_fusion.status`: `unsupported`
- `persistent_device_uccl_ep_runtime_fusion.actual_fused_cross_gpu_execution`:
  `false`
- `persistent_device_uccl_ep_runtime_fusion.shared_ownership_token`: `null`
- `persistent_device_uccl_ep_runtime_fusion.payload_lifetime_transition_log`:
  `[]`
- `persistent_device_uccl_ep_runtime_fusion.reason`: runtime component has not
  created or transferred shared payload ownership
- `boundary_validation.handoff_passed`: `true`
- `boundary_validation.actual_fused_cross_gpu_execution`: `false`
- `boundary_validation.structured_unsupported_boundary`: `true`
- `missing_boundaries`: includes
  `persistent_device_uccl_ep_runtime_fusion`,
  `shared_dispatch_combine_payload_between_persistent_graph_and_uccl_ep`, and
  `device_side_cross_gpu_expert_parallel_routing`
- `persistent_moe_validation.all_devices_passed`: `true`
- `persistent_moe_validation.completed_count_is_5`: `true`
- `persistent_moe_validation.scheduler_errors_zero`: `true`
- `persistent_moe_validation.fanin_remaining_zero`: `true`
- `persistent_moe_validation.source_digests_match`: `true`
- `persistent_moe_validation.bridge_metadata_match`: `true`
- `uccl_ep_adapter_validation.adapter_passed`: `true`
- `uccl_ep_adapter_validation.transport_is_ep`: `true`
- `uccl_ep_adapter_validation.operation_is_dispatch_combine`: `true`
- `uccl_ep_adapter_validation.descriptor_metadata_present`: `true`
- `uccl_ep_adapter_validation.all_ranks_passed`: `true`
- `uccl_ep_adapter_validation.max_abs_error_zero`: `true`
- `uccl_ep_adapter_validation.topk_weight_error_zero`: `true`
- `uccl_ep_adapter.rank_results[0].recv_tokens`: `[88]`
- `uccl_ep_adapter.rank_results[1].recv_tokens`: `[88]`
- `payload_provenance.uccl_ep_adapter.capability_id`:
  `uccl:rank0->cuda6,rank1->cuda7`
- `payload_provenance.uccl_ep_adapter.descriptor.metadata_shapes.topk_idx`:
  `[64, 4]`
- `payload_provenance.uccl_ep_adapter.rank_results[0].recv_tokens`:
  `[88]`
- `payload_provenance.uccl_ep_adapter.rank_results[1].recv_tokens`:
  `[88]`
- `payload_provenance.persistent_device_graph.graph_descriptor_id`:
  `graph_descriptor_moe_dispatch_combine`
- `payload_provenance.persistent_device_graph.source_digests`:
  recorded from the persistent-device aggregate result
- `payload_provenance.shared_payload_ownership.exists`: `false`
- `payload_provenance.shared_payload_ownership.ownership_token`: `null`
- `payload_provenance.shared_payload_ownership.lifetime_transition_log`:
  `[]`
- `persistent_moe_source_digests.dispatch_source_sha256`:
  `c096ede6d4ab5e1a9a33070bc1fcf988b9fb9c405d929a770c962308b396b209`
- `persistent_moe_source_digests.gluon_expert_bridge_sha256`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `persistent_moe_source_digests.task_body_func12_sha256`:
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `evidence_statement`: structured unsupported boundary and non-evidence for
  fused cross-GPU expert-parallel MoE

This is non-evidence for actual fused cross-GPU expert-parallel MoE. It does
not validate distributed expert parallelism, CUDA host-runtime UCCL dispatch,
RDMA, multi-node transport, serving, DeepSeek/vLLM integration, throughput, or
latency. A later implementation must add the
`persistent_device_uccl_ep_runtime_fusion` boundary before this can become
fused evidence.

The current provenance fields are dependency evidence only. The UCCL-EP
adapter provenance is copied from the adapter result's capability, descriptor,
metadata shapes, and rank results. The persistent-device graph provenance is
copied from the existing graph descriptor id, device ids, rank/device mapping,
source digests, and bridge digest. No runtime component creates or transfers a
shared ownership token, so the result explicitly reports no ownership token
and an empty payload lifetime transition log.

## Runtime-Owned Descriptor Implementation Handoff

The `nvidia-uccl-ep-runtime-fusion-impl-runtime-owned-descriptor` worker
audited the current implementation surface and found no real
`persistent_device_uccl_ep_runtime_fusion` coordinator behind the CUDA
runtime / `ChipWorker` boundary. The compiled
`src/cuda/runtime/persistent_device/` files are placeholders, and the runnable
persistent-device path is generated PTX launched by
`src/cuda/platform/onboard/host/pto_runtime_c_api.cpp::CudaDeviceRunner`.
That runner accepts persistent DAG args, but it does not own a UCCL-EP
dispatch/combine payload descriptor or transfer that descriptor to a UCCL-EP
runtime component.

Because the lower-level boundary is missing, this slice keeps the normal
`--with-uccl-ep-fused-boundary` status `unsupported`. It adds local guard
coverage in `examples/cuda/persistent_moe_dispatch_combine.py`:

- `_validate_runtime_fusion_evidence` accepts only a runtime-owned descriptor
  produced by `persistent_device_uccl_ep_runtime_fusion`;
- missing ownership tokens, mismatched tokens, double release,
  use-after-release, leaked ownership, and rank/device mismatches produce
  `status: failed` with populated `failure_fields`;
- pass-like runtime-fusion fields copied from UCCL-EP adapter output,
  payload provenance, or handoff metadata are rejected with
  `fabricated_or_untrusted_pass_evidence`;
- the normal unsupported result records
  `failure_fields.unsupported_boundary:
  persistent_device_uccl_ep_runtime_fusion`;
- `actual_fused_cross_gpu_execution` remains `false` unless the trusted
  runtime-owned guard passes.

No fresh H200 fused-boundary run is recorded for this blocked implementation
handoff because the code still cannot truthfully emit real runtime-owned
descriptor evidence. The last H200 fused-boundary artifact remains the PR #147
payload-provenance result above, which exited `unsupported` and is not fused
execution evidence.

## Implementation-Readiness Map

The next implementation attempt must keep the shared payload descriptor behind
the existing CUDA runtime and `ChipWorker` boundary. The runtime-owned
descriptor may live in the persistent-device CUDA runtime run context, with a
host-side control record and a device-visible descriptor buffer created for one
`ChipWorker::run` invocation. Python-side UCCL-EP adapter provenance and
persistent-device graph provenance remain inputs to the result, not ownership
evidence.

The persistent-device/UCCL-EP runtime fusion coordinator records the
ownership token and payload lifetime transition log. It must issue one shared
token for the dispatch/combine payload, validate that dispatch and combine use
that token, and record transitions through `allocated`, `dispatch_ready`,
`dispatch_in_flight`, `combine_ready`, `combine_in_flight`, `complete`, and
`released`. Missing token, mismatched token, illegal transition, double
release, use-after-release, or leaked in-flight ownership is a failure.

Mandatory failure states for the future result are setup failure, unsupported
boundary, descriptor mismatch, rank/device mismatch, payload lifetime failure,
transport failure, persistent-device scheduler failure, numeric validation
failure, and stale or incomplete evidence. Non-evidence states include
`unsupported`, `setup_failed`, `failed`, adapter-only pass, independent
two-device persistent MoE pass, NCCL worker-control pass, provenance without a
shared ownership token, empty lifetime transition log, and any result where
`actual_fused_cross_gpu_execution` is `false`.

Local evidence required before a later implementation reports
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`:

- focused tests that reject `passed` when the shared ownership token is
  missing, mismatched, double released, used after release, or left in flight;
- focused tests that reject `passed` when rank/device mapping differs between
  the persistent-device graph and UCCL-EP runtime;
- review-artifact tests that require the pass result to include dispatch and
  combine descriptors, ownership token, lifetime transition log, failure
  fields, and non-claims;
- the NVIDIA review guard and `test_nvidia_review_artifacts.py`.

H200 evidence required before those fields may report passed/true is one fresh
remote command using the fused-boundary path:

```bash
REMOTE_PTO_CU=<remote-checkout> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    source .venv/bin/activate && \
    pip install scikit-build-core nanobind cmake ninja nvidia-nccl-cu12 \
      >/tmp/pto-cu-uccl-ep-fusion-build-deps-install.log && \
    pip install --no-build-isolation -e . \
      >/tmp/pto-cu-uccl-ep-fusion-pip-install.log && \
    UCCL_SITE=$PWD/.venv/lib/python3.12/site-packages && \
    mkdir -p "$UCCL_SITE/uccl" && \
    cp <external-uccl-ep-bench>/uccl/__init__.py \
      "$UCCL_SITE/uccl/__init__.py" && \
    cp <external-uccl-ep-bench>/ep/build/lib.linux-x86_64-cpython-312/ep*.so \
      "$UCCL_SITE/uccl/" && \
    UCCL_EP_BENCH_DIR=<external-uccl-ep-bench>/ep/bench \
    NCCL_DEBUG=WARN \
    PYTHONPATH=$PWD:$PWD/python:$UCCL_SITE:<uccl-python-site-packages> \
    .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
      --device-ids 6,7 --n 4096 --arch compute_90 \
      --with-uccl-ep-fused-boundary --tensor-numel 1024 \
      --require-cuda \
      --output-json tmp/persistent-moe-uccl-ep-runtime-fusion-h200.json'
```

That H200 result must show `status: passed`,
`persistent_device_uccl_ep_runtime_fusion.status: passed`,
`actual_fused_cross_gpu_execution: true`, matching rank/device mapping,
zero persistent MoE and UCCL-EP validation errors, a non-null shared ownership
token, a complete lifetime transition log, and no failure fields set.

## Coordinator Boundary Map

PR #150 confirmed that guards alone cannot implement the boundary because the
runtime-owned coordinator is absent. The next honest implementation dependency
is therefore a concrete map for where the coordinator enters and what evidence
it owns.

Coordinator owner:
`persistent_device_uccl_ep_runtime_fusion`, created inside the CUDA
persistent-device runtime run context for exactly one `ChipWorker::run`
invocation.

Reviewable entry point:

1. L3+ scheduling still dispatches an opaque chip task through the normal
   `WorkerThread` mailbox path.
2. The chip child calls `ChipWorker::run` with the callable id, decoded
   `TaskArgsView`, and `CallConfig`.
3. `ChipWorker::run` builds `ChipStorageTaskArgs` and calls the CUDA
   host-runtime `.so`.
4. The persistent-device runtime run context creates the coordinator when the
   callable and private CUDA communication metadata opt into UCCL-EP fusion.
5. The coordinator allocates the shared dispatch/combine descriptor, emits the
   ownership token, validates transitions, and writes
   `persistent_device_uccl_ep_runtime_fusion` result fields.

Descriptor allocation site:
the CUDA persistent-device runtime run context allocates a host-side control
record and device-visible descriptor buffer before launching the
persistent-device scheduler. The example, adapter result, and provenance JSON
may describe payloads, but they do not allocate or own the shared descriptor.

Ownership token issuer:
only the coordinator issues the token. Dispatch and combine descriptors must
carry the same token, and the final result must show that the token was
released after `complete`.

Lifetime transition state machine:

```text
allocated
  -> dispatch_ready
  -> dispatch_in_flight
  -> combine_ready
  -> combine_in_flight
  -> complete
  -> released
```

Failure-field responsibilities:

- `unsupported_boundary`: missing coordinator or missing runtime fusion
  capability.
- `descriptor`: descriptor shape, dtype, metadata, allocation, or ownership
  mismatch.
- `rank_device`: graph, UCCL capability, and Worker-local device ordering
  disagree.
- `payload_lifetime`: missing token, mismatched token, illegal transition,
  double release, use-after-release, leaked owner, or missing release.
- `transport`: UCCL-EP runtime dispatch/combine failure.
- `scheduler`: persistent-device scheduler error, incomplete DAG, or nonzero
  fan-in remaining.
- `validation`: numeric mismatch or missing required pass field.
- `fabricated_or_untrusted_pass_evidence`: adapter, provenance, handoff, or
  example-side fields try to stand in for coordinator-owned evidence.

Local tests required before pass/true:

- reject `passed` without a runtime-owned descriptor allocation site;
- reject `passed` without exactly one coordinator-issued token shared by
  dispatch and combine;
- reject missing, skipped, or out-of-order lifetime transitions;
- reject rank/device mismatches against Worker-local device ordering;
- reject any pass-like fields supplied by handoff metadata or adapter
  provenance;
- require failure fields and non-claims to remain present in unsupported,
  setup-failed, and failed outputs.

Future H200 evidence required before pass/true:
run the existing `--with-uccl-ep-fused-boundary` command on a fresh branch
checkout and record a JSON artifact produced by the runtime coordinator. The
artifact must show top-level `status: passed`,
`persistent_device_uccl_ep_runtime_fusion.status: passed`,
`actual_fused_cross_gpu_execution: true`, matching rank/device mapping, zero
persistent-device and UCCL-EP validation errors, one non-null shared ownership
token, the complete lifetime transition log, and empty failure fields. If any
of those are missing, the result remains `unsupported`, `setup_failed`, or
`failed` and is not fused execution evidence.

This map preserves the accepted evidence boundary. PR #147 remains
provenance-only unsupported-boundary evidence. PR #150 remains guard-only
blocked implementation evidence. PR #151 remains a post-PR150 status refresh.
None of those PRs accepted
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

## Coordinator Entry Contract

The coordinator-boundary map above is still not enough for an implementation
branch to construct trusted runtime-fusion evidence. The next dependency is a
private CUDA persistent-device entry contract for the host-callable path:
`persistent_device_uccl_ep_runtime_fusion_entry`. It is reached only after
`ChipWorker::run` decodes the mailbox task, builds `ChipStorageTaskArgs`, and
enters the CUDA host runtime for the selected callable id.

The entry contract does not widen the public runtime API. It does not add
public `TaskArgs` fields, public `CallConfig` fields, or a UCCL host-runtime
ABI. The coordinator request is assembled from private runtime state already
available behind the chip-child boundary:

- callable id;
- chip-local rank/device map derived from Worker-local device ordering;
- persistent graph descriptor handle;
- UCCL-EP capability metadata and adapter provenance handles;
- descriptor allocation policy for the host-control record and
  device-visible dispatch/combine descriptor buffer;
- validation policy for descriptor, rank/device, payload lifetime, transport,
  scheduler, and numeric checks;
- runtime-owned output sink for the fused-boundary status artifact.

The result returned to the host/runtime status artifact must include:

- coordinator status;
- descriptor allocation provenance and runtime-owned allocation flag;
- one coordinator-issued ownership token shared by dispatch and combine;
- ordered state transitions with actor, state, token, descriptor id,
  rank/device map, and status;
- rank/device map used by the coordinator;
- validation summary;
- explicit failure fields for setup, unsupported, descriptor, rank/device,
  payload lifetime, transport, scheduler, validation, and fabricated or
  untrusted pass evidence.

The forbidden evidence paths stay explicit. Example-side JSON, adapter-only
provenance, handoff metadata, public `TaskArgs`, and public `CallConfig` must
not synthesize pass status, `actual_fused_cross_gpu_execution: true`,
allocation provenance, an ownership token, or a transition log. If those
paths supply pass-like fields, the status artifact must report `failed` with
`failure_fields.fabricated_or_untrusted_pass_evidence`.

No fresh H200 fused-boundary run is recorded for this entry-contract slice.
The current H200 fused-boundary evidence remains the structured unsupported
payload-provenance result above. A later implementation must keep the same
command shape and may report pass/true only when the fresh result is emitted
by the runtime coordinator through this private entry.

PR #153 accepted this entry contract only. It did not implement CUDA runtime
behavior, expand a UCCL host-runtime ABI, create descriptor memory, claim
fresh H200 fused success, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, or set
`actual_fused_cross_gpu_execution: true`.

## Accepted Private Entry Unsupported Scaffold

PR #155 accepted
`nvidia-uccl-ep-runtime-fusion-private-entry-unsupported`. It implemented
only the smallest private entry scaffold behind `ChipWorker::run` and
`ChipStorageTaskArgs`. The expected review-safe result remains
`unsupported` until the runtime coordinator itself allocates a shared
dispatch/combine descriptor, issues the ownership token, records the complete
lifetime transition log, and emits coordinator-owned validation fields.

The implementation must keep public `TaskArgs`, public `CallConfig`, and UCCL
host-runtime ABI fields unchanged. Adapter-only provenance, example-side
JSON, handoff metadata, and public API fields remain forbidden pass-evidence
paths.

## Private Entry Unsupported Scaffold

This implementation slice adds the first private CUDA host-side scaffold for
`persistent_device_uccl_ep_runtime_fusion_entry` without accepting pass
evidence. The private request/result shape lives in
`src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h`, and the CUDA
host runtime calls it from
`src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` when the persistent
DAG run path has a graph descriptor.

The scaffold records only unsupported or failed status. It derives available
private fields from the callable id, the persistent graph descriptor pointer,
the private CUDA communication descriptor when one is configured, and a
runtime-owned result sink. `PtoCudaRuntimeFusionRequest` and
`PtoCudaRuntimeFusionResult` carry that private request/result boundary.
When the coordinator, descriptor allocator, UCCL-EP runtime path, validation
policy, UCCL-EP capability metadata, or `ChipStorageTaskArgs` surface is
absent, the result records explicit failure bits and keeps
`persistent_device_uccl_ep_runtime_fusion.status: unsupported`.

Forbidden pass-evidence paths remain rejected. Adapter provenance,
example-side JSON, handoff metadata, payload provenance, public `TaskArgs`,
and public `CallConfig` map to
`fabricated_or_untrusted_pass_evidence` and cannot set
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

No fresh H200 fused-boundary run is recorded for this unsupported scaffold
slice. The branch does not add public `TaskArgs` fields, public `CallConfig`
fields, or UCCL host-runtime ABI fields, and it does not claim
`persistent_device_uccl_ep_runtime_fusion.status: passed`.

## Closed Invalid ChipStorageTaskArgs Request Boundary Attempt

PR #157 (`nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request`) is
closed invalid. The branch assigned the persistent DAG run `args` pointer to
`PtoCudaRuntimeFusionRequest::chip_storage_task_args` and recorded
`sizeof(ChipStorageTaskArgs)`. That pointer is a
`PtoCudaPersistentDagArgs *` in the CUDA host-runtime persistent DAG path, not
a `ChipStorageTaskArgs *` from `ChipWorker::run`.

The `nvidia-uccl-ep-runtime-fusion-private-request-envelope` slice now narrows
that handoff. `pto_cuda_private_run_envelope.h` can represent runtime-specific
task args separately from a typed `ChipStorageTaskArgs` pointer, and
`PtoCudaRuntimeFusionRequest::chip_storage_task_args` is typed as
`const ChipStorageTaskArgs *`. The CUDA host runtime copies only
`envelope->chip_storage_task_args` into the runtime-fusion request when a valid
private envelope is supplied. `ChipWorker::run` cannot honestly provide the
runtime-specific `PtoCudaPersistentDagArgs *` for this private envelope, so it
explicitly rejects the private-envelope path instead of passing
`ChipStorageTaskArgs *` as persistent DAG runtime args.

The expected review-safe result remains `unsupported`; missing coordinator,
descriptor allocator, UCCL-EP runtime path, validation policy, UCCL-EP
capability metadata, and pass evidence remain unsupported or failed states.
The slice does not add public `TaskArgs`, public `CallConfig`, common runtime
C API fields, UCCL host-runtime ABI fields, RDMA, multi-node, serving, vLLM,
DeepSeek, throughput, latency, fresh H200 fused-success evidence, or
pass/true fused-boundary status.

## Runtime Args Handoff Map

The selected post-PR161 dependency slice maps the private handoff that must
exist before another implementation attempt. It does not change the H200
result shape and does not record a fresh fused-boundary run.

The valid association is:

- `ChipWorker::run` builds the real `ChipStorageTaskArgs` from the decoded
  mailbox `TaskArgs` view and keeps that pointer typed as
  `const ChipStorageTaskArgs *`;
- the CUDA persistent DAG host-runtime path builds the real
  `PtoCudaPersistentDagArgs *` after resolving the prepared persistent DAG
  callable;
- `PtoCudaPrivateRunArgsEnvelope` associates those two private pointers only
  inside the CUDA host runtime and only for the same `ChipWorker::run`
  invocation;
- `persistent_device_uccl_ep_runtime_fusion_entry` may consume the
  association only after the envelope proves the chip-storage size,
  runtime-task-args size, callable type, rank/device metadata, and private
  output sink are all valid.

This map keeps PR #160's separation intact. `PtoCudaPersistentDagArgs *` is
not a `ChipStorageTaskArgs *`, `ChipStorageTaskArgs *` is not persistent DAG
runtime args, and neither pointer can be synthesized from public `TaskArgs`,
public `CallConfig`, example-side JSON, adapter provenance, payload
provenance, or handoff metadata.

A later implementation must fail or remain unsupported for null pointers,
wrong sizes, stale envelopes, mismatched callable types, cross-invocation
envelopes, missing coordinator, missing descriptor allocator, missing
UCCL-EP runtime path, missing validation policy, missing UCCL-EP capability
metadata, and forbidden pass-evidence sources. This dependency map therefore
does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`,
does not set `actual_fused_cross_gpu_execution: true`, and does not claim
RDMA, multi-node, serving, vLLM, DeepSeek, throughput, latency, or fresh H200
fused-success evidence.

PR #162 accepted this map as docs/test dependency evidence only. PR #164
accepted
`nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff`, which adds only
the private CUDA persistent DAG host-runtime association for real
same-invocation `ChipStorageTaskArgs *` and `PtoCudaPersistentDagArgs *`
pointers. It keeps the fused-boundary result unsupported or failed until a
later coordinator slice emits real fused execution evidence.

That implementation slice now adds the private association plumbing only:
`ChipWorker::run` carries the real chip-storage pointer, expected size,
callable id, and private invocation id into the CUDA host runtime, and the
CUDA persistent DAG path completes the envelope only after resolving the
prepared callable as a persistent DAG. The implementation keeps
`PtoCudaPersistentDagArgs *` separate from `ChipStorageTaskArgs *`, rejects
null, stale, wrong-size, wrong-callable, and cross-invocation envelopes, and
continues to reject forbidden public/API evidence paths.

This is still not fused execution evidence. The coordinator, descriptor
allocator, UCCL-EP runtime path, validation policy, UCCL-EP capability
metadata, pass evidence, and fresh H200 fused-success result are absent.
PR #165 accepted only the post-PR164 docs/test status refresh and selected
the capability metadata map.

## Capability Metadata Map Slice

This branch maps only the private UCCL-EP capability metadata dependency for
the later `persistent_device_uccl_ep_runtime_fusion_entry` request. It does
not change the H200 command shape, does not run fresh H200 fused-success
evidence, and does not change `persistent_device_uccl_ep_runtime_fusion` from
unsupported.

The minimum metadata fields are capability id, world size, rank-to-device
map, descriptor vocabulary, transport mode, adapter provenance handles, and
setup/validation failure ownership. The descriptor vocabulary names the
dispatch/combine payload terms a later coordinator must validate; it does not
allocate descriptors or transfer ownership.

The metadata source is private to the CUDA persistent-device runtime path and
chip-child private metadata. It is valid only after the PR #164 association
between real same-invocation `ChipStorageTaskArgs *` and
`PtoCudaPersistentDagArgs *` exists for the same invocation. Capability
metadata does not replace that association and cannot be copied from public
or example-owned result fields.

Failure ownership is split conservatively:

- missing private metadata is `unsupported`;
- stale metadata is `failed` when an invocation tries to consume it;
- mismatched-rank metadata is `failed`;
- mismatched-world-size metadata is `failed`;
- public/API-sourced capability metadata is `failed` as fabricated or
  untrusted pass evidence.

The cases missing, stale, mismatched-rank, mismatched-world-size, or
public/API-sourced capability metadata must remain unsupported or failed, not
pass evidence.

The forbidden pass-evidence paths remain public `TaskArgs`, public
`CallConfig`, common runtime C API, UCCL host-runtime ABI, example JSON,
adapter provenance, and handoff metadata. They must not provide private UCCL
capability metadata and must not set
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

This is a docs/test dependency map with no CUDA runtime behavior change. It
has no runtime-fusion coordinator implementation, no descriptor allocator
implementation, no UCCL-EP runtime path implementation, no validation policy
implementation, and no fresh H200 fused-success evidence. PR #166 accepted
only this private metadata dependency map and merged as
`42b996666e279024b43f490a310c490a591a897d`.

## Validation Policy Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-validation-policy-map`.

PR #168 accepted only the private validation policy dependency map required
after PR #166's capability metadata map and before descriptor allocation,
UCCL-EP runtime dispatch, coordinator implementation, pass evidence, or H200
fused-success evidence.

The validation policy remains private to the CUDA persistent-device runtime
path. It validates PR #164 same-invocation request args and PR #166 capability
metadata together before a later coordinator can consume either dependency.

Failure ownership is explicit: missing metadata is unsupported, stale metadata
is failed, mismatched-rank metadata is failed, and mismatched-world-size
metadata is failed. descriptor-vocabulary mismatch is failed because
descriptor vocabulary must match dispatch/combine payload terms.
transport-mode mismatch is failed because transport mode must be `ep`.
adapter-provenance mismatch is failed because adapter provenance handles must
match the private capability id, invocation id, and rank/device map.
public/API-sourced metadata is failed as fabricated or untrusted pass
evidence.

This slice has no descriptor allocation policy implementation, no UCCL-EP
runtime dispatch, no coordinator implementation, no pass evidence, and no
H200 fused-success evidence. Public `TaskArgs`, public `CallConfig`, common
runtime C API, UCCL host-runtime ABI, example JSON, adapter provenance, and
handoff metadata remain forbidden pass-evidence paths.

PR #168 merged this validation policy scope as
`e33d232deccdf947b9c382a3605191d0d5ae0004`.

## Descriptor Allocation Policy Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map`.

This dependency slice maps only the private descriptor allocation policy
required after PR #168's validation policy and before UCCL-EP runtime
dispatch, coordinator implementation, pass evidence, or H200 fused-success
evidence. It does not allocate descriptors.

The descriptor allocation policy remains private to the CUDA
persistent-device runtime path. It preserves PR #164 same-invocation request
args, PR #166 UCCL-EP capability metadata, and PR #168 validation policy as
prerequisites rather than pass evidence.

The allocator owner is the future private
`persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
persistent-device runtime run context. The host-control record policy defines
private per-invocation records for invocation id, persistent graph descriptor
id, UCCL capability id, validated rank/device map, descriptor vocabulary,
allocation state, runtime owner, and shared ownership token slot.

The device-visible descriptor buffer policy defines future coordinator-owned
buffers allocated through the CUDA persistent-device runtime allocator and
visible only to the persistent-device scheduler and UCCL-EP runtime path.
The dispatch descriptor identity is the validated graph descriptor id,
capability id, invocation id, rank/device map, dispatch vocabulary, payload
shape, and coordinator-issued shared token. The combine descriptor identity
uses the same validated ids, rank/device map, combine vocabulary, payload
shape, and the same shared token as dispatch.

Failure ownership is explicit: missing policy is unsupported, stale policy is
failed, non-runtime-owned allocation is failed, descriptor-vocabulary mismatch
is failed, token-sharing mismatch is failed, rank/device mismatch is failed,
and public/API-sourced policy fields are failed as fabricated or untrusted
pass evidence. The shared-token requirement is strict because dispatch and
combine descriptors must carry one coordinator-issued token. The allocation
lifetime failure ownership belongs to that same private runtime owner.

This slice must not allocate descriptors, implement UCCL-EP runtime dispatch,
construct the coordinator, change CUDA runtime behavior, claim pass evidence,
or claim H200 fused-success evidence. Public `TaskArgs`, public `CallConfig`,
common runtime C API, UCCL host-runtime ABI, example JSON, adapter provenance,
and handoff metadata remain forbidden pass-evidence paths.

PR #170 accepted only the private descriptor allocation policy dependency map
as `bd0b59ee8d5afc969020d3aea047aafc9f3152be`: allocator owner,
host-control record policy, device-visible descriptor buffer policy,
dispatch/combine descriptor identity, shared-token requirement, and allocation
lifetime failure ownership. It did not implement CUDA runtime behavior,
descriptor allocation, UCCL-EP runtime dispatch, a coordinator, pass evidence,
or H200 fused-success evidence.

## UCCL-EP Runtime Path Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map`.

This accepted dependency slice maps the private UCCL-EP runtime path that a
later coordinator must use after PR #170 descriptor allocation policy. It
defines how coordinator-owned dispatch and combine descriptor views would
reach UCCL-EP runtime logic, without implementing UCCL-EP runtime dispatch or
constructing the coordinator.

The runtime path remains private to the CUDA persistent-device runtime path.
PR #164 same-invocation request args, PR #166 UCCL-EP capability metadata,
PR #168 validation policy, and PR #170 descriptor allocation policy remain
prerequisites, not pass evidence.

The runtime-path owner is the future private
`persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
persistent-device runtime run context. The dispatch descriptor handoff uses
the PR #170 dispatch descriptor identity: invocation id, persistent graph
descriptor id, UCCL capability id, validated rank/device map, descriptor
vocabulary, dispatch payload shape, and coordinator-issued shared token. The
combine descriptor handoff uses the PR #170 combine descriptor identity with
the same invocation id, graph descriptor id, UCCL capability id, rank/device
map, descriptor vocabulary, combine payload shape, and exactly the same token.

descriptor-token checks fail unless dispatch and combine descriptor views
carry the same coordinator-issued token from the current same-invocation
request. Rank/device checks fail unless persistent graph descriptor metadata,
private UCCL-EP capability metadata, and Worker-local CUDA device ordering
agree. transport-mode checks fail unless the private capability metadata
declares `transport mode: ep`.

Runtime-path failure ownership remains private to the future coordinator.
missing runtime path is unsupported. stale descriptor views are failed,
descriptor-token mismatch is failed, rank/device mismatch is failed,
transport-mode mismatch is failed, descriptor-vocabulary mismatch is failed,
and public/API-sourced runtime-path fields are failed as fabricated or
untrusted pass evidence.

This accepted slice must not implement UCCL-EP runtime dispatch, construct a
coordinator, allocate descriptors, change CUDA runtime behavior, claim pass
evidence, or claim H200 fused-success evidence. It must not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or set
`actual_fused_cross_gpu_execution: true`.

PR #172 accepted only the private UCCL-EP runtime path dependency map as
`21b2b32a475dc04e19700115af74510daef70859`: runtime-path owner, dispatch
descriptor handoff, combine descriptor handoff, descriptor-token checks,
rank/device checks, transport-mode checks, and runtime-path failure
ownership. It did not implement CUDA runtime behavior, UCCL-EP runtime
dispatch, a coordinator, descriptor allocation, pass evidence, or H200
fused-success evidence.

## Accepted UCCL-EP Runtime Path Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl`.

This slice implements only the private UCCL-EP runtime path scaffold
needed before a later coordinator can route dispatch/combine descriptor views
into UCCL-EP runtime logic. It starts from the accepted PR #172 runtime-path
map and keeps the result non-evidence until real coordinator-owned runtime
fusion fields exist.

Required boundaries:

- keep the implementation private to the CUDA persistent-device runtime path;
- consume PR #164 same-invocation request args, PR #166 capability metadata,
  PR #168 validation policy, PR #170 descriptor allocation policy, and PR #172
  runtime-path map as prerequisites rather than pass evidence;
- keep missing descriptor allocation and missing coordinator as unsupported
  or failed states;
- reject public/API-sourced runtime-path fields, example JSON, adapter
  provenance, and handoff metadata as fabricated or untrusted pass evidence;
- make no fresh H200 fused-success claim.

The slice must not implement descriptor allocation, construct the
runtime-fusion coordinator, emit pass evidence, or report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

The scaffold adds private runtime-path descriptor views and checks for
same-invocation id, coordinator-owned source, shared descriptor token,
rank/device agreement, transport mode `ep`, descriptor vocabulary, stale
descriptor views, and public/API-sourced runtime-path fields. Valid scaffold
inputs still produce an unsupported fused-boundary status until descriptor
allocation and coordinator-owned UCCL-EP runtime dispatch exist.

PR #174 accepted this private runtime-path scaffold as
`3b4b19a04855d27289fb9cdad802fee0c47d8265`. The accepted surface is limited
to `PtoCudaUcclEpRuntimePath`, `PtoCudaUcclEpRuntimeDescriptorView`, private
descriptor-view validation, and invocation-id propagation through private
CUDA runtime-fusion request state. It did not implement the runtime-fusion
coordinator, descriptor allocation, UCCL-EP runtime dispatch, pass evidence,
fresh H200 fused-success evidence, public `TaskArgs`, public `CallConfig`,
common runtime C API fields, UCCL host-runtime ABI fields, serving, vLLM,
DeepSeek, throughput, or latency evidence.

## Accepted Descriptor Allocation Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl`.

This slice implements only private descriptor allocation mechanics:
the host-control record and device-visible dispatch/combine descriptor buffer
required by the PR #170 policy, bound to the same invocation id carried by
the PR #174 runtime-path scaffold.

The implemented private ABI surface is
`PtoCudaUcclEpDescriptorHostControl`,
`PtoCudaUcclEpDeviceDescriptorBuffer`,
`PtoCudaUcclEpDescriptorAllocation`, and
`pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors` in
`src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h`. The CUDA
host runtime calls that allocator from
`CudaDeviceRunner::record_runtime_fusion_unsupported`, then attaches the
resulting allocation and PR #174 `PtoCudaUcclEpRuntimePath` to the private
runtime-fusion request state.

This is narrower than runtime-fusion coordinator construction and narrower
than UCCL-EP runtime dispatch. Missing coordinator behavior and missing
UCCL-EP runtime dispatch remain unsupported or failed states. The slice must
not change the H200 fused-boundary claim state, must not report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, and must not set
`actual_fused_cross_gpu_execution: true`.

PR #176 accepted this private descriptor allocation scaffold only, merged as
`6e0cecc174ae9db47573c4c0f1698be7accb295c`. It accepted the private
host-control record, device-visible dispatch/combine descriptor buffer
mechanics, allocation bundle, same-invocation binding, and private
runtime-path handoff into request state. It did not implement coordinator
construction, UCCL-EP runtime dispatch, pass evidence, fresh H200
fused-success evidence, public `TaskArgs`, public `CallConfig`, common
runtime C API fields, UCCL host-runtime ABI fields, examples, stable docs,
serving, vLLM, DeepSeek, throughput, or latency.

## Accepted Runtime Fusion Coordinator Scaffold Status Slice

PR #178 merged as `aea89cc9dea8560602c72f84e5ff6e78ca526434` and accepted
only the private coordinator-construction scaffold/status slice from
`nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status`. It wires private
coordinator state needed to own the PR #176 descriptor allocation and PR #174
runtime path for one private `ChipWorker::run` invocation, but it does not
run or claim UCCL-EP runtime dispatch.

The implemented scaffold owns the same invocation id, descriptor allocation,
runtime path, unsupported/failure status, and runtime-owned output sink. The
new focused private-entry test proves the `missing_coordinator` failure clears
only when coordinator-shaped state is present. The final status remains
`unsupported`, `actual_fused_cross_gpu_execution` remains false, and the
UCCL-EP runtime-dispatch/pass-evidence surfaces remain absent.

This slice records no fresh H200 fused-boundary command. The existing H200
fused-boundary evidence remains structured unsupported evidence, not fused
success. The coordinator scaffold/status slice cannot claim
`persistent_device_uccl_ep_runtime_fusion.status: passed`,
`actual_fused_cross_gpu_execution: true`, pass evidence, or fresh H200
fused-success evidence until real UCCL-EP runtime dispatch exists and a fresh
H200 result proves coordinator-owned fused execution.

## Accepted Runtime Dispatch Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status`.

PR #180 merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba` and accepted
only the private UCCL-EP runtime-dispatch scaffold/status gate from
coordinator-owned state. It consumes the PR #178 coordinator-owned descriptor
allocation and runtime path, validates dispatch eligibility for one private
invocation, and records explicit unsupported or failed status through the
runtime-owned output sink.

This branch implements only the private gate. A missing
`runtime_dispatch_scaffold_status` on the coordinator records
`missing_runtime_dispatch_scaffold` and a failed private result. A prepared,
eligible gate records the scaffold/status check but still leaves
`persistent_device_uccl_ep_runtime_fusion` unsupported with
`actual_fused_cross_gpu_execution` false.

This remains narrower than scheduler/runtime pass evidence and narrower than
fresh H200 fused success. It must not run real UCCL-EP dispatch/combine work,
claim UCCL-EP runtime dispatch success, emit pass evidence, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, set
`actual_fused_cross_gpu_execution: true`, or add public API, RDMA, multi-node,
serving, vLLM, DeepSeek, throughput, or latency claims.

## Selected Runtime Dispatch Request Handoff Map Slice

The selected next PR-sized dependency slice is
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map`. It may
map only the private UCCL-EP runtime dispatch request/driver handoff from the
PR #180 coordinator-owned scaffold/status gate to a later runtime driver. It
must stay narrower than scheduler/runtime pass evidence and cannot claim
fused success.

This next slice should define private request ownership, driver ownership,
status dependency, and unsupported/failed handoff states before any real
UCCL-EP dispatch/combine work can run. Missing request state, missing driver
state, stale invocation ids, mismatched rank/device state, or public/API
handoff metadata remain non-pass evidence.

It must not run UCCL-EP dispatch/combine work, emit scheduler/runtime pass
evidence, claim fresh H200 fused success, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, set
`actual_fused_cross_gpu_execution: true`, or add public API, RDMA,
multi-node, serving, vLLM, DeepSeek, throughput, or latency claims.

## Future Fused Execution Evidence Shape

PR #174 defines only the private runtime-path scaffold. It does not implement
`persistent_device_uccl_ep_runtime_fusion`, does not change the example
command, and does not relabel the PR #143 structured unsupported boundary as
fused evidence.

A later implementation PR may claim actual fused cross-GPU expert-parallel
MoE execution only when one fresh H200 command emits all of these fields:

- top-level `status: passed`;
- `fused_boundary_scope`:
  `reduced-fused-cross-gpu-expert-parallel-moe-boundary`;
- `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- `actual_fused_cross_gpu_execution: true`;
- `device_ids`, `rank_to_device`, `world_size`, and UCCL capability id;
- persistent MoE validation fields with completion count `5`, zero scheduler
  errors, zero fan-in remaining, zero max error, and matching source/bridge
  digests;
- UCCL-EP runtime validation fields for `transport: ep`,
  `operation: ep_dispatch_combine`, all ranks passed, descriptor metadata
  present, zero max error, and zero top-k weight error;
- dispatch and combine payload descriptors with token count, hidden size,
  top-k, expert count, dtype, metadata shapes, and a shared ownership token;
- payload ownership/lifetime transition log showing
  `persistent_device_graph -> uccl_ep_runtime -> persistent_device_graph ->
  released` with no mismatched token, double release, leaked in-flight owner,
  or use-after-release;
- failure fields for setup, descriptor, rank/device, payload lifetime,
  transport, validation, unsupported boundary, and numeric correctness states;
- non-claims for CUDA host-runtime UCCL dispatch, RDMA, multi-node transport,
  serving, DeepSeek/vLLM integration, throughput, and latency unless that same
  PR also adds separate evidence for those claims.

The expected future command shape remains the PR #143 H200 command with
`--with-uccl-ep-fused-boundary`. The result may be called fused evidence only
when the boundary status is `passed` and the JSON records
`actual_fused_cross_gpu_execution: true`. `unsupported`, `setup_failed`, and
`failed` remain useful review outcomes, but they are not fused execution
evidence.
