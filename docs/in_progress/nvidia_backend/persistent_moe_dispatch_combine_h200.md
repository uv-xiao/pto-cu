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

| Device | Status | Max error | Completed | Scheduler errors | Fan-in | Host ns | Device ns |
| ------ | ------ | --------- | --------- | ---------------- | ------ | ------- | --------- |
| 6 | passed | 0.0 | 5 | `{count: 0, code: 0, task_id: 0}` | `[0, 0, 0, 0, 0]` | 2446295 | 46144 |
| 7 | passed | 0.0 | 5 | `{count: 0, code: 0, task_id: 0}` | `[0, 0, 0, 0, 0]` | 60360 | 49888 |

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
