# pypto-serving simpler-nv Shim Local Evidence

This note records the first synthetic `pypto-serving` simpler-nv shim
implementation. It follows
`docs/in_progress/nvidia_backend/pypto_serving_nv_shim_design.md` and keeps the
same serving boundaries as the source `pypto-serving` contracts.

## Artifact

- Example:
  `examples/cuda/pypto_serving_nv_shim.py`
- Tests:
  `tests/ut/py/test_pypto_serving_nv_shim.py`
- Synthetic model id:
  `synthetic-simpler-nv`
- Executor boundary:
  `SimplerNvExecutor` follows the `ModelExecutor` shape.
- Runner boundary:
  `SimplerNvModelRunner` follows the `ModelRunner` shape.
- Launch contract:
  `KernelLaunchRequest` records phase, platform, runtime, device, op, launch
  size, and PTX arch before calling the selected launcher.
- Launcher selection:
  Use `--kernel-launcher cuda-seed` by default. The
  `--kernel-launcher gluon-moe-expert` mode calls the generated Gluon MoE
  expert correctness harness for `moe_expert_affine_f32` and records
  generated-kernel metadata in `launch_results`. The
  `--kernel-launcher gluon-topk-sampling` mode calls the generated Gluon
  Top-K sampling correctness harness for `topk_sampling_f32` and records
  artifact/source digest plus validation metadata in `launch_results`. The
  `--kernel-launcher gluon-topp-sampling` mode calls the generated Gluon
  Top-P sampling correctness harness for `topp_sampling_f32` and records
  artifact/source digest plus validation metadata in `launch_results`. The
  `--kernel-launcher gluon-minp-sampling` mode calls the generated Gluon
  Min-P sampling correctness harness for `minp_sampling_f32` and records
  artifact/source digest plus validation metadata in `launch_results`. The
  `--kernel-launcher persistent-moe-dispatch-combine` mode calls the existing
  persistent-device dispatch/combine graph example through
  `run_moe_dispatch_combine(...)` and records bounded DAG metadata in
  `launch_results`. The aggregate `--pypto-serving-vllm-compat` CLI isolates
  persistent source-route checks in child processes so repeated route checks do
  not rebind the persistent runtime through `ctypes` in one TestClient process.
- Source-route loading:
  The shim loads the cloned `pypto-serving` `python/core/server.py` route file
  directly with minimal server-route stubs for the synthetic adapter. This
  keeps the source HTTP route contract real while avoiding an unnecessary
  import of the Torch-backed real model engine for these synthetic tests.

The public smoke entry point is `run_synthetic_serving_request(...)`. It
returns deterministic logits for one request and produces:

```text
text: NV
token_ids: [1, 2]
```

The shim keeps the OpenAI-compatible API as a future integration boundary. It
does not copy `pypto-serving`'s HTTP server into this repository.

## Verification

Focused local unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
4 passed
```

Remote H200 synthetic CUDA seed check:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90 --require-cuda'
```

Result:

```text
status: passed
backend: simpler-nv
model_id: synthetic-simpler-nv
text: NV
token_ids: [1, 2]
launch_count: 2
prefill cuda_seed status: pass
decode cuda_seed status: pass
runtime: host_schedule
ptx_arch: compute_90
op: add
```

The launcher normalizes the owned CUDA seed's raw `status: pass` into shim
launch `status: passed` while preserving the raw CUDA seed payload in the JSON
result.

Generated Gluon launch mode is skip-safe on machines without CUDA, torch CUDA,
or Triton Gluon. Its local JSON metadata records:

```text
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill|decode
shape.n: 16
source_sha256: <generated-source-digest-when-available>
```

Generated Top-K sampling launch mode is skip-safe on machines without CUDA,
torch CUDA, or Triton Gluon. Its local JSON metadata records:

```text
launch_kind: gluon-topk-sampling
kernel_name: topk_sampling_f32
phase: prefill|decode
shape: {rows: 3, vocab: 16, k: 5}
source_sha256: <generated-source-digest-when-available>
validation.max_abs_error: <max-absolute-error-when-present>
```

Generated Top-P sampling launch mode is skip-safe on machines without CUDA,
torch CUDA, or Triton Gluon. Its local JSON metadata records:

```text
launch_kind: gluon-topp-sampling
kernel_name: topp_sampling_f32
phase: prefill|decode
shape: {rows: 3, vocab: 16, max_k: 6}
p: 0.80
source_sha256: <generated-source-digest-when-available>
validation.max_abs_error: <max-absolute-error-when-present>
validation.max_cumulative_probability_error: <max-error-when-present>
```

Generated Min-P sampling launch mode is skip-safe on machines without CUDA,
torch CUDA, or Triton Gluon. Its local JSON metadata records:

```text
launch_kind: gluon-minp-sampling
kernel_name: minp_sampling_f32
phase: prefill|decode
shape: {rows: 3, vocab: 16, max_k: 6}
min_p: 0.50
source_sha256: <generated-source-digest-when-available>
validation.max_abs_error: <max-absolute-error-when-present>
```

Persistent MoE dispatch/combine launch mode is skip-safe on machines without
CUDA, runtime build prerequisites, or a visible NVIDIA GPU. Its local JSON
metadata records:

```text
launch_kind: persistent-moe-dispatch-combine
phase: prefill|decode
dag_shape: graph_descriptor_moe_dispatch_combine
shape.n: 16
completed_count: <completed-count-when-present>
max_abs_error: <max-absolute-error-when-present>
scheduler_error_summary: <device-scheduler-errors-when-present>
task_body_digest.source_sha256: <Gluon expert task-body digest when present>
```

When the aggregate compatibility command selects the persistent launcher, each
fixture summary also preserves per-route `pto_launch_results` metadata under
`fixtures[*].observed`, including launch kind, route phase, DAG shape, status,
completed count, max absolute error, scheduler error summary, and task-body
digest when available. If a selected launcher fails or skips locally, the
aggregate stops after the first non-passed PTO launch and returns structured
JSON instead of continuing into later routes.

## Interpretation

This is local and H200 synthetic shim evidence. It proves the repository has a
small `ModelExecutor`/`ModelRunner`-shaped simpler-nv boundary that can call a
known CUDA seed on H200 and return deterministic token output. The generated
Gluon and persistent MoE dispatch/combine launch modes are local contract
evidence unless an H200 selected-launcher run is recorded in the source
contract note. Top-K sampling launcher H200 evidence is recorded separately in
`docs/in_progress/nvidia_backend/pypto_serving_topk_sampling_launcher_h200.md`.
Top-P sampling launcher H200 evidence is recorded separately in
`docs/in_progress/nvidia_backend/pypto_serving_topp_sampling_launcher_h200.md`.
Min-P sampling launcher H200 evidence is recorded separately in
`docs/in_progress/nvidia_backend/pypto_serving_minp_sampling_launcher_h200.md`.

This is not serving evidence. It is not DeepSeek-V4-Flash correctness. It is
not vLLM plugin evidence. It is not a throughput, latency, UCCL serving, or
multi-node claim. It is not fused MoE dispatch/combine serving readiness.
