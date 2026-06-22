# NVIDIA Backend Goal Status Rollup

This rollup audits `origin/main` at `3338a239` against the acceptance
criteria in `docs/in_progress/001-nvidia-backend.md`. It uses checked-in
files and accepted merge history only. No new H200 or serving command was run
for this audit slice.

## Status By Acceptance Area

- Source acquisition from the restart instruction:
  `accepted evidence`.
  `docs/in_progress/nvidia_backend/source_inventory.md` records the clean
  acquisition pass from PR #4, and
  `docs/in_progress/nvidia_backend/source_manifest.md` was restored on `main`
  by PR #139 and recorded by PR #140.
- Selected Codex skill stack:
  `accepted evidence`.
  `.agents/skills/codex-goal-monitor/`,
  `.agents/skills/cuda-backend-eval/`,
  `.agents/rules/ultimate-goal-dispatch.md`, and
  `docs/in_progress/nvidia_backend/skill_selection.md` cover the Codex
  dispatcher, worker, GitHub, CUDA evaluation, and monitor surface. Relevant
  accepted refs are PR #3, PR #137, PR #138, PR #139, and PR #140.
- NVIDIA platform and runtime contracts:
  `accepted evidence`.
  `docs/nvidia-backend/status.md`,
  `docs/nvidia-backend/overall.md`, and
  `docs/nvidia-backend/persistent-device.md` describe the accepted
  `cuda/onboard`, `host_schedule`, and `persistent_device` contracts.
  Implementation and tests are anchored by `tests/ut/py/test_cuda_backend.py`,
  `tests/ut/py/test_cuda_kernel_compiler.py`,
  `tests/ut/py/test_cuda_persistent_codegen.py`,
  `tests/ut/py/test_runtime_builder.py`, and the current review guard.
- Gluon-generated GEMM and FlashAttention:
  `accepted evidence`.
  `docs/in_progress/nvidia_backend/gluon_gemm_h200.md`,
  `docs/in_progress/nvidia_backend/gluon_tensor_core_gemm.md`,
  `docs/in_progress/nvidia_backend/gluon_flashattention_h200.md`, and
  `docs/in_progress/nvidia_backend/gluon_performance_h200.md` record accepted
  H200 correctness and microbenchmark evidence. Relevant accepted refs include
  PR #8, PR #9, PR #10, PR #11, PR #79 through PR #93, and PR #118 through
  PR #136. This remains bounded kernel evidence, not serving evidence.
- Simpler-programmed fused attention or MoE megakernels on H200:
  `partial evidence`.
  `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`
  records a persistent-device MoE dispatch/combine graph on one H200 and an
  independent same-node two-H200 baseline. PR #14, PR #55, PR #57, PR #95,
  and PR #101 are the main accepted refs. It is still not fused cross-GPU
  expert-parallel MoE, serving, or DeepSeek evidence.
- Distributed compiler/runtime direction:
  `partial evidence`.
  `docs/in_progress/nvidia_backend/communication_selection.md`,
  `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`,
  `docs/in_progress/nvidia_backend/nccl_two_h200_baseline.md`,
  `docs/in_progress/nvidia_backend/nccl_worker_control_h200.md`, and
  `docs/in_progress/nvidia_backend/uccl_ep_adapter_h200.md` cover Ray, NCCL,
  and UCCL role selection plus H200 NCCL and UCCL adapter probes. PR #59 and
  PR #101 are accepted refs for worker-control and handoff evidence. UCCL
  host-runtime dispatch, RDMA, multi-node, and serving communication remain
  incomplete.
- Multi-GPU MoE dispatch/combine:
  `partial evidence`.
  The accepted path composes persistent MoE with NCCL worker-control and
  Python-side UCCL-EP adapter handoff on the same H200 device pair in
  `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`.
  It does not yet run fused cross-GPU expert-parallel dispatch/combine.
- Serving launches simpler NVIDIA kernels:
  `partial evidence`.
  `docs/in_progress/nvidia_backend/serving_target_selection.md`,
  `docs/in_progress/nvidia_backend/pypto_serving_nv_shim_local.md`, and
  `docs/in_progress/nvidia_backend/pypto_serving_source_contract_h200.md`
  show pypto-serving source routes launching synthetic simpler-nv,
  generated Gluon, and persistent MoE seeds on H200. This is separate from
  vLLM/DeepSeek evidence and is not real DeepSeek serving.
- DeepSeek-V4-Flash serving on two H200 GPUs:
  `partial evidence`.
  vLLM evidence under
  `docs/in_progress/nvidia_backend/vllm_remote_*` and
  `docs/in_progress/nvidia_backend/deepseek_v4_flash_serving_readiness.md`
  proves complete artifacts, vLLM model load, server health, inference smoke,
  response contracts, and bounded 256K needle/chat gates on two H200 GPUs.
  Relevant accepted refs include PR #16 through PR #18, PR #21 through PR #31,
  PR #37 through PR #54, PR #63 through PR #71, and PR #94. This is not
  simpler-nv/vLLM kernel integration evidence.
- GitHub PRs, dispatch logs, docs, tests, and hardware artifacts:
  `accepted evidence`.
  `docs/in_progress/nvidia_backend/dispatch_log.md`,
  `docs/in_progress/nvidia_backend/pr_slicing_plan.md`,
  `.agents/checks/check_nvidia_review_ready.py`, and
  `tests/ut/py/test_nvidia_review_artifacts.py` keep accepted slices visible
  on `main`. PR #139 restored the tracking surface, PR #140 recorded that
  merge, and PR #141 recorded the status rollup worker dispatch on current
  `main`.

## Evidence Separation

DeepSeek/vLLM serving evidence proves that the external vLLM path can load and
serve `deepseek-ai/DeepSeek-V4-Flash` on two H200 GPUs under bounded
contracts. It does not prove that vLLM launches simpler-nv kernels.

Simpler-nv integration evidence proves that pypto-serving source routes can
launch synthetic CUDA seeds, generated Gluon kernels, and persistent MoE seeds
on H200. It does not prove DeepSeek model correctness, tokenizer semantics,
or vLLM plugin integration.

## Recommended Next Slice

Recommended branch:
`nvidia-moe-uccl-ep-fused-boundary-h200`.

Objective: convert the accepted persistent-MoE plus UCCL-EP handoff into the
first reduced fused cross-GPU expert-parallel MoE boundary. The slice should
route one bounded dispatch/combine payload through the UCCL-EP adapter and
the persistent MoE graph in one H200 command, then record whether the result
is a pass, a structured unsupported boundary, or a setup failure.

Owned paths:

- `examples/cuda/persistent_moe_dispatch_combine.py`
- `examples/cuda/uccl_ep_dispatch_combine_adapter.py`
- `simpler_setup/cuda_comm.py`, only if descriptor metadata must grow
- `tests/ut/py/test_cuda_comm.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`
- `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`
- `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`
- `docs/in_progress/nvidia_backend/communication_selection.md`

Verification commands:

```bash
git diff --check
```

```bash
npx --no-install markdownlint-cli2 \
  docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md \
  docs/in_progress/nvidia_backend/communication_runtime_boundary.md \
  docs/in_progress/nvidia_backend/communication_selection.md
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_comm.py -q
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

H200 evidence requirement: run the new fused-boundary command through
`.agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync` on the
same two H200 device ids used by the UCCL-EP handoff, recording command,
commit, GPU model, driver, CUDA toolkit, UCCL dependency path policy, status,
per-rank validation, and sanitized source digests. A skip or dependency setup
failure must be recorded as non-evidence.

Non-claims for the next slice:

- not DeepSeek serving or correctness;
- not vLLM plugin integration;
- not RDMA or multi-node evidence;
- not throughput or latency evidence;
- not production fused MoE readiness unless the command proves that exact
  boundary.
