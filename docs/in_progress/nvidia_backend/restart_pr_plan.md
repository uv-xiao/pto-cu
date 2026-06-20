# Restart PR Execution Plan

This plan converts the dirty orphan `restart` workspace into a sequence of
reviewable child PRs. It is intentionally an execution plan, not a claim that
the current workspace is ready to merge.

## Current State

- Current branch: `restart`.
- Branch shape: orphan branch with no commits; `git rev-parse --verify HEAD`
  fails because the branch has no `HEAD` commit.
- Remote base: `origin/main` at `71bdc089` is available and should be the base
  for every child PR branch.
- Dirty shape observed on 2026-06-20:
  - `1015` paths in `git status --porcelain=v1 -uall`;
  - `950` index entries;
  - `922` staged added paths;
  - `28` staged-and-modified paths;
  - `27` untracked paths.
- The dirty state mixes the reconstructed base repository, `.agents/`
  operating material, CUDA runtime/platform changes, generated-kernel
  examples, serving probes, tests, in-progress design notes, external-source
  manifests, and old review-site artifacts.

## Why This Workspace Is Not PR-Qualified

The current `restart` branch should not be pushed as one PR.

- It has no commit history, so a PR from the branch would look like a new
  repository import rather than a reviewable delta from `origin/main`.
- The index already stages broad base-repository reconstruction. A normal
  `git commit` would bundle unrelated files across `.claude/`, `.agents/`,
  docs, examples, `src/`, tests, packaging, and CI.
- Many restart-specific artifacts are still untracked, including
  `docs/in_progress/nvidia_backend/`, new CUDA examples, new probe tests,
  `simpler_setup/cuda_comm.py`, `simpler_setup/gluon_gen.py`, and
  `src/cuda/platform/include/host/pto_cuda_comm_descriptor_abi.h`.
- The dispatch log records many slices on the same mutable branch, but those
  slices are not separated by branch, commit, PR, or review boundary.
- Existing evidence is historical workspace evidence. Every child PR must
  rerun its own verification or explicitly label older evidence as context.
- The active implementation surface and the temporary research surface are
  interleaved. Files under `tmp/` are correctly untracked, but several
  examples, docs, and `.agents/` scripts still need owner decisions before
  review.

## Branch Strategy

Do not create child PRs directly from the orphan `restart` index.

For each child PR:

1. Create a fresh branch from `origin/main`.
2. Import only the owned paths listed for that PR from the restart workspace.
3. Run the verification commands for that PR in the fresh branch.
4. Commit only the owned paths.
5. Open a draft PR until the merge-readiness evidence is present in the PR
   body and, where applicable, in `docs/in_progress/nvidia_backend/`.

This keeps the orphan restart workspace as a source workspace, not the review
branch.

## Child PR Sequence

### PR 0 - Restart PR Execution Plan

**Objective:** land the PR-slicing decision record before moving any code,
tests, or `.agents/` material.

**Owned file/path scope:**

- `docs/in_progress/nvidia_backend/restart_pr_plan.md`

**Dependencies:**

- None beyond `origin/main`.

**Verification commands:**

```bash
test -s docs/in_progress/nvidia_backend/restart_pr_plan.md
```

```bash
git status --short -- docs/in_progress/nvidia_backend/restart_pr_plan.md
```

**Merge readiness evidence:**

- The PR contains exactly this plan file.
- The plan states why the orphan workspace is not PR-qualified.
- The plan lists owned paths, dependencies, verification commands, evidence,
  and risks for the next child PRs.

**Risks/unknowns:**

- This PR changes process only; it does not preserve implementation work.
- Later PRs still need fresh branch construction from `origin/main`.

### PR 1 - Dispatcher Bootstrap And Minimal Agent Surface

**Objective:** preserve the durable dispatcher/source/skill operating surface
that all later CUDA child PRs depend on, while trimming the historical
workspace log and avoiding one-off script sprawl.

**Owned file/path scope:**

- `.agents/AGENT.md`
- `.agents/coding-guidance.md`
- `.agents/checks/check_nvidia_review_ready.py`
- `.agents/rules/ultimate-goal-dispatch.md`
- `.agents/rules/nvidia-backend-review.md`
- `.agents/rules/quality-evidence.md`
- `.agents/rules/remote-evaluation.md`
- `.agents/rules/requirements-first.md`
- `.agents/rules/testing-and-verification.md`
- `.agents/templates/ultimate-goal.md`
- `.agents/skills/codex-goal-monitor/SKILL.md`
- `.agents/skills/codex-goal-monitor/scripts/inject-codex-steer.sh`
- `.agents/skills/cuda-backend-eval/SKILL.md`
- `.agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh`
- `docs/in_progress/001-nvidia-backend.md`
- `docs/in_progress/nvidia_backend/source_manifest.md`
- `docs/in_progress/nvidia_backend/skill_selection.md`
- `docs/in_progress/nvidia_backend/cuda_eval_script_selection.md`
- `docs/in_progress/nvidia_backend/current_cuda_artifact_audit.md`
- `docs/in_progress/nvidia_backend/work_preparation.md`
- `docs/in_progress/nvidia_backend/dispatch_log.md`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 0.
- External sources remain in `tmp/`; this PR records them but does not commit
  cloned repositories, papers, model artifacts, or generated outputs.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/check_nvidia_review_ready.py
```

**Merge readiness evidence:**

- `dispatch_log.md` is reduced or rewritten so it only records bootstrap,
  source/skill selection, and explicit future handoffs that match this PR.
- The CUDA eval skill points at the active remote runner, not the accumulated
  historical script set.
- The review guard covers the committed docs and selected agent surface.
- The PR body states that no CUDA implementation, serving, H200 correctness,
  or DeepSeek model-load claim is being made.

**Risks/unknowns:**

- The existing `dispatch_log.md` is large and contains evidence for later
  slices. Committing it wholesale would blur PR boundaries.
- Some `.agents/` files are staged from earlier work but may be unrelated to
  the NVIDIA restart path.
- The review guard may need narrowing if it currently assumes files that are
  intentionally postponed.

### PR 2 - CUDA Communication Descriptor And NCCL Worker Control

**Objective:** land the private CUDA communication descriptor boundary and the
descriptor-backed NCCL worker-control path.

**Owned file/path scope:**

- `simpler_setup/cuda_comm.py`
- `src/cuda/platform/include/host/pto_cuda_comm_descriptor_abi.h`
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp`
- `src/common/platform_comm/comm.h`
- `src/common/platform_comm/comm_context.h`
- `src/common/platform_comm/comm_sim.cpp`
- `src/common/worker/chip_worker.cpp`
- `src/common/worker/chip_worker.h`
- `src/common/worker/pto_runtime_c_api.h`
- `python/simpler/worker.py`
- `python/simpler/task_interface.py`
- `python/bindings/task_interface.cpp`
- `python/bindings/worker_bind.h`
- `examples/cuda/nccl_two_gpu_baseline.py`
- `examples/cuda/nccl_worker_control_ops.py`
- `docs/in_progress/nvidia_backend/communication_selection.md`
- `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`
- `docs/in_progress/nvidia_backend/nccl_two_h200_baseline.md`
- `docs/in_progress/nvidia_backend/nccl_worker_control_h200.md`
- `tests/ut/py/test_cuda_comm.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 1 for the remote runner and review guard.
- Local venv prepared with repo rules.
- H200 remote environment with NCCL and CUDA access for hardware evidence.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_comm.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    pip install --no-build-isolation -e . >/tmp/pto-cu-pip-install.log && \
    NCCL_DEBUG=WARN .venv/bin/python \
      examples/cuda/nccl_worker_control_ops.py \
      --device-ids 6,7 --tensor-numel 1024 --build --require-cuda'
```

**Merge readiness evidence:**

- Unit tests exercise descriptor serialization and public/private boundary
  behavior.
- H200 command output is recorded in
  `docs/in_progress/nvidia_backend/nccl_worker_control_h200.md`.
- PR body says whether the evidence is same-node only, which GPU ids were
  used, and whether the build happened from the PR branch.
- No UCCL host-runtime ABI is included in this PR.

**Risks/unknowns:**

- Host-runtime descriptor changes can accidentally widen public APIs.
- H200 device ids and NCCL availability are environment-dependent.
- Existing code touches both Python bindings and common worker code, so this
  PR needs stricter review than a pure example harness.

### PR 3 - UCCL Adapter Evidence

**Objective:** preserve the UCCL-P2P and UCCL-EP adapter experiments as
Python-side evidence without committing to a CUDA host-runtime UCCL ABI.

**Owned file/path scope:**

- `simpler_setup/cuda_comm.py`
- `examples/cuda/uccl_p2p_ipc_adapter.py`
- `examples/cuda/uccl_ep_dispatch_combine_adapter.py`
- `docs/in_progress/nvidia_backend/uccl_ep_p2p_probe_plan.md`
- `docs/in_progress/nvidia_backend/uccl_ep_p2p_h200.md`
- `docs/in_progress/nvidia_backend/uccl_adapter_boundary.md`
- `docs/in_progress/nvidia_backend/uccl_p2p_adapter_h200.md`
- `docs/in_progress/nvidia_backend/uccl_ep_adapter_h200.md`
- `docs/in_progress/nvidia_backend/uccl_ep_nccl_worker_control_comparison.md`
- `tests/ut/py/test_cuda_comm.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 1 for source/evidence discipline.
- PR 2 if this reuses descriptor helpers from `simpler_setup/cuda_comm.py`.
- UCCL source remains under `tmp/sources/` and is not committed.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_comm.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/uccl_ep_dispatch_combine_adapter.py --require-cuda'
```

**Merge readiness evidence:**

- Docs clearly label UCCL evidence as adapter/probe evidence.
- PR body states that RDMA, multi-node, and serving integration are not
  proven.
- The host-runtime C ABI remains unchanged from PR 2.

**Risks/unknowns:**

- The adapter may depend on temporary source checkout layout under `tmp/`.
- UCCL evidence is same-node H200 only until a later RDMA slice.
- Sharing `simpler_setup/cuda_comm.py` with PR 2 creates rebase risk.

### PR 4 - Gluon Generator And GEMM Milestones

**Objective:** land `gluon-gen` plus scalar and tensor-core GEMM correctness
milestones before bundling flash-attention or broader performance claims.

**Owned file/path scope:**

- `simpler_setup/gluon_gen.py`
- `simpler_setup/kernel_compiler.py`
- `examples/cuda/gluon_gemm_f32.py`
- `examples/cuda/gluon_gemm_tensor_core.py`
- `examples/cuda/gluon_gemm_tensor_core_tiled.py`
- `docs/in_progress/nvidia_backend/gluon_gen_adapter.md`
- `docs/in_progress/nvidia_backend/gluon_gemm_h200.md`
- `docs/in_progress/nvidia_backend/gluon_tensor_core_gemm.md`
- `tests/ut/py/test_cuda_kernel_compiler.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 1 for H200 remote runner and source manifest.
- Triton/Gluon source remains in `tmp/sources/`; generated outputs remain in
  `tmp/gluon-*`.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_gemm_f32.py \
      --output-dir tmp/gluon-gemm-h200 --m 16 --n 16 --k 16 \
      --arch compute_90 --require-cuda'
```

**Merge readiness evidence:**

- The PR separates scalar GEMM correctness from tensor-core performance.
- Generated source and JSON manifests are written under `tmp/`, not committed.
- H200 output artifacts are summarized in the matching in-progress docs.

**Risks/unknowns:**

- Triton/Gluon APIs may change; generated code should be treated as a source
  artifact, not a stable compiler ABI.
- Tensor-core evidence may be correctness-only if performance work is not
  ready.
- `kernel_compiler.py` is a shared compiler surface and needs focused review.

### PR 5 - Gluon Flash-Attention And Benchmark Evidence

**Objective:** add generated flash-attention and benchmark harness evidence
after GEMM generation is reviewable.

**Owned file/path scope:**

- `simpler_setup/gluon_gen.py`
- `examples/cuda/gluon_flashattention_fwd.py`
- `examples/cuda/gluon_benchmark.py`
- `docs/in_progress/nvidia_backend/gluon_flashattention_h200.md`
- `docs/in_progress/nvidia_backend/gluon_performance_h200.md`
- `tests/ut/py/test_cuda_kernel_compiler.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 4.
- H200 runtime access for correctness and benchmark evidence.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_flashattention_fwd.py --require-cuda'
```

**Merge readiness evidence:**

- Flash-attention correctness is documented separately from performance.
- Benchmark docs include command, GPU model, architecture flag, tensor shapes,
  and result artifact location.
- PR body avoids "perfect performance" language unless benchmark comparisons
  actually justify it.

**Risks/unknowns:**

- Flash-attention correctness and performance have different failure modes.
- Benchmarks may be noisy across shared H200 devices.
- This PR should not include MoE or serving work.

### PR 6 - Persistent MoE Dispatch/Combine Seed

**Objective:** land the first persistent-device MoE dispatch/combine seed and
Gluon expert affine harness without claiming serving or full distributed MoE.

**Owned file/path scope:**

- `examples/cuda/gluon_moe_expert_affine.py`
- `examples/cuda/persistent_moe_dispatch_combine.py`
- `examples/cuda/persistent_layered_cross.py`
- `docs/in_progress/nvidia_backend/gluon_moe_expert_h200.md`
- `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 4 for Gluon generation.
- PR 2 or PR 3 only if the example depends on communication descriptors.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/persistent_moe_dispatch_combine.py --require-cuda'
```

**Merge readiness evidence:**

- H200 output is recorded with tensor shapes, devices, and command.
- Docs state whether this is a single-GPU, same-node multi-GPU, or synthetic
  local expert proof.
- PR body names the cited MoE dispatch/combine paper as design context only
  unless the implementation actually matches the paper's distributed path.

**Risks/unknowns:**

- Persistent megakernel shape can drift from the simpler runtime boundary.
- Example code may be too synthetic to justify promotion without clear limits.
- This PR should not claim DeepSeek, vLLM, or pypto-serving integration.

### PR 7 - pypto-Serving Fixture Path

**Objective:** preserve the serving target decision and pypto-serving source
contract fixture as the first serving path, while keeping it separate from
DeepSeek and vLLM model-load work.

**Owned file/path scope:**

- `examples/cuda/pypto_serving_nv_shim.py`
- `examples/cuda/README.md`
- `docs/in_progress/nvidia_backend/serving_target_selection.md`
- `docs/in_progress/nvidia_backend/pypto_serving_nv_shim_design.md`
- `docs/in_progress/nvidia_backend/pypto_serving_nv_shim_local.md`
- `docs/in_progress/nvidia_backend/pypto_serving_openai_completion_fixture.md`
- `docs/in_progress/nvidia_backend/pypto_serving_engine_fixture.md`
- `docs/in_progress/nvidia_backend/pypto_serving_http_fixture.md`
- `docs/in_progress/nvidia_backend/pypto_serving_source_contract_h200.md`
- `tests/ut/py/test_pypto_serving_nv_shim.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 1 for source manifest and H200 runner.
- PR 4 or PR 6 if the fixture launches generated simpler-nv kernels instead
  of a minimal CUDA smoke.
- `pypto-serving` source remains under `tmp/sources/` and must not be
  committed.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source --require-cuda \
      --prompt hello --max-new-tokens 2 --device 0 --arch compute_90'
```

**Merge readiness evidence:**

- Docs label the fixture as source-contract or synthetic serving evidence.
- PR body states that no DeepSeek-V4-Flash weights are loaded and no real text
  correctness is proven.
- If temporary source sync is required for `tmp/sources/`, the exact sync
  command is documented.

**Risks/unknowns:**

- The source contract may diverge from upstream `pypto-serving`.
- FastAPI/httpx dependencies may not be present in every local venv.
- Fixture success is not equivalent to production serving.

### PR 8 - vLLM DeepSeek-V4-Flash Readiness Gates

**Objective:** land weight-free vLLM DeepSeek V4 import/config/artifact probes
and the weight-manifest gate before any real model-load or serving attempt.

**Owned file/path scope:**

- `examples/cuda/vllm_deepseek_v4_import_probe.py`
- `examples/cuda/vllm_deepseek_v4_config_probe.py`
- `examples/cuda/deepseek_v4_flash_weight_manifest.py`
- `examples/cuda/README.md`
- `docs/in_progress/nvidia_backend/deepseek_v4_flash_serving_readiness.md`
- `docs/in_progress/nvidia_backend/vllm_remote_install_probe.md`
- `docs/in_progress/nvidia_backend/vllm_deepseek_v4_import_probe.md`
- `docs/in_progress/nvidia_backend/vllm_deepseek_v4_config_probe.md`
- `docs/in_progress/nvidia_backend/vllm_deepseek_v4_artifact_probe.md`
- `docs/in_progress/nvidia_backend/deepseek_v4_flash_weight_manifest_gate.md`
- `tests/ut/py/test_vllm_deepseek_v4_import_probe.py`
- `tests/ut/py/test_vllm_deepseek_v4_config_probe.py`
- `tests/ut/py/test_deepseek_v4_flash_weight_manifest.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`

**Dependencies:**

- PR 1 for source manifest and H200 runner.
- The `.venv-vllm-probe` remote venv or an equivalent documented venv.
- Small model artifacts remain under `tmp/model-artifacts/` and are not
  committed.

**Verification commands:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_vllm_deepseek_v4_import_probe.py \
  tests/ut/py/test_vllm_deepseek_v4_config_probe.py \
  tests/ut/py/test_deepseek_v4_flash_weight_manifest.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv-vllm-probe/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    python examples/cuda/vllm_deepseek_v4_config_probe.py \
      --require-vllm --max-position-embeddings 262144'
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/deepseek_v4_flash_weight_manifest.py --require-complete
```

**Merge readiness evidence:**

- Import/config probes are weight-free and say so.
- The weight manifest gate blocks serving when shards are incomplete.
- Docs record current shard count, present shard count, and indexed bytes.
- PR body states that no vLLM server is started and no output text is
  generated.

**Risks/unknowns:**

- The full DeepSeek-V4-Flash shard download needs a storage and checksum plan.
- Remote vLLM dependencies are large and may drift.
- A passing config probe does not imply model-load feasibility.

## Smallest First PR

The smallest next PR should be PR 0, `Restart PR Execution Plan`.

It should be created from `origin/main` and include only:

- `docs/in_progress/nvidia_backend/restart_pr_plan.md`

This PR gives reviewers a stable decomposition before any worker imports code
from the dirty orphan workspace. It also avoids the trap of making the first
restart PR a giant bootstrap containing `.agents/`, CUDA code, tests, docs,
and historical evidence at once.

## Remove, Move To `tmp/`, Or Postpone

### Remove From Child PR Staging

- Do not stage the whole orphan index or use `git add -A` from `restart`.
- Do not include the full reconstructed base repo in any NVIDIA child PR.
  Base files such as `src/a2a3/`, `src/a5/`, existing Ascend examples,
  existing ST/UT suites, CI, packaging, and stable docs need a separate branch
  strategy if they truly differ from `origin/main`.
- Do not include historical CUDA eval scripts by default:
  `.agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_pair_stream_benchmark.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_lifecycle_matrix.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_scheduler_error_matrix.py`,
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py`, and
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py`.
- Do not commit old promoted review-site captures under
  `docs/nvidia-backend/` as part of early restart PRs.

### Keep Or Move To `tmp/`

- Keep external source checkouts under `tmp/sources/`.
- Keep small DeepSeek artifact subsets under `tmp/model-artifacts/`.
- Keep generated Gluon files and JSON manifests under `tmp/gluon-*`.
- Keep worker transcripts and raw logs under `tmp/dispatch/`.
- If a future worker creates raw benchmark output, copy only summarized
  evidence into the matching in-progress doc and leave raw artifacts under
  `tmp/`.

### Postpone

- Stable-doc promotion into `docs/nvidia-backend/` until the corresponding
  child PR is merged and the in-progress doc has been reviewed.
- RDMA and multi-node UCCL evidence.
- FlashInfer-like serving kernels.
- Real DeepSeek-V4-Flash shard download, model load, vLLM server start, and
  2-H200 correct-output serving.
- Any claim that generated kernels have "perfect performance" until benchmark
  comparisons and hardware artifacts support that wording.

