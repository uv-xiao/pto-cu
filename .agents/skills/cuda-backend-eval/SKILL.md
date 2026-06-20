---
name: cuda-backend-eval
description: Use when implementing, testing, or evaluating the PTO CUDA backend with local CUDA probes or remote H200 execution, especially when Codex needs to run CUDA commands through SSH, collect evidence, or avoid unsupported performance and serving claims.
---

# CUDA Backend Evaluation

Use this skill for PTO CUDA backend evaluation workflows, not for recording
one-off benchmark recipes. Keep task-specific experiments in `tmp/` unless a
reviewed PR promotes them into durable tests, docs, or scripts.

## Local Environment

- Create a project-local venv before any `pip` command:
  `python3 -m venv --system-site-packages .venv`.
- Use the venv Python directly for Python checks:
  `.venv/bin/python -m pytest ...`.
- Set `PYTHONPATH=$PWD:$PWD/python` for source-tree CUDA probes so both
  `simpler_setup` and the runtime package resolve from the checkout under
  test.
- Do not rely on bare `pytest` or global Python tools when reporting
  verification.

## Local CUDA Probes

- First check whether CUDA is visible before running hardware work:
  `command -v nvcc` and
  `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader`.
- Prefer existing focused pytest selectors for committed behavior. Use
  `--platform cuda` when the test expects the CUDA platform.
- Report the actual local GPU shown by `nvidia-smi`; do not describe local
  results as H200 results unless the command ran on the remote H200 host.
- Treat CUDA microbenchmarks as launch, wall-time, device-event, copy, or
  dependency/concurrency evidence only. Do not turn them into model-serving or
  paper-level claims.

## Remote H200 Commands

Use `scripts/run-remote-cuda.sh` for non-interactive remote CUDA commands. It
sets `CUDA_HOME`, CUDA `PATH`, and `PYTHONPATH` in the remote checkout before
running the command.

```bash
REMOTE_PTO_CU=/work/pto-cu \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- nvidia-smi
```

Pass `--host` or `REMOTE_HOST` when the SSH host is not the default. Pass
`--remote-dir` or `REMOTE_PTO_CU` to select the remote checkout:

```bash
.agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh \
  --host h200-host --remote-dir /work/pto-cu -- \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
    -q --platform cuda
```

Use `--sync` only when the remote checkout must mirror the local tree before
the run. The sync excludes `.venv`, `build`, `tmp`, Python caches, and pytest
caches. Record whether evidence came from a remote Git refresh, an existing
remote checkout, or `--sync`.

## Evidence Rules

- Record the exact command, git commit, machine/GPU, CUDA toolkit path, driver
  version, and pass/fail outcome for every reported CUDA result.
- Keep raw command output and exploratory artifacts under `tmp/` unless the
  user explicitly asks for committed artifacts.
- Separate local CUDA evidence from remote H200 evidence in summaries and PR
  bodies.
- State when CUDA hardware, SSH access, model weights, vLLM, or serving
  systems were not run.
- Do not claim DeepSeek model load, vLLM server health, serving output,
  paper-level performance, or H200 validation without fresh command output
  from that exact run.

## Skill Maintenance

- Keep `SKILL.md` concise and durable. Add narrowly reusable scripts only when
  Codex would otherwise rewrite the same remote-evaluation plumbing.
- Do not commit capture-specific smoke, benchmark, validator, scheduler, or
  report-generation scripts in this skill. Put one-off experiments under
  `tmp/` until they are promoted through a reviewed PR.
