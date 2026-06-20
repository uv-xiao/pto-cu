# vLLM Remote H200 Install Probe

This note records a bounded remote H200 readiness check for the existing
weight-free vLLM DeepSeek V4 probes. It does not install vLLM, initialize a
vLLM engine, load model weights, start a server, or run inference.

Follow-up evidence after creating a checkout-local vLLM probe environment is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`.
That later gate passes the import/config probes and still fails the artifact
gates because the repo-relative artifact path exposes only metadata/tokenizer
files, not the indexed weight shards.

## Remote Environment

The probe used the checked-in remote runner with `--sync`, so the remote
checkout mirrored this branch's working tree before command execution.

Local branch commit synced for the run:

```text
716ba9c12c7d530bee2b40d1e0e824ac2870a14f
```

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-readiness/`.

Remote hardware/tooling reported:

```text
GPU: 8 x NVIDIA H200 NVL
compute capability: 9.0
driver: 580.126.20
memory per GPU: 143771 MiB
CUDA_HOME: /usr/local/cuda
nvcc: CUDA 12.8, V12.8.61
python3: Python 3.12.3
```

## vLLM Environment Check

The synced remote checkout did not have `.venv-vllm-probe/bin/python`.
System Python also did not have an importable `vllm` package.

No remote vLLM installation was attempted in this slice. The current evidence
is therefore an environment failure for the `--require-vllm` gates, not a
model-load or serving failure.

## Required Probe Results

The required probes were run through the existing scripts with active system
Python because no `.venv-vllm-probe` or equivalent vLLM venv was present.

Import probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 examples/cuda/vllm_deepseek_v4_import_probe.py --require-vllm
```

Result:

```text
status: skipped
vllm_import: missing
source_status: incomplete
exit: 2
```

Config probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 examples/cuda/vllm_deepseek_v4_config_probe.py \
  --require-vllm --max-position-embeddings 262144
```

Result:

```text
status: skipped
vllm_import: missing
source_status: incomplete
requested_max_position_embeddings: 262144
exit: 2
```

Composed artifact/vLLM probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm
```

Result:

```text
status: failed
artifact_probe.status: failed
artifact_probe.reason: artifact directory is missing
vllm_status: missing
failure_reasons:
- artifacts are required but incomplete
- vLLM is required but not available
exit: 2
```

## Interpretation

The remote H200 is reachable through the checked-in runner and reports CUDA
12.8 tooling on H200 NVL GPUs. The remote checkout does not currently provide
the vLLM environment or repo-relative DeepSeek-V4-Flash artifact directory
needed to pass the existing `--require-vllm` and `--require-artifacts` gates.

## Next Gate

Follow-up evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`. That slice
created a remote-local `.venv-vllm-probe`, installed vLLM only into that venv,
and got the weight-free DeepSeek V4 import and config probes passing on the
remote H200.

The remaining blocker is artifact-path readiness: after sync, the remote
checkout still does not expose
`tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash`. The artifact probe and
weight manifest gate now fail explicitly on that missing repo-relative path.
The next gate is to expose a complete artifact directory, including all
indexed safetensors shards, at that path and rerun the same probes. That gate
should still stop before model load, server startup, or inference unless a
later PR explicitly owns those steps.

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not vLLM server health evidence.
- This is not serving correctness or correct-text evidence.
- This is not long-context, latency, throughput, or production readiness
  evidence.
- This did not copy, sync, download, or inspect raw model weights.
