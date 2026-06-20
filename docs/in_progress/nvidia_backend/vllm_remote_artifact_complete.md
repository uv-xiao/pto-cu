# vLLM Remote H200 Artifact Complete

This note records the remote H200 artifact-completeness gate for
`deepseek-ai/DeepSeek-V4-Flash`. It is a weight-free artifact and vLLM
import/config gate. It does not initialize a vLLM engine, load model weights,
start a server, run inference, validate generated text, or measure latency,
throughput, or long-context behavior.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-artifact-complete/`.

## Remote Environment

The remote checkout was refreshed from local commit:

```text
cf8fabaae2f25fb58b04d74f5c57517a9dab4ea3
```

The run reused the existing checkout-local `.venv-vllm-probe`. No venv repair
or package installation was needed in this slice.

Remote hardware/tooling reported:

```text
GPU: 8 x NVIDIA H200 NVL
compute capability: 9.0
driver: 580.126.20
memory per GPU: 143771 MiB
CUDA_HOME: /usr/local/cuda
nvcc: CUDA 12.8, V12.8.61
```

The vLLM probe environment reported:

```text
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0
transformers: 5.12.1
flashinfer-python: 0.6.12
triton: 3.6.0
```

## Artifact Setup

A complete local gitignored `deepseek-ai/DeepSeek-V4-Flash` artifact directory
was copied into the remote checkout's ignored repo-relative artifact path:

```text
tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

The transfer used a bounded `rsync --delete --partial` copy of the artifact
directory only. The source tree refresh used the checked-in remote helper with
`--sync`, which excludes `tmp/` and `.venv-*`, so the copied model artifacts
and vLLM probe venv remained uncommitted.

Post-sync remote inventory found 46 safetensors shard files at the
repo-relative artifact path.

## Required Probe Results

Import probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_import_probe.py --require-vllm
```

Result:

```text
status: passed
vllm_import: available
source_status: available
imported_symbols:
- DeepseekV4Config
- DeepseekV4FP8Config
- DeepseekV4ForCausalLM
- DeepseekV4Tokenizer
exit: 0
```

Config probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_config_probe.py \
  --require-vllm --max-position-embeddings 262144
```

Result:

```text
status: passed
vllm_import: available
source_status: available
config_class: DeepseekV4Config
model_type: deepseek_v4
quantization_method: deepseek_v4_fp8
requested_max_position_embeddings: 262144
exit: 0
```

Composed artifact/vLLM probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm
```

Result:

```text
status: passed
vllm_status: available
artifact_probe.status: passed
indexed_tensors: 69187
indexed_shards: 46
present_shards: 46
missing_shards: 0
present_bytes: 159617149040
index_total_size: 159609485896
exit: 0
```

Manifest gate:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-complete
```

Result:

```text
status: complete
indexed_tensors: 69187
indexed_shards: 46
present_shards: 46
missing_shards: 0
present_bytes: 159617149040
index_total_size: 159609485896
exit: 0
```

## Interpretation

The remote H200 checkout now exposes a complete
`deepseek-ai/DeepSeek-V4-Flash` artifact directory at the repo-relative path
required by the existing probes. The same remote vLLM probe venv that
previously passed the import/config gates still passes those gates, and the
artifact and manifest gates now pass with all 46 indexed safetensors shards
present.

This clears the remote pre-load artifact gate. The next reviewable gate is a
separate model-load or serving probe with its own command, resource plan,
failure boundary, and non-claims.

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not H200 vLLM engine initialization evidence.
- This is not vLLM server startup or server health evidence.
- This is not inference, prompt, tokenizer semantic, correct-text, or 256K
  context behavior evidence.
- This is not throughput, latency, or production readiness evidence.
- This did not commit raw model artifacts, venvs, command dumps, or `tmp/`
  symlinks.
