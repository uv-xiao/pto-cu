# vLLM Remote H200 Environment And Artifact Probe

This note records the follow-up remote H200 vLLM environment and artifact
readiness gate for `deepseek-ai/DeepSeek-V4-Flash`. It is a weight-free
environment/artifact check. It does not initialize a vLLM engine, load model
weights, start a server, run inference, validate generated text, or measure
latency, throughput, or long-context behavior.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-env-artifact/`.

## Remote Environment

The remote checkout used a checkout-local `.venv-vllm-probe` environment.
The probe venv was created under the remote checkout selected by
`REMOTE_PTO_CU`; committed documentation intentionally omits the absolute
remote path and host name.

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

The vLLM probe environment installed:

```text
vllm: 0.23.0
torch: 2.11.0
transformers: 5.12.1
flashinfer-python: 0.6.12
triton: 3.6.0
```

Because the venv used `--system-site-packages`, it initially exposed
ABI-incompatible system packages. Venv-local repairs installed newer
environment-local packages for the failing dependencies:

- `scipy>=1.16`, `blosc2`, and related dependencies fixed `pip check`.
- `pandas>=2.3` fixed the NumPy ABI error raised while importing pandas.
- `bottleneck>=1.4` fixed the remaining optional pandas ABI issue.

The final `pip check` result was clean.

## Required Probe Results

The final required probes are recorded in
`tmp/vllm-remote-env-artifact/required_probe_results_final.txt`.

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
status: failed
vllm_status: available
artifact_probe.status: failed
artifact_probe.reason: indexed weight shards are missing
failure_reasons:
- artifacts are required but incomplete
exit: 2
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
status: incomplete
indexed_tensors: 69187
indexed_shards: 46
index_total_size: 159609485896
present_shards: 0
missing_shards: 46
present_bytes: 0
exit: 2
```

## Artifact State

The repo-relative artifact path used by the probes was:

```text
tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

The path exposed a prior scratch metadata-only artifact directory through a
repo-relative `tmp/model-artifacts/...` symlink for probing. Required
metadata/tokenizer files were present, including:

- `config.json`
- `model.safetensors.index.json`
- `tokenizer.json`
- `tokenizer_config.json`

The indexed shard set was not present:

```text
indexed_shards: 46
present_shards: 0
missing_shards: 46
present_bytes: 0
```

## Interpretation

The remote H200 vLLM import and synthetic DeepSeek V4 config gates now pass in
the checkout-local vLLM probe environment. The artifact gates still fail
correctly because the repo-relative artifact path exposes only metadata and
tokenizer files, not the 46 indexed weight shards.

The next reviewable gate is remote artifact completion under the same
repo-relative path, followed by the same required artifact and manifest probes.
That gate should still stop before model load, server startup, or inference
unless a later PR explicitly owns those steps.

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not H200 vLLM engine initialization evidence.
- This is not vLLM server health evidence.
- This is not prompt, tokenizer semantic, correct-text, long-context,
  throughput, latency, or production readiness evidence.
- This did not copy, sync, download, or commit raw model weights.
