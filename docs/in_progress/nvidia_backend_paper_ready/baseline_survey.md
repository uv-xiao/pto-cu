# Baseline Source Survey

## Purpose

This survey turns the paper-baseline requirement into reviewable source state.
It records which external systems have been inspected, where their local source
copies live under `tmp/`, and what each child evaluation slice must reproduce.

## Local Source State

| System | Upstream | Local source | Commit | Status |
| --- | --- | --- | --- | --- |
| MPK | `mirage-project/mirage` branch `mpk` | `tmp/baselines/mirage-mpk` | `bde2dec1736d612f7a2e4c89e6182560a863072f` | cloned for survey |
| VDCores | `vdcores/vdcores` branch `main` | `tmp/baselines/vdcores` | `5247328cf3f893ed9df95f9f38e7e9a97f0cbfb1` | cloned for survey |
| vLLM | `vllm-project/vllm` branch `main` | `tmp/baselines/vllm` | `27fa5aa3b952a6108de127423397e50364a95fcb` | cloned for survey |
| SGLang | `sgl-project/sglang` branch `main` | `tmp/baselines/sglang` | `7ed53d15f357ea4d722c1980c2cb35e8367d8bb0` | cloned for survey |
| ThunderKittens | `HazyResearch/ThunderKittens` branch `main` | `tmp/baselines/thunderkittens` | `34b15f7e7012de25ae162c8d9dc85296dd342676` | cloned for survey |

Comparator dependency sources used by those baseline runs are also kept under
`tmp/` for review. The current FlashAttention source clone is
`tmp/baselines/flash-attention` at commit
`6dba0373b775196039aedda01cd14c51662965d8`. It was used to build the
FlashAttention-3 Hopper module required by the official ThunderKittens H100
MHA benchmark.

The committed viewer data mirrors this table in
`docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json` so the
human-reviewable benchmark viewer can show baseline readiness without relying
on private terminal history. Reproduction commands for these systems live in
`docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json` so the
viewer can show which setup/run commands and tmp artifacts are expected before
a baseline can be imported as paper evidence.
Serving workload policies live in
`docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json`. The MPK
and VDCores papers use different context and decode lengths, so the current
survey records two comparable policy IDs instead of pretending there is one
universal serving row:

- `mpk_offline_decode`: Qwen3-8B primary, prompt target 64, decode 1024,
  offline batch sizes 1, 2, 4, 8, and 16.
- `vdcores_offline_decode`: Qwen3-8B cross-paper target through the VDCores
  `qwen3` schedule path, context target 128, decode 64, offline batch sizes
  1, 2, 4, 8, and 16.

The current primary-model launch plan for those policies is materialized at
`tmp/cuda-backend/paper-baselines/serving-runs/plan-43b927ed.json` by
`.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`.
It expands the two policy IDs into MPK, VDCores, vLLM, and SGLang command rows
for each batch size, including the raw artifact paths expected by the viewer
importer after the long baseline runs complete. SGLang commands in the plan
prefix `PYTHONPATH=$PWD/tmp/baselines/sglang/python:$PYTHONPATH` so benchmark
modules resolve from the pinned source checkout.

The persistent-device scheduler comparison has separate run contracts in
`paper_baseline_runs.json`: `mpk_persistent_scheduler_trace` and
`vdcores_resource_policy_trace`. These are not serving throughput rows; they
name the planned MPK and VDCores artifacts needed to compare generated
persistent kernels, virtual-core queue/resource policy, dispatch traces, and
scheduler overhead against PTO CUDA persistent-device captures.

## MPK Notes

MPK is sourced from `https://github.com/mirage-project/mirage/tree/mpk`.
The local clone at `tmp/baselines/mirage-mpk` is on commit
`bde2dec1736d612f7a2e4c89e6182560a863072f`.

Observed entry points:

- `README.md`: MPK quickstart and `PersistentKernel` API.
- `demo/qwen3/demo.py`: Qwen3 demo referenced by the README.
- `benchmark/benchmark_serving.py`: serving endpoint benchmark.
- `src/kernel/runtime.cc`: persistent-kernel runtime code generation.
- `src/kernel/task_register.cc`: generated task implementation registry.

Paper baselines to reproduce or compare against:

- vLLM;
- SGLang;
- FlashInfer or FlashAttention;
- cuBLAS or cuTLASS;
- CUDA operator path;
- Triton operator path.

First reproduction command candidates:

```bash
python demo/qwen3/demo.py
python demo/qwen3/demo.py --use-mirage
python demo/qwen3/demo.py --use-mirage --profiling
```

## VDCores Notes

VDCores is sourced from `https://github.com/vdcores/vdcores`.
The local clone at `tmp/baselines/vdcores` is on commit
`5247328cf3f893ed9df95f9f38e7e9a97f0cbfb1`.

Observed entry points:

- `README.md`: CUDA 13.0, Hopper `sm_90a`, and Llama 3.1 decode demo.
- `Makefile` and `setup.py`: Python extension build path.
- `src/runtime.cu` and `src/torch_runtime.cu`: runtime and PyTorch binding.
- `include/dae/`: virtual core runtime, queues, allocator, and launcher.
- `include/task/`: attention, GEMV, RMSNorm, RoPE, SiLU, WGMMA, and argmax.
- `app/python/qwen3/sched.py`: Qwen/Qwen3-8B decode schedule used for
  the current paper-target preflight.
- `app/python/qwen3_1p7b/sched.py`: Qwen/Qwen3-1.7B bring-up and
  runtime-diagnostic schedule.
- `app/python/llama3/sched.py`: Llama decode schedule retained as an
  upstream demo path, but no longer the paper-target VDCores row.
- `agents/workflows/development-and-test.md`: correctness and benchmark
  commands used by the VDCores repo.

Paper baselines to reproduce or compare against:

- vLLM;
- SGLang;
- Mirage;
- ThunderKittens variants;
- Torch plus ThunderKittens.

First reproduction command candidates:

```bash
make pyext
HF_TOKEN= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python app/python/qwen3/sched.py \
  --hf-cache-dir <shared-hf-cache>/hub \
  --correctness
```

The first Qwen3-8B preflight loaded the offline H200 checkpoint and reached
`dae.launch()`, then stopped because the compiled `dae.runtime` did not include
the Qwen3-8B compute-operator set reported by the launcher. After rebuilding
with that operator set, the bounded correctness path passed and a token-1
benchmark emitted timing. A later capacity diagnostic measured the full
`-N 64` schedule at up to `2177` compute instructions and `15042` memory
instructions per SM. Switching VDCores to a temporary global-instruction
runtime with `numInsts=16384` allowed `-N 64 -b 5` to run, but that runtime
failed Qwen3-8B correctness thresholds. The remaining VDCores paper-serving
blocker is therefore correctness for the global-instruction path, or an
equivalent segmented schedule that preserves the shared-instruction runtime.

## vLLM Notes

vLLM is sourced from `https://github.com/vllm-project/vllm`.
The local clone at `tmp/baselines/vllm` is on commit
`27fa5aa3b952a6108de127423397e50364a95fcb`.

Observed entry points:

- `README.md`: serving features, PagedAttention, CUDA/HIP graphs, and
  optimized attention backends.
- `benchmarks/README.md`: benchmark categories and CLI documentation links.
- `benchmarks/benchmark_serving.py`: online serving latency/throughput harness.
- `benchmarks/benchmark_throughput.py`: offline throughput harness.
- `benchmarks/backend_request_func.py`: OpenAI-compatible request adapters,
  including vLLM and SGLang backend labels.
- `csrc/`: CUDA kernels and launch utilities.

First reproduction command candidates:

```bash
vllm serve <model> --port 8000
vllm bench serve --backend vllm --model <model> --host 127.0.0.1 --port 8000
vllm bench throughput --model <model>
```

## SGLang Notes

SGLang is sourced from `https://github.com/sgl-project/sglang`.
The local clone at `tmp/baselines/sglang` is on commit
`7ed53d15f357ea4d722c1980c2cb35e8367d8bb0`.

Observed entry points:

- `README.md`: runtime features such as RadixAttention, continuous batching,
  paged attention, prefill/decode disaggregation, and OpenAI API support.
- `docs/developer_guide/benchmark_and_profiling.md`: benchmark tool taxonomy.
- `python/sglang/check_env.py`: package checks for FlashInfer and Triton.
- `sgl-kernel/CMakeLists.txt`: FlashInfer, FlashAttention, Triton, and custom
  CUDA kernel integration.
- `benchmark/`: model, serving, and task benchmark scripts.

First reproduction command candidates:

```bash
python -m sglang.launch_server --model-path <model> --port 30000
python -m sglang.bench_serving --backend sglang --model <model> --port 30000
python -m sglang.bench_offline_throughput --model-path <model>
python -m sglang.bench_one_batch --model-path <model>
```

## ThunderKittens Notes

ThunderKittens is sourced from `https://github.com/HazyResearch/ThunderKittens`.
The local clone at `tmp/baselines/thunderkittens` is on commit
`34b15f7e7012de25ae162c8d9dc85296dd342676`.
The official H100 MHA benchmark now also has a FlashAttention-3 comparator
capture. FlashAttention-3 was built from
`tmp/baselines/flash-attention/hopper` with an SM90 BF16 head-dim-128 scoped
build, then the unmodified ThunderKittens `benchmark.py` and
`test_correctness.py` were run through a local `PYTHONPATH` compatibility shim
that requests `return_attn_probs=True` from `flash_attn_interface`. This
produced all FA3 forward and backward rows across sequence lengths 768, 1536,
3072, 6144, and 12288. A follow-up isolated PyTorch reference capture ran each
large reference cell in a fresh H200 process with expandable allocator
segments. That recovered every selected 6144-token cell. The remaining
official-sweep gap is true 12288-token dense PyTorch reference capacity, not
missing FA3 bindings or monolithic benchmark fragmentation. The tensor-core
claim records this as an accepted evidence-policy exception: paper tables may
show those 12288-token dense reference cells only as OOM/not-applicable
footnotes, while measured FA3, ThunderKittens, PTO, cuBLAS/CUTLASS, and Triton
rows remain eligible for comparison.

Observed entry points:

- `README.md`: tile DSL overview, CUDA 12.8+ requirement, Hopper/Blackwell
  focus, and pre-implemented kernel workflow.
- `include/kittens.cuh`: top-level header-only library entry point.
- `kernels/`: self-contained kernel directories with Makefiles, tests, and
  benchmarks.
- `kernels/layernorm/benchmark.py`: example benchmark structure with Torch and
  Triton references.
- `demos/`: Llama, Qwen, LoLCATS, and Based demos.

First reproduction command candidates:

```bash
cd tmp/baselines/thunderkittens/kernels/<selected-kernel>
make
python benchmark.py
python test_correctness.py
```

## PTO Comparison Mapping

| Baseline concept | PTO comparison target | Required evidence |
| --- | --- | --- |
| MPK persistent megakernel | `cuda/persistent_device` | device scheduler overhead, task dispatch trace, and graph lifecycle |
| VDCores virtual cores | `cuda/persistent_device` | memory/compute task split, queue pressure, and resource policy |
| vLLM and SGLang serving | end-to-end PTO CUDA workload | same model, prompts, batch/decode lengths, and serving metric |
| CUDA Graph replay | `cuda/host_schedule` | host launch overhead and stream semantics |
| cuBLAS/cuBLASLt | tensor tile and GEMM workloads | library throughput and device elapsed time |
| Triton or torch.compile | generated-kernel baseline | compile path, launch path, and correctness |

The current PTO serving comparison has explicit lifecycle artifacts at
`tmp/cuda-backend/pto-serving-lifecycle-b95ff321/qwen-serving-lifecycle-plan.json`,
`tmp/cuda-backend/pto-serving-tokenizer-b95ff321/qwen-prompt-accounting.json`,
`tmp/cuda-backend/pto-serving-weights-edbd4390/qwen-weight-inventory.json`,
`tmp/cuda-backend/pto-serving-scaffold-edbd4390/qwen-serving-scaffold.json`,
and
`tmp/cuda-backend/pto-serving-preflight-edbd4390/pto-serving-preflight.json`.
They record the proxy-only execution state plus the new partial runtime plan:
the benchmark viewer has a controlled attention-tile PTO serving-equivalent
row, and the repo-owned PTO CUDA path now has a reviewable Qwen3-8B model
shape, KV-cache capacity ladder, weight-binding plan, and persistent-device
task mapping, tokenizer-observed prompt counts, and safetensors shard/tensor
inventory. It still does not open safetensors tensors, validate shapes/dtypes,
bind Qwen weights to CUDA buffers, bind token IDs to runtime buffers, bind real
CUDA buffers, run generated Qwen kernel bodies, or execute a decode loop for
`Qwen/Qwen3-8B`.

## Next Dispatcher Actions

1. Build MPK on a compatible GPU host and record Qwen3 native versus MPK
   command outputs from the generated serving command plan.
2. Build VDCores on H100/H200-class hardware and record correctness plus
   decode benchmark outputs from the generated serving command plan.
3. Build the selected ThunderKittens kernel baseline and capture Torch plus
   ThunderKittens comparison data.
4. Add baseline-result import scripts so raw JSON can feed the benchmark
   viewer without hand editing.
