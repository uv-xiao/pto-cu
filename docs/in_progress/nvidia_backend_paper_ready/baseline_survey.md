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

The committed viewer data mirrors this table in
`docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json` so the
human-reviewable benchmark viewer can show baseline readiness without relying
on private terminal history. Reproduction commands for these systems live in
`docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json` so the
viewer can show which setup/run commands and tmp artifacts are expected before
a baseline can be imported as paper evidence.

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
- `app/python/llama3/sched.py`: Llama decode schedule.
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
python app/python/llama3/sched.py -w
make pyext
python app/python/llama3/sched.py -N 256 "Write a hello world in Python."
python app/python/llama3/sched.py --correctness
```

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

## Next Dispatcher Actions

1. Choose the first common model, prompt length, decode length, and batch or
   concurrency policy shared by PTO, MPK, VDCores, vLLM, and SGLang.
2. Build MPK on a compatible GPU host and record Qwen3 native versus MPK
   command outputs.
3. Build VDCores on H100/H200-class hardware and record correctness plus
   decode benchmark outputs.
4. Build the selected ThunderKittens kernel baseline and capture Torch plus
   ThunderKittens comparison data.
5. Add baseline-result import scripts so raw JSON can feed the benchmark
   viewer without hand editing.
