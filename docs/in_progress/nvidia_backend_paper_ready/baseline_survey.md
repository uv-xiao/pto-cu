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
| vLLM | `vllm-project/vllm` | `tmp/baselines/vllm` | pending | planned |
| SGLang | `sgl-project/sglang` | `tmp/baselines/sglang` | pending | planned |
| ThunderKittens | `HazyResearch/ThunderKittens` | `tmp/baselines/thunderkittens` | pending | planned |

The committed viewer data mirrors this table in
`docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json` so the
human-reviewable benchmark viewer can show baseline readiness without relying
on private terminal history.

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

1. Capture source notes for vLLM, SGLang, and ThunderKittens under
   `tmp/baselines/`.
2. Convert each baseline source state into viewer data with commit, status,
   compatible hardware, and run commands.
3. Build MPK on a compatible GPU host and record Qwen3 native versus MPK
   command outputs.
4. Build VDCores on H100/H200-class hardware and record correctness plus
   decode benchmark outputs.
5. Add baseline-result import scripts so raw JSON can feed the benchmark
   viewer without hand editing.
