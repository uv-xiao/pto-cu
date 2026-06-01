# MPK And VDCores Source Notes

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
The shared-window analysis script derives the minimum segmented runtime
requirement from the same H200 artifact: keeping the default 512-instruction
shared table needs at least 5 compute-instruction windows and 30
memory-instruction windows per SM for the Qwen3-8B decode64 path.
