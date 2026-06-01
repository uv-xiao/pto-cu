# vLLM And SGLang Source Notes

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
