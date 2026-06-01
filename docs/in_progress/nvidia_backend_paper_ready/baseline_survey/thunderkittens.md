# ThunderKittens Source Notes

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
