# PTO Comparison Mapping And Actions

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
`tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json`,
`tmp/cuda-backend/pto-serving-shards-a16851f6/qwen-safetensors-shards.json`,
`tmp/cuda-backend/pto-serving-safetensors-a16851f6/qwen-safetensors-metadata.json`,
`tmp/cuda-backend/pto-serving-weight-residency-1ae913c9/qwen-cuda-weight-residency.json`,
`tmp/cuda-backend/pto-serving-weight-args-21589e81/qwen-persistent-weight-args.json`,
`tmp/cuda-backend/pto-serving-weight-materialization-2026-06-01/qwen-persistent-weight-materialization.json`,
`tmp/cuda-backend/pto-serving-resident-weight-table-2026-06-01/qwen-resident-weight-table.json`,
`tmp/cuda-backend/pto-serving-input-binding-2026-06-01/qwen-runtime-input-binding.json`,
`tmp/cuda-backend/pto-serving-scaffold-2026-06-01/qwen-serving-scaffold.json`,
and
`tmp/cuda-backend/pto-serving-preflight-2026-06-01/pto-serving-preflight.json`.
They record the proxy-only execution state plus the new partial runtime plan:
the benchmark viewer has a controlled attention-tile PTO serving-equivalent
row, and the repo-owned PTO CUDA path now has a reviewable Qwen3-8B model
shape, KV-cache capacity ladder, weight-binding plan, and persistent-device
task mapping, tokenizer-observed prompt counts, padded target-length
`input_ids`, matching `attention_mask`, and decode `output_ids` buffer plans,
CUDA token-buffer allocation/copy-back verification, safetensors shard/tensor
persistent decode token argument binding through `a`, `b`, and `out`,
token pointer-table ownership through decode-arg materialization,
KV-cache key/value binding through persistent DAG `c` and `d`,
controlled proxy numeric-oracle outputs for the generated task-body scaffold,
dry-run decode-loop owner ordering and persistent DAG submission planning,
safetensors shard/tensor inventory, and the config-derived expected weight
shape/dtype contract. It also has local Qwen shard placement plus actual
safetensors shape/dtype
validation for 399 tensors across five shards. The CUDA binding artifact maps
all 399 tensors to stable binding slots, file byte ranges, and readonly
persistent-device argument roles, held all 16.38 GB of Qwen weights resident
on an A100 at once, and verified 16 small norm tensors by copying them back
from device memory. The persistent weight-argument artifact decomposes Qwen
work into 255 persistent DAG task descriptors that cover all 399 weights while
staying within the four-pointer `tensor_args` ABI. The materialization artifact
maps those descriptors through the `CudaPersistentDagTask` ctypes layout and
records the symbolic `resident_weight_ptrs[slot_id]` source for each weight
argument. The resident table artifact adds a process-scoped owner that keeps
399 dry-run pointers live through materialization and frees all of them after
close. The decode-loop runner artifact orders those owners around persistent
DAG submission and output-token accounting. The task-body artifact renders
source-level Qwen persistent-device callables through the existing persistent
DAG source generator and records token, KV-cache, and weight field
consumption, including mutable `c`/`d` writeback fields. It still does not
provide numerically correct Qwen kernels or a `cuda_live` decode loop for
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
