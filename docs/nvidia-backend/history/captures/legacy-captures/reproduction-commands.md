# Legacy Reproduction Commands

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 8,16,32,64 \
    --label a100-wide-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-wide-$(git rev-parse --short HEAD)
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 8,16,32,64 \
     --label h200-wide-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-wide-$(git rev-parse --short HEAD)'
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100-wide-e430bc1b/cuda-benchmark.json \
    tmp/cuda-backend/h200-wide-e430bc1b/cuda-benchmark.json \
    --label cuda-wide-a100-h200-e430bc1b \
    --output-dir tmp/cuda-backend/combined-wide-e430bc1b
```

DAG-chain capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
    --label a100-dag-323f4587 \
    --output-dir tmp/cuda-backend/a100-dag-323f4587

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
     --label h200-dag-323f4587 \
     --output-dir tmp/cuda-backend/h200-dag-323f4587'
```

Scratch-reuse DAG capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
    --label a100-reuse-bcf54a88 \
    --output-dir tmp/cuda-backend/a100-reuse-bcf54a88

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
     --label h200-reuse-bcf54a88 \
     --output-dir tmp/cuda-backend/h200-reuse-bcf54a88'
```

Tensor-tile DAG capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
    --label a100-tensor-8950e029 \
    --output-dir tmp/cuda-backend/a100-tensor-8950e029

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
     --label h200-tensor-8950e029 \
     --output-dir tmp/cuda-backend/h200-tensor-8950e029'
```

Extended worker-grid capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 6 \
    --worker-blocks-per-task 32,64,128,256 \
    --label a100-gridext-3eeb399a \
    --output-dir tmp/cuda-backend/a100-gridext-3eeb399a

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 6 \
     --worker-blocks-per-task 32,64,128,256 \
     --label h200-gridext-3eeb399a \
     --output-dir tmp/cuda-backend/h200-gridext-3eeb399a'

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-gridext-3eeb399a/cuda-benchmark.json \
    tmp/cuda-backend/h200-gridext-3eeb399a/cuda-benchmark.json \
    --label cuda-gridext-a100-h200-3eeb399a \
    --output-dir tmp/cuda-backend/combined-gridext-3eeb399a
```

Task-count sweep capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 2,6,12 \
    --worker-blocks-per-task 128,256 \
    --label a100-taskcount-7194bfc9 \
    --output-dir tmp/cuda-backend/a100-taskcount-7194bfc9

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 65536,1048576 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 2,6,12 \
     --worker-blocks-per-task 128,256 \
     --label h200-taskcount-7194bfc9 \
     --output-dir tmp/cuda-backend/h200-taskcount-7194bfc9'

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-taskcount-7194bfc9/cuda-benchmark.json \
    tmp/cuda-backend/h200-taskcount-7194bfc9/cuda-benchmark.json \
    --label cuda-taskcount-a100-h200-7194bfc9 \
    --output-dir tmp/cuda-backend/combined-taskcount-7194bfc9
```

Wider range capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 16384,262144,4194304 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 4,8,16 \
    --worker-blocks-per-task 128,256 \
    --label a100-rangewide-cc6869f7 \
    --output-dir tmp/cuda-backend/a100-rangewide-cc6869f7

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 16384,262144,4194304 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 4,8,16 \
     --worker-blocks-per-task 128,256 \
     --label h200-rangewide-cc6869f7 \
     --output-dir tmp/cuda-backend/h200-rangewide-cc6869f7'

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-rangewide-cc6869f7/cuda-benchmark.json \
    tmp/cuda-backend/h200-rangewide-cc6869f7/cuda-benchmark.json \
    --label cuda-rangewide-a100-h200-cc6869f7 \
    --output-dir tmp/cuda-backend/combined-rangewide-cc6869f7
```

Stream concurrency:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --stream-concurrency --device 0 --repeats 7 --arch compute_80 \
    --label a100-stream-37bebf44 \
    --output-dir tmp/cuda-backend/a100-stream-37bebf44

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd <remote-pto-cu> && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --stream-concurrency --device 0 --repeats 7 --arch compute_90 \
     --label h200-stream-37bebf44 \
     --output-dir tmp/cuda-backend/h200-stream-37bebf44'
```
