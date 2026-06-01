# CUDA Persistent Device Runtime Analysis: Core Constraint

## Core Constraint

In the a2a3/a5 runtime, the AICPU is a device-side control processor. It can
run scheduler code, observe task readiness, and hand work to AICore workers.
CUDA has no equivalent control processor. A CUDA GPU can run kernels launched
from the host, and CUDA Dynamic Parallelism can launch child grids from device
code, but using child kernel launches as the normal task dispatch mechanism is
not the right default for PTO:

- child launches add CUDA device-runtime state and launch-management overhead;
- parent/child completion is nested, which fights a long-lived scheduler loop;
- stream/event objects created on device have grid scope;
- the device runtime has resource limits that become scheduler limits;
- it still launches grids, not lightweight task functions onto reserved worker
  warps.

The CUDA `persistent_device` runtime should therefore launch one persistent
executor kernel from the host. Scheduler warps/blocks inside that executor
manage ready queues and dispatch tasks by calling linked task functions on
worker warps/blocks.

```text
host
  |
  | cuLaunchKernel(pto_persistent_executor<<<grid, block, smem, stream>>>)
  v
CUDA persistent executor grid
  |
  | scheduler warps/blocks
  |  - build or consume TensorMap/ring metadata
  |  - update fanin/fanout counters
  |  - push ready task descriptors
  v
ready queues in global memory
  |
  | worker warps/blocks
  |  - pop descriptor
  |  - decode func_id + args
  |  - call linked task function through generated dispatch
  |  - publish completion
  v
completion queues/counters in global memory
```

This is not a direct AICPU replacement. The scheduler and worker roles share
the same CUDA grid and compete for SM resources. The runtime must decide how
many blocks/warps are scheduler roles and how many are workers.

