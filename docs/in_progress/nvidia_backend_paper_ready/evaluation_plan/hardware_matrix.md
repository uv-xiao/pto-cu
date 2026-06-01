# NVIDIA Backend Paper-Ready Evaluation Plan: Hardware Matrix

## Hardware Matrix

Minimum paper-development hardware:

- A100 for local development and compatibility with current captures.
- H200 for Hopper-class scheduling, tensor-core, and remote evaluation.

Optional paper extensions:

- H100 when available for direct comparison with MPK and VDCores paper
  hardware.
- B200 or Blackwell-class hardware only after the A100/H200 matrix is stable.

Every result records GPU model, CUDA toolkit, driver version, compute target,
clock policy when known, and whether Multi-Process Service or exclusive mode
was active.

