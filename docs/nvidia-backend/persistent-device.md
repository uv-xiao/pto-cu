# CUDA Persistent Device Runtime Analysis

This document analyzes the largest CUDA design gap: NVIDIA GPUs do not expose
an AICPU-like device scheduler that can launch independent worker kernels.
That changes both runtime architecture and build architecture.

## Review Map

- [Core Constraint](persistent-device/core-constraint.md) (50 lines)
- [Host-Schedule Runtime](persistent-device/host-schedule-runtime.md) (62 lines)
- [Persistent Device Runtime](persistent-device/persistent-device-runtime.md) (98 lines)
- [One Kernel Body Across Runtimes](persistent-device/one-kernel-body.md) (278 lines)
- [Static NVCC Linking Feasibility](persistent-device/static-nvcc-linking.md) (51 lines)
- [Orchestrator Separation](persistent-device/orchestrator-separation.md) (73 lines)
- [Build and Architecture Changes](persistent-device/build-architecture-changes.md) (54 lines)
- [NVRTC and nvJitLink Position](persistent-device/nvrtc-nvjitlink-position.md) (21 lines)
- [Sources](persistent-device/sources.md) (12 lines)

## Design Position

The CUDA `persistent_device` runtime should launch one persistent executor
kernel from the host. Scheduler warps or blocks inside that executor manage
ready queues and dispatch linked task functions onto worker warps or blocks.
This is the CUDA path that replaces the Ascend AICPU scheduler assumption.

The stable build direction is ordinary offline `nvcc` plus relocatable device
code and device linking. NVRTC plus nvJitLink remains a useful optional
experimentation path, not the primary production path.
