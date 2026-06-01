# CUDA Backend Status

This page tracks current implementation status against the CUDA backend design.
It distinguishes verified tracer-bullet behavior from remaining design work so
evaluation results are not mistaken for a complete backend.

## Implemented And Verified

- [Platform and runtime discovery](status/platform-runtime-discovery.md)
- [Host-schedule runtime](status/host-schedule-runtime.md)
- [Persistent-device runtime](status/persistent-device-runtime.md)
- [Evaluation and reporting](status/evaluation-and-reporting/index.md)

## Latest Local Verification

- [Verification archive](status/latest-local-verification/index.md)

## Remaining Gaps

- [Kernel compiler integration](status/remaining-gaps/kernel-compiler-integration/index.md)
- [Fourth-tensor persistent DAG verification](status/remaining-gaps/fourth-tensor-persistent-dag-verification.md)
- [Target role cleanup](status/remaining-gaps/target-role-cleanup.md)
- [Persistent scheduler generalization](status/remaining-gaps/persistent-scheduler-generalization/index.md)
- [Tuned tensor workloads](status/remaining-gaps/tuned-tensor-workloads.md)
- [CI coverage](status/remaining-gaps/ci-coverage.md)

## Review Policy

The linked files preserve the previous status evidence while keeping each page
short enough for human review. New CUDA status updates should go to the focused
file that owns the runtime, verification, or remaining-gap topic.
