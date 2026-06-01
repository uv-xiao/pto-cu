# NVIDIA Backend Paper-Ready Evaluation Plan: Metrics

## Metrics

Collect at least:

- correctness status and checksum or reference comparison;
- end-to-end latency;
- device-only elapsed time;
- host launch overhead;
- scheduler overhead;
- throughput;
- p50, p90, p99, mean, standard deviation, min, max, and sample count;
- occupancy or resource policy where available;
- stream count, graph node count, scheduler blocks, worker blocks, block
  dimension, and queue capacity for PTO runtimes.

Paper figures should separate launch overhead, device execution, and scheduler
overhead so CUDA Graph, host-schedule, and persistent-device claims are not
collapsed into one number.

