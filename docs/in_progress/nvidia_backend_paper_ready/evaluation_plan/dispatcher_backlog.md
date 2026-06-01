# NVIDIA Backend Paper-Ready Evaluation Plan: First Dispatcher Backlog

## First Dispatcher Backlog

1. Audit current viewer data against the shared schema.
2. Add missing fields for method category, launch model, hardware, and raw
   artifact paths.
3. Create a baseline source index under `tmp/` for MPK, VDCores, and paper
   baselines.
4. Clone or inspect MPK and VDCores repositories under `tmp/baselines/`.
5. Use `serving_workloads.json` to run MPK, VDCores, vLLM, and SGLang with
   comparable model, prompt, decode, and batch-size policies.
6. Extend viewer export scripts as new benchmark families and paper baselines
   produce raw JSON.
7. Run A100 and H200 paired captures for the current PTO workloads.
8. Add controlled CUDA Graph, cuBLAS, and direct launch baselines.

