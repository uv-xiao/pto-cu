# NVIDIA Backend Flow Details: CUDA Runtime API vs Driver API

## CUDA Runtime API vs Driver API

NVIDIA exposes two host APIs that can be mixed carefully.

| Concern | CUDA Runtime API | CUDA Driver API |
| ------- | ---------------- | --------------- |
| Context | implicit primary context management | explicit current context model |
| Memory | `cudaMalloc`, `cudaFree`, `cudaMemcpyAsync` | `cuMemAlloc`, `cuMemFree`, `cuMemcpy*Async` |
| Static kernel launch | natural `<<<...>>>` or `cudaLaunchKernel` | `cuLaunchKernel` / `cuLaunchKernelEx` |
| Dynamic module loading | newer `cudaLibraryLoadData` and kernel handles | mature `cuModuleLoadDataEx`, `cuModuleGetFunction` |
| Control level | simpler host code | finer control over module/context lifetime |
| Fit for plugin-style runtime | acceptable with primary context discipline | better ownership and loading semantics |

Relevant NVIDIA constraints:

- Runtime API code is simpler because it manages the primary context and
  modules implicitly.
- Driver API gives more direct control over module loading and lets the runtime
  keep only the modules it needs loaded.
- A Driver API context owns modules, memory, and actions; a host thread needs
  the right current context for most operations.
- Runtime and Driver API can interoperate through the primary context.

Recommended PTO CUDA split:

- Use Driver API for context retention/currentness and module/library loading.
  This matches `ChipWorker` as a plugin-style host runtime loaded by `dlopen`.
- Use Runtime API only where it materially simplifies ordinary memory/stream
  operations and does not hide module ownership.
- Avoid `cudaDeviceReset` inside the backend. It can invalidate shared primary
  context state used by other libraries in the process.
- Treat one `ChipWorker` as owning one CUDA context attachment, stream set, and
  callable-module cache.

The key design choice is not "Runtime API or Driver API everywhere". It is:
make context/module ownership explicit enough for a `dlopen`-loaded runtime,
while keeping simple memory and stream code readable.

