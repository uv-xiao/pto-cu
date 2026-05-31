#!/usr/bin/env python3
"""Capture a CUTLASS 16x16x16 tensor-tile baseline for the viewer."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
CUTLASS_ROOT = ROOT / "tmp" / "baselines" / "cutlass"
DEFAULT_SHAPE = "n=1024, tensor tile 16x16x16"
DEFAULT_DTYPE = "tf32 CUTLASS Gemm tensor op, f32 accumulator"
DEFAULT_TOLERANCE = 1.0e-3


def fail(message: str) -> None:
    raise SystemExit(f"cutlass tensor tile capture failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def percentile_int(values: list[int], quantile: float) -> int:
    if not values:
        fail("cannot summarize an empty sample list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return int(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def latency_summary(samples: list[dict[str, Any]], field: str, prefix: str) -> dict[str, int]:
    values = [int(sample[field]) for sample in samples]
    return {
        f"{prefix}_p50_ns": percentile_int(values, 0.50),
        f"{prefix}_p90_ns": percentile_int(values, 0.90),
        f"{prefix}_p99_ns": percentile_int(values, 0.99),
        f"{prefix}_mean_ns": int(statistics.fmean(values)),
        f"{prefix}_stdev_ns": int(statistics.stdev(values)) if len(values) > 1 else 0,
        f"{prefix}_min_ns": min(values),
        f"{prefix}_max_ns": max(values),
    }


def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        fail(f"{owner} missing {key}")
    return value


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} missing {key}")
    return value


def require_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        fail("capture has no samples")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(f"sample {index} is not an object")
        for key in ("host_wall_ns", "device_wall_ns", "max_abs_error"):
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                fail(f"sample {index} has invalid {key}")
    return samples


def require_binary_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        fail("capture binary returned no samples")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(f"binary sample {index} is not an object")
        for key in ("host_wall_ns", "device_wall_ns", "max_abs_error"):
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                fail(f"binary sample {index} has invalid {key}")
    return samples


def viewer_record(payload: dict[str, Any], raw_artifact: str) -> dict[str, Any]:
    owner = "cutlass tensor tile capture"
    metadata = require_dict(payload, "metadata", owner)
    hardware = require_dict(payload, "hardware", owner)
    inputs = require_dict(payload, "inputs", owner)
    samples = require_samples(payload)
    host_summary = latency_summary(samples, "host_wall_ns", "host_wall")
    device_summary = latency_summary(samples, "device_wall_ns", "device_wall")
    max_abs_error = max(float(sample["max_abs_error"]) for sample in samples)
    tolerance = float(metadata.get("tolerance", DEFAULT_TOLERANCE))

    return {
        "benchmark_id": "tensor_core_tile",
        "method_id": "cutlass",
        "hardware": {
            "gpu": require_string(hardware, "gpu", owner),
            "machine": require_string(hardware, "machine", owner),
            "compute_target": require_string(hardware, "compute_target", owner),
            "driver": str(hardware.get("driver", "see raw artifact")),
            "cuda_toolkit": str(hardware.get("cuda_toolkit", "see raw artifact")),
            "clock_policy": str(hardware.get("clock_policy", "not recorded")),
        },
        "commit": str(metadata.get("pto_commit", metadata.get("git_commit", "unknown"))),
        "inputs": {
            "shape": require_string(inputs, "shape", owner),
            "dtype": require_string(inputs, "dtype", owner),
            "repeat_policy": require_string(inputs, "repeat_policy", owner),
        },
        "statistic": {
            "kind": "cutlass_cuda_event_distribution",
            "sample_count": len(samples),
            "host_wall_ns": host_summary["host_wall_p50_ns"],
            "device_wall_ns": device_summary["device_wall_p50_ns"],
            **host_summary,
            **device_summary,
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
        },
        "raw_artifact": raw_artifact,
        "correctness": "pass" if max_abs_error <= tolerance else "fail",
    }


def run_text(command: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        fail(f"{' '.join(command)} failed:\n{result.stdout}")
    return result.stdout.strip()


def driver_version(device: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(device),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "not recorded"
    if result.returncode != 0 or not result.stdout.strip():
        return "not recorded"
    return result.stdout.splitlines()[0].strip()


def gpu_name(device: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                str(device),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "not recorded"
    if result.returncode != 0 or not result.stdout.strip():
        return "not recorded"
    return result.stdout.splitlines()[0].strip().split("NVIDIA ", 1)[-1].split()[0]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def cutlass_commit() -> str:
    if not CUTLASS_ROOT.is_dir():
        return "missing"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CUTLASS_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def nvcc_path() -> str:
    for candidate in (
        shutil.which("nvcc"),
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    fail("nvcc was not found")


def nvcc_version(nvcc: str) -> str:
    result = subprocess.run(
        [nvcc, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        return "not recorded"
    for line in result.stdout.splitlines():
        if "release" in line:
            return line.strip()
    return result.stdout.splitlines()[-1].strip() if result.stdout.strip() else "not recorded"


def generated_source() -> str:
    return r'''
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"

#define CHECK_CUDA(expr)                                                     \
  do {                                                                       \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "cuda error: " << cudaGetErrorString(err) << "\n";       \
      return 2;                                                              \
    }                                                                        \
  } while (0)

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: <device> <rows> <cols> <inner> <tile_count> "
                 "<warmup> <repeats>\n";
    return 2;
  }
  int device = std::atoi(argv[1]);
  int rows = std::atoi(argv[2]);
  int cols = std::atoi(argv[3]);
  int inner = std::atoi(argv[4]);
  int tile_count = std::atoi(argv[5]);
  int warmup = std::atoi(argv[6]);
  int repeats = std::atoi(argv[7]);
  if (rows <= 0 || cols <= 0 || inner <= 0 || tile_count <= 0 ||
      warmup <= 0 || repeats <= 0) {
    std::cerr << "all dimensions and repeat counts must be positive\n";
    return 2;
  }
  CHECK_CUDA(cudaSetDevice(device));

  using ElementInput = cutlass::tfloat32_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using Gemm = cutlass::gemm::device::Gemm<
      ElementInput, cutlass::layout::ColumnMajor,
      ElementInput, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  size_t a_elems = static_cast<size_t>(tile_count) * rows * inner;
  size_t b_elems = static_cast<size_t>(tile_count) * inner * cols;
  size_t c_elems = static_cast<size_t>(tile_count) * rows * cols;
  std::vector<ElementInput> h_a(a_elems);
  std::vector<ElementInput> h_b(b_elems);
  std::vector<float> h_c(c_elems, 0.0f);
  std::vector<float> h_ref(c_elems, 0.0f);

  for (size_t i = 0; i < a_elems; ++i) {
    h_a[i] = ElementInput(-0.01f + 0.02f * float(i % 257) / 256.0f);
  }
  for (size_t i = 0; i < b_elems; ++i) {
    h_b[i] = ElementInput(0.01f - 0.02f * float(i % 251) / 250.0f);
  }
  for (int tile = 0; tile < tile_count; ++tile) {
    size_t a_base = static_cast<size_t>(tile) * rows * inner;
    size_t b_base = static_cast<size_t>(tile) * inner * cols;
    size_t c_base = static_cast<size_t>(tile) * rows * cols;
    for (int m = 0; m < rows; ++m) {
      for (int n = 0; n < cols; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < inner; ++k) {
          float a = static_cast<float>(h_a[a_base + m + k * rows]);
          float b = static_cast<float>(h_b[b_base + k + n * inner]);
          acc += a * b;
        }
        h_ref[c_base + m * cols + n] = acc;
      }
    }
  }

  ElementInput *d_a = nullptr;
  ElementInput *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(ElementInput) * a_elems));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(ElementInput) * b_elems));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * c_elems));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), sizeof(ElementInput) * a_elems,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), sizeof(ElementInput) * b_elems,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float) * c_elems));

  Gemm gemm;
  auto run_once = [&]() -> cutlass::Status {
    for (int tile = 0; tile < tile_count; ++tile) {
      size_t a_base = static_cast<size_t>(tile) * rows * inner;
      size_t b_base = static_cast<size_t>(tile) * inner * cols;
      size_t c_base = static_cast<size_t>(tile) * rows * cols;
      Gemm::Arguments args({rows, cols, inner},
                           {d_a + a_base, rows},
                           {d_b + b_base, inner},
                           {d_c + c_base, cols},
                           {d_c + c_base, cols},
                           {1.0f, 0.0f});
      cutlass::Status status = gemm(args);
      if (status != cutlass::Status::kSuccess) {
        return status;
      }
    }
    return cutlass::Status::kSuccess;
  };

  for (int i = 0; i < warmup; ++i) {
    if (run_once() != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS GEMM failed during warmup\n";
      return 3;
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t end;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&end));
  std::vector<long long> device_ns;
  std::vector<long long> host_ns;
  std::vector<double> errors;
  device_ns.reserve(repeats);
  host_ns.reserve(repeats);
  errors.reserve(repeats);

  for (int i = 0; i < repeats; ++i) {
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float) * c_elems));
    auto host_start = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(start));
    if (run_once() != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS GEMM failed during measurement\n";
      return 3;
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    auto host_end = std::chrono::steady_clock::now();
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, end));
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, sizeof(float) * c_elems,
                          cudaMemcpyDeviceToHost));
    double max_abs_error = 0.0;
    for (size_t idx = 0; idx < c_elems; ++idx) {
      max_abs_error = std::max(max_abs_error,
                               std::abs(double(h_c[idx] - h_ref[idx])));
    }
    device_ns.push_back(static_cast<long long>(elapsed_ms * 1000000.0f));
    host_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            host_end - host_start).count());
    errors.push_back(max_abs_error);
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(end));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  int runtime_version = 0;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));

  std::cout << "{\n";
  std::cout << "  \"compute_target\": \"compute_" << prop.major << prop.minor << "\",\n";
  std::cout << "  \"cuda_runtime_version\": \"" << runtime_version << "\",\n";
  std::cout << "  \"samples\": [\n";
  for (int i = 0; i < repeats; ++i) {
    std::cout << "    {\"host_wall_ns\": " << host_ns[i]
              << ", \"device_wall_ns\": " << device_ns[i]
              << ", \"max_abs_error\": " << std::setprecision(10) << errors[i]
              << "}";
    if (i + 1 != repeats) {
      std::cout << ",";
    }
    std::cout << "\n";
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
  return 0;
}
'''


def compile_capture_binary(args: argparse.Namespace, artifact_dir: Path, nvcc: str) -> Path:
    if not CUTLASS_ROOT.is_dir():
        fail(f"CUTLASS checkout is missing: {repo_relative(CUTLASS_ROOT)}")
    source = artifact_dir.resolve() / "cutlass_tensor_tile_capture.cu"
    binary = artifact_dir.resolve() / "cutlass_tensor_tile_capture"
    source.write_text(generated_source(), encoding="utf-8")
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        f"-I{CUTLASS_ROOT / 'include'}",
        f"-I{CUTLASS_ROOT / 'tools' / 'util' / 'include'}",
        f"-gencode=arch={args.arch},code=compute_80",
        str(source),
        "-o",
        str(binary),
    ]
    run_text(command)
    return binary


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.output.parent if args.output else ROOT / "tmp" / "cuda-backend" / "cutlass-capture"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    nvcc = nvcc_path()
    binary = compile_capture_binary(args, artifact_dir, nvcc)
    command = [
        str(binary),
        str(args.device),
        str(args.rows),
        str(args.cols),
        str(args.inner),
        str(args.tile_count),
        str(args.warmup),
        str(args.repeats),
    ]
    binary_payload = json.loads(run_text(command, cwd=artifact_dir))
    binary_samples = require_binary_samples(binary_payload)
    samples = [
        {
            "host_wall_ns": int(sample["host_wall_ns"]),
            "device_wall_ns": int(sample["device_wall_ns"]),
            "max_abs_error": float(sample["max_abs_error"]),
        }
        for sample in binary_samples
    ]

    return {
        "metadata": {
            "pto_commit": args.pto_commit or git_commit(),
            "source": "cutlass_tensor_tile_capture.py",
            "generated_source": repo_relative(artifact_dir / "cutlass_tensor_tile_capture.cu"),
            "cutlass_root": repo_relative(CUTLASS_ROOT),
            "cutlass_commit": cutlass_commit(),
            "rows": args.rows,
            "cols": args.cols,
            "inner": args.inner,
            "tile_count": args.tile_count,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "tolerance": args.tolerance,
            "nvcc_version": nvcc_version(nvcc),
        },
        "hardware": {
            "gpu": gpu_name(args.device),
            "machine": socket.gethostname(),
            "compute_target": str(binary_payload.get("compute_target", args.arch)),
            "driver": driver_version(args.device),
            "cuda_toolkit": str(binary_payload.get("cuda_runtime_version", "unknown")),
            "clock_policy": "not recorded",
        },
        "inputs": {
            "shape": DEFAULT_SHAPE,
            "dtype": DEFAULT_DTYPE,
            "repeat_policy": f"{args.repeats}-repeat CUTLASS tensor tile capture",
        },
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Convert an existing raw capture.")
    parser.add_argument("--output", type=Path, help="Write raw capture JSON here.")
    parser.add_argument("--viewer-output", type=Path, help="Write viewer result records here.")
    parser.add_argument("--artifact-root", help="Repo-relative raw artifact path.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--inner", type=int, default=16)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--pto-commit", help="Commit label to store in the raw artifact.")
    parser.add_argument("--arch", default="compute_80", help="PTX architecture used by nvcc.")
    args = parser.parse_args()
    for key in ("rows", "cols", "inner", "tile_count", "warmup", "repeats"):
        value = getattr(args, key)
        if value <= 0:
            fail(f"--{key.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json) if args.input_json else run_capture(args)
    if args.output:
        write_json(args.output, payload)
    raw_artifact = args.artifact_root
    if raw_artifact is None:
        source = args.output or args.input_json
        raw_artifact = repo_relative(source.parent) + "/" if source else "tmp/"
    records = [viewer_record(payload, raw_artifact)]
    if args.viewer_output:
        write_json(args.viewer_output, records)
    elif not args.output:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
