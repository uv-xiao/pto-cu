"""GPU metadata helpers for ThunderKittens captures."""

from __future__ import annotations

import subprocess


def read_gpu_metadata() -> dict[str, str]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # pragma: no cover - depends on target machine.
        return {
            "gpu": "unknown",
            "driver": "unknown",
            "compute_target": "unknown",
            "nvidia_smi_error": str(exc),
        }
    first = output.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    gpu = parts[0] if parts else "unknown"
    driver = parts[1] if len(parts) > 1 else "unknown"
    compute_cap = parts[2].replace(".", "") if len(parts) > 2 else "unknown"
    compute_target = (
        f"compute_{compute_cap}" if compute_cap != "unknown" else "unknown"
    )
    return {"gpu": gpu, "driver": driver, "compute_target": compute_target}
