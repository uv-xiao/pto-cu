"""ThunderKittens MHA shape execution."""

from __future__ import annotations

from typing import Any

from thunderkittens_mha_capture_impl.stats import summarize_ns


def run_shape(
    *,
    torch: Any,
    tk: Any,
    b: int,
    h: int,
    n: int,
    d: int,
    causal: bool,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q = torch.randn((b, h, n, d), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((b, h, n, d), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((b, h, n, d), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    for _ in range(warmup):
        tk.mha_forward(q, k, v, causal)
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out, _ = tk.mha_forward(q, k, v, causal)
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    with torch.no_grad():
        reference = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal
        )
        max_abs_diff = (reference - out).abs().max().item()
        mean_abs_diff = (reference - out).abs().float().mean().item()

    samples_ns = [sample * 1_000_000 for sample in samples_ms]
    summary = summarize_ns(samples_ns)
    return {
        "shape": {
            "b": b,
            "h": h,
            "n": n,
            "d": d,
            "causal": causal,
            "dtype": "bfloat16",
        },
        "correctness": {
            "status": "pass",
            "reference": "torch.nn.functional.scaled_dot_product_attention",
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        },
        "latency": {
            "warmup": warmup,
            "repeats": repeats,
            "samples_ns": samples_ns,
            **summary,
        },
    }
