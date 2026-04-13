# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Microbenchmark for NVFP4 activation quantization kernels.

Compares:
1. vLLM per-tensor global scaling (`ops.scaled_fp4_quant(..., False)`)
2. FlashInfer per-tensor global scaling (`flashinfer.nvfp4_quantize`)
3. FlashInfer per-token global scaling (`flashinfer.nvfp4_quant_and_per_token_scale`)
4. FlashInfer per-token global scaling with 128x4 swizzled SF layout
5. FlashInfer per-token global scaling with 8x4 swizzled SF layout

The benchmark uses explicit CUDA graph capture and replay to avoid Python-side
kernel launch overhead during timing. Read bytes are counted from the BF16 input
matrix only. Write bytes are counted from the actual output tensors returned by
each kernel, including packed activations, block scales, and per-token scales.

Run with:
    buck2 run @mode/opt -c fbcode.enable_gpu_sections=true \
        -c fbcode.nvcc_arch=b200a \
        //vllm/fb/plugins/tests:benchmark_nvfp4_quantization
"""

import argparse
import gc
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from flashinfer import (
    nvfp4_quant_and_per_token_scale,
    nvfp4_quantize,
    SfLayout,
)


DEFAULT_M_VALUES = [1, 8, 16, 64, 128, 256, 1024, 8192, 32768]
DEFAULT_K_VALUES = [2304, 3072, 4096]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)


@dataclass(frozen=True)
class BenchmarkResult:
    kernel: str
    m: int
    k: int
    median_us: float | None
    read_bytes: int | None
    write_bytes: int | None
    total_bytes: int | None
    bandwidth_tbps: float | None
    notes: str = ""


KERNEL_ORDER = [
    "vllm_per_tensor",
    "flashinfer_per_tensor",
    "flashinfer_per_token",
    "flashinfer_per_token_128x4",
    "flashinfer_per_token_8x4",
]


def _compute_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.float().abs().nan_to_num().max()
    return FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _nbytes(value: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _iter_tensors(value))


def _benchmark_cudagraph(
    fn: Callable[[], Any],
    warmup_iters: int,
    replay_iters: int,
) -> tuple[float, Any]:
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            fn()
    stream.synchronize()

    captured_output = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        captured_output = fn()
    stream.synchronize()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(replay_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(times_ms)

    if hasattr(graph, "reset"):
        graph.reset()
    del graph
    torch.cuda.synchronize()

    return median_ms, captured_output


def _format_float(value: float | None, precision: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _format_mb(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / (1024 * 1024):.3f}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    numeric_cols = set(range(0, len(headers)))

    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(row: Sequence[str], is_header: bool) -> str:
        cells = []
        for idx, cell in enumerate(row):
            if not is_header and idx in numeric_cols:
                cells.append(cell.rjust(widths[idx]))
            else:
                cells.append(cell.ljust(widths[idx]))
        return " | ".join(cells)

    divider = "-+-".join("-" * width for width in widths)
    out = [_format_row(headers, True), divider]
    out.extend(_format_row(row, False) for row in rows)
    return "\n".join(out)


def _group_results(
    results: Sequence[BenchmarkResult],
) -> dict[tuple[int, int], dict[str, BenchmarkResult]]:
    grouped: dict[tuple[int, int], dict[str, BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.m, result.k), {})[result.kernel] = result
    return grouped


def _shared_total_bytes(result_by_kernel: dict[str, BenchmarkResult]) -> int | None:
    for kernel_name in KERNEL_ORDER:
        result = result_by_kernel.get(kernel_name)
        if result is not None and result.total_bytes is not None:
            return result.total_bytes
    return None


def _run_single_benchmark(
    kernel_name: str,
    fn: Callable[[], Any],
    input_tensor: torch.Tensor,
    warmup_iters: int,
    replay_iters: int,
) -> BenchmarkResult:
    read_bytes = input_tensor.numel() * input_tensor.element_size()

    try:
        median_ms, outputs = _benchmark_cudagraph(fn, warmup_iters, replay_iters)
        write_bytes = _nbytes(outputs)
        total_bytes = read_bytes + write_bytes
        bandwidth_tbps = total_bytes / (median_ms * 1e9)
        return BenchmarkResult(
            kernel=kernel_name,
            m=input_tensor.shape[0],
            k=input_tensor.shape[1],
            median_us=median_ms * 1000.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            total_bytes=total_bytes,
            bandwidth_tbps=bandwidth_tbps,
        )
    except Exception as error:
        return BenchmarkResult(
            kernel=kernel_name,
            m=input_tensor.shape[0],
            k=input_tensor.shape[1],
            median_us=None,
            read_bytes=read_bytes,
            write_bytes=None,
            total_bytes=None,
            bandwidth_tbps=None,
            notes=str(error).splitlines()[0],
        )
    finally:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def run_benchmarks(
    m_values: Sequence[int],
    k_values: Sequence[int],
    warmup_iters: int,
    replay_iters: int,
    seed: int,
) -> list[BenchmarkResult]:
    results = []

    for k in k_values:
        for m in m_values:
            torch.manual_seed(seed + m * 100_000 + k)
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            global_scale = _compute_global_scale(x)

            benchmark_fns = [
                (
                    "flashinfer_per_tensor",
                    lambda x=x, global_scale=global_scale: nvfp4_quantize(
                        x,
                        global_scale,
                    ),
                ),
                (
                    "flashinfer_per_token",
                    lambda x=x: nvfp4_quant_and_per_token_scale(
                        x,
                        NVFP4_PER_TOKEN_SCALE_INV,
                        sf_layout=SfLayout.layout_linear,
                    ),
                ),
                (
                    "flashinfer_per_token_128x4",
                    lambda x=x: nvfp4_quant_and_per_token_scale(
                        x,
                        NVFP4_PER_TOKEN_SCALE_INV,
                        sf_layout=SfLayout.layout_128x4,
                    ),
                ),
                (
                    "flashinfer_per_token_8x4",
                    lambda x=x: nvfp4_quant_and_per_token_scale(
                        x,
                        NVFP4_PER_TOKEN_SCALE_INV,
                        sf_layout=SfLayout.layout_8x4,
                    ),
                ),
            ]

            for kernel_name, fn in benchmark_fns:
                results.append(
                    _run_single_benchmark(
                        kernel_name,
                        fn,
                        x,
                        warmup_iters=warmup_iters,
                        replay_iters=replay_iters,
                    )
                )

            del x
            gc.collect()
            torch.cuda.empty_cache()

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 quantization kernels with CUDA graphs."
    )
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=DEFAULT_M_VALUES,
        help="Batch sizes M to benchmark.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="Hidden sizes K to benchmark.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations before graph capture.",
    )
    parser.add_argument(
        "--replay-iters",
        type=int,
        default=100,
        help="Timed CUDA graph replays per benchmark case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for input generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"DType: {torch.bfloat16}")
    print(
        "Kernels: vllm_per_tensor, flashinfer_per_tensor, "
        "flashinfer_per_token, flashinfer_per_token_128x4, "
        "flashinfer_per_token_8x4"
    )
    print(f"Warmup iters: {args.warmup_iters}, timed replays: {args.replay_iters}")
    print()

    results = run_benchmarks(
        m_values=args.m_values,
        k_values=args.k_values,
        warmup_iters=args.warmup_iters,
        replay_iters=args.replay_iters,
        seed=args.seed,
    )
    grouped_results = _group_results(results)

    rows = []
    for k in args.k_values:
        for m in args.m_values:
            result_by_kernel = grouped_results.get((m, k), {})
            row = [str(m), str(k), _format_mb(_shared_total_bytes(result_by_kernel))]
            for kernel_name in KERNEL_ORDER:
                result = result_by_kernel.get(kernel_name)
                row.append(_format_float(result.bandwidth_tbps, 4) if result else "-")
            rows.append(row)

    headers = [
        "M",
        "K",
        "Total MB",
        "vllm HBM (TB/s)",
        "fi_tensor HBM (TB/s)",
        "fi_token HBM (TB/s)",
        "fi_token_128x4 HBM (TB/s)",
        "fi_token_8x4 HBM (TB/s)",
    ]
    print(_format_table(headers, rows))


if __name__ == "__main__":
    main()
