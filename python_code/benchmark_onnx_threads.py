#!/usr/bin/env python3
"""测试不同 ONNX Runtime 线程数下的模型延迟。"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from runtime_utils import resolve_project_dir


PROJECT_DIR = resolve_project_dir(__file__)


def concrete_shape(shape) -> list[int]:
    defaults = [1, 3, 224, 224]
    if len(shape) != 4:
        raise ValueError(f"仅支持四维输入，当前模型输入为 {shape}")
    return [value if isinstance(value, int) and value > 0 else defaults[index] for index, value in enumerate(shape)]


def benchmark(model_path: Path, threads: int, warmup: int, runs: int) -> tuple[float, float, float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    dtype = np.uint8 if "uint8" in input_info.type.lower() else np.float32
    shape = concrete_shape(input_info.shape)
    generator = np.random.default_rng(0)
    if dtype == np.uint8:
        sample = generator.integers(0, 256, size=shape, dtype=np.uint8)
    else:
        sample = generator.random(shape, dtype=np.float32)

    for _ in range(warmup):
        session.run(None, {input_info.name: sample})

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {input_info.name: sample})
        timings.append((time.perf_counter() - started) * 1000)

    return statistics.mean(timings), statistics.median(timings), min(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime CPU 线程基准测试")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("WECHAT_MODEL_PATH", PROJECT_DIR / "models/wechat_page.onnx")),
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    model_path = args.model.expanduser()
    if not model_path.is_file():
        parser.error(f"找不到模型: {model_path}")
    if args.warmup < 0 or args.runs < 1 or any(value < 1 for value in args.threads):
        parser.error("线程数和 --runs 必须大于 0，--warmup 不能小于 0")

    print(f"model: {model_path}")
    print(f"providers: {ort.get_available_providers()}")
    print("threads | mean(ms) | median(ms) | best(ms)")
    print("------- | -------- | ---------- | --------")
    for threads in args.threads:
        mean_ms, median_ms, best_ms = benchmark(
            model_path, threads, args.warmup, args.runs
        )
        print(f"{threads:7d} | {mean_ms:8.2f} | {median_ms:10.2f} | {best_ms:8.2f}")


if __name__ == "__main__":
    main()
