"""不依赖板端硬件的运行时工具，便于在 CI 中单独测试。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def resolve_project_dir(script_file: str, env_name: str = "ELDER_AI_HOME") -> Path:
    """返回项目目录；允许部署时通过环境变量覆盖。"""
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    return Path(script_file).resolve().parent


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是 1/0、true/false、yes/no 或 on/off，当前值为 {raw!r}")


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值为 {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"环境变量 {name} 不能小于 {minimum}，当前值为 {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"环境变量 {name} 不能大于 {maximum}，当前值为 {value}")
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字，当前值为 {raw!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"环境变量 {name} 必须是有限数字，当前值为 {value}")
    if minimum is not None and value < minimum:
        raise ValueError(f"环境变量 {name} 不能小于 {minimum}，当前值为 {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"环境变量 {name} 不能大于 {maximum}，当前值为 {value}")
    return value


def scores_to_probabilities(values: Sequence[float]) -> list[float]:
    """把模型输出转换为概率，同时避免对已有概率再次做 softmax。"""
    scores = [float(value) for value in values]
    if not scores:
        raise ValueError("模型输出不能为空")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("模型输出包含 NaN 或无穷值")

    total = sum(scores)
    looks_like_probabilities = all(value >= 0.0 for value in scores) and abs(total - 1.0) <= 1e-3
    if looks_like_probabilities:
        return [value / total for value in scores]

    maximum = max(scores)
    exponents = [math.exp(value - maximum) for value in scores]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


@dataclass
class PredictionGate:
    """过滤低置信度和单帧抖动，并控制重复播报间隔。"""

    confidence_threshold: float = 0.4
    required_hits: int = 2
    repeat_seconds: float = 6.0
    candidate_label: str | None = None
    candidate_hits: int = 0
    announced_label: str | None = None
    announced_at: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 必须在 0 到 1 之间")
        if self.required_hits < 1:
            raise ValueError("required_hits 必须大于等于 1")
        if self.repeat_seconds < 0:
            raise ValueError("repeat_seconds 不能小于 0")

    def observe(self, label: str, confidence: float, now: float) -> bool:
        """本次预测足够稳定且应该播报时返回 True。"""
        if confidence < self.confidence_threshold:
            self.candidate_label = None
            self.candidate_hits = 0
            return False

        if label == self.candidate_label:
            self.candidate_hits += 1
        else:
            self.candidate_label = label
            self.candidate_hits = 1

        if self.candidate_hits < self.required_hits:
            return False

        label_changed = label != self.announced_label
        repeat_due = self.announced_at is None or now - self.announced_at >= self.repeat_seconds
        if not label_changed and not repeat_due:
            return False

        self.announced_label = label
        self.announced_at = now
        return True
