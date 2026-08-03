#!/usr/bin/env python3
"""部署前自检：报告模型、依赖、GPIO、摄像头和音频状态。"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
from pathlib import Path

from runtime_utils import resolve_project_dir


def mark(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 elder_ai 板端运行环境")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=resolve_project_dir(__file__),
        help="包含 assistant_with_voice.py、models/ 和 tts_cache/ 的目录",
    )
    args = parser.parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    model_path = Path(
        os.environ.get("WECHAT_MODEL_PATH", project_dir / "models/wechat_page.onnx")
    ).expanduser()

    print(f"project_dir: {project_dir}")
    print("\n[Python 必需依赖]")
    required_modules = ["cv2", "numpy", "onnxruntime", "gpiozero", "lgpio"]
    missing_required = []
    for module in required_modules:
        ok = module_exists(module)
        print(f"  [{mark(ok):7}] {module}")
        if not ok:
            missing_required.append(module)

    print("\n[模型与语音缓存]")
    model_ok = model_path.is_file() and model_path.stat().st_size > 0
    print(f"  [{mark(model_ok):7}] model: {model_path}")
    cache_dir = project_dir / "tts_cache"
    wav_files = sorted(cache_dir.glob("*.wav")) if cache_dir.is_dir() else []
    print(f"  [{'OK' if wav_files else 'OPTIONAL':7}] wav cache: {len(wav_files)} files")

    print("\n[系统设备与命令]")
    cameras = sorted(glob.glob("/dev/video*"))
    gpio_ok = Path("/dev/gpiochip0").exists()
    aplay = shutil.which("aplay")
    print(f"  [{'OK' if cameras else 'MISSING':7}] cameras: {', '.join(cameras) or 'none'}")
    print(f"  [{mark(gpio_ok):7}] /dev/gpiochip0")
    print(f"  [{mark(aplay is not None):7}] aplay: {aplay or 'not found'}")

    print("\n[OCR / 动态语音（至少选择一种 OCR）]")
    paddle = module_exists("paddleocr")
    pytesseract = module_exists("pytesseract")
    tesseract = shutil.which("tesseract")
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    print(f"  [{'OK' if paddle else 'OPTIONAL':7}] paddleocr")
    print(f"  [{'OK' if pytesseract else 'OPTIONAL':7}] pytesseract")
    print(f"  [{'OK' if tesseract else 'OPTIONAL':7}] tesseract: {tesseract or 'not found'}")
    print(f"  [{'OK' if espeak else 'OPTIONAL':7}] dynamic TTS: {espeak or 'not found'}")

    if missing_required or not model_ok:
        print("\n自检未通过：请先补齐必需依赖和 ONNX 模型。")
        return 1
    print("\n基础自检通过。请再用 camera_probe.py 和 k1_buttons.py --test 验证硬件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
