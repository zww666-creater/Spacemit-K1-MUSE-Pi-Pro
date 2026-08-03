#!/usr/bin/env python3
"""独立的摄像头 + ONNX 页面分类调试工具。"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from runtime_utils import resolve_project_dir, scores_to_probabilities


CLASSES = ["home", "wechat_chat", "wechat_select", "wechat_videochat"]
PROJECT_DIR = resolve_project_dir(__file__)


def parse_camera(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def create_session(model_path: Path, threads: int) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 ONNX 模型: {model_path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def preprocess(frame, input_shape, input_type):
    if len(input_shape) != 4:
        raise ValueError(f"仅支持四维图像输入，模型输入为 {input_shape}")

    nchw = input_shape[1] in (1, 3) or isinstance(input_shape[1], str)
    if nchw:
        height = input_shape[2] if isinstance(input_shape[2], int) else 224
        width = input_shape[3] if isinstance(input_shape[3], int) else 224
    else:
        height = input_shape[1] if isinstance(input_shape[1], int) else 224
        width = input_shape[2] if isinstance(input_shape[2], int) else 224

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if "uint8" in input_type.lower():
        tensor = image.astype(np.uint8)
    else:
        tensor = image.astype(np.float32) / 255.0
    if nchw:
        tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0)


def predict(session: ort.InferenceSession, frame) -> tuple[str, float]:
    input_info = session.get_inputs()[0]
    tensor = preprocess(frame, input_info.shape, input_info.type)
    scores = np.asarray(session.run(None, {input_info.name: tensor})[0]).reshape(-1)
    if scores.size < len(CLASSES):
        raise ValueError(f"模型只输出 {scores.size} 个值，但配置了 {len(CLASSES)} 个类别")
    probabilities = scores_to_probabilities(scores[: len(CLASSES)])
    index = int(np.argmax(probabilities))
    return CLASSES[index], float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="测试摄像头上的微信页面分类模型")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("WECHAT_MODEL_PATH", PROJECT_DIR / "models/wechat_page.onnx")),
        help="ONNX 模型路径",
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("CAMERA_DEVICE", "/dev/video20"),
        help="摄像头索引或设备路径",
    )
    parser.add_argument("--threads", type=int, default=int(os.environ.get("ORT_INTRA_OP_THREADS", "8")))
    parser.add_argument("--headless", action="store_true", help="不创建 OpenCV 窗口")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_DIR / "captures")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads 必须大于等于 1")

    session = create_session(args.model.expanduser(), args.threads)
    camera = parse_camera(args.camera)
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头: {camera}")

    print(f"模型: {args.model}")
    print(f"摄像头: {camera}")
    print("按 q/ESC 退出，按 s 保存当前原始画面")
    last_print = 0.0
    current_label = "unknown"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("读取画面失败")
                time.sleep(0.1)
                continue

            current_label, confidence = predict(session, frame)
            now = time.monotonic()
            if now - last_print >= 1.0:
                print(f"label={current_label}, confidence={confidence:.3f}")
                last_print = now

            if args.headless:
                continue

            cv2.putText(
                frame,
                f"Page: {current_label}  Conf: {confidence:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("WeChat Page Assistant", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                target = args.save_dir / f"{int(time.time())}_{current_label}.jpg"
                cv2.imwrite(str(target), frame)
                print(f"已保存: {target}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
