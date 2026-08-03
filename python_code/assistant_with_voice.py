#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spacemit K1 MUSE Pi Pro elder_ai 主程序

功能：
    按键 1 / GPIO71：进入 OCR 拍照识别模式
    按键 2 / GPIO72：进入微信页面识别引导模式
    按键 4 / GPIO74：结束 OCR 或微信引导，释放摄像头，返回待机

说明：
    按键 3 / GPIO73 不在本文件中监听，它由 k1_buttons.py 启动器负责。
"""

import os

from runtime_utils import (
    PredictionGate,
    env_bool,
    env_float,
    env_int,
    resolve_project_dir,
    scores_to_probabilities,
)

ORT_INTRA_OP_THREADS = env_int("ORT_INTRA_OP_THREADS", 8, minimum=1)
os.environ.setdefault("OMP_NUM_THREADS", str(ORT_INTRA_OP_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(ORT_INTRA_OP_THREADS))

import signal
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# ===== OpenCV 调试窗口：只在 DEBUG_SHOW_CAMERA=1 时启用 =====
# 不改原业务逻辑，只包装 cv2.VideoCapture.read()，把读到的画面显示出来。
if os.environ.get("DEBUG_SHOW_CAMERA", "0") == "1" and not getattr(cv2, "_ELDER_DEBUG_CAMERA_PATCHED", False):
    cv2._ELDER_ORIG_VIDEOCAPTURE = cv2.VideoCapture

    class _ElderDebugVideoCapture:
        def __init__(self, *args, **kwargs):
            self._cap = cv2._ELDER_ORIG_VIDEOCAPTURE(*args, **kwargs)
            self._window_name = os.environ.get("DEBUG_CAMERA_WINDOW", "Elder AI Camera Debug")
            self._show_failed = False

        def read(self):
            ret, frame = self._cap.read()

            if ret and frame is not None:
                try:
                    cv2.imshow(self._window_name, frame)
                    cv2.waitKey(1)
                except Exception as e:
                    if not self._show_failed:
                        print(f"[DEBUG_CAMERA] OpenCV窗口显示失败: {e}", flush=True)
                        self._show_failed = True

            return ret, frame

        def release(self):
            try:
                return self._cap.release()
            finally:
                try:
                    cv2.destroyWindow(self._window_name)
                except Exception:
                    pass

        def __getattr__(self, name):
            return getattr(self._cap, name)

    cv2.VideoCapture = _ElderDebugVideoCapture
    cv2._ELDER_DEBUG_CAMERA_PATCHED = True
    print("[DEBUG_CAMERA] OpenCV摄像头调试窗口已启用", flush=True)
# ===== OpenCV 调试窗口补丁结束 =====

import numpy as np
import onnxruntime as ort

from gpiozero import Button, Device
from gpiozero.pins.lgpio import LGPIOFactory


# =========================
# 基础路径和配置
# =========================

BASE_DIR = resolve_project_dir(__file__)
MODEL_PATH = Path(os.environ.get("WECHAT_MODEL_PATH", BASE_DIR / "models" / "wechat_page.onnx"))
TTS_CACHE = Path(os.environ.get("TTS_CACHE_DIR", BASE_DIR / "tts_cache"))
OCR_CAPTURE_DIR = Path(os.environ.get("OCR_CAPTURE_DIR", BASE_DIR / "ocr_captures"))

AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "plughw:0,0")

KEY_OCR_GPIO = env_int("KEY_OCR_GPIO", 71, minimum=0)
KEY_WECHAT_GPIO = env_int("KEY_WECHAT_GPIO", 72, minimum=0)
KEY_STOP_GPIO = env_int("KEY_STOP_GPIO", 74, minimum=0)

CLASSES = [
    "home",
    "wechat_chat",
    "wechat_select",
    "wechat_videochat",
]

CLASS_TO_WAV = {
    "home": "home.wav",
    "wechat_chat": "wechat_chat.wav",
    "wechat_select": "wechat_select.wav",
    "wechat_videochat": "wechat_videochat.wav",
}

CLASS_TO_TEXT = {
    "home": "微信首页。",
    "wechat_chat": "聊天页面。",
    "wechat_select": "功能菜单。",
    "wechat_videochat": "视频通话页面。",
}

# 默认不弹出 OpenCV 窗口，因为你现在主要靠实体按键控制。
# 如果你想显示摄像头窗口，可以这样运行：
# SHOW_WINDOW=1 python assistant_with_voice.py
SHOW_WINDOW = env_bool("SHOW_WINDOW", False)

# 微信识别间隔，单位秒。数值越小识别越频繁，CPU 压力越大。
WECHAT_INFER_INTERVAL = env_float("WECHAT_INFER_INTERVAL", 0.8, minimum=0.05)

# 置信度阈值，太低时不播报。
WECHAT_CONF_THRESHOLD = env_float("WECHAT_CONF_THRESHOLD", 0.40, minimum=0.0, maximum=1.0)

# 连续识别到相同页面若干次后再播报，减少镜头抖动造成的误提示。
WECHAT_STABLE_FRAMES = env_int("WECHAT_STABLE_FRAMES", 2, minimum=1)
WECHAT_REPEAT_SECONDS = env_float("WECHAT_REPEAT_SECONDS", 6.0, minimum=0.0)
CAMERA_READ_FAILURE_LIMIT = env_int("CAMERA_READ_FAILURE_LIMIT", 30, minimum=1)
MODE_STOP_TIMEOUT = env_float("MODE_STOP_TIMEOUT", 8.0, minimum=0.1)


# =========================
# 全局状态
# =========================

app_stop_event = threading.Event()

mode_lock = threading.Lock()
current_mode = "standby"
current_worker_thread = None
current_worker_stop_event = None

buttons = []

ort_session = None
ort_input_name = None
ort_input_shape = None
ort_input_type = None

audio_lock = threading.Lock()
current_audio_process = None

paddle_ocr_instance = None


# =========================
# 音频相关
# =========================

def stop_audio():
    global current_audio_process

    with audio_lock:
        proc = current_audio_process
        current_audio_process = None

    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def play_wav(filename, wait=False):
    """
    播放 tts_cache 里面的 wav。
    filename 可以传 'home.wav'，也可以传完整路径。
    """
    global current_audio_process

    wav_path = Path(filename)
    if not wav_path.is_absolute():
        wav_path = TTS_CACHE / filename

    if not wav_path.exists():
        print(f"[AUDIO] wav not found: {wav_path}")
        return False

    stop_audio()

    cmd = ["aplay", "-q", "-D", AUDIO_DEVICE, str(wav_path)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with audio_lock:
            current_audio_process = proc

        if wait:
            proc.wait()

        return True

    except Exception as e:
        print(f"[AUDIO] play failed: {e}")
        return False


def speak_text_dynamic(text, stop_event=None):
    """
    用系统命令朗读动态文字。
    优先使用 espeak-ng，其次 espeak。

    注意：
        你的固定微信页面播报仍然走 wav 缓存。
        OCR 识别出来的文字是动态内容，如果机器上没有动态 TTS，
        这里会只打印文字，不会真正朗读。
    """
    text = (text or "").strip()
    if not text:
        return

    print("[SPEAK]", text)

    tts_cmd = shutil.which("espeak-ng") or shutil.which("espeak")

    if not tts_cmd:
        print("[SPEAK] 当前系统没有找到 espeak-ng/espeak，OCR文字只打印不朗读。")
        print("[SPEAK] 如需安装可后续尝试：sudo apt install espeak-ng")
        return

    chunks = []
    max_len = 120
    buf = ""

    for ch in text:
        buf += ch
        if len(buf) >= max_len or ch in "。！？!?；;\n":
            chunks.append(buf.strip())
            buf = ""

    if buf.strip():
        chunks.append(buf.strip())

    for chunk in chunks:
        if stop_event is not None and stop_event.is_set():
            return

        try:
            proc = subprocess.Popen(
                [tts_cmd, "-v", "zh", "-s", "150", chunk],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    proc.terminate()
                    return
                time.sleep(0.1)

        except Exception as e:
            print(f"[SPEAK] dynamic tts failed: {e}")
            return


def speak_message(text, wav_name=None, stop_event=None):
    """
    优先播放指定 wav；如果 wav 不存在，则尝试动态 TTS。
    """
    if wav_name:
        wav_path = TTS_CACHE / wav_name
        if wav_path.exists():
            play_wav(wav_name, wait=True)
            return

    speak_text_dynamic(text, stop_event=stop_event)


# =========================
# 摄像头相关
# =========================

def open_camera():
    """
    打开摄像头，优先使用 /dev/video20。
    """
    candidates = []

    env_device = os.environ.get("CAMERA_DEVICE")
    if env_device:
        candidates.append(env_device)

    env_index = os.environ.get("CAMERA_INDEX")
    if env_index is not None:
        try:
            candidates.append(int(env_index))
            candidates.append(f"/dev/video{int(env_index)}")
        except ValueError:
            candidates.append(env_index)

    default_candidates = [
        "/dev/video20",
        20,
        "/dev/video1",
        "/dev/video2",
        "/dev/video3",
        1,
        2,
        3,
        "/dev/video0",
        0,
    ]

    for item in default_candidates:
        if item not in candidates:
            candidates.append(item)

    last_error = None

    for dev in candidates:
        print(f"[CAMERA] trying {dev}")

        try:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        except Exception as e:
            last_error = e
            print(f"[CAMERA] open exception on {dev}: {e}")
            continue

        if not cap.isOpened():
            cap.release()
            print(f"[CAMERA] not opened: {dev}")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ok = False
        frame = None

        for _ in range(15):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.05)

        if ok:
            print(f"[CAMERA] opened {dev}, frame_shape={frame.shape}")
            return cap, dev

        print(f"[CAMERA] opened but cannot read frame: {dev}")
        cap.release()

    if last_error:
        raise RuntimeError(f"所有摄像头设备都无法打开: {last_error}")

    raise RuntimeError("所有摄像头设备都无法打开")

def close_cv_windows():
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def handle_window_key(stop_event):
    """
    保留 q / ESC，但不依赖它。
    现在真正可靠的停止方式是按键 4。
    """
    if not SHOW_WINDOW:
        return

    try:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("[KEYBOARD] q/ESC pressed")
            stop_event.set()
    except Exception:
        pass


# =========================
# ONNX 微信页面识别
# =========================

def load_wechat_model():
    global ort_session, ort_input_name, ort_input_shape, ort_input_type

    if ort_session is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

    print(f"[ONNX] loading model: {MODEL_PATH}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    sess_options.inter_op_num_threads = 1

    ort_session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    providers = ort_session.get_providers()
    print(f"[ONNX] providers: {providers}")

    input_info = ort_session.get_inputs()[0]
    ort_input_name = input_info.name
    ort_input_shape = input_info.shape
    ort_input_type = input_info.type

    print(f"[ONNX] input name: {ort_input_name}")
    print(f"[ONNX] input shape: {ort_input_shape}")
    print(f"[ONNX] input type: {ort_input_type}")


def _shape_dim(value, default):
    if isinstance(value, int) and value > 0:
        return value
    return default


def preprocess_for_onnx(frame):
    """
    通用分类模型预处理：
        - 支持 NCHW: [1, 3, H, W]
        - 支持 NHWC: [1, H, W, 3]
        - 默认 resize 到 224x224
        - 默认 float32 / 255
    """
    load_wechat_model()

    shape = ort_input_shape

    if len(shape) != 4:
        raise RuntimeError(f"不支持的 ONNX 输入维度: {shape}")

    is_nchw = False
    is_nhwc = False

    if shape[1] == 3 or shape[1] == 1 or isinstance(shape[1], str):
        is_nchw = True
        h = _shape_dim(shape[2], 224)
        w = _shape_dim(shape[3], 224)
    elif shape[3] == 3 or shape[3] == 1 or isinstance(shape[3], str):
        is_nhwc = True
        h = _shape_dim(shape[1], 224)
        w = _shape_dim(shape[2], 224)
    else:
        is_nchw = True
        h = 224
        w = 224

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if "uint8" in str(ort_input_type).lower():
        data = img.astype(np.uint8)
    else:
        data = img.astype(np.float32) / 255.0

    if is_nchw:
        data = np.transpose(data, (2, 0, 1))

    data = np.expand_dims(data, axis=0)

    return data


def classify_wechat_frame(frame):
    load_wechat_model()

    tensor = preprocess_for_onnx(frame)
    outputs = ort_session.run(None, {ort_input_name: tensor})

    arr = np.asarray(outputs[0]).reshape(-1)

    if arr.size < len(CLASSES):
        raise RuntimeError(f"模型输出数量小于类别数量: output={arr.size}, classes={len(CLASSES)}")

    arr = arr[:len(CLASSES)]
    probs = np.asarray(scores_to_probabilities(arr), dtype=np.float32)

    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    conf = float(probs[idx])

    return label, conf


# =========================
# OCR 相关
# =========================

def ocr_with_paddle(image_path):
    global paddle_ocr_instance

    try:
        from paddleocr import PaddleOCR
    except Exception:
        return None

    try:
        if paddle_ocr_instance is None:
            print("[OCR] loading PaddleOCR...")
            paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang="ch")

        result = paddle_ocr_instance.ocr(str(image_path), cls=True)

        texts = []

        if result is None:
            return ""

        for page in result:
            if page is None:
                continue
            for line in page:
                try:
                    # PaddleOCR 常见格式:
                    # [box, [text, score]]
                    text = line[1][0]
                    if text:
                        texts.append(str(text))
                except Exception:
                    continue

        return "\n".join(texts).strip()

    except Exception as e:
        print(f"[OCR] PaddleOCR failed: {e}")
        return None


def ocr_with_pytesseract(image_path):
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return None

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text.strip()
    except Exception as e:
        print(f"[OCR] pytesseract failed: {e}")
        return None


def ocr_with_tesseract_cli(image_path):
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return None

    try:
        result = subprocess.run(
            [tesseract, str(image_path), "stdout", "-l", "chi_sim+eng"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            print(f"[OCR] tesseract cli failed: {result.stderr.strip()}")
            return None

        return result.stdout.strip()

    except Exception as e:
        print(f"[OCR] tesseract cli exception: {e}")
        return None


def run_ocr(image_path):
    """
    OCR 优先级：
        1. PaddleOCR
        2. pytesseract
        3. tesseract 命令行
    """
    print(f"[OCR] image: {image_path}")

    text = ocr_with_paddle(image_path)
    if text is not None:
        return text.strip()

    text = ocr_with_pytesseract(image_path)
    if text is not None:
        return text.strip()

    text = ocr_with_tesseract_cli(image_path)
    if text is not None:
        return text.strip()

    print("[OCR] 没有找到可用 OCR 引擎。")
    print("[OCR] 可后续安装 PaddleOCR 或 tesseract/pytesseract。")
    return ""



def preprocess_ocr_frame(frame):
    """
    OCR 图像预处理：
    灰度化 -> 放大 -> 对比度增强 -> 去噪 -> 自适应二值化
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=1.8,
        fy=1.8,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )

    return binary


def ocr_sharpness_score(frame):
    """
    用 Laplacian 方差计算图像清晰度。
    分数越高，图像越清楚。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def capture_best_ocr_image(cap, stop_event=None):
    """
    OCR 增强拍照流程：
    1. 设置高分辨率 1280x720
    2. 等待 2 秒，让曝光、白平衡、画面稳定
    3. 连续采集 25 帧
    4. 选择最清晰的一帧
    5. 同时保存原图和预处理图
    6. 默认返回预处理图给 OCR 识别
    """
    OCR_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    width = env_int("OCR_CAMERA_WIDTH", 1280, minimum=1)
    height = env_int("OCR_CAMERA_HEIGHT", 720, minimum=1)
    warmup_seconds = env_float("OCR_WARMUP_SECONDS", 2.0, minimum=0.0)
    sample_frames = env_int("OCR_SAMPLE_FRAMES", 25, minimum=1)

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception as e:
        print(f"[OCR] set FOURCC MJPG failed: {e}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0

    print(
        f"[OCR] camera request={width}x{height}, actual={actual_w}x{actual_h}, fps={actual_fps:.1f}"
    )

    print(f"[OCR] 请保持画面稳定，预热 {warmup_seconds:.1f} 秒...")
    warmup_count = 0
    deadline = time.monotonic() + warmup_seconds

    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return None

        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_count += 1

        time.sleep(0.03)

    print(f"[OCR] warmup frames={warmup_count}")
    print(f"[OCR] sampling {sample_frames} frames for best sharpness...")

    best_frame = None
    best_score = -1.0

    for i in range(sample_frames):
        if stop_event is not None and stop_event.is_set():
            return None

        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"[OCR] sample {i + 1}/{sample_frames}: read failed")
            time.sleep(0.05)
            continue

        score = ocr_sharpness_score(frame)

        if score > best_score:
            best_score = score
            best_frame = frame.copy()

        print(
            f"[OCR] sample {i + 1}/{sample_frames}: sharpness={score:.1f}, best={best_score:.1f}"
        )

        time.sleep(0.04)

    if best_frame is None:
        print("[OCR] 拍照失败：连续采集没有拿到有效画面")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OCR_CAPTURE_DIR / f"ocr_{timestamp}_raw.jpg"
    processed_path = OCR_CAPTURE_DIR / f"ocr_{timestamp}_processed.jpg"

    processed = preprocess_ocr_frame(best_frame)

    raw_ok = cv2.imwrite(str(raw_path), best_frame)
    processed_ok = cv2.imwrite(str(processed_path), processed)

    if not raw_ok:
        print(f"[OCR] 保存原图失败: {raw_path}")

    if not processed_ok:
        print(f"[OCR] 保存预处理图失败: {processed_path}")

    print(f"[OCR] best sharpness={best_score:.1f}")
    print(f"[OCR] captured raw: {raw_path}")
    print(f"[OCR] captured processed: {processed_path}")

    use_processed = env_bool("OCR_USE_PROCESSED", True)

    if use_processed and processed_ok:
        print(f"[OCR] captured: {processed_path}")
        return processed_path

    if raw_ok:
        print(f"[OCR] captured: {raw_path}")
        return raw_path

    return None


def run_ocr_mode(stop_event):
    """
    按键 1 进入：
        打开摄像头 -> 拍照 -> 保存图片 -> OCR -> 朗读/打印结果
    """
    OCR_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    cap = None

    try:
        if stop_event.is_set():
            return

        speak_message("已进入拍照识别模式。", wav_name="ocr_start.wav", stop_event=stop_event)

        cap, camera_idx = open_camera()

        if stop_event.is_set():
            return

        print("[OCR] 请将文字对准摄像头，并保持稳定 2 秒")
        image_path = capture_best_ocr_image(cap, stop_event=stop_event)

        if stop_event.is_set():
            return

        if image_path is None:
            print("[OCR] 拍照失败：没有获取到可识别画面")
            speak_message("拍照失败。", wav_name="ocr_fail.wav", stop_event=stop_event)
            return

        speak_message("已拍照，正在识别。", wav_name="ocr_processing.wav", stop_event=stop_event)

        if stop_event.is_set():
            return

    finally:
        if cap is not None:
            cap.release()
            print("[CAMERA] OCR camera released")
        close_cv_windows()

    if stop_event.is_set():
        return

    text = run_ocr(image_path)

    if stop_event.is_set():
        return

    if text.strip():
        print("[OCR] result:")
        print(text)
        speak_text_dynamic(text, stop_event=stop_event)
    else:
        print("[OCR] 没有识别到文字")
        speak_message("没有识别到文字。", wav_name="ocr_no_text.wav", stop_event=stop_event)


# =========================
# 微信引导模式
# =========================

def run_wechat_mode(stop_event):
    """
    按键 2 进入：
        打开摄像头 -> 循环识别微信页面 -> 页面变化时播放 wav
    """
    load_wechat_model()

    cap = None
    last_infer_time = 0.0
    consecutive_read_failures = 0
    prediction_gate = PredictionGate(
        confidence_threshold=WECHAT_CONF_THRESHOLD,
        required_hits=WECHAT_STABLE_FRAMES,
        repeat_seconds=WECHAT_REPEAT_SECONDS,
    )

    try:
        speak_message("已进入微信引导模式。", wav_name="startup.wav", stop_event=stop_event)

        cap, camera_idx = open_camera()

        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                print("[CAMERA] read failed")
                if consecutive_read_failures >= CAMERA_READ_FAILURE_LIMIT:
                    raise RuntimeError(
                        f"摄像头连续读取失败 {consecutive_read_failures} 次，请检查连接后重新进入模式"
                    )
                time.sleep(0.1)
                continue

            consecutive_read_failures = 0
            now = time.monotonic()

            if now - last_infer_time >= WECHAT_INFER_INTERVAL:
                last_infer_time = now

                try:
                    label, conf = classify_wechat_frame(frame)
                    text = CLASS_TO_TEXT.get(label, label)

                    print(f"[WECHAT] {label}, conf={conf:.3f}, text={text}")

                    if prediction_gate.observe(label, conf, now):
                        wav_name = CLASS_TO_WAV.get(label)
                        if wav_name:
                            play_wav(wav_name, wait=False)
                        else:
                            speak_text_dynamic(text, stop_event=stop_event)

                except Exception as e:
                    print(f"[WECHAT] classify failed: {e}")
                    time.sleep(0.2)

            if SHOW_WINDOW:
                try:
                    cv2.imshow("elder_ai_wechat", frame)
                except Exception as e:
                    print(f"[WINDOW] imshow failed: {e}")
                handle_window_key(stop_event)

        print("[WECHAT] stop event received")

    finally:
        if cap is not None:
            cap.release()
            print("[CAMERA] WeChat camera released")
        close_cv_windows()


# =========================
# 模式控制：按键 1/2/4 的核心
# =========================

def stop_current_function(reason=""):
    """
    按键 4 调用这个函数。
    停止当前 OCR 或微信引导，并返回待机。
    """
    global current_mode, current_worker_thread, current_worker_stop_event

    with mode_lock:
        mode = current_mode
        thread = current_worker_thread
        stop_event = current_worker_stop_event

        if mode == "standby":
            print("[MODE] 当前已经是待机状态")
            return True

        print(f"[MODE] stopping current mode: {mode}")
        if reason:
            print(f"[MODE] reason: {reason}")

        if stop_event is not None:
            stop_event.set()

    stop_audio()

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=MODE_STOP_TIMEOUT)

    with mode_lock:
        still_alive = thread is not None and thread.is_alive()

        if still_alive:
            print("[MODE] 当前功能还在结束中，请稍后再按其他功能键")
            current_mode = "stopping"
            return False

        current_mode = "standby"
        current_worker_thread = None
        current_worker_stop_event = None

    print("[MODE] 已返回待机状态，等待按键 1 或按键 2")
    return True


def start_function_mode(mode_name, target_func):
    """
    启动 OCR 或微信引导。
    如果当前已经在其他模式，会先停止旧模式。
    """
    global current_mode, current_worker_thread, current_worker_stop_event

    with mode_lock:
        if current_mode == mode_name:
            print(f"[MODE] 已经在 {mode_name} 模式")
            return

        if current_mode == "stopping":
            print("[MODE] 当前功能还在停止中，请稍后再试")
            return

    ok = stop_current_function(reason=f"switch to {mode_name}")
    if not ok:
        return

    stop_event = threading.Event()

    def worker():
        global current_mode, current_worker_thread, current_worker_stop_event

        print(f"[MODE] entered {mode_name}")

        try:
            target_func(stop_event)
        except Exception as e:
            print(f"[ERROR] {mode_name} mode crashed: {e}")
        finally:
            stop_audio()

            with mode_lock:
                if current_worker_stop_event is stop_event:
                    current_mode = "standby"
                    current_worker_thread = None
                    current_worker_stop_event = None

            print("[MODE] 功能结束，已返回待机状态")

    thread = threading.Thread(target=worker, daemon=True)

    with mode_lock:
        current_mode = mode_name
        current_worker_thread = thread
        current_worker_stop_event = stop_event

    thread.start()


def start_ocr_mode_by_button():
    print("[BUTTON] 1号键：进入 OCR 模式")
    start_function_mode("ocr", run_ocr_mode)


def start_wechat_mode_by_button():
    print("[BUTTON] 2号键：进入微信引导模式")
    start_function_mode("wechat", run_wechat_mode)


def stop_mode_by_button():
    print("[BUTTON] 4号键：结束当前功能，返回待机")
    stop_current_function(reason="key4 pressed")


# =========================
# 按键初始化
# =========================

def setup_buttons():
    global buttons

    Device.pin_factory = LGPIOFactory(chip=0)

    btn_ocr = Button(KEY_OCR_GPIO, pull_up=True, bounce_time=0.08)
    btn_wechat = Button(KEY_WECHAT_GPIO, pull_up=True, bounce_time=0.08)
    btn_stop = Button(KEY_STOP_GPIO, pull_up=True, bounce_time=0.08)

    btn_ocr.when_pressed = start_ocr_mode_by_button
    btn_wechat.when_pressed = start_wechat_mode_by_button
    btn_stop.when_pressed = stop_mode_by_button

    buttons = [btn_ocr, btn_wechat, btn_stop]

    print("[BUTTON] 1号键=OCR，2号键=微信引导，4号键=结束当前功能，按键监听已启动")
    print(f"[BUTTON] key1 GPIO{KEY_OCR_GPIO}, key2 GPIO{KEY_WECHAT_GPIO}, key4 GPIO{KEY_STOP_GPIO}")


# =========================
# 信号退出
# =========================

def install_signal_handlers():
    def handler(signum, frame):
        print(f"[SIGNAL] received {signum}, exiting...")
        app_stop_event.set()

        with mode_lock:
            stop_event = current_worker_stop_event
            if stop_event is not None:
                stop_event.set()

        stop_audio()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def cleanup():
    print("[CLEANUP] stopping current function...")
    stop_current_function(reason="program cleanup")
    stop_audio()
    close_cv_windows()
    print("[CLEANUP] done")


# =========================
# 主入口
# =========================

def main():
    print("======================================")
    print("[elder_ai] assistant_with_voice.py started")
    print(f"[elder_ai] project dir: {BASE_DIR}")
    print("[elder_ai] 按键1=OCR，按键2=微信引导，按键4=结束当前功能")
    print("[elder_ai] 当前启动后进入待机状态")
    print("======================================")

    install_signal_handlers()

    try:
        setup_buttons()
    except PermissionError:
        print("[GPIO] Permission denied")
        print("[GPIO] 请先执行：sudo chmod a+rw /dev/gpiochip0")
        raise
    except Exception as e:
        print(f"[GPIO] setup failed: {e}")
        raise

    speak_message("项目已启动。", wav_name="startup.wav")

    print("[MODE] standby：等待按键 1 或按键 2")

    try:
        while not app_stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt")
        app_stop_event.set()
    finally:
        cleanup()
        print("[elder_ai] exited")


if __name__ == "__main__":
    main()
