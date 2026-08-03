import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from runtime_utils import resolve_project_dir


PROJECT_DIR = resolve_project_dir(__file__)
SPACEMIT_NLP_PATH = Path(
    os.environ.get("SPACEMIT_NLP_PATH", "/home/hnu/spacemit-demo/examples/NLP")
).expanduser()
CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", PROJECT_DIR / "tts_cache")).expanduser()


TEXTS = {
    "startup": "微信辅助已启动。",
    "home": "微信首页。",
    "wechat_chat": "聊天页面。",
    "wechat_select": "功能菜单。",
    "wechat_videochat": "视频通话页面。",
    "ocr_start": "已进入拍照识别模式。",
    "ocr_processing": "已拍照，正在识别。",
    "ocr_fail": "拍照失败。",
    "ocr_no_text": "没有识别到文字。",
    "exit": "程序已退出。",
}


def main():
    parser = argparse.ArgumentParser(description="生成项目所需的固定语音缓存")
    parser.add_argument("--force", action="store_true", help="覆盖已有且有效的 wav 文件")
    args = parser.parse_args()

    if not SPACEMIT_NLP_PATH.exists():
        raise SystemExit(f"找不到 SpaceMIT NLP 目录: {SPACEMIT_NLP_PATH}")

    sys.path.insert(0, str(SPACEMIT_NLP_PATH))
    try:
        from spacemit_tts import TTSModel
    except ImportError as exc:
        raise SystemExit(f"无法导入 spacemit_tts，请检查 SPACEMIT_NLP_PATH: {exc}") from exc

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("正在加载官方 TTS 模型，请稍候...")
    tts_model = TTSModel()
    print("TTS 模型加载完成")

    for name, text in TEXTS.items():
        target_path = CACHE_DIR / f"{name}.wav"

        if not args.force and target_path.exists() and target_path.stat().st_size > 1000:
            print(f"已存在，跳过: {target_path}")
            continue

        print("=" * 60)
        print(f"正在生成: {name} -> {text}")

        tmp_path = None
        start = time.monotonic()

        try:
            tmp_path = tts_model.ort_predict(text)
            shutil.copyfile(tmp_path, str(target_path))

            print(f"生成完成: {target_path}")
            print(f"耗时: {time.monotonic() - start:.2f} 秒")

        finally:
            if tmp_path is not None and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

    print("=" * 60)
    print("全部语音缓存生成完成")
    print(f"缓存目录: {CACHE_DIR}")


if __name__ == "__main__":
    main()
