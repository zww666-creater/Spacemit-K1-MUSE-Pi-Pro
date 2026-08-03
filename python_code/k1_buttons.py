#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spacemit K1 MUSE Pi Pro 按键启动器

默认模式：
    只监听按键 3 / GPIO73，用来启动 assistant_with_voice.py

测试模式：
    python3 k1_buttons.py --test
    可以测试按键 1/2/3/4 是否都能读取到

最终设计：
    按键 1 -> GPIO71 -> 由 assistant_with_voice.py 负责 OCR
    按键 2 -> GPIO72 -> 由 assistant_with_voice.py 负责微信引导
    按键 3 -> GPIO73 -> 由本启动器负责启动主项目
    按键 4 -> GPIO74 -> 由 assistant_with_voice.py 负责结束当前功能
"""

import argparse
import os
import sys
import time
import socket
import subprocess
from pathlib import Path

from gpiozero import Button, Device
from gpiozero.pins.lgpio import LGPIOFactory
from runtime_utils import env_int, resolve_project_dir


PROJECT_DIR = resolve_project_dir(__file__)
ASSISTANT = Path(os.environ.get("ASSISTANT_PATH", PROJECT_DIR / "assistant_with_voice.py"))
VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"
LOG_FILE = PROJECT_DIR / "assistant_with_voice.log"
NETWORK_WAIT_SECONDS = env_int("NETWORK_WAIT_SECONDS", 0, minimum=0)

KEY_OCR_GPIO = env_int("KEY_OCR_GPIO", 71, minimum=0)
KEY_WECHAT_GPIO = env_int("KEY_WECHAT_GPIO", 72, minimum=0)
KEY_PROJECT_GPIO = env_int("KEY_PROJECT_GPIO", 73, minimum=0)
KEY_STOP_GPIO = env_int("KEY_STOP_GPIO", 74, minimum=0)


def wait_network(timeout_seconds=60):
    print("[LAUNCHER] waiting for network...")

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection(("223.5.5.5", 53), timeout=2)
            sock.close()
            print("[LAUNCHER] network check finished")
            return True
        except OSError:
            time.sleep(2)

    print("[LAUNCHER] network check timeout, continue anyway")
    return False


def assistant_is_running():
    try:
        result = subprocess.run(
            ["pgrep", "-f", str(ASSISTANT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        pids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)
        return len(pids) > 0
    except Exception as e:
        print(f"[LAUNCHER] check running failed: {e}")
        return False


def start_project():
    if assistant_is_running():
        print("[LAUNCHER] elder_ai project is already running")
        return

    if not ASSISTANT.exists():
        print(f"[LAUNCHER] assistant not found: {ASSISTANT}")
        return

    python_bin = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    print("[LAUNCHER] starting elder_ai project...")
    print(f"[LAUNCHER] python: {python_bin}")
    print(f"[LAUNCHER] script: {ASSISTANT}")
    print(f"[LAUNCHER] log: {LOG_FILE}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", buffering=1, encoding="utf-8") as log_f:
        subprocess.Popen(
            [python_bin, str(ASSISTANT)],
            cwd=str(PROJECT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(0.5)

    if assistant_is_running():
        print("[LAUNCHER] elder_ai project started")
    else:
        print("[LAUNCHER] start command issued, but process not detected yet; check log file")


def run_test_mode():
    print("[TEST] 按键测试模式")
    print("[TEST] 按 Ctrl+C 退出")
    print("[TEST] key1=GPIO71, key2=GPIO72, key3=GPIO73, key4=GPIO74")

    Device.pin_factory = LGPIOFactory(chip=0)

    buttons = []

    def make_button(name, gpio):
        btn = Button(gpio, pull_up=True, bounce_time=0.08)

        def on_pressed():
            print(f"[BUTTON] {name} pressed, GPIO{gpio}")

        btn.when_pressed = on_pressed
        buttons.append(btn)

    make_button("ocr/key1", KEY_OCR_GPIO)
    make_button("wechat/key2", KEY_WECHAT_GPIO)
    make_button("project/key3", KEY_PROJECT_GPIO)
    make_button("stop/key4", KEY_STOP_GPIO)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[TEST] exit")


def run_launcher_mode():
    if NETWORK_WAIT_SECONDS:
        wait_network(NETWORK_WAIT_SECONDS)

    print("[LAUNCHER] press key 3 to start project")
    print("[LAUNCHER] 注意：启动器默认只监听按键3，按键1/2/4由 assistant_with_voice.py 监听")

    Device.pin_factory = LGPIOFactory(chip=0)

    btn_project = Button(KEY_PROJECT_GPIO, pull_up=True, bounce_time=0.08)

    def on_project_pressed():
        print("[BUTTON] project pressed")
        start_project()

    btn_project.when_pressed = on_project_pressed

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[LAUNCHER] exit")


def main():
    parser = argparse.ArgumentParser(description="Spacemit K1 MUSE Pi Pro 项目按键启动器")
    parser.add_argument("--test", action="store_true", help="监听并打印全部四个按键")
    args = parser.parse_args()

    print(f"[LAUNCHER] project dir: {PROJECT_DIR}")
    if args.test:
        run_test_mode()
    else:
        run_launcher_mode()


if __name__ == "__main__":
    main()
