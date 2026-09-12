"""Avvio desktop macOS per MatchScope.

Avvia Streamlit solo in locale e lo visualizza in una finestra nativa macOS.
Chiudendo la finestra viene terminato anche il server locale.
"""

from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import webview


if getattr(sys, "frozen", False):
    # Dentro MatchScope.app il codice compilato vive in Contents/MacOS;
    # il progetto locale è quattro livelli sopra, accanto a `dist`.
    ROOT = Path(sys.executable).resolve().parents[4]
    PYTHON = ROOT / "venv" / "bin" / "python"
else:
    ROOT = Path(__file__).resolve().parent
    PYTHON = Path(sys.executable)
SERVER: subprocess.Popen | None = None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_server() -> None:
    global SERVER
    if SERVER and SERVER.poll() is None:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=5)
        except subprocess.TimeoutExpired:
            SERVER.kill()
    SERVER = None


def wait_until_ready(url: str, timeout: int = 25) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("MatchScope non si è avviata. Controlla che le dipendenze siano installate.")


def main() -> None:
    global SERVER
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    SERVER = subprocess.Popen(
        [
            str(PYTHON), "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.headless", "true",
            "--server.address", "127.0.0.1",
            "--server.port", str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)
    wait_until_ready(url)
    window = webview.create_window("MatchScope", url, width=1440, height=920, min_size=(1100, 700))
    window.events.closed += lambda: stop_server()
    webview.start(gui="cocoa")


if __name__ == "__main__":
    main()
