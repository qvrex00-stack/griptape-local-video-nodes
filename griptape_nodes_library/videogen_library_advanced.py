import logging
import os
import subprocess
import sys
import threading
import time

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes")

SERVER_DIR = os.environ.get(
    "GRIPTAPE_WAN_SERVER_DIR",
    r"C:\Foundry\Griptape\video_gen_server"
)
SERVER_DIR = os.path.normpath(SERVER_DIR)
HEALTH_URL = "http://127.0.0.1:8090/health"
BAT_PATH = os.path.join(SERVER_DIR, "start_server.bat")


class VideoGenLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for Local GPU Video Generation Library."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        # Griptape 시작 시 video_gen_server의 CMD 프롬프트 창을 화면에 자동 실행
        threading.Thread(target=_ensure_server_window, daemon=True).start()


def _is_server_alive() -> bool:
    """서버가 HTTP 요청에 응답하는지 확인."""
    try:
        import urllib.request
        urllib.request.urlopen(HEALTH_URL, timeout=2)
        return True
    except Exception:
        return False


def _is_console_window_running() -> bool:
    """start_server.bat의 CMD 콘솔 창이 실제로 화면에 떠 있는지 확인."""
    if sys.platform != "win32":
        return False
    try:
        cmd = "Get-Process | Where-Object { $_.MainWindowTitle -like '*Video Generation Server*' } | Select-Object -ExpandProperty Id"
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=5
        )
        pids = [line.strip() for line in res.stdout.strip().splitlines() if line.strip().isdigit()]
        return len(pids) > 0
    except Exception:
        return False


def _ensure_server_window() -> None:
    """
    서버 콘솔 창이 열려있지 않거나 실행 중이 아닐 때 os.startfile()로 독립 콘솔 창 실행.
    """
    if _is_server_alive():
        if _is_console_window_running():
            logger.info("[VideoServer] Already running with console window at http://127.0.0.1:8090")
            return
        logger.info("[VideoServer] Port 8090 is responding but console window is missing. Restarting in visible console...")
        _stop_server()
        time.sleep(1)

    logger.info("[VideoServer] Server not running. Launching start_server.bat console window...")

    if sys.platform != "win32":
        logger.warning("[VideoServer] Auto-start only supported on Windows.")
        return

    if not os.path.exists(BAT_PATH):
        logger.warning(f"[VideoServer] start_server.bat not found: {BAT_PATH}")
        return

    try:
        os.startfile(BAT_PATH)
        logger.info("[VideoServer] Launched start_server.bat console window.")
    except Exception as e:
        logger.warning(f"[VideoServer] Failed to start server window via os.startfile: {e}")


def _ensure_server_running() -> None:
    """외부(노드)에서 호출 가능한 서버 시작 함수."""
    _ensure_server_window()


def _stop_server() -> None:
    """포트 8090 프로세스만 정밀 종료 (Griptape 본체 안전 보장)."""
    try:
        import requests
        requests.post("http://127.0.0.1:8090/server/stop", timeout=2)
        logger.info("[VideoServer] Stop signal sent via /server/stop")
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-NetTCPConnection -LocalPort 8090 -ErrorAction SilentlyContinue).OwningProcess"],
            capture_output=True, text=True, timeout=3
        )
        pids = [line.strip() for line in res.stdout.strip().splitlines() if line.strip().isdigit()]
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            logger.info(f"[VideoServer] Killed PID {pid} (port 8090)")
    except Exception as e:
        logger.warning(f"[VideoServer] Failed to stop port 8090 process: {e}")

