"""ComfyUI Server Process & Lifecycle Manager.

Manages starting, monitoring, and stopping a local or external ComfyUI server instance.
Provides robust health checks, process isolation, and port verification.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from backend.config import settings

logger = logging.getLogger(__name__)


class ComfyUIServerManager:
    """Controls the lifecycle of a real ComfyUI execution server."""

    def __init__(
        self,
        server_dir: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        is_external: bool = False,
    ):
        self.server_dir = server_dir or (settings.base_dir / "comfyui_server")
        self.host = host or settings.comfyui_host
        self.port = port or settings.comfyui_port
        self.is_external = is_external or (os.getenv("COMFYUI_EXTERNAL", "false").lower() == "true")
        self.process: Optional[subprocess.Popen] = None
        self.log_file_path = settings.data_dir / "comfyui_server.log"
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_port_in_use(self) -> bool:
        """Check if target host:port is already accepting TCP connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    async def check_health(self, timeout_sec: float = 2.5) -> Dict[str, Any]:
        """Verify if ComfyUI is actively responding on HTTP API."""
        url = f"{self.base_url}/system_stats"
        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    return {
                        "online": True,
                        "base_url": self.base_url,
                        "is_external": self.is_external,
                        "system_stats": data,
                        "process_running": self.process is not None and self.process.poll() is None,
                    }
        except Exception as exc:
            logger.debug("ComfyUI health check to %s failed: %s", url, exc)

        return {
            "online": False,
            "base_url": self.base_url,
            "is_external": self.is_external,
            "process_running": self.process is not None and self.process.poll() is None,
            "error": "ComfyUI server is unreachable",
        }

    async def start(self, wait_timeout_sec: int = 40) -> Dict[str, Any]:
        """Start the local ComfyUI server if not already running."""
        # 1. Check if already online
        health = await self.check_health()
        if health.get("online"):
            logger.info("ComfyUI server is already online and responding at %s", self.base_url)
            return {"status": "already_running", "base_url": self.base_url, "health": health}

        if self.is_external:
            raise RuntimeError(
                f"Configured for external ComfyUI server at {self.base_url}, but it is offline."
            )

        # 2. Verify server directory exists
        main_script = self.server_dir / "main.py"
        if not main_script.is_file():
            raise FileNotFoundError(
                f"ComfyUI main.py not found in {self.server_dir}. Cannot start local instance."
            )

        # 3. Check if port is occupied by an unresponsive process
        if self.is_port_in_use():
            logger.warning("Port %d is in use but not responding to /system_stats.", self.port)

        # 4. Prepare command line
        cmd = [
            sys.executable,
            str(main_script),
            "--listen",
            self.host,
            "--port",
            str(self.port),
            "--cpu",  # Strict CPU execution mode for systems without NVIDIA GPU
            "--disable-auto-launch",
            "--disable-cuda-malloc",
            "--offline",
            "--front-end-root",
            str(self.server_dir / "app"),
        ]

        logger.info("Starting ComfyUI process: %s", " ".join(cmd))
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_file_path, "a", buffering=1)

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.server_dir),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        # 5. Wait for server to become healthy
        start_time = time.time()
        while time.time() - start_time < wait_timeout_sec:
            if self.process.poll() is not None:
                exit_code = self.process.returncode
                log_snippet = self.get_recent_logs(lines=20)
                raise RuntimeError(
                    f"ComfyUI process terminated prematurely with code {exit_code}.\nRecent logs:\n{log_snippet}"
                )

            health = await self.check_health(timeout_sec=1.0)
            if health.get("online"):
                logger.info("ComfyUI server successfully initialized at %s (PID: %d)", self.base_url, self.process.pid)
                return {
                    "status": "started",
                    "base_url": self.base_url,
                    "pid": self.process.pid,
                    "health": health,
                }
            await asyncio.sleep(1.0)

        # Timeout reached
        self.stop()
        raise TimeoutError(
            f"ComfyUI did not respond within {wait_timeout_sec}s.\nLogs: {self.get_recent_logs(lines=25)}"
        )

    def stop(self) -> bool:
        """Gracefully terminate the local ComfyUI server process."""
        if not self.process:
            return True

        logger.info("Stopping ComfyUI server process (PID: %d)...", self.process.pid)
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()

            self.process.wait(timeout=5)
        except Exception as exc:
            logger.warning("Graceful stop failed: %s; forcing SIGKILL", exc)
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            except Exception:
                pass
        finally:
            if self._log_handle and not self._log_handle.closed:
                self._log_handle.close()
            self.process = None

        return True

    def get_recent_logs(self, lines: int = 30) -> str:
        """Retrieve the last N lines from the ComfyUI server log."""
        if not self.log_file_path.is_file():
            return "No log file found."
        try:
            with open(self.log_file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as exc:
            return f"Failed reading log: {exc}"


# Global server manager singleton
comfyui_server_manager = ComfyUIServerManager()
