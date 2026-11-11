#!/usr/bin/env python3
"""
tests/test_tier3_lifecycle.py
=============================
Tier 3: Process Lifecycle & Script Integration Coverage.

Covers lifecycle integration specifications from TEST_INFRA.md:
  T3.1: Server start -> PID created -> status reports RUNNING -> health probe passes
  T3.2: Server start idempotency (warning emitted, no duplicate daemon)
  T3.3: Server restart (old PID killed, new PID started, port rebound without EADDRINUSE)
  T3.4: Server stop (SIGTERM, wait, PID file deleted, process terminated)
  T3.5: Stop idempotency (calling stop.sh when stopped succeeds with diagnostic)
  T3.6: Log rotation check (log file > 10MB triggers rotation to .log.1 on start)
  T3.7: Port collision prevention via start.sh (exits non-zero with diagnostic)
  T3.8: Stale PID cleanup on start
  T3.9: status.sh exit codes (0 for running, 3 for stopped)
  T3.10: Custom port command-line argument propagation (--port)

Requirement: >= 10 test cases total.
"""

import os
import sys
import time
import json
import socket
import signal
import shutil
import tempfile
import platform
import subprocess
import urllib.request
import urllib.error
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_free_port() -> int:
    """Find an available unprivileged local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestTier3Lifecycle(unittest.TestCase):
    """Lifecycle integration testing for start/stop/restart/status scripts."""

    @classmethod
    def setUpClass(cls):
        # Locate lifecycle scripts directory
        cls.scripts_dir = REPO_ROOT / "scripts"
        cls.start_sh = cls.scripts_dir / "start.sh"
        cls.stop_sh = cls.scripts_dir / "stop.sh"
        cls.restart_sh = cls.scripts_dir / "restart.sh"
        cls.status_sh = cls.scripts_dir / "status.sh"

        # Check if scripts exist and environment is capable of executing bash
        if not cls.start_sh.exists() or platform.system() != "Linux":
            raise unittest.SkipTest("Lifecycle bash scripts require Linux environment and scripts/*.sh")

        cls.test_port = find_free_port()
        cls.pid_file = REPO_ROOT / "logs" / "dashboard.pid"
        cls.log_file = REPO_ROOT / "logs" / "dashboard.log"

        # Ensure any leftover daemon is cleaned up before testing
        if cls.pid_file.exists():
            try:
                pid = int(cls.pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
            except Exception:
                pass
            cls.pid_file.unlink(missing_ok=True)

    def setUp(self):
        # Ensure clean state before each test
        self.tearDown()

    def tearDown(self):
        # Safety cleanup: stop any running server on test port
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if is_port_in_use(self.test_port):
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            self.pid_file.unlink(missing_ok=True)

    def test_t3_01_start_script_creates_pid_and_passes_health(self):
        """T3.1: start.sh launches daemon, records PID, and /api/health responds 200."""
        cmd = ["bash", str(self.start_sh), "--port", str(self.test_port)]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0, f"start.sh failed: {res.stderr}\n{res.stdout}")

        # Assert PID file exists
        self.assertTrue(self.pid_file.exists(), "dashboard.pid was not created")
        pid = int(self.pid_file.read_text().strip())
        self.assertGreater(pid, 0)

        # Assert process is alive
        try:
            os.kill(pid, 0)
        except OSError:
            self.fail(f"PID {pid} is not a running process")

        # Assert health probe passes
        url = f"http://127.0.0.1:{self.test_port}/api/health"
        with urllib.request.urlopen(url, timeout=3) as resp:
            self.assertEqual(resp.status, 200)

    def test_t3_02_status_script_reports_running(self):
        """T3.1: status.sh outputs RUNNING and returns 0 when daemon is active."""
        # Start daemon first
        subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)

        res = subprocess.run(["bash", str(self.status_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5)
        self.assertEqual(res.returncode, 0)
        self.assertIn("RUNNING", res.stdout)

    def test_t3_03_start_idempotency_keeps_single_instance(self):
        """T3.2: Calling start.sh when daemon is already running exits 0 with ALREADY running warning."""
        subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)
        initial_pid = int(self.pid_file.read_text().strip())

        # Second start call
        res = subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ALREADY running", res.stdout)

        # PID should remain identical
        current_pid = int(self.pid_file.read_text().strip())
        self.assertEqual(initial_pid, current_pid)

    def test_t3_04_restart_terminates_old_and_updates_pid(self):
        """T3.3: restart.sh kills old PID, launches new PID, and rebounds port cleanly."""
        subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)
        old_pid = int(self.pid_file.read_text().strip())

        res = subprocess.run(["bash", str(self.restart_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)

        new_pid = int(self.pid_file.read_text().strip())
        self.assertNotEqual(old_pid, new_pid, "Restart did not generate a new PID")

        # Verify old PID is dead
        with self.assertRaises(OSError):
            os.kill(old_pid, 0)

        # Verify new PID is alive and port is responsive
        url = f"http://127.0.0.1:{self.test_port}/api/health"
        with urllib.request.urlopen(url, timeout=3) as resp:
            self.assertEqual(resp.status, 200)

    def test_t3_05_stop_script_terminates_and_deletes_pid_file(self):
        """T3.4: stop.sh sends SIGTERM, verifies termination, and removes PID file."""
        subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)
        pid = int(self.pid_file.read_text().strip())

        res = subprocess.run(["bash", str(self.stop_sh)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)
        self.assertFalse(self.pid_file.exists(), "dashboard.pid was not removed after stop.sh")

        # Verify process is dead
        with self.assertRaises(OSError):
            os.kill(pid, 0)

    def test_t3_06_stop_idempotency_succeeds(self):
        """T3.5: Calling stop.sh when server is already stopped returns exit code 0."""
        # Ensure stopped
        self.pid_file.unlink(missing_ok=True)
        res = subprocess.run(["bash", str(self.stop_sh)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5)
        self.assertEqual(res.returncode, 0)
        self.assertIn("not running", res.stdout)

    def test_t3_07_status_script_reports_stopped(self):
        """T3.4: status.sh returns code 3 and reports STOPPED when server is not running."""
        self.pid_file.unlink(missing_ok=True)
        res = subprocess.run(["bash", str(self.status_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5)
        self.assertEqual(res.returncode, 3)
        self.assertIn("STOPPED", res.stdout)

    def test_t3_08_log_rotation_on_start_when_exceeding_10mb(self):
        """T3.6: When log file > 10MB, start.sh rotates it to dashboard.log.1."""
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        rotated_log = logs_dir / "dashboard.log.1"
        rotated_log.unlink(missing_ok=True)

        # Create dummy log > 10MB (10.5 MB)
        target_size = int(10.5 * 1024 * 1024)
        with open(self.log_file, "wb") as f:
            f.seek(target_size - 1)
            f.write(b"\0")

        # Launch server
        subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)

        # Assert rotated log was created
        self.assertTrue(rotated_log.exists(), "dashboard.log.1 was not created upon 10MB log rotation")
        self.assertGreaterEqual(rotated_log.stat().st_size, 10 * 1024 * 1024)

        # Clean up
        rotated_log.unlink(missing_ok=True)

    def test_t3_09_port_collision_detected_by_start_script(self):
        """T3.7: start.sh aborts with exit code 1 when requested port is occupied by another process."""
        collision_port = find_free_port()
        # Bind a mock socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mock_sock:
            mock_sock.bind(("127.0.0.1", collision_port))
            mock_sock.listen(1)

            res = subprocess.run(
                ["bash", str(self.start_sh), "--port", str(collision_port)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("already in use", res.stdout.lower() + res.stderr.lower())

    def test_t3_10_stale_pid_cleanup_on_start(self):
        """T3.8: If dashboard.pid contains a dead PID, start.sh removes it and launches cleanly."""
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Write dead PID (e.g. 9999999)
        self.pid_file.write_text("9999999")

        res = subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0)
        new_pid = int(self.pid_file.read_text().strip())
        self.assertNotEqual(new_pid, 9999999)

    def test_t3_11_metrics_collection_post_restart(self):
        """T3.7: After restart, /api/metrics immediately serves fresh telemetry."""
        subprocess.run(["bash", str(self.start_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)
        subprocess.run(["bash", str(self.restart_sh), "--port", str(self.test_port)], cwd=str(REPO_ROOT), check=True)

        url = f"http://127.0.0.1:{self.test_port}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("gpu", data)
            self.assertIn("memory", data)

    def test_t3_12_stop_escalates_to_sigkill_if_unresponsive(self):
        """T3.4: stop.sh escalates to SIGKILL if process ignores SIGTERM."""
        # Launch a dummy unresponsive process that traps SIGTERM
        dummy_code = "import signal, time; signal.signal(signal.SIGTERM, lambda s,f: None); [time.sleep(1) for _ in range(60)]"
        proc = subprocess.Popen([sys.executable, "-c", dummy_code])
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(proc.pid))

        try:
            # Stop should escalate and terminate
            res = subprocess.run(["bash", str(self.stop_sh)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=25)
            self.assertEqual(res.returncode, 0)
            # Process must be dead
            time.sleep(0.5)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main(verbosity=2)
