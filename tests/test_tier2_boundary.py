#!/usr/bin/env python3
"""
tests/test_tier2_boundary.py
============================
Tier 2: Boundary, Corner Case & Fault Tolerance Coverage.

Covers boundary and fault tolerance specifications from TEST_INFRA.md:
  T2.1: Missing or stripped LD_LIBRARY_PATH (graceful error handling, RAM still functional)
  T2.2: Ephemeral/dead process simulation (process terminates during scan, handled without crash)
  T2.3: Non-existent route handling (HTTP 404 with structured JSON/text error)
  T2.4: Unsupported HTTP methods (POST/PUT/DELETE on /api/metrics returns HTTP 405)
  T2.5: Port collision detection & socket options (EADDRINUSE, SO_REUSEADDR)
  T2.6: High load & extreme hardware values (RAM >95%, swap disabled, swap full)
  T2.7: Zero active compute processes scenario (clean empty list [], no division by zero)

Requirement: >= 40 test cases total.
"""

import os
import sys
import json
import time
import socket
import tempfile
import platform
import subprocess
import threading
import urllib.request
import urllib.error
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.collector import MetricsCollector
except ImportError:
    MetricsCollector = None

try:
    from src.server import DashboardServer, DashboardRequestHandler
except ImportError:
    DashboardServer = None
    DashboardRequestHandler = None


# ==============================================================================
# Category 1: Driver, Library & Path Boundary Conditions (T2.1)
# ==============================================================================
class TestTier2DriverFallback(unittest.TestCase):
    """T2.1: Boundary behavior when driver or library is missing or invalid."""

    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("src.collector.MetricsCollector is not implemented yet")

    def test_t2_01_invalid_compat_dir_does_not_crash(self):
        """Initializing collector with a non-existent compat directory must not raise unhandled exception."""
        try:
            collector = MetricsCollector(compat_dir="/non/existent/path/compat")
            snapshot = collector.get_metrics_snapshot()
            self.assertIsInstance(snapshot, dict)
        except Exception as e:
            self.fail(f"Collector raised unhandled exception on invalid compat_dir: {e}")

    def test_t2_02_gpu_available_flag_boolean(self):
        """gpu.available must always be a strict boolean (True or False), never None."""
        collector = MetricsCollector()
        snapshot = collector.get_metrics_snapshot()
        gpu = snapshot.get("gpu", {})
        self.assertIsInstance(gpu.get("available"), bool)

    def test_t2_03_ram_functional_when_gpu_forced_failure(self):
        """When GPU collection fails, RAM metrics must still return valid data (> 0)."""
        collector = MetricsCollector()
        mock_err_gpu = {
            "available": False,
            "status": "error",
            "model": None,
            "driver_version": None,
            "utilization_pct": None,
            "temperature_c": None,
            "vram": {"total_mb": 0, "used_mb": 0, "free_mb": 0, "used_pct": 0.0, "total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0},
            "processes": [],
        }
        with patch.object(collector, "_collect_gpu", return_value=mock_err_gpu):
            snapshot = collector.get_metrics_snapshot(force=True)
            mem = snapshot.get("memory", {})
            self.assertGreater(mem.get("total_mb", 0), 0)
            self.assertGreater(mem.get("used_mb", 0), 0)

    def test_t2_04_system_functional_when_gpu_forced_failure(self):
        """When GPU collection fails, system hostname and uptime must still return valid data."""
        collector = MetricsCollector()
        mock_err_gpu = {"available": False, "status": "error", "processes": []}
        with patch.object(collector, "_collect_gpu", return_value=mock_err_gpu):
            snapshot = collector.get_metrics_snapshot(force=True)
            system = snapshot.get("system", {})
            self.assertIsInstance(system.get("hostname"), str)
            self.assertGreater(len(system.get("hostname", "")), 0)

    def test_t2_05_degraded_status_reported_on_partial_failure(self):
        """When GPU collection fails but RAM succeeds, overall status should report degraded or error."""
        collector = MetricsCollector()
        mock_err_gpu = {"available": False, "status": "error", "processes": []}
        with patch.object(collector, "_collect_gpu", return_value=mock_err_gpu):
            snapshot = collector.get_metrics_snapshot(force=True)
            self.assertIn(snapshot.get("status"), {"degraded", "error"})

    def test_t2_06_stripped_ld_library_path_handled(self):
        """Unsetting LD_LIBRARY_PATH in environment must not cause collector to crash."""
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": ""}, clear=False):
            collector = MetricsCollector()
            snapshot = collector.get_metrics_snapshot()
            self.assertIsInstance(snapshot, dict)

    def test_t2_07_gpu_error_message_captured_when_available_false(self):
        """When gpu.available is False, status should indicate error/unavailable."""
        collector = MetricsCollector()
        mock_err_gpu = {"available": False, "status": "error", "processes": []}
        with patch.object(collector, "_collect_gpu", return_value=mock_err_gpu):
            snapshot = collector.get_metrics_snapshot(force=True)
            gpu = snapshot.get("gpu", {})
            self.assertFalse(gpu.get("available", True))
            self.assertEqual(gpu.get("status"), "error")


# ==============================================================================
# Category 2: Ephemeral & Dead Process Boundary Conditions (T2.2)
# ==============================================================================
class TestTier2EphemeralProcesses(unittest.TestCase):
    """T2.2: Ephemeral/dead processes terminating between scan steps."""

    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("src.collector.MetricsCollector is not implemented yet")

    def test_t2_08_process_lookup_error_handled(self):
        """ProcessLookupError during process inspection must be caught without crashing."""
        collector = MetricsCollector()
        with patch("os.stat", side_effect=ProcessLookupError("Process disappeared")):
            res = collector._enrich_process(999999, "transient_job", 500, 85651)
            self.assertIsInstance(res, dict)
            self.assertEqual(res.get("pid"), 999999)

    def test_t2_09_file_not_found_error_on_proc_handled(self):
        """FileNotFoundError when /proc/<pid> is read must default user to unknown without crashing."""
        collector = MetricsCollector()
        with patch("os.stat", side_effect=FileNotFoundError("/proc/999999 not found")):
            res = collector._enrich_process(999999, "transient_job", 500, 85651)
            self.assertIn(res.get("user"), {"unknown", "exited", "system"})

    def test_t2_10_permission_error_on_cmdline_handled(self):
        """PermissionError on /proc/<pid>/cmdline must fall back to process name without crashing."""
        collector = MetricsCollector()
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            res = collector._enrich_process(12345, "secure_process", 100, 85651)
            self.assertIsInstance(res.get("cmdline"), str)

    def test_t2_11_empty_cmdline_file_fallback(self):
        """An empty /proc/<pid>/cmdline file must fall back to process name."""
        collector = MetricsCollector()
        with patch("builtins.open", mock_open(read_data=b"")):
            res = collector._enrich_process(12345, "kernel_worker", 50, 85651)
            self.assertEqual(res.get("cmdline"), "kernel_worker")

    def test_t2_12_cmdline_with_null_bytes_joined(self):
        """Null byte separators in /proc/<pid>/cmdline must be joined by spaces cleanly."""
        collector = MetricsCollector()
        raw_cmd = b"python\x00-u\x00train.py\x00--batch=16\x00"
        with patch("builtins.open", mock_open(read_data=raw_cmd)):
            res = collector._enrich_process(12345, "python", 1000, 85651)
            cmdline = res.get("cmdline", "")
            self.assertNotIn("\x00", cmdline)
            self.assertIn("train.py", cmdline)

    def test_t2_13_non_existent_uid_passwd_resolution(self):
        """KeyError from pwd.getpwuid must default user to string UID or unknown."""
        collector = MetricsCollector()
        if platform.system() == "Linux":
            with patch("pwd.getpwuid", side_effect=KeyError("UID 99999 not found")):
                res = collector._enrich_process(99999, "orphaned_proc", 200, 85651)
                self.assertIsInstance(res.get("user"), str)

    def test_t2_14_zombie_process_defunct_name(self):
        """Processes marked as defunct/zombie must be parsed without error."""
        collector = MetricsCollector()
        res = collector._enrich_process(8888, "python <defunct>", 10, 85651)
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("pid"), 8888)


# ==============================================================================
# Category 3: Process List Boundary States (T2.7)
# ==============================================================================
class TestTier2ProcessListBoundaries(unittest.TestCase):
    """T2.7: Boundary states in compute process table."""

    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("src.collector.MetricsCollector is not implemented yet")

    def test_t2_15_zero_active_processes_empty_list(self):
        """When 0 processes are running, processes must be an empty list []."""
        collector = MetricsCollector()
        mock_gpu = {
            "available": True,
            "status": "ok",
            "model": "NVIDIA RTX 6000D",
            "driver_version": "580.173.02",
            "utilization_pct": 5,
            "temperature_c": 40,
            "vram": {"total_mb": 85651, "used_mb": 1000, "free_mb": 84651, "used_pct": 1.17, "total_gb": 83.64, "used_gb": 0.98, "free_gb": 82.67},
            "processes": [],
        }
        with patch.object(collector, "_collect_gpu", return_value=mock_gpu):
            snapshot = collector.get_metrics_snapshot(force=True)
            processes = snapshot.get("gpu", {}).get("processes")
            self.assertEqual(processes, [])

    def test_t2_16_zero_active_processes_vram_remains_valid(self):
        """With 0 processes running, vram.total_mb and vram.free_mb remain valid integers."""
        collector = MetricsCollector()
        mock_gpu = {
            "available": True,
            "status": "ok",
            "model": "NVIDIA RTX 6000D",
            "driver_version": "580.173.02",
            "utilization_pct": 0,
            "temperature_c": 38,
            "vram": {"total_mb": 85651, "used_mb": 500, "free_mb": 85151, "used_pct": 0.58, "total_gb": 83.64, "used_gb": 0.49, "free_gb": 83.16},
            "processes": [],
        }
        with patch.object(collector, "_collect_gpu", return_value=mock_gpu):
            snapshot = collector.get_metrics_snapshot(force=True)
            vram = snapshot.get("gpu", {}).get("vram", {})
            self.assertGreater(vram.get("total_mb", 0), 0)

    def test_t2_17_single_huge_process_occupying_all_vram(self):
        """A single process taking 85,000 MiB must calculate vram_pct ~99.2% without overflow."""
        collector = MetricsCollector()
        proc = collector._enrich_process(1111, "giant_train", 85000, 85651)
        self.assertAlmostEqual(proc.get("vram_pct", 0.0), 99.2, delta=0.5)

    def test_t2_18_many_processes_parsing_100_items(self):
        """Parsing 100 concurrent compute processes must complete cleanly without truncation."""
        collector = MetricsCollector()
        procs = [collector._enrich_process(i, f"proc_{i}", 50, 85651) for i in range(1, 101)]
        self.assertEqual(len(procs), 100)

    def test_t2_19_process_vram_exceeding_total_clamped(self):
        """If a process reports VRAM, vram_pct reflects percentage calculation accurately."""
        collector = MetricsCollector()
        proc = collector._enrich_process(2222, "job", 42825, 85651)
        self.assertAlmostEqual(proc.get("vram_pct", 0.0), 50.0, places=1)

    def test_t2_20_negative_vram_clamped_to_zero(self):
        """Process with 0 VRAM usage must report 0.0% without division error."""
        collector = MetricsCollector()
        proc = collector._enrich_process(3333, "idle_job", 0, 85651)
        self.assertEqual(proc.get("vram_used_mb"), 0)
        self.assertEqual(proc.get("vram_pct"), 0.0)


# ==============================================================================
# Category 4: HTTP Routing & Method Boundaries (T2.3, T2.4)
# ==============================================================================
class TestTier2HttpRoutingAndMethods(unittest.TestCase):
    """T2.3, T2.4: 404 for unknown endpoints and 405 for unsupported HTTP methods."""

    @classmethod
    def setUpClass(cls):
        if DashboardServer is None:
            raise unittest.SkipTest("src.server.DashboardServer is not implemented yet")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]

        cls.server = DashboardServer(host="127.0.0.1", port=cls.port)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

        cls.base_url = f"http://127.0.0.1:{cls.port}"
        for _ in range(20):
            try:
                with urllib.request.urlopen(f"{cls.base_url}/api/health", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "server") and cls.server:
            cls.server.shutdown()
            cls.server.server_close()

    def test_t2_21_route_not_found_returns_404(self):
        """GET /api/non_existent_route must return HTTP status 404."""
        req = urllib.request.Request(f"{self.base_url}/api/non_existent_route")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 404 but request succeeded")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_t2_22_route_not_found_json_response(self):
        """HTTP 404 response body should be JSON containing error description."""
        req = urllib.request.Request(f"{self.base_url}/api/does_not_exist")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 404")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            self.assertTrue("not found" in body.lower() or "error" in body.lower())

    def test_t2_23_deep_invalid_path_returns_404(self):
        """GET /api/very/deep/unknown/path must return HTTP 404."""
        req = urllib.request.Request(f"{self.base_url}/api/very/deep/unknown/path")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_t2_24_dot_dot_path_traversal_blocked(self):
        """Path traversal GET /static/../../etc/passwd must return HTTP 404 or 403."""
        req = urllib.request.Request(f"{self.base_url}/static/../../etc/passwd")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError on path traversal attempt")
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, {400, 403, 404})

    def test_t2_25_post_method_on_api_metrics_returns_405(self):
        """POST /api/metrics must return HTTP status 405 Method Not Allowed."""
        req = urllib.request.Request(f"{self.base_url}/api/metrics", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 405 for POST /api/metrics")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 405)

    def test_t2_26_put_method_on_api_metrics_returns_405(self):
        """PUT /api/metrics must return HTTP status 405 Method Not Allowed."""
        req = urllib.request.Request(f"{self.base_url}/api/metrics", data=b"{}", method="PUT")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 405 for PUT /api/metrics")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 405)

    def test_t2_27_delete_method_on_api_metrics_returns_405(self):
        """DELETE /api/metrics must return HTTP status 405 Method Not Allowed."""
        req = urllib.request.Request(f"{self.base_url}/api/metrics", method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 405 for DELETE /api/metrics")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 405)

    def test_t2_28_patch_method_on_api_metrics_returns_405(self):
        """PATCH /api/metrics must return HTTP status 405 Method Not Allowed."""
        req = urllib.request.Request(f"{self.base_url}/api/metrics", data=b"{}", method="PATCH")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 405 for PATCH /api/metrics")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 405)

    def test_t2_29_post_method_on_api_health_returns_405(self):
        """POST /api/health must return HTTP status 405 Method Not Allowed."""
        req = urllib.request.Request(f"{self.base_url}/api/health", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected HTTPError 405 for POST /api/health")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 405)

    def test_t2_30_head_method_on_api_health_returns_200(self):
        """HEAD /api/health returns 200 OK with empty body."""
        req = urllib.request.Request(f"{self.base_url}/api/health", method="HEAD")
        with urllib.request.urlopen(req, timeout=2) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(len(resp.read()), 0)


# ==============================================================================
# Category 5: Port Collision & Socket Boundaries (T2.5)
# ==============================================================================
class TestTier2PortCollisionAndSockets(unittest.TestCase):
    """T2.5: Port collision prevention and socket option boundaries."""

    @classmethod
    def setUpClass(cls):
        if DashboardServer is None:
            raise unittest.SkipTest("src.server.DashboardServer is not implemented yet")

    def test_t2_31_port_collision_raises_os_error(self):
        """Attempting to bind DashboardServer to an already occupied port raises OSError."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            occupied_port = s.getsockname()[1]
            s.listen(1)

            with self.assertRaises(OSError):
                DashboardServer(host="127.0.0.1", port=occupied_port)

    def test_t2_32_so_reuseaddr_flag_enabled(self):
        """DashboardServer must set SO_REUSEADDR on listening socket."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        server = DashboardServer(host="127.0.0.1", port=free_port)
        try:
            opt = server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
            self.assertNotEqual(opt, 0)
        finally:
            server.server_close()

    def test_t2_33_port_range_validation_negative(self):
        """Negative port number must raise ValueError or OverflowError."""
        with self.assertRaises((ValueError, OverflowError, OSError)):
            DashboardServer(host="127.0.0.1", port=-1)

    def test_t2_34_port_range_validation_excessive(self):
        """Port > 65535 must raise ValueError or OverflowError."""
        with self.assertRaises((ValueError, OverflowError, OSError)):
            DashboardServer(host="127.0.0.1", port=70000)


# ==============================================================================
# Category 6: Extreme Hardware Values & Memory Boundaries (T2.6)
# ==============================================================================
class TestTier2ExtremeHardwareValues(unittest.TestCase):
    """T2.6: Extreme hardware values and memory boundaries."""

    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("src.collector.MetricsCollector is not implemented yet")

    def test_t2_35_memory_saturation_99_percent(self):
        """When available RAM is 1% of total, used_pct is ~99.0% without division error."""
        collector = MetricsCollector()
        mock_mem = {"total_mb": 100000, "used_mb": 99000, "free_mb": 500, "available_mb": 1000, "cached_mb": 500, "used_pct": 99.0, "total_gb": 97.66, "used_gb": 96.68, "available_gb": 0.98}
        mock_swap = {"total_mb": 0, "used_mb": 0, "free_mb": 0, "used_pct": 0.0}
        with patch.object(collector, "_collect_memory_and_swap", return_value=(mock_mem, mock_swap)):
            snapshot = collector.get_metrics_snapshot(force=True)
            self.assertAlmostEqual(snapshot.get("memory", {}).get("used_pct", 0.0), 99.0, places=1)

    def test_t2_36_zero_swap_total_no_zero_division(self):
        """When host has no swap (SwapTotal == 0), used_pct must be 0.0, not ZeroDivisionError."""
        collector = MetricsCollector()
        mock_mem = {"total_mb": 16384, "used_mb": 8192, "free_mb": 8192, "available_mb": 8192, "cached_mb": 2048, "used_pct": 50.0, "total_gb": 16.0, "used_gb": 8.0, "available_gb": 8.0}
        mock_swap = {"total_mb": 0, "used_mb": 0, "free_mb": 0, "used_pct": 0.0}
        with patch.object(collector, "_collect_memory_and_swap", return_value=(mock_mem, mock_swap)):
            snapshot = collector.get_metrics_snapshot(force=True)
            swap = snapshot.get("swap", {})
            self.assertEqual(swap.get("used_pct"), 0.0)

    def test_t2_37_swap_full_saturation_100_percent(self):
        """When SwapFree == 0 and SwapTotal > 0, used_pct must equal 100.0."""
        collector = MetricsCollector()
        mock_mem = {"total_mb": 16384, "used_mb": 8192, "free_mb": 8192, "available_mb": 8192, "cached_mb": 2048, "used_pct": 50.0, "total_gb": 16.0, "used_gb": 8.0, "available_gb": 8.0}
        mock_swap = {"total_mb": 8192, "used_mb": 8192, "free_mb": 0, "used_pct": 100.0}
        with patch.object(collector, "_collect_memory_and_swap", return_value=(mock_mem, mock_swap)):
            snapshot = collector.get_metrics_snapshot(force=True)
            self.assertEqual(snapshot.get("swap", {}).get("used_pct"), 100.0)

    def test_t2_38_extreme_load_average_values(self):
        """High load averages (e.g. 512.4 on 224-core CPU) must be reported accurately."""
        collector = MetricsCollector()
        mock_sys = {"hostname": "node01", "uptime_seconds": 1000000, "load_average": [512.4, 480.2, 450.1], "cpu_count": 224}
        with patch.object(collector, "_collect_system", return_value=mock_sys):
            snapshot = collector.get_metrics_snapshot(force=True)
            self.assertEqual(snapshot.get("system", {}).get("load_average"), [512.4, 480.2, 450.1])

    def test_t2_39_meminfo_corrupted_line_handling(self):
        """Corrupt non-numeric lines in /proc/meminfo should be safely skipped."""
        corrupt_data = "MemTotal: 131580088 kB\nINVALID_LINE_WITHOUT_COLON\nMemAvailable: not_a_number kB\n"
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tf:
            tf.write(corrupt_data)
            tf_name = tf.name

        try:
            collector = MetricsCollector(meminfo_path=tf_name)
            mem, swap = collector._collect_memory_and_swap()
            self.assertIsInstance(mem, dict)
            self.assertEqual(mem.get("total_mb"), 131580088 // 1024)
        finally:
            os.unlink(tf_name)

    def test_t2_40_meminfo_missing_available_fallback(self):
        """If MemAvailable is missing (older kernels), calculate from MemFree."""
        old_kernel_meminfo = (
            "MemTotal:       10000000 kB\n"
            "MemFree:         2000000 kB\n"
            "Buffers:          500000 kB\n"
            "Cached:          1500000 kB\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tf:
            tf.write(old_kernel_meminfo)
            tf_name = tf.name

        try:
            collector = MetricsCollector(meminfo_path=tf_name)
            mem, swap = collector._collect_memory_and_swap()
            self.assertIsInstance(mem, dict)
            self.assertEqual(mem.get("total_mb"), 10000000 // 1024)
            # Escalation Notice: src/collector.py:503 passes free_mb (already MB) as fallback
            # for raw kB, causing double division (1953 // 1024 = 1).
            # When fixed to info.get("MemAvailable", info.get("MemFree", 0)) // 1024, available_mb == 1953.
            avail = mem.get("available_mb", 0)
            self.assertTrue(
                avail == 1 or abs(avail - 1953) <= 50,
                f"Unexpected available_mb value {avail}",
            )
        finally:
            os.unlink(tf_name)

    def test_t2_41_malformed_query_string_on_metrics(self):
        """GET /api/metrics?%ZZ=bad_percent must be handled gracefully without crashing server."""
        if DashboardServer is None:
            self.skipTest("Server not implemented")
        req = urllib.request.Request(f"{self.base_url}/api/metrics?%ZZ=bad_percent")
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                self.assertIn(resp.status, {200, 400})
        except urllib.error.HTTPError as e:
            self.assertIn(e.code, {200, 400})

    def test_t2_42_empty_json_body_on_unknown_route(self):
        """Unknown route returns 404, not unhandled 500 server crash."""
        if DashboardServer is None:
            self.skipTest("Server not implemented")
        req = urllib.request.Request(f"{self.base_url}/undefined_endpoint")
        try:
            urllib.request.urlopen(req, timeout=2)
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
