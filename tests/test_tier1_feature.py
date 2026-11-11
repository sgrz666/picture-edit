#!/usr/bin/env python3
"""
tests/test_tier1_feature.py
===========================
Tier 1: Feature & Hardware Metric Precision Coverage.

Covers all 8 features inventoried in PROJECT.md and TEST_INFRA.md:
  F1. GPU VRAM Core Metrics (total, used, free, %, conversions)
  F2. System RAM Core Metrics (total, used, free, available, %, conversions)
  F3. GPU Core Utilization % and Temperature ℃
  F4. Compute Process List (PID, name, VRAM used, VRAM %)
  F5. Process User Resolution and Commandline Inspection
  F6. Driver Compatibility & System Host Info (uptime, load average, swap)
  F7. Zero-Dependency HTTP Server & /api/health Endpoint
  F8. /api/metrics JSON Schema & Snapshot Completeness

Requirement: >= 5 tests per feature for all 8 features (>= 40 tests total).
"""

import os
import sys
import json
import time
import socket
import platform
import subprocess
import threading
import urllib.request
import urllib.error
import unittest
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional import of collector and server modules
try:
    from src.collector import MetricsCollector
except ImportError:
    MetricsCollector = None

try:
    from src.server import DashboardServer, DashboardRequestHandler
except ImportError:
    DashboardServer = None
    DashboardRequestHandler = None


def get_live_nvidia_smi_memory():
    """Helper to query native nvidia-smi if available on Linux."""
    compat_paths = [
        "/data/shangguanrz/仪表盘/compat",
        str(REPO_ROOT / "compat"),
    ]
    env = os.environ.copy()
    for cp in compat_paths:
        if os.path.isdir(cp):
            env["LD_LIBRARY_PATH"] = f"{cp}:{env.get('LD_LIBRARY_PATH', '')}"
            break
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [int(p.strip()) for p in res.stdout.strip().split(",")]
            if len(parts) == 3:
                return {"total": parts[0], "used": parts[1], "free": parts[2]}
    except Exception:
        pass
    return None


def get_live_proc_meminfo():
    """Helper to parse /proc/meminfo directly if on Linux."""
    if not os.path.exists("/proc/meminfo"):
        return None
    data = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    data[key.strip()] = int(val.strip().split()[0])
        return data
    except Exception:
        return None


class BaseCollectorTestCase(unittest.TestCase):
    """Base class for collector tests providing a valid snapshot."""

    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("src.collector.MetricsCollector is not yet implemented")
        cls.collector = MetricsCollector()
        cls.snapshot = cls.collector.get_metrics_snapshot()


# ==============================================================================
# Feature 1: GPU VRAM Core Metrics
# ==============================================================================
class TestFeature1GPUVRAM(BaseCollectorTestCase):
    """F1: GPU VRAM Total, Used, Free and Percentage accuracy."""

    def test_f1_01_vram_total_nominal_85651(self):
        """RTX 6000D nominal VRAM is 85651 MiB (~83.6 GiB)."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        vram = gpu.get("vram", {})
        self.assertEqual(vram.get("total_mb"), 85651)

    def test_f1_02_vram_used_positive_and_within_bounds(self):
        """Used VRAM must be > 0 and <= total VRAM."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        vram = gpu.get("vram", {})
        used = vram.get("used_mb", 0)
        total = vram.get("total_mb", 0)
        self.assertGreater(used, 0)
        self.assertLessEqual(used, total)

    def test_f1_03_vram_free_bounds(self):
        """Free VRAM must be >= 0 and <= total VRAM."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        vram = gpu.get("vram", {})
        free = vram.get("free_mb", 0)
        total = vram.get("total_mb", 0)
        self.assertGreaterEqual(free, 0)
        self.assertLessEqual(free, total)

    def test_f1_04_vram_conservation_equation(self):
        """Total VRAM must equal Used VRAM + Free VRAM within 10 MiB."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        vram = gpu.get("vram", {})
        total = vram.get("total_mb", 0)
        used = vram.get("used_mb", 0)
        free = vram.get("free_mb", 0)
        self.assertAlmostEqual(total, used + free, delta=10)

    def test_f1_05_vram_percentage_calculation(self):
        """used_pct must equal round((used_mb / total_mb) * 100, 1)."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        vram = gpu.get("vram", {})
        total = vram.get("total_mb", 1)
        used = vram.get("used_mb", 0)
        expected_pct = round((used / total) * 100, 1)
        self.assertAlmostEqual(vram.get("used_pct", 0.0), expected_pct, delta=0.15)

    def test_f1_06_vram_gb_conversions(self):
        """GB values must correctly convert MB / 1024."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        vram = gpu.get("vram", {})
        total_mb = vram.get("total_mb", 0)
        used_mb = vram.get("used_mb", 0)
        free_mb = vram.get("free_mb", 0)
        self.assertAlmostEqual(vram.get("total_gb", 0.0), round(total_mb / 1024, 2), places=2)
        self.assertAlmostEqual(vram.get("used_gb", 0.0), round(used_mb / 1024, 2), places=2)
        self.assertAlmostEqual(vram.get("free_gb", 0.0), round(free_mb / 1024, 2), places=2)

    def test_f1_07_vram_parity_with_native_nvidia_smi(self):
        """Collector VRAM usage must match native nvidia-smi within live fluctuation tolerance."""
        ref = get_live_nvidia_smi_memory()
        if ref is None:
            self.skipTest("Native nvidia-smi query not available in current environment")
        fresh = self.collector.get_metrics_snapshot(force=True)
        gpu = fresh.get("gpu", {})
        vram = gpu.get("vram", {})
        self.assertEqual(vram.get("total_mb"), ref["total"])
        self.assertLessEqual(abs(vram.get("used_mb", 0) - ref["used"]), 1000)


# ==============================================================================
# Feature 2: System RAM Core Metrics
# ==============================================================================
class TestFeature2SystemRAM(BaseCollectorTestCase):
    """F2: System RAM Total, Used, Free, Available and Percentage accuracy."""

    def test_f2_01_ram_total_from_meminfo(self):
        """RAM total must be greater than 1024 MB (node01 is 128 GB)."""
        mem = self.snapshot.get("memory", {})
        total_mb = mem.get("total_mb", 0)
        self.assertGreater(total_mb, 1024)
        if os.path.exists("/proc/meminfo"):
            self.assertGreater(total_mb, 64000)

    def test_f2_02_ram_available_and_used_relationship(self):
        """used_mb must be calculated as total_mb - available_mb."""
        mem = self.snapshot.get("memory", {})
        total = mem.get("total_mb", 0)
        avail = mem.get("available_mb", 0)
        used = mem.get("used_mb", 0)
        self.assertGreater(total, 0)
        self.assertGreater(avail, 0)
        self.assertEqual(used, total - avail)

    def test_f2_03_ram_percentage_accuracy(self):
        """RAM used_pct must equal round((used_mb / total_mb) * 100, 1)."""
        mem = self.snapshot.get("memory", {})
        total = mem.get("total_mb", 1)
        used = mem.get("used_mb", 0)
        expected_pct = round((used / total) * 100, 1)
        self.assertAlmostEqual(mem.get("used_pct", 0.0), expected_pct, delta=0.15)

    def test_f2_04_ram_gb_conversions(self):
        """RAM GB values must equal MB / 1024."""
        mem = self.snapshot.get("memory", {})
        total_mb = mem.get("total_mb", 0)
        used_mb = mem.get("used_mb", 0)
        avail_mb = mem.get("available_mb", 0)
        self.assertAlmostEqual(mem.get("total_gb", 0.0), round(total_mb / 1024, 2), places=2)
        self.assertAlmostEqual(mem.get("used_gb", 0.0), round(used_mb / 1024, 2), places=2)
        self.assertAlmostEqual(mem.get("available_gb", 0.0), round(avail_mb / 1024, 2), places=2)

    def test_f2_05_ram_cached_buffers_non_negative(self):
        """cached_mb and free_mb must be non-negative integers."""
        mem = self.snapshot.get("memory", {})
        self.assertGreaterEqual(mem.get("cached_mb", 0), 0)
        self.assertGreaterEqual(mem.get("free_mb", 0), 0)

    def test_f2_06_ram_parity_with_proc_meminfo(self):
        """Collector RAM metrics must match /proc/meminfo within 100 MB."""
        info = get_live_proc_meminfo()
        if info is None or "MemTotal" not in info:
            self.skipTest("/proc/meminfo not available in this environment")
        mem = self.snapshot.get("memory", {})
        ref_total = info["MemTotal"] // 1024
        ref_avail = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
        self.assertAlmostEqual(mem.get("total_mb", 0), ref_total, delta=50)
        self.assertAlmostEqual(mem.get("available_mb", 0), ref_avail, delta=100)


# ==============================================================================
# Feature 3: GPU Utilization & Temperature
# ==============================================================================
class TestFeature3GPUUtilizationAndTemp(BaseCollectorTestCase):
    """F3: GPU Core Utilization % and Temperature ℃."""

    def test_f3_01_gpu_utilization_range_0_to_100(self):
        """GPU utilization percentage must be an integer between 0 and 100."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        util = gpu.get("utilization_pct", -1)
        self.assertIsInstance(util, int)
        self.assertGreaterEqual(util, 0)
        self.assertLessEqual(util, 100)

    def test_f3_02_gpu_temperature_reasonable_range_20_to_95(self):
        """GPU core temperature must be within 20℃ to 95℃ under normal operation."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        temp = gpu.get("temperature_c", -1)
        self.assertIsInstance(temp, int)
        self.assertGreaterEqual(temp, 20)
        self.assertLessEqual(temp, 95)

    def test_f3_03_gpu_utilization_is_int(self):
        """utilization_pct must be int type."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        self.assertIsInstance(gpu.get("utilization_pct"), int)

    def test_f3_04_gpu_temperature_is_int(self):
        """temperature_c must be int type."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        self.assertIsInstance(gpu.get("temperature_c"), int)

    def test_f3_05_gpu_model_string_contains_rtx_6000d(self):
        """GPU model string should identify the RTX 6000D card."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        model = gpu.get("model", "")
        self.assertTrue("RTX 6000D" in model or "NVIDIA" in model)

    def test_f3_06_gpu_driver_version_string_valid(self):
        """Driver version should be non-empty string."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available on current host")
        driver = gpu.get("driver_version", "")
        self.assertTrue(len(driver) > 0)


# ==============================================================================
# Feature 4: Compute Process Enumeration
# ==============================================================================
class TestFeature4ComputeProcesses(BaseCollectorTestCase):
    """F4: Compute Process List (PID, process name, VRAM used, VRAM %)."""

    def test_f4_01_processes_is_list(self):
        """gpu.processes must always be a list."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", None)
        self.assertIsInstance(processes, list)

    def test_f4_02_process_pids_positive_integers(self):
        """Every running compute process must have integer PID > 0."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            self.assertIsInstance(proc.get("pid"), int)
            self.assertGreater(proc.get("pid"), 0)

    def test_f4_03_process_names_non_empty(self):
        """Every process must report a non-empty name."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            name = proc.get("name", "")
            self.assertIsInstance(name, str)
            self.assertGreater(len(name.strip()), 0)

    def test_f4_04_process_vram_used_mb_positive(self):
        """Every compute process must report positive VRAM usage in MB."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            used_mb = proc.get("vram_used_mb", 0)
            self.assertGreater(used_mb, 0)

    def test_f4_05_process_vram_percentage_range(self):
        """Each process vram_pct must be between 0.0 and 100.0."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            pct = proc.get("vram_pct", -1.0)
            self.assertGreaterEqual(pct, 0.0)
            self.assertLessEqual(pct, 100.0)

    def test_f4_06_process_vram_sum_not_exceeding_total(self):
        """Sum of all process VRAM usages must not exceed total VRAM."""
        gpu = self.snapshot.get("gpu", {})
        if not gpu.get("available", False):
            self.skipTest("GPU not available")
        total_mb = gpu.get("vram", {}).get("total_mb", 85651)
        processes = gpu.get("processes", [])
        total_proc_vram = sum(p.get("vram_used_mb", 0) for p in processes)
        self.assertLessEqual(total_proc_vram, total_mb)


# ==============================================================================
# Feature 5: Process User & Commandline
# ==============================================================================
class TestFeature5ProcessUserAndCmdline(BaseCollectorTestCase):
    """F5: Process User Resolution and Commandline Inspection."""

    def test_f5_01_process_user_is_non_empty_string(self):
        """Each process user must be a non-empty string."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            user = proc.get("user", "")
            self.assertIsInstance(user, str)
            self.assertGreater(len(user.strip()), 0)

    def test_f5_02_process_user_matches_proc_status_uid(self):
        """When on Linux, verify user matches pwd resolution of process UID."""
        if platform.system() != "Linux":
            self.skipTest("Linux specific test")
        import pwd
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            pid = proc.get("pid")
            proc_status = f"/proc/{pid}/status"
            if os.path.exists(proc_status):
                try:
                    with open(proc_status, "r") as f:
                        for line in f:
                            if line.startswith("Uid:"):
                                uid = int(line.split()[1])
                                expected_user = pwd.getpwuid(uid).pw_name
                                self.assertEqual(proc.get("user"), expected_user)
                                break
                except (PermissionError, FileNotFoundError, KeyError):
                    pass

    def test_f5_03_process_cmdline_is_string(self):
        """Each process cmdline must be a string."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            self.assertIsInstance(proc.get("cmdline"), str)

    def test_f5_04_process_cmdline_not_empty_for_active_jobs(self):
        """Active jobs should report command line or fall back to process name."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            cmd = proc.get("cmdline", "")
            name = proc.get("name", "")
            self.assertTrue(len(cmd) > 0 or len(name) > 0)

    def test_f5_05_process_vram_gb_precision(self):
        """vram_used_gb must equal round(vram_used_mb / 1024, 2)."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        for proc in processes:
            used_mb = proc.get("vram_used_mb", 0)
            used_gb = proc.get("vram_used_gb", 0.0)
            self.assertAlmostEqual(used_gb, round(used_mb / 1024, 2), places=2)

    def test_f5_06_process_schema_keys_complete(self):
        """Each process item must contain all contract keys."""
        gpu = self.snapshot.get("gpu", {})
        processes = gpu.get("processes", [])
        required_keys = {"pid", "user", "name", "cmdline", "vram_used_mb", "vram_used_gb", "vram_pct"}
        for proc in processes:
            self.assertTrue(required_keys.issubset(proc.keys()))


# ==============================================================================
# Feature 6: Driver Compatibility & Host Telemetry
# ==============================================================================
class TestFeature6DriverCompatAndSystemInfo(BaseCollectorTestCase):
    """F6: Driver Compatibility, Hostname, Uptime, Load Average, Swap."""

    def test_f6_01_compat_dir_parameter_accepted(self):
        """MetricsCollector must accept custom compat_dir without error."""
        custom_collector = MetricsCollector(compat_dir="/data/shangguanrz/仪表盘/compat")
        self.assertIsNotNone(custom_collector)

    def test_f6_02_system_hostname_non_empty(self):
        """system.hostname must be a non-empty string."""
        system = self.snapshot.get("system", {})
        hostname = system.get("hostname", "")
        self.assertIsInstance(hostname, str)
        self.assertGreater(len(hostname.strip()), 0)

    def test_f6_03_system_uptime_seconds_positive(self):
        """system.uptime_seconds must be a non-negative integer (positive on Linux)."""
        system = self.snapshot.get("system", {})
        uptime = system.get("uptime_seconds", -1)
        self.assertIsInstance(uptime, int)
        self.assertGreaterEqual(uptime, 0)
        if os.path.exists("/proc/uptime"):
            self.assertGreater(uptime, 0)

    def test_f6_04_system_load_average_three_floats(self):
        """system.load_average must contain 3 non-negative floats [1m, 5m, 15m]."""
        system = self.snapshot.get("system", {})
        load = system.get("load_average", [])
        self.assertIsInstance(load, list)
        self.assertEqual(len(load), 3)
        for val in load:
            self.assertGreaterEqual(val, 0.0)

    def test_f6_05_system_cpu_count_positive(self):
        """system.cpu_count must be > 0 (node01 has 224 cores)."""
        system = self.snapshot.get("system", {})
        cpu_count = system.get("cpu_count", 0)
        self.assertGreater(cpu_count, 0)

    def test_f6_06_swap_memory_metrics_structure(self):
        """swap dictionary must contain total_mb, used_mb, free_mb, used_pct."""
        swap = self.snapshot.get("swap", {})
        self.assertIn("total_mb", swap)
        self.assertIn("used_mb", swap)
        self.assertIn("free_mb", swap)
        self.assertIn("used_pct", swap)
        self.assertGreaterEqual(swap.get("total_mb", 0), 0)
        self.assertGreaterEqual(swap.get("used_pct", 0.0), 0.0)
        self.assertLessEqual(swap.get("used_pct", 0.0), 100.0)


# ==============================================================================
# Feature 7: Zero-Dependency HTTP Server & /api/health
# ==============================================================================
class TestFeature7ZeroDepHttpServerHealth(unittest.TestCase):
    """F7: Zero-Dependency HTTP Server & /api/health Endpoint."""

    @classmethod
    def setUpClass(cls):
        if DashboardServer is None:
            raise unittest.SkipTest("src.server.DashboardServer is not implemented yet")
        # Find a free local port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]

        cls.server = DashboardServer(host="127.0.0.1", port=cls.port)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

        # Wait for server to become responsive
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

    def test_f7_01_health_endpoint_http_200(self):
        """GET /api/health must return HTTP status 200 OK."""
        url = f"{self.base_url}/api/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            self.assertEqual(resp.status, 200)

    def test_f7_02_health_content_type_json(self):
        """GET /api/health Content-Type header must be application/json."""
        url = f"{self.base_url}/api/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("application/json", content_type)

    def test_f7_03_health_status_healthy(self):
        """GET /api/health response body must report status: healthy."""
        url = f"{self.base_url}/api/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "healthy")

    def test_f7_04_health_service_name(self):
        """GET /api/health must identify service as gpu-ram-dashboard."""
        url = f"{self.base_url}/api/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("service"), "gpu-ram-dashboard")

    def test_f7_05_health_version_string(self):
        """GET /api/health version must be 1.0.0."""
        url = f"{self.base_url}/api/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("version"), "1.0.0")

    def test_f7_06_health_timestamp_recent(self):
        """GET /api/health timestamp must be within 10s of current time."""
        url = f"{self.base_url}/api/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ts = data.get("timestamp", 0)
            self.assertAlmostEqual(ts, time.time(), delta=10)


# ==============================================================================
# Feature 8: REST API /api/metrics Endpoint
# ==============================================================================
class TestFeature8RestApiMetricsEndpoint(unittest.TestCase):
    """F8: /api/metrics JSON Schema and Snapshot Completeness."""

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

    def test_f8_01_metrics_endpoint_http_200(self):
        """GET /api/metrics must return HTTP status 200."""
        url = f"{self.base_url}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            self.assertEqual(resp.status, 200)

    def test_f8_02_metrics_content_type_json(self):
        """GET /api/metrics Content-Type must be application/json."""
        url = f"{self.base_url}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("application/json", content_type)

    def test_f8_03_metrics_cache_control_no_cache(self):
        """GET /api/metrics must have Cache-Control header preventing caching."""
        url = f"{self.base_url}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            cache_ctrl = resp.headers.get("Cache-Control", "")
            self.assertTrue("no-cache" in cache_ctrl or "no-store" in cache_ctrl)

    def test_f8_04_metrics_top_level_keys_complete(self):
        """GET /api/metrics must contain all required top-level contract keys."""
        url = f"{self.base_url}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            required = {"timestamp", "formatted_time", "status", "gpu", "memory", "swap", "system"}
            self.assertTrue(required.issubset(data.keys()))

    def test_f8_05_metrics_status_enum_valid(self):
        """status must be one of 'ok', 'degraded', 'error'."""
        url = f"{self.base_url}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn(data.get("status"), {"ok", "degraded", "error"})

    def test_f8_06_metrics_json_serializable(self):
        """Snapshot must cleanly serialize and deserialize as valid JSON."""
        url = f"{self.base_url}/api/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            re_encoded = json.dumps(data)
            self.assertIsInstance(re_encoded, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
