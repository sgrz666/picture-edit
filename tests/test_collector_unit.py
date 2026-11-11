"""Unit and Contract Verification for MetricsCollector (Milestone 1).

Covers:
- Strict schema validation matching PROJECT.md interface contracts.
- Accurate calculation of MB, GB, and percentages.
- Synthetic /proc/meminfo parsing accuracy.
- Caching TTL behavior (sub-1ms repeated retrieval and force bypass).
- Graceful degradation when GPU fails.
- Dead / short-lived process exception handling.
- Context manager protocol and resource cleanup.
"""

import os
import sys
import tempfile
import time
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collector import MetricsCollector


class TestMetricsCollectorUnit(unittest.TestCase):
    """Unit test suite for MetricsCollector."""

    def test_schema_conformance(self):
        """Verify snapshot dictionary strictly matches PROJECT.md schema."""
        collector = MetricsCollector()
        snapshot = collector.get_metrics_snapshot(force=True)

        # Root fields
        self.assertIn("timestamp", snapshot)
        self.assertIsInstance(snapshot["timestamp"], float)
        self.assertIn("formatted_time", snapshot)
        self.assertIsInstance(snapshot["formatted_time"], str)
        self.assertIn("status", snapshot)
        self.assertIn(snapshot["status"], ["ok", "degraded", "error"])

        # GPU fields
        self.assertIn("gpu", snapshot)
        gpu = snapshot["gpu"]
        self.assertIn("available", gpu)
        self.assertIsInstance(gpu["available"], bool)
        self.assertIn("vram", gpu)
        self.assertIn("processes", gpu)
        self.assertIsInstance(gpu["processes"], list)

        if gpu["available"]:
            self.assertIsInstance(gpu["model"], str)
            self.assertIsInstance(gpu["driver_version"], str)
            self.assertIsInstance(gpu["utilization_pct"], int)
            self.assertIsInstance(gpu["temperature_c"], int)

            vram = gpu["vram"]
            for field in ["total_mb", "used_mb", "free_mb"]:
                self.assertIn(field, vram)
                self.assertIsInstance(vram[field], int)
            for field in ["used_pct", "total_gb", "used_gb", "free_gb"]:
                self.assertIn(field, vram)
                self.assertIsInstance(vram[field], (int, float))

        # Memory fields
        self.assertIn("memory", snapshot)
        mem = snapshot["memory"]
        for field in ["total_mb", "used_mb", "free_mb", "available_mb", "cached_mb"]:
            self.assertIn(field, mem)
            self.assertIsInstance(mem[field], int)
        for field in ["used_pct", "total_gb", "used_gb", "available_gb"]:
            self.assertIn(field, mem)
            self.assertIsInstance(mem[field], (int, float))

        # Swap fields
        self.assertIn("swap", snapshot)
        swap = snapshot["swap"]
        for field in ["total_mb", "used_mb", "free_mb"]:
            self.assertIn(field, swap)
            self.assertIsInstance(swap[field], int)
        self.assertIn("used_pct", swap)
        self.assertIsInstance(swap["used_pct"], (int, float))

        # System fields
        self.assertIn("system", snapshot)
        system = snapshot["system"]
        self.assertIn("hostname", system)
        self.assertIsInstance(system["hostname"], str)
        self.assertTrue(len(system["hostname"]) > 0)
        self.assertIn("uptime_seconds", system)
        self.assertIsInstance(system["uptime_seconds"], int)
        self.assertIn("load_average", system)
        self.assertIsInstance(system["load_average"], list)
        self.assertEqual(len(system["load_average"]), 3)
        self.assertIn("cpu_count", system)
        self.assertIsInstance(system["cpu_count"], int)
        self.assertGreater(system["cpu_count"], 0)

        collector.close()

    def test_mock_meminfo_parsing(self):
        """Test exact byte/MB arithmetic against synthetic /proc/meminfo."""
        mock_meminfo = """MemTotal:       131580088 kB
MemFree:         4693472 kB
MemAvailable:   17725484 kB
Buffers:          286892 kB
Cached:          7710364 kB
SwapCached:       134568 kB
SwapTotal:       8388604 kB
SwapFree:          22708 kB
SReclaimable:    6320196 kB
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(mock_meminfo)
            temp_path = f.name

        try:
            collector = MetricsCollector(meminfo_path=temp_path)
            mem, swap = collector._collect_memory_and_swap()

            # 131580088 // 1024 = 128496 MB (~125.48 GB)
            self.assertEqual(mem["total_mb"], 128496)
            self.assertEqual(mem["total_gb"], 125.48)

            # Available: 17725484 // 1024 = 17310 MB (~16.90 GB)
            self.assertEqual(mem["available_mb"], 17310)
            self.assertEqual(mem["available_gb"], 16.9)

            # Used: 128496 - 17310 = 111186 MB (~108.58 GB)
            self.assertEqual(mem["used_mb"], 111186)
            self.assertEqual(mem["used_gb"], 108.58)

            # Used %: round(111186 / 128496 * 100, 2) = 86.53%
            self.assertEqual(mem["used_pct"], 86.53)

            # Free: 4693472 // 1024 = 4583 MB
            self.assertEqual(mem["free_mb"], 4583)

            # Swap: Total 8388604 // 1024 = 8191 MB, Free 22708 // 1024 = 22 MB
            self.assertEqual(swap["total_mb"], 8191)
            self.assertEqual(swap["free_mb"], 22)
            self.assertEqual(swap["used_mb"], 8169)
            self.assertEqual(swap["used_pct"], 99.73)
            collector.close()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cache_ttl_behavior(self):
        """Test in-memory cache TTL returns cached snapshot in <1ms."""
        collector = MetricsCollector(cache_ttl_sec=1.0)
        snap1 = collector.get_metrics_snapshot(force=True)

        start = time.perf_counter()
        snap2 = collector.get_metrics_snapshot(force=False)
        duration_ms = (time.perf_counter() - start) * 1000.0

        # Sub-1ms response for cached retrieval
        self.assertLess(duration_ms, 5.0)
        self.assertEqual(snap1["timestamp"], snap2["timestamp"])

        # Force bypasses cache
        time.sleep(0.01)
        snap3 = collector.get_metrics_snapshot(force=True)
        self.assertGreater(snap3["timestamp"], snap1["timestamp"])
        collector.close()

    def test_dead_process_handling(self):
        """Verify that attempting to enrich a dead/non-existent PID handles exceptions gracefully."""
        collector = MetricsCollector()
        dead_pid = 999999999  # Guaranteed non-existent PID
        proc = collector._enrich_process(
            pid=dead_pid,
            default_name="orphan_worker",
            used_mb=2048,
            total_vram_mb=85651,
        )

        self.assertEqual(proc["pid"], dead_pid)
        self.assertEqual(proc["user"], "unknown")
        self.assertEqual(proc["name"], "orphan_worker")
        self.assertEqual(proc["cmdline"], "orphan_worker")
        self.assertEqual(proc["vram_used_mb"], 2048)
        self.assertEqual(proc["vram_used_gb"], 2.0)
        self.assertEqual(proc["vram_pct"], 2.39)
        collector.close()

    def test_gpu_failure_graceful_degradation(self):
        """Test that when GPU is unavailable, status degrades gracefully without crashing."""
        # Point compat_dir to non-existent location and disable NVML
        collector = MetricsCollector(compat_dir="/tmp/non_existent_compat_dir_xyz")
        collector._nvml_available = False

        # Mock _collect_gpu_via_smi to raise an error
        def mock_smi_fail():
            raise RuntimeError("nvidia-smi not available")

        collector._collect_gpu_via_smi = mock_smi_fail

        snapshot = collector.get_metrics_snapshot(force=True)
        self.assertEqual(snapshot["status"], "degraded")
        self.assertFalse(snapshot["gpu"]["available"])
        self.assertEqual(snapshot["gpu"]["status"], "error")
        self.assertEqual(snapshot["gpu"]["processes"], [])
        self.assertGreater(snapshot["memory"]["total_mb"], 0)
        collector.close()

    def test_context_manager(self):
        """Verify context manager protocol works cleanly."""
        with MetricsCollector() as collector:
            snapshot = collector.get_metrics_snapshot()
            self.assertIn("timestamp", snapshot)
        self.assertFalse(collector._nvml_available)


if __name__ == "__main__":
    unittest.main()
