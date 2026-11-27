#!/usr/bin/env python3
"""
tests/test_memory_dashboard.py
==============================
Automated Verification Suite for Memory Dashboard v2.0 Features:
- Kernel /proc/meminfo deep breakdown (Anon, Cache, Slab, PageTables, CommitLimit)
- Kernel PSI (/proc/pressure/memory) parsing & fallback
- /proc/vmstat delta rate computation (Swap In/Out KB/s, Major Faults/s, OOM kills)
- /dev/shm tmpfs utilization check
- Per-process /proc/<pid> memory & swap parsing + per-user aggregation (by_user)
- HTTP /api/history ring-buffer endpoint & HTTP method boundaries
- Zero-CDN & DOM element completeness for Memory Deep View
"""

import json
import os
import platform
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collector import MetricsCollector
from src.server import DashboardServer


class TestMemoryCollectorDeep(unittest.TestCase):
    """Deep telemetry parsing tests for System Memory, Swap, PSI, and Per-Process stats."""

    def test_meminfo_deep_breakdown_and_commit(self):
        """Parse full kernel /proc/meminfo breakdown and CommitLimit ratio."""
        mock_meminfo = """MemTotal:       131580088 kB
MemFree:         58535236 kB
MemAvailable:    88572348 kB
Buffers:          4053148 kB
Cached:          20212528 kB
SwapCached:       5064484 kB
Active:          19910680 kB
Inactive:        36461884 kB
SwapTotal:      125829108 kB
SwapFree:        83573244 kB
Dirty:               6144 kB
Writeback:              0 kB
AnonPages:       30240596 kB
Shmem:             207164 kB
Slab:            11266556 kB
SReclaimable:     7490516 kB
SUnreclaim:       3776040 kB
KernelStack:       257936 kB
PageTables:        559596 kB
Mlocked:             2048 kB
AnonHugePages:    1128448 kB
CommitLimit:    191619152 kB
Committed_AS:   282957148 kB
"""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write(mock_meminfo)
            mem_path = tf.name

        try:
            collector = MetricsCollector(meminfo_path=mem_path, compat_dir="/nonexistent")
            mem, swap = collector._collect_memory_and_swap()

            bd = mem["breakdown"]
            self.assertEqual(bd["anon_mb"], 30240596 // 1024)
            self.assertEqual(bd["buffers_mb"], 4053148 // 1024)
            self.assertEqual(bd["shmem_mb"], 207164 // 1024)
            self.assertEqual(bd["slab_mb"], 11266556 // 1024)
            self.assertEqual(bd["sreclaimable_mb"], 7490516 // 1024)
            self.assertEqual(bd["sunreclaim_mb"], 3776040 // 1024)
            self.assertEqual(bd["page_tables_mb"], 559596 // 1024)
            self.assertEqual(bd["dirty_mb"], 6144 // 1024)
            self.assertEqual(bd["mlocked_mb"], 2)
            self.assertGreater(bd["reclaimable_cache_mb"], 0)
            self.assertEqual(bd["process_anon_mb"], bd["anon_mb"])

            commit = mem["commit"]
            self.assertEqual(commit["limit_mb"], 191619152 // 1024)
            self.assertEqual(commit["committed_mb"], 282957148 // 1024)
            self.assertGreater(commit["commit_pct"], 140.0)

            self.assertEqual(swap["cached_mb"], 5064484 // 1024)
            self.assertGreater(swap["total_gb"], 100.0)
            collector.close()
        finally:
            if os.path.exists(mem_path):
                os.unlink(mem_path)

    def test_psi_memory_pressure_parsing_and_fallback(self):
        """Parse Linux /proc/pressure/memory PSI values and handle missing file."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write("some avg10=1.25 avg60=0.80 avg300=0.45 total=123456\n")
            tf.write("full avg10=0.50 avg60=0.20 avg300=0.10 total=65432\n")
            psi_path = tf.name

        try:
            collector = MetricsCollector(pressure_path=psi_path, compat_dir="/nonexistent")
            psi = collector._collect_pressure()
            self.assertTrue(psi["available"])
            self.assertEqual(psi["some_avg10"], 1.25)
            self.assertEqual(psi["some_avg60"], 0.8)
            self.assertEqual(psi["full_avg10"], 0.5)
            self.assertEqual(psi["full_avg300"], 0.1)

            collector.pressure_path = "/nonexistent/pressure/memory"
            missing_psi = collector._collect_pressure()
            self.assertFalse(missing_psi["available"])
            self.assertEqual(missing_psi["some_avg10"], 0.0)
            collector.close()
        finally:
            if os.path.exists(psi_path):
                os.unlink(psi_path)

    def test_vmstat_swap_thrashing_rate_delta(self):
        """Verify swap I/O KB/s and oom_kill count across vmstat delta ticks."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write("pswpin 1000\npswpout 2000\npgmajfault 500\noom_kill 3\n")
            vm_path = tf.name

        try:
            collector = MetricsCollector(vmstat_path=vm_path, compat_dir="/nonexistent")
            t0 = 1000.0
            first = collector._collect_vmstat_rates(t0)
            self.assertEqual(first["oom_kills"], 3)
            self.assertEqual(first["swap_in_kbps"], 0.0)
            self.assertFalse(first["thrashing"])

            # Simulate 2 seconds elapsed with +1000 pages in and +2000 pages out (12MB / 2s = 6MB/s)
            with open(vm_path, "w", encoding="utf-8") as wf:
                wf.write("pswpin 2000\npswpout 4000\npgmajfault 550\noom_kill 4\n")

            second = collector._collect_vmstat_rates(t0 + 2.0)
            self.assertGreaterEqual(second["swap_in_kbps"], 2000.0)
            self.assertGreaterEqual(second["swap_out_kbps"], 4000.0)
            self.assertEqual(second["pgmajfault_per_sec"], 25.0)
            self.assertEqual(second["oom_kills"], 4)
            self.assertTrue(second["thrashing"])
            collector.close()
        finally:
            if os.path.exists(vm_path):
                os.unlink(vm_path)

    def test_proc_process_scan_and_user_aggregation(self):
        """Verify synthetic /proc directory scanning, filtering of 0MB kernel threads, and user sums."""
        with tempfile.TemporaryDirectory() as proc_dir:
            # Kernel thread (0 RSS, 0 Swap -> must be excluded)
            kdir = os.path.join(proc_dir, "2")
            os.makedirs(kdir)
            with open(os.path.join(kdir, "status"), "w", encoding="utf-8") as f:
                f.write("Name:\tkthreadd\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\nVmRSS:\t0 kB\nVmSwap:\t0 kB\nThreads:\t1\n")

            # Process 101: python training job
            p1 = os.path.join(proc_dir, "101")
            os.makedirs(p1)
            with open(os.path.join(p1, "status"), "w", encoding="utf-8") as f:
                f.write(
                    "Name:\tpython3\nState:\tR (running)\nUid:\t0\t0\t0\t0\n"
                    "VmSize:\t4194304 kB\nVmRSS:\t2097152 kB\nRssAnon:\t1572864 kB\n"
                    "RssShmem:\t10240 kB\nVmSwap:\t1048576 kB\nThreads:\t8\n"
                )
            with open(os.path.join(p1, "cmdline"), "wb") as f:
                f.write(b"python3\x00-u\x00train.py\x00")
            with open(os.path.join(p1, "oom_score"), "w", encoding="utf-8") as f:
                f.write("750\n")

            # Process 102: small worker without cmdline/oom_score
            p2 = os.path.join(proc_dir, "102")
            os.makedirs(p2)
            with open(os.path.join(p2, "status"), "w", encoding="utf-8") as f:
                f.write(
                    "Name:\tworker\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\n"
                    "VmSize:\t204800 kB\nVmRSS:\t102400 kB\nVmSwap:\t0 kB\nThreads:\t2\n"
                )

            collector = MetricsCollector(proc_root=proc_dir, compat_dir="/nonexistent")
            procs, by_user, summary = collector._collect_memory_processes(
                total_mem_mb=131072, total_swap_mb=65536, force=True
            )

            self.assertEqual(summary["active_count"], 2)
            self.assertEqual(len(procs), 2)
            # Sorted descending by total footprint (101 = 2048+1024 = 3072 MB)
            self.assertEqual(procs[0]["pid"], 101)
            self.assertEqual(procs[0]["rss_mb"], 2048.0)
            self.assertEqual(procs[0]["swap_mb"], 1024.0)
            self.assertEqual(procs[0]["total_mb"], 3072.0)
            self.assertEqual(procs[0]["oom_score"], 750)
            self.assertEqual(procs[0]["threads"], 8)
            self.assertIn("train.py", procs[0]["cmdline"])

            # Fallback when cmdline missing
            self.assertEqual(procs[1]["pid"], 102)
            self.assertEqual(procs[1]["cmdline"], "worker")
            self.assertEqual(procs[1]["oom_score"], 0)
            self.assertFalse(procs[0]["swap_is_pss"])

            # Aggregated by user
            self.assertEqual(len(by_user), 1)
            self.assertEqual(by_user[0]["process_count"], 2)
            self.assertAlmostEqual(by_user[0]["rss_mb"], 2148.0, places=1)
            self.assertAlmostEqual(by_user[0]["swap_mb"], 1024.0, places=1)
            collector.close()

    def test_smaps_rollup_deduplicates_forked_cow_swap(self):
        """Verify smaps_rollup SwapPss prevents forked workers from double-counting VmSwap."""
        with tempfile.TemporaryDirectory() as proc_dir:
            # Parent training process: VmRSS=2000MB, VmSwap=1000MB, SwapPss=1000MB
            parent = os.path.join(proc_dir, "201")
            os.makedirs(parent)
            with open(os.path.join(parent, "status"), "w", encoding="utf-8") as f:
                f.write(
                    "Name:\tpython\nState:\tR (running)\nUid:\t0\t0\t0\t0\n"
                    "VmSize:\t4194304 kB\nVmRSS:\t2048000 kB\nRssAnon:\t1536000 kB\n"
                    "RssShmem:\t0 kB\nVmSwap:\t1024000 kB\nThreads:\t4\n"
                )
            with open(os.path.join(parent, "smaps_rollup"), "w", encoding="utf-8") as f:
                f.write("Rss:\t2048000 kB\nPss:\t2040000 kB\nSwap:\t1024000 kB\nSwapPss:\t1024000 kB\n")

            # Two forked worker clones sharing parent VmSwap (VmSwap=1000MB each, SwapPss=20MB each)
            for pid in ("202", "203"):
                wdir = os.path.join(proc_dir, pid)
                os.makedirs(wdir)
                with open(os.path.join(wdir, "status"), "w", encoding="utf-8") as f:
                    f.write(
                        "Name:\tpython\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\n"
                        "VmSize:\t4194304 kB\nVmRSS:\t2048 kB\nRssAnon:\t1024 kB\n"
                        "RssShmem:\t0 kB\nVmSwap:\t1024000 kB\nThreads:\t1\n"
                    )
                with open(os.path.join(wdir, "smaps_rollup"), "w", encoding="utf-8") as f:
                    f.write("Rss:\t2048 kB\nPss:\t200 kB\nSwap:\t1024000 kB\nSwapPss:\t20480 kB\n")

            collector = MetricsCollector(proc_root=proc_dir, compat_dir="/nonexistent")
            procs, by_user, summary = collector._collect_memory_processes(
                total_mem_mb=131072, total_swap_mb=65536, force=True
            )

            self.assertEqual(len(procs), 3)
            self.assertEqual(procs[0]["pid"], 201)
            self.assertTrue(all(p["swap_is_pss"] for p in procs))
            # Workers report SwapPss=20MB while retaining raw vm_swap_mb=1000MB
            self.assertEqual(procs[1]["swap_mb"], 20.0)
            self.assertEqual(procs[1]["vm_swap_mb"], 1000.0)
            # User and summary totals must be ~1040MB (1000 + 20 + 20), not 3000MB
            self.assertAlmostEqual(by_user[0]["swap_mb"], 1040.0, places=1)
            self.assertAlmostEqual(by_user[0]["vm_swap_mb"], 3000.0, places=1)
            self.assertTrue(by_user[0]["swap_exact"])
            self.assertAlmostEqual(summary["total_swap_mb"], 1040.0, places=1)
            self.assertAlmostEqual(summary["total_vm_swap_mb"], 3000.0, places=1)
            self.assertEqual(summary["swap_exact_procs"], 3)
            collector.close()

    def test_shm_collection_bounds(self):
        """Verify /dev/shm telemetry reports sane capacity on Linux and clean fallback."""
        collector = MetricsCollector(compat_dir="/nonexistent")
        shm = collector._collect_shm()
        if platform.system() == "Linux" and os.path.isdir("/dev/shm"):
            self.assertTrue(shm["available"])
            self.assertGreater(shm["total_mb"], 0)
            self.assertGreaterEqual(shm["used_pct"], 0.0)
            self.assertLessEqual(shm["used_pct"], 100.0)

        collector.shm_path = "/nonexistent/shm_path"
        fallback = collector._collect_shm()
        self.assertFalse(fallback["available"])
        self.assertEqual(fallback["total_mb"], 0)
        collector.close()

    def test_multi_gpu_metrics_schema(self):
        """Verify multi-GPU metrics expose devices, all_vram, and per-process gpu_index."""
        collector = MetricsCollector()
        snap = collector.get_metrics_snapshot(force=True)
        gpu = snap.get("gpu", {})
        self.assertIn("devices", gpu)
        self.assertIsInstance(gpu["devices"], list)
        self.assertIn("all_vram", gpu)
        self.assertIn("gpus", snap)
        self.assertEqual(snap["gpus"], gpu["devices"])
        if gpu.get("available", False):
            self.assertGreaterEqual(gpu.get("count", 0), 1)
            for dev in gpu["devices"]:
                self.assertIn("index", dev)
                self.assertIn("model", dev)
                self.assertIn("vram", dev)
                self.assertIn("utilization_pct", dev)
                self.assertIn("temperature_c", dev)
            for p in gpu.get("processes", []):
                self.assertIn("gpu_index", p)
        collector.close()


class TestMemoryDashboardServerHistory(unittest.TestCase):
    """HTTP /api/history and extended /api/metrics integration checks."""

    @classmethod
    def setUpClass(cls):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]

        cls.server = DashboardServer(host="127.0.0.1", port=cls.port, quiet=True)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        for _ in range(25):
            try:
                with urllib.request.urlopen(f"{cls.base_url}/api/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_api_history_returns_points(self):
        """GET /api/history returns 200 JSON with valid timeline points."""
        with urllib.request.urlopen(f"{self.base_url}/api/history", timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("points", data)
            self.assertGreaterEqual(len(data["points"]), 1)
            pt = data["points"][-1]
            for key in ("timestamp", "time", "ram_used_gb", "ram_used_pct", "anon_pct", "swap_used_pct"):
                self.assertIn(key, pt)

    def test_api_history_rejects_post_with_405(self):
        """POST /api/history must return 405 Method Not Allowed."""
        req = urllib.request.Request(f"{self.base_url}/api/history", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2)
        self.assertEqual(ctx.exception.code, 405)

    def test_api_metrics_extended_memory_fields(self):
        """GET /api/metrics exposes deep memory breakdown, pressure, shm, and process list."""
        with urllib.request.urlopen(f"{self.base_url}/api/metrics", timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            mem = data["memory"]
            for key in ("breakdown", "commit", "shm", "pressure", "by_user", "processes", "process_summary"):
                self.assertIn(key, mem)
            self.assertIsInstance(mem["processes"], list)
            self.assertIsInstance(mem["by_user"], list)

    def test_frontend_memory_dashboard_elements_present(self):
        """Verify index.html keeps Memory Dashboard DOM hooks and omits removed panels."""
        html_path = REPO_ROOT / "src" / "static" / "index.html"
        content = html_path.read_text(encoding="utf-8")
        for token in (
            "card-ram",
            "user-quota-bar",
            "mem-process-table",
            "tab-memory",
            "mem-quick-filters",
            "quick-shmem-gb",
            "ram-anon-shmem-stat",
            'data-sort="ram_pct"',
        ):
            self.assertIn(token, content)
        for removed in ("memory-analytics-section", "trend-chart-section", "mem-stacked-bar", "trend-svg"):
            self.assertNotIn(removed, content)

        js_content = (REPO_ROOT / "src" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("memSortColumn: 'rss_mb'", js_content)
        self.assertIn("quickShmemGb", js_content)

    def test_frontend_html_js_css_never_cached_and_memory_route_works(self):
        """Verify /, /memory, /static/app.js, and /static/style.css send no-store headers."""
        for route in ("/", "/memory", "/gpu", "/static/index.html", "/static/app.js", "/static/style.css"):
            with urllib.request.urlopen(f"{self.base_url}{route}", timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                cache_ctrl = resp.headers.get("Cache-Control", "").lower()
                self.assertIn("no-store", cache_ctrl, f"Route {route} had stale cache header: {cache_ctrl}")
                self.assertIn("max-age=0", cache_ctrl, f"Route {route} had non-zero max-age: {cache_ctrl}")
                body = resp.read().decode("utf-8")
                if route in ("/", "/memory", "/gpu", "/static/index.html"):
                    self.assertIn("mem-process-table", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
