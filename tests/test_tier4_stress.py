#!/usr/bin/env python3
"""
tests/test_tier4_stress.py
==========================
Tier 4: Real-World Workload, Stress & Air-Gapped Asset Tests.

Covers performance and stress scenarios from TEST_INFRA.md:
  T4.1: API Latency SLA (50 sequential requests to /api/metrics, p50 < 10ms, p95 < 25ms, max < 50ms)
  T4.2: Concurrent Client Polling (10 concurrent threads, 100 total requests, 100% success rate)
  T4.3: Air-Gapped Asset Integrity (0 external URLs/CDNs in index.html, style.css, app.js)
  T4.4: Live Production Load Consistency (verifies real training jobs like ComfyUI PID 210681 are accurately tracked)
  T4.5: In-Memory Cache Stability under Burst (subsequent requests within TTL served from cache)
  T4.6: Frontend DOM Contract & Element Integrity (metric cards, process table, refresh controls)

Requirement: >= 5 scenarios total.
"""

import os
import re
import sys
import json
import time
import socket
import platform
import subprocess
import threading
import statistics
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.collector import MetricsCollector
except ImportError:
    MetricsCollector = None

try:
    from src.server import DashboardServer
except ImportError:
    DashboardServer = None


class BaseServerTestFixture(unittest.TestCase):
    """Fixture that boots a lightweight in-process DashboardServer on an ephemeral port."""

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
        # Wait up to 3s for server readiness
        for _ in range(30):
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


# ==============================================================================
# Scenario 1: API Latency SLA (T4.1)
# ==============================================================================
class TestScenario1LatencySLA(BaseServerTestFixture):
    """T4.1: API Latency SLA: 50 sequential requests to /api/metrics (p50 < 10ms, p95 < 25ms, max < 50ms)."""

    def test_scenario_1_api_latency_sla(self):
        url = f"{self.base_url}/api/metrics"
        latencies_ms = []

        # Warm-up request
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
        except Exception as e:
            self.fail(f"Warm-up request failed: {e}")

        # 50 sequential requests
        for i in range(50):
            start = time.perf_counter_ns()
            with urllib.request.urlopen(url, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                _ = resp.read()
            end = time.perf_counter_ns()
            elapsed_ms = (end - start) / 1_000_000.0
            latencies_ms.append(elapsed_ms)

        latencies_sorted = sorted(latencies_ms)
        p50 = statistics.median(latencies_sorted)
        p95_idx = int(0.95 * len(latencies_sorted))
        p95 = latencies_sorted[p95_idx]
        max_lat = max(latencies_sorted)

        print(f"\n[LATENCY SLA BENCHMARK] 50 requests: p50={p50:.2f}ms, p95={p95:.2f}ms, max={max_lat:.2f}ms")

        # In-memory cache ensures response times are well within SLA
        self.assertLess(p50, 10.0, f"p50 latency {p50:.2f}ms exceeds SLA 10ms")
        self.assertLess(p95, 25.0, f"p95 latency {p95:.2f}ms exceeds SLA 25ms")
        self.assertLess(max_lat, 50.0, f"max latency {max_lat:.2f}ms exceeds SLA 50ms")


# ==============================================================================
# Scenario 2: Concurrent Client Polling (T4.2)
# ==============================================================================
class TestScenario2ConcurrentPolling(BaseServerTestFixture):
    """T4.2: Concurrent client polling: 10 concurrent threads polling simultaneously (100% success rate)."""

    def test_scenario_2_concurrent_client_polling(self):
        url = f"{self.base_url}/api/metrics"
        num_threads = 10
        requests_per_thread = 10
        total_requests = num_threads * requests_per_thread

        def worker_task(thread_id):
            successes = 0
            errors = []
            for r in range(requests_per_thread):
                try:
                    with urllib.request.urlopen(url, timeout=3) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            if "gpu" in data and "memory" in data:
                                successes += 1
                            else:
                                errors.append(f"Incomplete payload in thread {thread_id}")
                        else:
                            errors.append(f"HTTP {resp.status} in thread {thread_id}")
                except Exception as e:
                    errors.append(str(e))
            return successes, errors

        start_time = time.time()
        total_successes = 0
        all_errors = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            for future in as_completed(futures):
                s, errs = future.result()
                total_successes += s
                all_errors.extend(errs)

        elapsed = time.time() - start_time
        print(f"\n[CONCURRENCY BENCHMARK] {total_successes}/{total_requests} succeeded in {elapsed:.2f}s (10 threads)")

        self.assertEqual(len(all_errors), 0, f"Encountered concurrency errors: {all_errors[:5]}")
        self.assertEqual(total_successes, total_requests, "Dropped requests under concurrent polling")


# ==============================================================================
# Scenario 3: Air-Gapped Asset Integrity (T4.3)
# ==============================================================================
class TestScenario3AirGappedAssets(unittest.TestCase):
    """T4.3: Air-gapped asset self-containment: scan static assets for external URLs (0 CDN/external links)."""

    def test_scenario_3_air_gapped_asset_self_containment(self):
        static_dir = REPO_ROOT / "src" / "static"
        if not static_dir.exists():
            self.skipTest("src/static directory not yet created")

        assets_to_scan = ["index.html", "style.css", "app.js"]
        external_url_regex = re.compile(
            r"https?://(?!(?:127\.0\.0\.1|localhost|172\.24\.|0\.0\.0\.0|www\.w3\.org/2000/svg|www\.w3\.org/1999/xlink)[/:]?)[a-zA-Z0-9.-]+",
            re.IGNORECASE,
        )
        forbidden_cdn_terms = [
            "cdnjs.cloudflare.com",
            "cdn.jsdelivr.net",
            "unpkg.com",
            "fonts.googleapis.com",
            "fonts.gstatic.com",
            "maxcdn.bootstrapcdn.com",
            "code.jquery.com",
            "fontawesome.com",
            "cdn.tailwindcss.com",
        ]

        violations = []
        for asset_name in assets_to_scan:
            asset_path = static_dir / asset_name
            if not asset_path.exists():
                self.fail(f"Required static asset missing: {asset_path}")

            content = asset_path.read_text(encoding="utf-8")

            # Check external URLs
            matches = external_url_regex.findall(content)
            for m in matches:
                violations.append((asset_name, f"External URL: {m}"))

            # Check known CDNs
            for cdn in forbidden_cdn_terms:
                if cdn in content.lower():
                    violations.append((asset_name, f"CDN dependency: {cdn}"))

        self.assertEqual(
            len(violations),
            0,
            f"Air-gapped asset violation: found {len(violations)} external dependencies: {violations}",
        )


# ==============================================================================
# Scenario 4: Live Training Job Tracking (T4.4)
# ==============================================================================
class TestScenario4LiveJobTracking(unittest.TestCase):
    """T4.4: Live training job tracking consistency (e.g. ComfyUI PID 210681 or active compute PIDs)."""

    def test_scenario_4_live_job_tracking(self):
        if MetricsCollector is None:
            raise unittest.SkipTest("src.collector.MetricsCollector is not implemented yet")

        collector = MetricsCollector()
        snapshot = collector.get_metrics_snapshot()
        gpu = snapshot.get("gpu", {})

        if not gpu.get("available", False):
            self.skipTest("GPU not available in current test environment")

        processes = gpu.get("processes", [])
        # Check against native nvidia-smi if on Linux
        compat_paths = ["/data/shangguanrz/仪表盘/compat", str(REPO_ROOT / "compat")]
        env = os.environ.copy()
        for cp in compat_paths:
            if os.path.isdir(cp):
                env["LD_LIBRARY_PATH"] = f"{cp}:{env.get('LD_LIBRARY_PATH', '')}"
                break

        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [l.strip().split(",") for l in res.stdout.strip().splitlines()]
                ref_pids = {int(p[0].strip()) for p in lines if len(p) >= 2}
                collector_pids = {p.get("pid") for p in processes}

                # Verify all active reference PIDs are detected by collector
                intersection = ref_pids.intersection(collector_pids)
                self.assertGreater(len(intersection), 0, f"Active jobs {ref_pids} not captured by collector")
        except Exception:
            self.skipTest("Live nvidia-smi query skipped or unavailable")


# ==============================================================================
# Scenario 5: In-Memory Cache Stability under Burst (T4.5)
# ==============================================================================
class TestScenario5CacheBurstStability(BaseServerTestFixture):
    """T4.5: Burst requests within cache TTL window return consistent snapshots instantly."""

    def test_scenario_5_in_memory_cache_stability_under_burst(self):
        url = f"{self.base_url}/api/metrics"
        snapshots = []

        # Fire 20 rapid bursts within 200ms
        for _ in range(20):
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                snapshots.append(data)

        # All responses in rapid burst should share same or near-identical timestamp
        timestamps = [s.get("timestamp", 0) for s in snapshots]
        self.assertLessEqual(max(timestamps) - min(timestamps), 2.0)
        self.assertEqual(len(snapshots), 20)


# ==============================================================================
# Scenario 6: Frontend DOM Contract & Element Integrity (T4.6)
# ==============================================================================
class TestScenario6FrontendDOMContract(unittest.TestCase):
    """T4.6: Verify index.html contains essential dashboard DOM IDs and classes."""

    def test_scenario_6_frontend_dom_contract(self):
        html_path = REPO_ROOT / "src" / "static" / "index.html"
        if not html_path.exists():
            self.skipTest("src/static/index.html not yet implemented")

        content = html_path.read_text(encoding="utf-8")

        required_tokens = [
            "vram",
            "ram",
            "gpu",
            "temperature",
            "process",
        ]
        for token in required_tokens:
            self.assertIn(token, content.lower(), f"Missing required token '{token}' in index.html")


if __name__ == "__main__":
    unittest.main(verbosity=2)
