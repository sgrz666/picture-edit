#!/usr/bin/env python3
"""
tests/test_adversarial_load.py
==============================
Challenger 1 Adversarial Load, Latency & Concurrency Stress Suite.

Empirically challenges the live deployed dashboard daemon on ssh sg:9527:
1. 100 sequential requests latency distribution (p50, p95, p99, max, min, mean)
2. 20 concurrent threads high concurrency load (200 requests) & error rate
3. Cache consistency & freshness verification (sub-millisecond cache hits vs 1s refresh)
4. Fuzzing HTTP methods (PUT, DELETE, PATCH, OPTIONS, CONNECT, custom verbs)
5. Malformed URLs, path traversal (%00, %2e%2e%2f, etc.), massive query strings (100KB)
6. Rapid connect/disconnect & broken pipe / TCP RST injection
"""

import sys
import time
import socket
import struct
import urllib.request
import urllib.error
import urllib.parse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 9527
BASE_URL = f"http://{TARGET_HOST}:{TARGET_PORT}"

def check_server_alive():
    """Verify live server health probe."""
    url = f"{BASE_URL}/api/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Challenger1-Probe"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "healthy"
    except Exception as e:
        print(f"[FATAL] Server not reachable at {url}: {e}")
        return False
    return False


def test_100_sequential_latency():
    """Benchmark 100 sequential requests latency distribution to /api/metrics."""
    print("\n" + "=" * 70)
    print("TEST 1: 100 SEQUENTIAL REQUESTS LATENCY BENCHMARK (/api/metrics)")
    print("=" * 70)
    url = f"{BASE_URL}/api/metrics"
    latencies_ms = []
    status_codes = []
    payload_sizes = []

    # Warm-up request
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            assert resp.status == 200
            _ = resp.read()
    except Exception as e:
        print(f"[FAIL] Warmup failed: {e}")
        return False, {}

    for i in range(100):
        t0 = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = resp.read()
                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1_000_000.0)
                status_codes.append(resp.status)
                payload_sizes.append(len(data))
        except Exception as e:
            print(f"[FAIL] Request #{i+1} failed: {e}")
            latencies_ms.append(float('nan'))
            status_codes.append(500)

    valid_latencies = sorted([lat for lat in latencies_ms if not (lat != lat)])
    if len(valid_latencies) < 100:
        print(f"[FAIL] Only {len(valid_latencies)}/100 requests succeeded.")
        return False, {}

    min_lat = min(valid_latencies)
    max_lat = max(valid_latencies)
    mean_lat = statistics.mean(valid_latencies)
    median_lat = statistics.median(valid_latencies)
    p90 = valid_latencies[int(0.90 * len(valid_latencies))]
    p95 = valid_latencies[int(0.95 * len(valid_latencies))]
    p99 = valid_latencies[int(0.99 * len(valid_latencies))]
    stdev_lat = statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0.0

    print(f"Total Requests:     100")
    print(f"Success Rate:       100% (100/100)")
    print(f"Avg Payload Size:   {statistics.mean(payload_sizes):.1f} bytes")
    print(f"Min Latency:        {min_lat:.3f} ms")
    print(f"Median (p50):       {median_lat:.3f} ms")
    print(f"p90 Latency:        {p90:.3f} ms")
    print(f"p95 Latency:        {p95:.3f} ms")
    print(f"p99 Latency:        {p99:.3f} ms")
    print(f"Max Latency:        {max_lat:.3f} ms")
    print(f"Mean ± Stdev:       {mean_lat:.3f} ms ± {stdev_lat:.3f} ms")

    # SLA Checks
    # In-memory cache should deliver sub-5ms median, p95 < 25ms, max < 50ms
    sla_p50 = median_lat < 10.0
    sla_p95 = p95 < 25.0
    sla_max = max_lat < 50.0

    print(f"\nSLA Verdict:")
    print(f"  p50 < 10ms:       {'[PASS]' if sla_p50 else '[FAIL]'} ({median_lat:.3f} ms)")
    print(f"  p95 < 25ms:       {'[PASS]' if sla_p95 else '[FAIL]'} ({p95:.3f} ms)")
    print(f"  max < 50ms:       {'[PASS]' if sla_max else '[FAIL]'} ({max_lat:.3f} ms)")

    stats = {
        "count": 100,
        "min_ms": min_lat,
        "p50_ms": median_lat,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max_lat,
        "mean_ms": mean_lat,
        "stdev_ms": stdev_lat,
        "sla_passed": sla_p50 and sla_p95 and sla_max,
    }
    return (sla_p50 and sla_p95 and sla_max), stats


def test_20_threads_concurrency():
    """Stress test 20 concurrent threads firing 200 requests to /api/metrics."""
    print("\n" + "=" * 70)
    print("TEST 2: HIGH CONCURRENCY STRESS (20 THREADS x 10 = 200 REQUESTS)")
    print("=" * 70)
    url = f"{BASE_URL}/api/metrics"
    num_threads = 20
    reqs_per_thread = 10
    total_reqs = num_threads * reqs_per_thread

    def worker(thread_idx):
        latencies = []
        errors = []
        for r in range(reqs_per_thread):
            t0 = time.perf_counter_ns()
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = resp.read()
                    t1 = time.perf_counter_ns()
                    if resp.status == 200:
                        parsed = json.loads(data.decode("utf-8"))
                        if "gpu" in parsed and "memory" in parsed:
                            latencies.append((t1 - t0) / 1_000_000.0)
                        else:
                            errors.append(f"Incomplete JSON schema")
                    else:
                        errors.append(f"HTTP {resp.status}")
            except Exception as e:
                errors.append(str(e))
        return thread_idx, latencies, errors

    t_start = time.perf_counter()
    all_latencies = []
    all_errors = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in as_completed(futures):
            tid, lats, errs = f.result()
            all_latencies.extend(lats)
            all_errors.extend(errs)

    t_total = time.perf_counter() - t_start
    throughput = len(all_latencies) / t_total if t_total > 0 else 0

    print(f"Total Requests Dispatched:  {total_reqs}")
    print(f"Successful Requests:        {len(all_latencies)}")
    print(f"Failed / Dropped:           {len(all_errors)}")
    print(f"Error Rate:                 {len(all_errors) / total_reqs * 100:.2f}%")
    print(f"Wall Clock Elapsed:         {t_total:.3f} s")
    print(f"Concurrency Throughput:     {throughput:.1f} req/s")

    if all_latencies:
        sorted_lats = sorted(all_latencies)
        p50 = statistics.median(sorted_lats)
        p95 = sorted_lats[int(0.95 * len(sorted_lats))]
        max_lat = max(sorted_lats)
        print(f"Concurrent p50 Latency:     {p50:.3f} ms")
        print(f"Concurrent p95 Latency:     {p95:.3f} ms")
        print(f"Concurrent Max Latency:     {max_lat:.3f} ms")

    if all_errors:
        print(f"[FAIL] Concurrency errors observed: {all_errors[:5]}")
        return False, {"errors": all_errors}

    # Verify server responsiveness immediately after stress
    alive = check_server_alive()
    print(f"Post-Stress Health Check:   {'[OK]' if alive else '[FAILED]'}")

    stats = {
        "threads": num_threads,
        "total_requests": total_reqs,
        "success_count": len(all_latencies),
        "error_count": len(all_errors),
        "wall_time_s": t_total,
        "throughput_rps": throughput,
        "post_alive": alive,
    }
    return (len(all_errors) == 0 and alive), stats


def test_cache_freshness_and_consistency():
    """Verify in-memory cache freshness and sub-millisecond cached responses."""
    print("\n" + "=" * 70)
    print("TEST 3: CACHE FRESHNESS & SUB-MS HIT VERIFICATION")
    print("=" * 70)
    url = f"{BASE_URL}/api/metrics"

    # Sample over 3.2 seconds at 100ms intervals
    samples = []
    latencies = []
    for _ in range(32):
        t0 = time.perf_counter_ns()
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1_000_000.0)
            samples.append(data.get("timestamp", 0))
        time.sleep(0.1)

    unique_timestamps = sorted(set(samples))
    print(f"Total Samples Collected:   {len(samples)}")
    print(f"Unique Metric Timestamps: {len(unique_timestamps)}")
    print(f"Timestamp Progression:     {unique_timestamps}")
    print(f"Median Burst Latency:      {statistics.median(latencies):.3f} ms")
    print(f"Min Burst Latency:         {min(latencies):.3f} ms")

    # In 3.2s with ~1s TTL, we expect between 2 and 5 distinct timestamp updates
    updates_ok = 2 <= len(unique_timestamps) <= 6
    # Sub-millisecond or low single-digit cache response latency
    cache_fast = statistics.median(latencies) < 5.0

    print(f"Cache Refresh Invariant:   {'[PASS]' if updates_ok else '[FAIL]'} ({len(unique_timestamps)} distinct epochs in 3.2s)")
    print(f"Sub-5ms Cache Latency:     {'[PASS]' if cache_fast else '[FAIL]'} ({statistics.median(latencies):.3f} ms)")

    return (updates_ok and cache_fast), {
        "unique_epochs": len(unique_timestamps),
        "median_latency_ms": statistics.median(latencies),
    }


def test_http_method_fuzzing():
    """Fuzz HTTP methods: PUT, DELETE, PATCH, OPTIONS, CONNECT, invalid verbs."""
    print("\n" + "=" * 70)
    print("TEST 4: HTTP METHOD & VERB FUZZING")
    print("=" * 70)

    methods_to_test = [
        ("GET", "/api/metrics", [200]),
        ("HEAD", "/api/metrics", [200]),
        ("OPTIONS", "/api/metrics", [204]),
        ("POST", "/api/metrics", [405]),
        ("PUT", "/api/metrics", [405]),
        ("DELETE", "/api/metrics", [405]),
        ("PATCH", "/api/metrics", [405]),
        ("GET", "/api/health", [200]),
        ("HEAD", "/api/health", [200]),
        ("OPTIONS", "/api/health", [204]),
        ("POST", "/api/health", [405]),
        ("PUT", "/api/health", [405]),
        ("DELETE", "/api/health", [405]),
        ("PATCH", "/api/health", [405]),
    ]

    all_passed = True
    results = []

    for method, path, expected_codes in methods_to_test:
        url = f"{BASE_URL}{path}"
        req = urllib.request.Request(url, method=method)
        status_got = None
        headers_got = {}
        body = b""
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                status_got = resp.status
                headers_got = dict(resp.headers)
                body = resp.read()
        except urllib.error.HTTPError as e:
            status_got = e.code
            headers_got = dict(e.headers)
            body = e.read()
        except Exception as e:
            status_got = f"EXCEPTION: {e}"

        passed = status_got in expected_codes
        if not passed:
            all_passed = False
        print(f"  {method:<8} {path:<15} -> Got: {status_got:<5} Expected: {expected_codes}  [{'PASS' if passed else 'FAIL'}]")

        # For 405, check Allow header
        if status_got == 405:
            allow = headers_got.get("Allow") or headers_got.get("allow")
            if not allow or "GET" not in allow:
                print(f"    [WARN] 405 missing valid 'Allow' header: {allow}")
                all_passed = False
        results.append((method, path, status_got, passed))

    # Test non-standard and custom verbs via raw socket
    raw_verbs = ["TRACE", "CONNECT", "FOOBAR", "PROPFIND", "MKCOL", "INVENTED"]
    for verb in raw_verbs:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((TARGET_HOST, TARGET_PORT))
            raw_req = f"{verb} /api/metrics HTTP/1.1\r\nHost: {TARGET_HOST}\r\nConnection: close\r\n\r\n"
            s.sendall(raw_req.encode("utf-8"))
            resp_raw = s.recv(4096).decode("utf-8", errors="replace")
            s.close()

            # First line: HTTP/1.1 501 Not Implemented or 405 Method Not Allowed
            first_line = resp_raw.split("\r\n")[0] if resp_raw else "EMPTY"
            # We accept 405, 501, or proper error status (must not be 500 internal error or crash)
            valid = any(code in first_line for code in ["405", "501", "400"])
            print(f"  {verb:<8} /api/metrics   -> Raw: {first_line[:30]}  [{'PASS' if valid else 'FAIL'}]")
            if not valid:
                all_passed = False
        except Exception as e:
            print(f"  {verb:<8} /api/metrics   -> EXCEPTION: {e} [FAIL]")
            all_passed = False

    return all_passed, results


def test_malformed_urls_and_injections():
    """Test path traversal, null bytes, massive query strings, URL encoding fuzzing."""
    print("\n" + "=" * 70)
    print("TEST 5: MALFORMED URLS, PATH TRAVERSAL & EXPLOIT INJECTIONS")
    print("=" * 70)

    test_cases = [
        # Path traversal attempts
        ("/static/../server.py", [403, 404]),
        ("/static/..%2fserver.py", [403, 404]),
        ("/static/%2e%2e%2fserver.py", [403, 404]),
        ("/static/%252e%252e%252fserver.py", [403, 404]),
        ("/static/../../../../../../etc/passwd", [403, 404]),
        ("/static/..\\..\\server.py", [403, 404]),
        ("/static/./././style.css", [200]),
        # Null bytes
        ("/static/index.html%00.png", [403, 400, 404]),
        ("/api/metrics%00", [400, 404]),
        # Nonexistent paths
        ("/nonexistent_endpoint_12345", [404]),
        ("/api/unknown_subservice", [404]),
        ("/api/v1/metrics", [404]),
        # Malformed query strings
        ("/api/metrics?%ZZ=bad_hex", [200, 400]),
        ("/api/metrics?&&&===&&&", [200, 400]),
        ("/api/metrics?utf8=" + urllib.parse.quote("你好世界_显卡监控"), [200]),
        # 100KB massive query string
        ("/api/metrics?" + "param=" + "X" * 102400, [200, 414, 400]),
        # Extremely long path
        ("/" + "a" * 8192, [404, 414, 400]),
    ]
    all_passed = True

    for path, allowed_codes in test_cases:
        url = f"{BASE_URL}{path}"
        status_got = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Challenger1-Fuzz"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                status_got = resp.status
        except urllib.error.HTTPError as e:
            status_got = e.code
        except Exception as e:
            status_got = f"EXCEPTION: {type(e).__name__}"

        passed = status_got in allowed_codes
        if not passed:
            all_passed = False
        display_path = path if len(path) <= 45 else path[:42] + "..."
        print(f"  {display_path:<45} -> Got: {str(status_got):<12} Allowed: {allowed_codes} [{'PASS' if passed else 'FAIL'}]")

    # Verify server is still alive and responsive after attacks
    alive = check_server_alive()
    print(f"\nServer Liveness Post-Fuzzing: {'[ALIVE]' if alive else '[CRASHED]'}")
    return (all_passed and alive), {}


def test_rapid_disconnects_and_broken_pipe():
    """Simulate rapid client disconnects, half-open sockets, and TCP RST injections."""
    print("\n" + "=" * 70)
    print("TEST 6: RAPID DISCONNECTS & BROKEN PIPE SIMULATION")
    print("=" * 70)

    # 1. Connect and immediately close (zero bytes sent) x 20
    print("  1. Connect and instant FIN (zero bytes) x 20:")
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TARGET_HOST, TARGET_PORT))
        s.close()
    print("     -> Completed cleanly without server exception.")

    # 2. Connect, send partial request line, and abruptly close x 20
    print("  2. Connect, partial request line ('GET /api/'), instant FIN x 20:")
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TARGET_HOST, TARGET_PORT))
        s.sendall(b"GET /api/")
        s.close()
    print("     -> Completed cleanly.")

    # 3. Connect, send partial headers, and abruptly close x 20
    print("  3. Connect, partial headers, instant FIN x 20:")
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TARGET_HOST, TARGET_PORT))
        s.sendall(b"GET /api/metrics HTTP/1.1\r\nHost: 127.0.0.1\r\n")
        s.close()
    print("     -> Completed cleanly.")

    # 4. Connect, send complete request, read 1 byte of response, send TCP RST
    print("  4. Complete request, read 1 byte, send TCP RST (SO_LINGER=0) x 20:")
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TARGET_HOST, TARGET_PORT))
        s.sendall(b"GET /api/metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        # Read initial byte of HTTP status line
        _ = s.recv(1)
        # Enable SO_LINGER with timeout 0 -> generates TCP RST upon close()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        s.close()
    print("     -> Completed cleanly.")

    # 5. Verify server responds normally to legitimate requests immediately after RST flood
    print("\n  Verifying post-disconnect server responsiveness:")
    alive = check_server_alive()
    t0 = time.perf_counter_ns()
    with urllib.request.urlopen(f"{BASE_URL}/api/metrics", timeout=3) as resp:
        data = resp.read()
        status = resp.status
    t1 = time.perf_counter_ns()
    lat_ms = (t1 - t0) / 1_000_000.0

    print(f"  Post-Disconnect Status:    {status} (Latency: {lat_ms:.2f} ms)")
    print(f"  Post-Disconnect Health:    {'[HEALTHY]' if alive else '[UNHEALTHY]'}")

    passed = alive and (status == 200) and (lat_ms < 10.0)
    return passed, {"latency_ms": lat_ms, "status": status}


def main():
    print("=" * 70)
    print("   CHALLENGER 1: ADVERSARIAL STRESS & CONCURRENCY HARNESS")
    print(f"   Target: {BASE_URL} (Live Deployed Dashboard)")
    print("=" * 70)

    if not check_server_alive():
        print(f"[FATAL] Target server at {BASE_URL} is NOT alive! Aborting.")
        sys.exit(1)

    suite_results = {}

    r1_pass, r1_data = test_100_sequential_latency()
    suite_results["sequential_latency_100"] = (r1_pass, r1_data)

    r2_pass, r2_data = test_20_threads_concurrency()
    suite_results["concurrency_20_threads"] = (r2_pass, r2_data)

    r3_pass, r3_data = test_cache_freshness_and_consistency()
    suite_results["cache_freshness"] = (r3_pass, r3_data)

    r4_pass, r4_data = test_http_method_fuzzing()
    suite_results["method_fuzzing"] = (r4_pass, r4_data)

    r5_pass, r5_data = test_malformed_urls_and_injections()
    suite_results["malformed_and_exploit"] = (r5_pass, r5_data)

    r6_pass, r6_data = test_rapid_disconnects_and_broken_pipe()
    suite_results["rapid_disconnect_rst"] = (r6_pass, r6_data)

    print("\n" + "=" * 70)
    print("                 SUITE SUMMARY & FINAL VERDICT")
    print("=" * 70)
    all_passed = True
    for test_name, (passed, data) in suite_results.items():
        verdict = "PASS" if passed else "FAIL"
        print(f"  {test_name:<30}: [{verdict}]")
        if not passed:
            all_passed = False

    print("-" * 70)
    final_verdict = "APPROVE" if all_passed else "REQUEST_CHANGES"
    print(f"  OVERALL CHALLENGER 1 VERDICT: {final_verdict}")
    print("=" * 70)

    # Save empirical summary to json for handoff
    summary_path = "/data/shangguanrz/仪表盘/logs/challenger_1_results.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "verdict": final_verdict,
                "timestamp": time.time(),
                "all_passed": all_passed,
                "results": suite_results
            }, f, indent=2)
        print(f"Empirical results exported to: {summary_path}")
    except Exception as e:
        print(f"Failed to export results json: {e}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
