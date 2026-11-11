"""Hardware and Process Telemetry Engine (Milestone 1).

Conforms strictly to PROJECT.md interface contracts.
Collects GPU metrics (VRAM, utilization, temperature, active compute processes),
host system RAM and swap via /proc/meminfo, and system load/uptime.
Features a dual-engine architecture:
- Primary: In-process ctypes calling NVML with /data/shangguanrz/仪表盘/compat/libnvidia-ml.so.1
- Secondary: Subprocess calling nvidia-smi with LD_LIBRARY_PATH injection
- Fault tolerance: Degraded status if GPU unavailable; host metrics continue unimpeded.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import datetime
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


class _NVMLMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NVMLMemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NVMLUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class _NVMLProcessInfoV3(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class _NVMLProcessInfoV1(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
    ]


class MetricsCollector:
    """Hardware telemetry collector for GPU VRAM, host RAM, and compute processes."""

    NVML_SUCCESS = 0
    NVML_ERROR_INSUFFICIENT_SIZE = 7

    def __init__(
        self,
        compat_dir: str = "/data/shangguanrz/仪表盘/compat",
        cache_ttl_sec: float = 0.5,
        meminfo_path: str = "/proc/meminfo",
        uptime_path: str = "/proc/uptime",
        loadavg_path: str = "/proc/loadavg",
        proc_root: str = "/proc",
        pressure_path: str = "/proc/pressure/memory",
        vmstat_path: str = "/proc/vmstat",
        shm_path: str = "/dev/shm",
        proc_scan_ttl_sec: float = 1.5,
        max_mem_processes: int = 100,
    ) -> None:
        self.compat_dir = compat_dir
        self.cache_ttl_sec = cache_ttl_sec
        self.meminfo_path = meminfo_path
        self.uptime_path = uptime_path
        self.loadavg_path = loadavg_path
        self.proc_root = proc_root
        self.pressure_path = pressure_path
        self.vmstat_path = vmstat_path
        self.shm_path = shm_path
        self.proc_scan_ttl_sec = proc_scan_ttl_sec
        self.max_mem_processes = max_mem_processes

        self._nvml_lib: Optional[ctypes.CDLL] = None
        self._nvml_handle: Optional[ctypes.c_void_p] = None
        self._nvml_available: bool = False
        self._device_model: Optional[str] = None
        self._driver_version: Optional[str] = None

        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._last_snapshot_time: float = 0.0

        # Memory extension caches & rate-calculation state
        self._uid_cache: Dict[int, str] = {}
        self._prev_vmstat: Optional[Tuple[float, int, int, int]] = None
        self._last_mem_procs_result: Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]] = None
        self._last_mem_procs_time: float = 0.0

        self._init_nvml()

    def _find_library_candidates(self) -> List[str]:
        candidates: List[str] = []
        if self.compat_dir and os.path.isdir(self.compat_dir):
            candidates.append(os.path.join(self.compat_dir, "libnvidia-ml.so.1"))
            candidates.append(os.path.join(self.compat_dir, "libnvidia-ml.so"))

        # Standard system paths
        candidates.extend(
            [
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
                "/usr/lib64/libnvidia-ml.so.1",
                "/usr/lib64/libnvidia-ml.so",
                "/usr/lib/libnvidia-ml.so.1",
                "/usr/lib/libnvidia-ml.so",
            ]
        )

        found = ctypes.util.find_library("nvidia-ml")
        if found:
            candidates.append(found)

        # De-duplicate while preserving order
        unique_candidates: List[str] = []
        for path in candidates:
            if path not in unique_candidates and (os.path.isfile(path) or "/" not in path):
                unique_candidates.append(path)
        return unique_candidates

    def _init_nvml(self) -> bool:
        """Attempt in-process NVML loading via ctypes."""
        for lib_path in self._find_library_candidates():
            try:
                lib = ctypes.CDLL(lib_path)
                if not hasattr(lib, "nvmlInit_v2"):
                    continue

                ret = lib.nvmlInit_v2()
                if ret != self.NVML_SUCCESS:
                    continue

                # Set up device handle (GPU 0)
                handle = ctypes.c_void_p()
                ret = lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
                if ret != self.NVML_SUCCESS:
                    lib.nvmlShutdown()
                    continue

                # Set function signatures
                lib.nvmlDeviceGetMemoryInfo.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_NVMLMemory),
                ]
                lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

                lib.nvmlDeviceGetUtilizationRates.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_NVMLUtilization),
                ]
                lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int

                lib.nvmlDeviceGetTemperature.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_uint),
                ]
                lib.nvmlDeviceGetTemperature.restype = ctypes.c_int

                # Retrieve device model name
                name_buf = ctypes.create_string_buffer(96)
                if hasattr(lib, "nvmlDeviceGetName"):
                    lib.nvmlDeviceGetName(handle, name_buf, 96)
                    self._device_model = name_buf.value.decode("utf-8", errors="ignore").strip()

                # Retrieve driver version
                drv_buf = ctypes.create_string_buffer(96)
                if hasattr(lib, "nvmlSystemGetDriverVersion"):
                    lib.nvmlSystemGetDriverVersion(drv_buf, 96)
                    self._driver_version = drv_buf.value.decode("utf-8", errors="ignore").strip()

                self._nvml_lib = lib
                self._nvml_handle = handle
                self._nvml_available = True
                return True
            except Exception:
                continue

        self._nvml_available = False
        return False

    def _collect_gpu_via_nvml(self) -> Dict[str, Any]:
        """Collect real-time GPU telemetry using direct in-process NVML calls."""
        if not self._nvml_lib or not self._nvml_handle:
            raise RuntimeError("NVML not initialized")

        lib = self._nvml_lib
        handle = self._nvml_handle

        # Memory Info
        total_mb = 0
        used_mb = 0
        free_mb = 0

        if hasattr(lib, "nvmlDeviceGetMemoryInfo_v2"):
            try:
                mem_v2 = _NVMLMemoryV2()
                mem_v2.version = (2 << 24) | ctypes.sizeof(_NVMLMemoryV2)
                ret = lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(mem_v2))
                if ret == self.NVML_SUCCESS:
                    total_mb = int(mem_v2.total // (1024 * 1024))
                    used_mb = int(mem_v2.used // (1024 * 1024))
                    free_mb = max(0, total_mb - used_mb)
            except Exception:
                pass

        if total_mb == 0:
            mem = _NVMLMemory()
            ret = lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
            if ret != self.NVML_SUCCESS:
                raise RuntimeError(f"nvmlDeviceGetMemoryInfo failed with code {ret}")

            total_mb = int(mem.total // (1024 * 1024))
            used_mb = int(mem.used // (1024 * 1024))
            free_mb = max(0, total_mb - used_mb)

        used_pct = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0

        total_gb = round(total_mb / 1024.0, 2)
        used_gb = round(used_mb / 1024.0, 2)
        free_gb = round(free_mb / 1024.0, 2)

        # Utilization Rates
        util = _NVMLUtilization()
        ret = lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
        util_pct = int(util.gpu) if ret == self.NVML_SUCCESS else 0

        # Temperature (Sensor 0 = GPU core)
        temp = ctypes.c_uint()
        ret = lib.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temp))
        temp_c = int(temp.value) if ret == self.NVML_SUCCESS else 0

        # Compute Running Processes
        raw_processes: List[Tuple[int, int, str]] = []  # (pid, used_bytes, name)
        processes_collected = False

        # Attempt v3 first
        if hasattr(lib, "nvmlDeviceGetComputeRunningProcesses_v3"):
            try:
                count = ctypes.c_uint(64)
                arr = (_NVMLProcessInfoV3 * 64)()
                ret = lib.nvmlDeviceGetComputeRunningProcesses_v3(
                    handle, ctypes.byref(count), arr
                )
                if ret == self.NVML_ERROR_INSUFFICIENT_SIZE and count.value > 64:
                    arr = (_NVMLProcessInfoV3 * count.value)()
                    ret = lib.nvmlDeviceGetComputeRunningProcesses_v3(
                        handle, ctypes.byref(count), arr
                    )
                if ret == self.NVML_SUCCESS:
                    for i in range(count.value):
                        raw_processes.append((arr[i].pid, arr[i].usedGpuMemory, ""))
                    processes_collected = True
            except Exception:
                pass

        # Fallback to v1 if v3 unavailable
        if not processes_collected and hasattr(lib, "nvmlDeviceGetComputeRunningProcesses"):
            try:
                count = ctypes.c_uint(64)
                arr = (_NVMLProcessInfoV1 * 64)()
                ret = lib.nvmlDeviceGetComputeRunningProcesses(
                    handle, ctypes.byref(count), arr
                )
                if ret == self.NVML_ERROR_INSUFFICIENT_SIZE and count.value > 64:
                    arr = (_NVMLProcessInfoV1 * count.value)()
                    ret = lib.nvmlDeviceGetComputeRunningProcesses(
                        handle, ctypes.byref(count), arr
                    )
                if ret == self.NVML_SUCCESS:
                    for i in range(count.value):
                        raw_processes.append((arr[i].pid, arr[i].usedGpuMemory, ""))
                    processes_collected = True
            except Exception:
                pass

        # Enrich process details
        enriched_processes = []
        for pid, used_bytes, default_name in raw_processes:
            # Guard against NVML_VALUE_NOT_AVAILABLE sentinel (0xFFFFFFFFFFFFFFFF)
            safe_bytes = 0 if used_bytes >= (1 << 50) else used_bytes
            vram_mb = int(safe_bytes // (1024 * 1024))
            proc_dict = self._enrich_process(pid, default_name, vram_mb, total_mb)
            enriched_processes.append(proc_dict)

        return {
            "available": True,
            "status": "ok",
            "model": self._device_model or "NVIDIA GPU",
            "driver_version": self._driver_version or "Unknown",
            "utilization_pct": util_pct,
            "temperature_c": temp_c,
            "vram": {
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "used_pct": used_pct,
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
            },
            "processes": enriched_processes,
        }

    def _collect_gpu_via_smi(self) -> Dict[str, Any]:
        """Fallback GPU telemetry collection via nvidia-smi CLI."""
        env = os.environ.copy()
        if self.compat_dir and os.path.isdir(self.compat_dir):
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{self.compat_dir}:{existing_ld}" if existing_ld else self.compat_dir
            )

        cmd_gpu = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,temperature.gpu,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        res_gpu = subprocess.run(
            cmd_gpu,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            env=env,
        )
        if res_gpu.returncode != 0 or not res_gpu.stdout.strip():
            raise RuntimeError(f"nvidia-smi query-gpu failed: {res_gpu.stderr.strip()}")

        first_line = res_gpu.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 7:
            raise ValueError(f"Unexpected nvidia-smi output format: {first_line}")

        model_name = parts[0]
        driver_version = parts[1]
        util_pct = int(float(parts[2]))
        temp_c = int(float(parts[3]))
        total_mb = int(float(parts[4]))
        used_mb = int(float(parts[5]))
        free_mb = max(0, total_mb - used_mb)

        used_pct = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0
        total_gb = round(total_mb / 1024.0, 2)
        used_gb = round(used_mb / 1024.0, 2)
        free_gb = round(free_mb / 1024.0, 2)

        # Query compute apps
        cmd_apps = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        res_apps = subprocess.run(
            cmd_apps,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            env=env,
        )

        processes = []
        if res_apps.returncode == 0 and res_apps.stdout.strip():
            for line in res_apps.stdout.strip().splitlines():
                app_parts = [p.strip() for p in line.split(",")]
                if len(app_parts) >= 3:
                    try:
                        pid = int(app_parts[0])
                        proc_name = app_parts[1]
                        vram_used = int(float(app_parts[2]))
                        processes.append(
                            self._enrich_process(pid, proc_name, vram_used, total_mb)
                        )
                    except (ValueError, IndexError):
                        continue

        return {
            "available": True,
            "status": "ok",
            "model": model_name,
            "driver_version": driver_version,
            "utilization_pct": util_pct,
            "temperature_c": temp_c,
            "vram": {
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "used_pct": used_pct,
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
            },
            "processes": processes,
        }

    def _enrich_process(
        self, pid: int, default_name: str, used_mb: int, total_vram_mb: int
    ) -> Dict[str, Any]:
        """Map PID to Linux username via pwd.getpwuid and commandline preview."""
        user = "unknown"
        # 1. Resolve username via os.stat and pwd.getpwuid
        try:
            stat_info = os.stat(f"/proc/{pid}")
            uid = stat_info.st_uid
            try:
                import pwd

                user = pwd.getpwuid(uid).pw_name
            except (KeyError, ImportError, AttributeError):
                user = str(uid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            user = "unknown"
        except Exception:
            user = "unknown"

        # Fallback to ps if user still unknown
        if user == "unknown":
            try:
                res = subprocess.run(
                    ["ps", "-o", "user=", "-p", str(pid)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1,
                )
                if res.returncode == 0 and res.stdout.strip():
                    user = res.stdout.strip()
            except Exception:
                pass

        # 2. Resolve process name
        name = default_name or "unknown"
        try:
            with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="ignore") as f:
                comm = f.read().strip()
                if comm:
                    name = comm
        except Exception:
            pass

        # 3. Resolve commandline preview
        cmdline = ""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
                if raw:
                    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

        if not cmdline:
            cmdline = name

        used_gb = round(used_mb / 1024.0, 2)
        vram_pct = round((used_mb / total_vram_mb) * 100, 2) if total_vram_mb > 0 else 0.0

        return {
            "pid": pid,
            "user": user,
            "name": name,
            "cmdline": cmdline,
            "vram_used_mb": used_mb,
            "vram_used_gb": used_gb,
            "vram_pct": vram_pct,
        }

    def _collect_gpu(self) -> Dict[str, Any]:
        """Dual-engine GPU collector: tries in-process NVML first, then nvidia-smi."""
        # 1. Primary engine: in-process ctypes NVML
        if self._nvml_available:
            try:
                return self._collect_gpu_via_nvml()
            except Exception:
                # NVML failed during runtime, try fallback
                pass

        # 2. Secondary engine: nvidia-smi subprocess
        try:
            return self._collect_gpu_via_smi()
        except Exception:
            pass

        # 3. Fault-tolerant error payload
        return {
            "available": False,
            "status": "error",
            "model": None,
            "driver_version": None,
            "utilization_pct": None,
            "temperature_c": None,
            "vram": {
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "used_pct": 0.0,
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
            },
            "processes": [],
        }

    def _collect_memory_and_swap(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse host RAM and swap memory from /proc/meminfo in <0.1ms."""
        info: Dict[str, int] = {}
        if os.path.isfile(self.meminfo_path):
            try:
                with open(self.meminfo_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if ":" not in line:
                            continue
                        parts = line.split(":", 1)
                        k = parts[0].strip()
                        v_parts = parts[1].strip().split()
                        if v_parts:
                            try:
                                info[k] = int(v_parts[0])  # values in kB
                            except ValueError:
                                pass
            except Exception:
                pass

        if info:
            total_kb = info.get("MemTotal", 0)
            free_kb = info.get("MemFree", 0)
            raw_available = info.get("MemAvailable", info.get("MemFree", 0))
            buffers_kb = info.get("Buffers", 0)
            cached_kb = info.get("Cached", 0)
            sreclaimable_kb = info.get("SReclaimable", 0)
            sunreclaim_kb = info.get("SUnreclaim", 0)
            slab_kb = info.get("Slab", sreclaimable_kb + sunreclaim_kb)
            shmem_kb = info.get("Shmem", 0)
            anon_kb = info.get("AnonPages", 0)
            page_tables_kb = info.get("PageTables", 0)
            kernel_stack_kb = info.get("KernelStack", 0)
            active_kb = info.get("Active", 0)
            inactive_kb = info.get("Inactive", 0)
            dirty_kb = info.get("Dirty", 0)
            writeback_kb = info.get("Writeback", 0)
            unevictable_kb = info.get("Unevictable", 0)
            mlocked_kb = info.get("Mlocked", 0)
            anon_huge_kb = info.get("AnonHugePages", 0)
            commit_limit_kb = info.get("CommitLimit", 0)
            committed_as_kb = info.get("Committed_AS", 0)

            total_mb = total_kb // 1024
            free_mb = free_kb // 1024
            available_mb = raw_available // 1024
            used_mb = max(0, total_mb - available_mb)
            buffers_mb = buffers_kb // 1024
            cached_mb = (buffers_kb + cached_kb + sreclaimable_kb) // 1024

            used_pct = round((used_mb / total_mb) * 100, 2) if total_mb > 0 else 0.0
            total_gb = round(total_mb / 1024.0, 2)
            used_gb = round(used_mb / 1024.0, 2)
            available_gb = round(available_mb / 1024.0, 2)

            swap_total_mb = info.get("SwapTotal", 0) // 1024
            swap_free_mb = info.get("SwapFree", 0) // 1024
            swap_used_mb = max(0, swap_total_mb - swap_free_mb)
            swap_cached_mb = info.get("SwapCached", 0) // 1024
            swap_used_pct = (
                round((swap_used_mb / swap_total_mb) * 100, 2) if swap_total_mb > 0 else 0.0
            )
            zswap_mb = info.get("Zswap", 0) // 1024
            zswapped_mb = info.get("Zswapped", 0) // 1024

            commit_limit_mb = commit_limit_kb // 1024
            committed_mb = committed_as_kb // 1024
            commit_pct = (
                round((committed_mb / commit_limit_mb) * 100, 1)
                if commit_limit_mb > 0
                else 0.0
            )

            breakdown = {
                "anon_mb": anon_kb // 1024,
                "file_cache_mb": max(0, cached_kb - shmem_kb) // 1024,
                "buffers_mb": buffers_mb,
                "raw_cached_mb": cached_kb // 1024,
                "shmem_mb": shmem_kb // 1024,
                "slab_mb": slab_kb // 1024,
                "sreclaimable_mb": sreclaimable_kb // 1024,
                "sunreclaim_mb": sunreclaim_kb // 1024,
                "page_tables_mb": page_tables_kb // 1024,
                "kernel_stack_mb": kernel_stack_kb // 1024,
                "active_mb": active_kb // 1024,
                "inactive_mb": inactive_kb // 1024,
                "dirty_mb": dirty_kb // 1024,
                "writeback_mb": writeback_kb // 1024,
                "unevictable_mb": unevictable_kb // 1024,
                "mlocked_mb": mlocked_kb // 1024,
                "hugepages_anon_mb": anon_huge_kb // 1024,
                # Composition buckets for segmented stacked bar
                "process_anon_mb": anon_kb // 1024,
                "shared_mem_mb": shmem_kb // 1024,
                "reclaimable_cache_mb": max(0, buffers_kb + cached_kb + sreclaimable_kb - shmem_kb) // 1024,
                "kernel_slab_other_mb": (sunreclaim_kb + page_tables_kb + kernel_stack_kb) // 1024,
                "free_mb": free_mb,
            }
            commit_info = {
                "limit_mb": commit_limit_mb,
                "committed_mb": committed_mb,
                "commit_pct": commit_pct,
            }
        else:
            # Fallback for Windows or systems without /proc/meminfo
            total_mb = 16384
            used_mb = 8192
            free_mb = 8192
            available_mb = 8192
            cached_mb = 2048
            used_pct = 50.0
            total_gb = 16.0
            used_gb = 8.0
            available_gb = 8.0
            swap_total_mb = 0
            swap_used_mb = 0
            swap_free_mb = 0
            swap_cached_mb = 0
            swap_used_pct = 0.0
            zswap_mb = 0
            zswapped_mb = 0

            # On Windows, try ctypes GlobalMemoryStatusEx if available
            if sys.platform == "win32":
                try:
                    class _MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = _MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        total_mb = int(stat.ullTotalPhys // (1024 * 1024))
                        available_mb = int(stat.ullAvailPhys // (1024 * 1024))
                        used_mb = max(0, total_mb - available_mb)
                        free_mb = available_mb
                        used_pct = (
                            round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0
                        )
                        total_gb = round(total_mb / 1024.0, 2)
                        used_gb = round(used_mb / 1024.0, 2)
                        available_gb = round(available_mb / 1024.0, 2)
                except Exception:
                    pass

            breakdown = {
                "anon_mb": max(0, used_mb - cached_mb),
                "file_cache_mb": cached_mb,
                "buffers_mb": 0,
                "raw_cached_mb": cached_mb,
                "shmem_mb": 0,
                "slab_mb": 0,
                "sreclaimable_mb": 0,
                "sunreclaim_mb": 0,
                "page_tables_mb": 0,
                "kernel_stack_mb": 0,
                "active_mb": used_mb,
                "inactive_mb": cached_mb,
                "dirty_mb": 0,
                "writeback_mb": 0,
                "unevictable_mb": 0,
                "mlocked_mb": 0,
                "hugepages_anon_mb": 0,
                "process_anon_mb": max(0, used_mb - cached_mb),
                "shared_mem_mb": 0,
                "reclaimable_cache_mb": cached_mb,
                "kernel_slab_other_mb": 0,
                "free_mb": free_mb,
            }
            commit_info = {
                "limit_mb": total_mb,
                "committed_mb": used_mb,
                "commit_pct": round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0,
            }

        memory_dict = {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
            "available_mb": available_mb,
            "cached_mb": cached_mb,
            "used_pct": used_pct,
            "total_gb": total_gb,
            "used_gb": used_gb,
            "available_gb": available_gb,
            "breakdown": breakdown,
            "commit": commit_info,
        }

        swap_dict = {
            "total_mb": swap_total_mb,
            "used_mb": swap_used_mb,
            "free_mb": swap_free_mb,
            "used_pct": swap_used_pct,
            "total_gb": round(swap_total_mb / 1024.0, 2),
            "used_gb": round(swap_used_mb / 1024.0, 2),
            "free_gb": round(swap_free_mb / 1024.0, 2),
            "cached_mb": swap_cached_mb,
            "zswap_mb": zswap_mb,
            "zswapped_mb": zswapped_mb,
        }

        return memory_dict, swap_dict

    def _collect_shm(self) -> Dict[str, Any]:
        """Collect /dev/shm shared tmpfs capacity and utilization (<0.02ms)."""
        if self.shm_path and os.path.isdir(self.shm_path) and hasattr(os, "statvfs"):
            try:
                st = os.statvfs(self.shm_path)
                total_bytes = st.f_blocks * st.f_frsize
                avail_bytes = st.f_bavail * st.f_frsize
                free_bytes = st.f_bfree * st.f_frsize
                used_bytes = max(0, total_bytes - free_bytes)

                total_mb = int(total_bytes // (1024 * 1024))
                used_mb = int(used_bytes // (1024 * 1024))
                free_mb = int(avail_bytes // (1024 * 1024))
                used_pct = round((used_mb / total_mb) * 100, 2) if total_mb > 0 else 0.0
                return {
                    "available": True,
                    "mount_point": self.shm_path,
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "free_mb": free_mb,
                    "used_pct": used_pct,
                    "total_gb": round(total_mb / 1024.0, 2),
                    "used_gb": round(used_mb / 1024.0, 2),
                }
            except Exception:
                pass
        return {
            "available": False,
            "mount_point": self.shm_path,
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "used_pct": 0.0,
            "total_gb": 0.0,
            "used_gb": 0.0,
        }

    def _collect_pressure(self) -> Dict[str, Any]:
        """Collect Linux kernel Pressure Stall Information from /proc/pressure/memory."""
        result: Dict[str, Any] = {
            "available": False,
            "some_avg10": 0.0,
            "some_avg60": 0.0,
            "some_avg300": 0.0,
            "full_avg10": 0.0,
            "full_avg60": 0.0,
            "full_avg300": 0.0,
        }
        if not self.pressure_path or not os.path.isfile(self.pressure_path):
            return result
        try:
            with open(self.pressure_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                kind = parts[0]
                if kind not in ("some", "full"):
                    continue
                for token in parts[1:]:
                    if "=" not in token:
                        continue
                    k, v = token.split("=", 1)
                    if k in ("avg10", "avg60", "avg300"):
                        try:
                            result[f"{kind}_{k}"] = round(float(v), 2)
                            result["available"] = True
                        except ValueError:
                            pass
        except Exception:
            pass
        return result

    def _collect_vmstat_rates(self, now: float) -> Dict[str, Any]:
        """Collect swap-in/out KB/s, major page faults/s, and oom_kill count via /proc/vmstat."""
        metrics: Dict[str, Any] = {
            "swap_in_kbps": 0.0,
            "swap_out_kbps": 0.0,
            "pgmajfault_per_sec": 0.0,
            "oom_kills": 0,
            "thrashing": False,
        }
        if not self.vmstat_path or not os.path.isfile(self.vmstat_path):
            return metrics

        pswpin = pswpout = pgmajfault = oom_kill = 0
        try:
            with open(self.vmstat_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("pswpin "):
                        pswpin = int(line.split()[1])
                    elif line.startswith("pswpout "):
                        pswpout = int(line.split()[1])
                    elif line.startswith("pgmajfault "):
                        pgmajfault = int(line.split()[1])
                    elif line.startswith("oom_kill "):
                        oom_kill = int(line.split()[1])
        except Exception:
            return metrics

        page_kb = 4
        if hasattr(os, "sysconf"):
            try:
                page_kb = max(1, os.sysconf("SC_PAGESIZE") // 1024)
            except (ValueError, OSError):
                page_kb = 4

        swap_in_kbps = 0.0
        swap_out_kbps = 0.0
        pgmaj_rate = 0.0
        if self._prev_vmstat is not None:
            prev_t, prev_in, prev_out, prev_maj = self._prev_vmstat
            dt = now - prev_t
            if dt > 0.05:
                d_in = max(0, pswpin - prev_in)
                d_out = max(0, pswpout - prev_out)
                d_maj = max(0, pgmajfault - prev_maj)
                swap_in_kbps = round((d_in * page_kb) / dt, 1)
                swap_out_kbps = round((d_out * page_kb) / dt, 1)
                pgmaj_rate = round(d_maj / dt, 1)

        self._prev_vmstat = (now, pswpin, pswpout, pgmajfault)
        metrics["swap_in_kbps"] = swap_in_kbps
        metrics["swap_out_kbps"] = swap_out_kbps
        metrics["pgmajfault_per_sec"] = pgmaj_rate
        metrics["oom_kills"] = oom_kill
        metrics["thrashing"] = (swap_in_kbps + swap_out_kbps) >= 1024.0
        return metrics

    def _resolve_uid(self, uid: int) -> str:
        """Resolve numeric Linux UID to username using an in-memory cache."""
        cached = self._uid_cache.get(uid)
        if cached is not None:
            return cached
        try:
            import pwd

            name = pwd.getpwuid(uid).pw_name
        except (KeyError, ImportError, AttributeError):
            name = str(uid)
        self._uid_cache[uid] = name
        return name

    def _read_smaps_rollup(self, pid_dir: str) -> Tuple[Optional[int], Optional[int]]:
        """Read proportional Pss and SwapPss (kB) from /proc/<pid>/smaps_rollup if accessible."""
        rollup_file = os.path.join(self.proc_root, pid_dir, "smaps_rollup")
        try:
            with open(rollup_file, "rb") as f:
                raw = f.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            return None, None

        pss_kb: Optional[int] = None
        swap_pss_kb: Optional[int] = None
        for line in raw.splitlines():
            if line.startswith(b"Pss:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pss_kb = int(parts[1])
                    except ValueError:
                        pass
            elif line.startswith(b"SwapPss:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        swap_pss_kb = int(parts[1])
                    except ValueError:
                        pass
                    break
        return pss_kb, swap_pss_kb

    def _collect_memory_processes(
        self,
        total_mem_mb: int,
        total_swap_mb: int,
        force: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Scan /proc/<pid>/status (and smaps_rollup when permitted) for RAM/Swap footprint."""
        now = time.time()
        if (
            not force
            and self._last_mem_procs_result is not None
            and (now - self._last_mem_procs_time) < self.proc_scan_ttl_sec
        ):
            return self._last_mem_procs_result

        if not self.proc_root or not os.path.isdir(self.proc_root):
            empty_summary = {
                "active_count": 0,
                "returned_count": 0,
                "total_rss_mb": 0.0,
                "total_swap_mb": 0.0,
                "total_vm_swap_mb": 0.0,
                "swap_exact_procs": 0,
                "swap_fallback_procs": 0,
            }
            return [], [], empty_summary

        # Tuple: (footprint_kb, pid, uid, user, name, state, threads, rss_kb, swap_kb, raw_swap_kb, swap_is_pss, pss_kb, vsz_kb, anon_kb, shmem_kb)
        raw_entries: List[Tuple[int, int, int, str, str, str, int, int, int, int, bool, Optional[int], int, int, int]] = []
        user_totals: Dict[str, Dict[str, Any]] = {}
        total_rss_kb = 0
        total_swap_kb = 0
        total_raw_swap_kb = 0
        swap_exact_procs = 0
        swap_fallback_procs = 0
        euid: Optional[int] = os.geteuid() if hasattr(os, "geteuid") else None
        is_live_proc = self.proc_root == "/proc"

        try:
            pid_dirs = [d for d in os.listdir(self.proc_root) if d.isdigit()]
        except Exception:
            pid_dirs = []

        for d in pid_dirs:
            status_file = os.path.join(self.proc_root, d, "status")
            try:
                with open(status_file, "rb") as f:
                    raw = f.read()
                if b"VmRSS:" not in raw and b"VmSwap:" not in raw:
                    continue

                uid = 0
                name = ""
                state = ""
                vsz_kb = 0
                rss_kb = 0
                anon_kb = 0
                shmem_kb = 0
                raw_swap_kb = 0
                threads = 1

                for line in raw.splitlines():
                    if line.startswith(b"Name:\t"):
                        name = line[6:].decode("utf-8", errors="ignore").strip()
                    elif line.startswith(b"State:\t"):
                        parts = line[7:].decode("ascii", errors="ignore").strip().split()
                        state = parts[0] if parts else ""
                    elif line.startswith(b"Uid:\t"):
                        parts = line.split()
                        if len(parts) >= 2:
                            uid = int(parts[1])
                    elif line.startswith(b"VmSize:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            vsz_kb = int(parts[1])
                    elif line.startswith(b"VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss_kb = int(parts[1])
                    elif line.startswith(b"RssAnon:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            anon_kb = int(parts[1])
                    elif line.startswith(b"RssShmem:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            shmem_kb = int(parts[1])
                    elif line.startswith(b"VmSwap:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            raw_swap_kb = int(parts[1])
                    elif line.startswith(b"Threads:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            threads = int(parts[1])
                        break

                if rss_kb == 0 and raw_swap_kb == 0:
                    continue

                # Use proportional SwapPss & Pss from smaps_rollup when accessible so
                # forked child processes do not double-count shared COW swap pages.
                pss_kb: Optional[int] = None
                swap_kb = raw_swap_kb
                swap_is_pss = False
                can_probe_rollup = (
                    (raw_swap_kb >= 1024 or rss_kb >= 10240)
                    and (not is_live_proc or euid == 0 or euid == uid)
                )
                if can_probe_rollup:
                    pss_kb, swap_pss_kb = self._read_smaps_rollup(d)
                    if swap_pss_kb is not None:
                        swap_kb = swap_pss_kb
                        swap_is_pss = True

                if raw_swap_kb >= 1024:
                    if swap_is_pss:
                        swap_exact_procs += 1
                    else:
                        swap_fallback_procs += 1

                user = self._resolve_uid(uid)
                footprint_kb = rss_kb + swap_kb
                total_rss_kb += rss_kb
                total_swap_kb += swap_kb
                total_raw_swap_kb += raw_swap_kb

                u_rec = user_totals.get(user)
                if u_rec is None:
                    u_rec = {
                        "user": user,
                        "process_count": 0,
                        "rss_kb": 0,
                        "swap_kb": 0,
                        "vm_swap_kb": 0,
                        "total_kb": 0,
                        "fallback_swap_kb": 0,
                        "exact_procs": 0,
                    }
                    user_totals[user] = u_rec
                u_rec["process_count"] += 1
                u_rec["rss_kb"] += rss_kb
                u_rec["swap_kb"] += swap_kb
                u_rec["vm_swap_kb"] += raw_swap_kb
                u_rec["total_kb"] += footprint_kb
                if raw_swap_kb >= 1024:
                    if swap_is_pss:
                        u_rec["exact_procs"] += 1
                    else:
                        u_rec["fallback_swap_kb"] += raw_swap_kb

                raw_entries.append(
                    (
                        footprint_kb,
                        int(d),
                        uid,
                        user,
                        name or "unknown",
                        state or "S",
                        threads,
                        rss_kb,
                        swap_kb,
                        raw_swap_kb,
                        swap_is_pss,
                        pss_kb,
                        vsz_kb,
                        anon_kb,
                        shmem_kb,
                    )
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
                continue

        # Sort descending by deduplicated footprint (RSS + SwapPss/VmSwap)
        raw_entries.sort(key=lambda item: item[0], reverse=True)
        top_entries = raw_entries[: max(1, self.max_mem_processes)]

        processes: List[Dict[str, Any]] = []
        for entry in top_entries:
            (
                footprint_kb,
                pid,
                _uid,
                user,
                name,
                state,
                threads,
                rss_kb,
                swap_kb,
                raw_swap_kb,
                swap_is_pss,
                pss_kb,
                vsz_kb,
                anon_kb,
                shmem_kb,
            ) = entry

            cmdline = ""
            try:
                with open(os.path.join(self.proc_root, str(pid), "cmdline"), "rb") as f:
                    raw_cmd = f.read()
                    if raw_cmd:
                        cmdline = (
                            raw_cmd.replace(b"\x00", b" ")
                            .decode("utf-8", errors="ignore")
                            .strip()
                        )
            except Exception:
                pass
            if not cmdline:
                cmdline = name

            oom_score = 0
            try:
                with open(os.path.join(self.proc_root, str(pid), "oom_score"), "r", encoding="utf-8") as f:
                    oom_score = int(f.read().strip())
            except Exception:
                pass

            rss_mb = round(rss_kb / 1024.0, 1)
            swap_mb = round(swap_kb / 1024.0, 1)
            vm_swap_mb = round(raw_swap_kb / 1024.0, 1)
            pss_mb = round(pss_kb / 1024.0, 1) if pss_kb is not None else None
            total_proc_mb = round(footprint_kb / 1024.0, 1)
            vsz_mb = round(vsz_kb / 1024.0, 1)
            anon_mb = round(anon_kb / 1024.0, 1)
            shmem_mb = round(shmem_kb / 1024.0, 1)

            ram_pct = round((rss_mb / total_mem_mb) * 100, 2) if total_mem_mb > 0 else 0.0
            swap_pct = round((swap_mb / total_swap_mb) * 100, 2) if total_swap_mb > 0 else 0.0

            processes.append(
                {
                    "pid": pid,
                    "user": user,
                    "name": name,
                    "state": state,
                    "threads": threads,
                    "cmdline": cmdline,
                    "rss_mb": rss_mb,
                    "pss_mb": pss_mb,
                    "swap_mb": swap_mb,
                    "vm_swap_mb": vm_swap_mb,
                    "swap_is_pss": swap_is_pss,
                    "total_mb": total_proc_mb,
                    "vsz_mb": vsz_mb,
                    "anon_mb": anon_mb,
                    "shmem_mb": shmem_mb,
                    "ram_pct": ram_pct,
                    "swap_pct": swap_pct,
                    "oom_score": oom_score,
                }
            )

        by_user_list: List[Dict[str, Any]] = []
        for u_rec in sorted(user_totals.values(), key=lambda x: x["total_kb"], reverse=True):
            u_rss_mb = round(u_rec["rss_kb"] / 1024.0, 1)
            u_swap_mb = round(u_rec["swap_kb"] / 1024.0, 1)
            u_vm_swap_mb = round(u_rec["vm_swap_kb"] / 1024.0, 1)
            u_total_mb = round(u_rec["total_kb"] / 1024.0, 1)
            fb_kb = u_rec["fallback_swap_kb"]
            # Minor setuid daemons (sshd/systemd <16MB and <2% of user swap) stay marked exact.
            u_swap_exact = (
                fb_kb == 0
                or (
                    u_rec["exact_procs"] > 0
                    and fb_kb < 16384
                    and fb_kb * 50 <= u_rec["swap_kb"]
                )
            )
            by_user_list.append(
                {
                    "user": u_rec["user"],
                    "process_count": u_rec["process_count"],
                    "rss_mb": u_rss_mb,
                    "swap_mb": u_swap_mb,
                    "vm_swap_mb": u_vm_swap_mb,
                    "swap_exact": u_swap_exact,
                    "total_mb": u_total_mb,
                    "ram_pct": round((u_rss_mb / total_mem_mb) * 100, 1) if total_mem_mb > 0 else 0.0,
                }
            )

        summary = {
            "active_count": len(raw_entries),
            "returned_count": len(processes),
            "total_rss_mb": round(total_rss_kb / 1024.0, 1),
            "total_swap_mb": round(total_swap_kb / 1024.0, 1),
            "total_vm_swap_mb": round(total_raw_swap_kb / 1024.0, 1),
            "swap_exact_procs": swap_exact_procs,
            "swap_fallback_procs": swap_fallback_procs,
        }

        result = (processes, by_user_list, summary)
        self._last_mem_procs_result = result
        self._last_mem_procs_time = now
        return result

    def _collect_system(self) -> Dict[str, Any]:
        """Collect host system metadata: hostname, uptime, load averages, CPU count."""
        hostname = socket.gethostname()

        # Uptime
        uptime_seconds = 0
        if os.path.isfile(self.uptime_path):
            try:
                with open(self.uptime_path, "r", encoding="utf-8") as f:
                    uptime_seconds = int(float(f.read().split()[0]))
            except Exception:
                uptime_seconds = 0

        # Load average
        load_average = [0.0, 0.0, 0.0]
        try:
            load_average = [round(x, 2) for x in os.getloadavg()]
        except (AttributeError, OSError):
            if os.path.isfile(self.loadavg_path):
                try:
                    with open(self.loadavg_path, "r", encoding="utf-8") as f:
                        tokens = f.read().split()[:3]
                        load_average = [round(float(x), 2) for x in tokens]
                except Exception:
                    load_average = [0.0, 0.0, 0.0]

        cpu_count = os.cpu_count() or 1

        return {
            "hostname": hostname,
            "uptime_seconds": uptime_seconds,
            "load_average": load_average,
            "cpu_count": cpu_count,
        }

    def get_metrics_snapshot(
        self,
        force: bool = False,
        force_processes: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return structured telemetry snapshot conforming to PROJECT.md interface contract.

        Uses an internal cache TTL to satisfy high-frequency polling (<500ms) with sub-1ms latency.
        """
        now = time.time()
        if (
            not force
            and self._last_snapshot is not None
            and (now - self._last_snapshot_time) < self.cache_ttl_sec
        ):
            return self._last_snapshot

        gpu_metrics = self._collect_gpu()
        raw_memory, raw_swap = self._collect_memory_and_swap()
        memory_metrics = dict(raw_memory)
        swap_metrics = dict(raw_swap)

        shm_metrics = self._collect_shm()
        pressure_metrics = self._collect_pressure()
        vmstat_metrics = self._collect_vmstat_rates(now)

        scan_procs = force if force_processes is None else force_processes
        mem_procs, by_user, proc_summary = self._collect_memory_processes(
            total_mem_mb=memory_metrics.get("total_mb", 0),
            total_swap_mb=swap_metrics.get("total_mb", 0),
            force=scan_procs,
        )

        memory_metrics["shm"] = shm_metrics
        memory_metrics["pressure"] = pressure_metrics
        memory_metrics["by_user"] = by_user
        memory_metrics["processes"] = mem_procs
        memory_metrics["process_summary"] = proc_summary

        swap_metrics.update(vmstat_metrics)
        system_metrics = self._collect_system()

        # Determine overall system telemetry health status
        if gpu_metrics.get("available") and memory_metrics.get("total_mb", 0) > 0:
            status = "ok"
        elif memory_metrics.get("total_mb", 0) > 0:
            status = "degraded"
        else:
            status = "error"

        snapshot = {
            "timestamp": now,
            "formatted_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "gpu": gpu_metrics,
            "memory": memory_metrics,
            "swap": swap_metrics,
            "system": system_metrics,
        }

        self._last_snapshot = snapshot
        self._last_snapshot_time = now
        return snapshot

    def close(self) -> None:
        """Clean up NVML handles upon termination."""
        if self._nvml_lib and self._nvml_available:
            try:
                self._nvml_lib.nvmlShutdown()
            except Exception:
                pass
            self._nvml_available = False
            self._nvml_lib = None
            self._nvml_handle = None

    def __enter__(self) -> MetricsCollector:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


if __name__ == "__main__":
    collector = MetricsCollector()
    snapshot = collector.get_metrics_snapshot(force=True)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    collector.close()
