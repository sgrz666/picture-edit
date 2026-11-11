"""Sample host-wide GPU/RAM plus this process, without touching other jobs."""
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from core import write_json


def memory_kib(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            result[fields[0].rstrip(':')] = int(fields[1])
    return result


class ResourceMonitor(AbstractContextManager):
    def __init__(self, config, report_path):
        self.config, self.report_path = config, Path(report_path)
        self.stop_event = threading.Event()
        self.violation = None
        self.samples = []
        self.thread = None

    def sample(self):
        info = memory_kib('/proc/meminfo')
        own = memory_kib('/proc/self/status')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = self.config['nvml_compat'] + ':' + env.get('LD_LIBRARY_PATH', '')
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits'], env=env, text=True, timeout=10)
        used, free, util = [float(v.strip()) for v in value.splitlines()[0].split(',')]
        return {'time': time.time(), 'pid': os.getpid(), 'gpu_used_mib': used, 'gpu_free_mib': free,
                'gpu_utilization': util, 'ram_available_mib': info['MemAvailable'] / 1024,
                'rss_mib': own.get('VmRSS', 0) / 1024, 'swap_mib': own.get('VmSwap', 0) / 1024,
                'host_swap_used_mib': (info['SwapTotal'] - info['SwapFree']) / 1024}

    def tick(self):
        record = self.sample()
        self.samples.append(record)
        if record['gpu_free_mib'] < self.config['min_free_gpu_mib']:
            self.violation = f'GPU free {record["gpu_free_mib"]} MiB below configured reserve'
        if record['ram_available_mib'] < self.config['min_available_ram_mib']:
            self.violation = f'Available RAM {record["ram_available_mib"]} MiB below configured reserve'

    def check(self):
        if self.violation:
            raise RuntimeError(self.violation)

    def __enter__(self):
        self.tick()
        self.check()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self.stop_event.wait(2):
            try:
                self.tick()
            except Exception as error:
                self.violation = f'Resource monitoring failed: {error}'
                break

    def __exit__(self, kind, error, traceback):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=15)
        try:
            self.tick()
        except Exception:
            pass
        summary = {'pid': os.getpid(), 'samples': len(self.samples), 'violation': self.violation,
                   'error': str(error) if error else None, 'records': self.samples}
        if self.samples:
            summary.update({'peak_host_gpu_used_mib': max(s['gpu_used_mib'] for s in self.samples),
                            'min_gpu_free_mib': min(s['gpu_free_mib'] for s in self.samples),
                            'peak_process_rss_mib': max(s['rss_mib'] for s in self.samples),
                            'min_ram_available_mib': min(s['ram_available_mib'] for s in self.samples)})
        try:
            import torch
            if torch.cuda.is_initialized():
                summary['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
                summary['peak_cuda_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        except ImportError:
            pass
        write_json(self.report_path, summary)
        return False
