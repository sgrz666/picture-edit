"""Run preparation and the staged SFT independently of the login session."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from select_pairs import read, ref, write


def start_ticks(pid):
    try:
        state, *fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if state == "Z" else fields[18]


def wait_for(path, pid, ticks, boot_id):
    while start_ticks(pid) == ticks and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot_id:
        time.sleep(5)
    if not path.is_file():
        raise RuntimeError(f"Process {pid} exited without publishing {path}")
    read(path)


def available(gpu, needed_mib):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return int(raw.strip()) >= needed_mib


def wait_gpu(gpu, needed_mib):
    while not available(gpu, needed_mib):
        print(f"Waiting for GPU {gpu} with {needed_mib} MiB free", flush=True)
        time.sleep(30)


def main(args):
    root = args.root.resolve()
    scripts = root / "scripts"
    with (root / f"{args.mode}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        env = {**os.environ, "PYTHONPATH": str(args.repo), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        write(root / "logs" / f"{args.mode}-supervisor.json", {
            "pid": os.getpid(), "start_ticks": start_ticks(os.getpid()), "boot_id": boot,
            "source": str(args.repo), "selection": str(args.selection),
            "scripts": {path.name: ref(path) for path in scripts.glob("*.py")}})

        def cmd(file, *values):
            return [sys.executable, "-u", str(scripts / file), *map(str, values)]

        def gpu_env(gpu):
            return {**env, "CUDA_VISIBLE_DEVICES": str(gpu)}

        if args.mode == "smoke":
            work = root / "smoke-v2"
            wait_for(work / "native-ready.json", args.wait_pid, args.wait_ticks, boot)
            subprocess.run(cmd("qwen_cache.py", "finalize", "--root", root / "smoke"), check=True, env=env)
            wait_gpu(args.train_gpu, 79500)
            subprocess.run(cmd("train_sft.py", "--root", work, "--smoke"), check=True, env=gpu_env(args.train_gpu))
            return

        work = root / "production"
        wait_for(args.selection, args.wait_pid, args.wait_ticks, boot)
        selection = read(args.selection)
        print(selection["report"], flush=True)
        work.mkdir()
        subprocess.run(cmd("qwen_cache.py", "prepare", "--root", work, "--selection", args.selection),
                       check=True, env=env)
        wait_gpu(args.cache_gpu, 40000)
        wait_gpu(args.assets_gpu, 52000)
        with ExitStack() as stack:
            logs = [stack.enter_context((root / "logs" / f"production-{name}.log").open("x", encoding="utf-8"))
                    for name in ("qwen", "native")]
            qwen = cmd("qwen_cache.py", "encode", "--root", work)
            native = cmd("native_assets.py", "--root", work, "--selection", args.selection,
                         "--face-models", args.face_models, "--appearance-module", scripts / "appearance.py")
            processes = [subprocess.Popen(action, env=gpu_env(gpu), stdout=log, stderr=subprocess.STDOUT)
                         for action, gpu, log in zip((qwen, native), (args.cache_gpu, args.assets_gpu), logs)]
            write(root / "logs" / "production-preparation-pids.json", {
                name: {"pid": p.pid, "start_ticks": start_ticks(p.pid), "gpu": gpu}
                for name, p, gpu in zip(("qwen", "native"), processes, (args.cache_gpu, args.assets_gpu))})
            while any(p.poll() is None for p in processes):
                if any(p.returncode not in (None, 0) for p in processes):
                    raise RuntimeError("Production preparation failed. Check the independent Qwen and Native logs.")
                time.sleep(5)
            if any(p.returncode for p in processes):
                raise RuntimeError("Production preparation failed. Check the independent Qwen and Native logs.")
        subprocess.run(cmd("qwen_cache.py", "finalize", "--root", work), check=True, env=env)
        wait_gpu(args.train_gpu, 79500)
        subprocess.run(cmd("train_sft.py", "--root", work), check=True, env=gpu_env(args.train_gpu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "production"))
    for option in ("root", "repo", "selection"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--face-models", type=Path,
                        default=Path("/home/shangguanrz/project/pic-edit/models/antelopev2"))
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-ticks", required=True)
    parser.add_argument("--cache-gpu", type=int, default=1)
    parser.add_argument("--assets-gpu", type=int, default=2)
    parser.add_argument("--train-gpu", type=int, default=2)
    main(parser.parse_args())
