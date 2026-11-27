"""Parallelize Native assembly for an already locked XHS run with a finished Qwen cache."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from qwen_cache import INIT, INIT_SHA, MODEL_SHA, pinned
from run_pipeline import available, start_ticks, wait_gpu
from select_pairs import read, ref, write


def pair_key(pair):
    return pair["split"] + "\0" + pair["id"]


def partition_by_note(pairs, count):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["note"]].append(pair)
    shards = [[] for _ in range(count)]
    weights = [0] * count

    def cost(note_pairs):
        frames = {pair[role]["rgb"]["sha256"] for pair in note_pairs for role in ("source", "target")}
        return len(note_pairs) + 2 * len(frames)

    for group in sorted(groups.values(), key=cost, reverse=True):
        index = min(range(count), key=weights.__getitem__)
        shards[index].extend(group)
        weights[index] += cost(group)
    return [shard for shard in shards if shard]


def publish_cache(original, work, inputs):
    from src.pose_control.qwen21_full.native_v62.conditioning import CACHE_SCHEMA

    entries = read(original / "cache/shard-0/complete.json")
    condition_keys = {key: row["condition"]["sha256"] for key, row in inputs.items()}
    if condition_keys != {key: row["condition_sha256"] for key, row in entries.items()}:
        raise ValueError("The finished Qwen shard does not bind exactly the locked conditions")
    cache = work / "cache"
    cache.mkdir()
    for key, entry in entries.items():
        if any(entry[field] != inputs[key][field] for field in (
                "source_rgb_sha256", "driver_rgb_sha256", "source_canvas_id", "target_canvas_id")):
            raise ValueError(f"Frozen RGB/Canvas provenance differs for {key!r}")
        old = Path(entry["file"]["path"])
        if (original / "cache/shard-0").resolve() != old.resolve().parent:
            raise ValueError(f"Cached file is outside the verified Qwen shard: {old}")
        new = cache / old.name
        os.link(old, new)
        entry["file"] = {"path": str(new), "sha256": entry["file"]["sha256"]}
    write(cache / "index.json", {"schema": CACHE_SCHEMA, "profile": "dual_4ref",
          "model_manifest_sha256": MODEL_SHA, "condition_bindings": condition_keys, "entries": entries})


def assembled_prefix(original, pairs):
    count = 0
    for pair in pairs:
        if not (original / "native" / pair["id"] / "source-preservation.pt").is_file():
            break
        count += 1
    # The next pair can be in flight. Borrow only records followed by another completed record.
    return min(max(0, count - 1), len(pairs))


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def publish_native(work, selection, pairs, inputs, roots, statuses):
    from src.pose_control.qwen21_full.native_v7.preservation_assets import INDEX_SCHEMA

    if len(roots) != len(pairs) or len({pair_key(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Every locked pair must have exactly one Native asset directory")
    provenance = {k: v for k, v in statuses[0].items() if k not in ("selection", "splits")}
    if any({k: v for k, v in s.items() if k not in ("selection", "splits")} != provenance for s in statuses):
        raise ValueError("Shards disagree about Native model, evaluator or condition provenance")
    manifests, groups = defaultdict(list), defaultdict(dict)
    preserved = {}
    for pair in pairs:
        pid, split, key = pair["id"], pair["split"], pair_key(pair)
        directory = roots[key] / "native" / pid
        v4 = directory / "v4.jsonl"
        row = json.loads(v4.read_text(encoding="utf-8"))
        if row["pair_id"] != pid or row["source_rgb"] != pair["source"]["rgb"] or row["target_rgb"] != pair["target"]["rgb"]:
            raise ValueError(f"Native V4 pairing differs for {pid}")
        manifests[split].append({"schema": "native_full_pair_v5", "pair_id": pid,
                                 "v4_manifest": ref(v4), "detail": ref(directory / "detail.npz"),
                                 "supervision": ref(directory / "supervision.npz"),
                                 "qwen_condition": inputs[key]["condition"]})
        groups[split][pid] = pair["group"]
        preserved[key] = ref(directory / "source-preservation.pt")
    for split, rows in manifests.items():
        write_rows(work / f"{split}.jsonl", rows)
        write(work / f"{split}-groups.json", groups[split])
    write(work / "preservation.json", {"schema": INDEX_SCHEMA, "entries": preserved,
                                      "evaluator": provenance["identity_evaluator"]})
    write(work / "native-ready.json", {**provenance, "selection": ref(selection),
          "splits": {split: len(rows) for split, rows in manifests.items()}})


def handoff(root, original, work):
    old = read(root / "logs/production-supervisor.json")
    prepare = read(root / "logs/production-preparation-pids.json")["native"]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot != old["boot_id"]:
        raise RuntimeError("Original process identity belongs to a different boot")
    if (original / "native-ready.json").is_file() or (original / "sft").exists():
        return False
    if start_ticks(old["pid"]) == old["start_ticks"] and start_ticks(prepare["pid"]) == prepare["start_ticks"]:
        process = os.pidfd_open(prepare["pid"])
        try:
            if start_ticks(prepare["pid"]) != prepare["start_ticks"]:
                raise RuntimeError("Original Native worker changed during verification")
            signal.pidfd_send_signal(process, signal.SIGTERM)
        finally:
            os.close(process)
    else:
        if start_ticks(old["pid"]) == old["start_ticks"]:
            raise RuntimeError("Original supervisor is live without the expected Native worker")
    write(work / "handoff.json", {"old_supervisor": old["pid"], "old_native": prepare["pid"],
          "old_root": str(original), "selection": ref(root / "data/selection.json")})
    return True


def main(args):
    root = args.root.resolve()
    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    original, work = root / "production", root / f"production-parallel-v{args.attempt}"
    selection = root / "data/selection.json"
    pairs = read(selection)["pairs"]
    qwen = read(original / "qwen-inputs.json")
    if qwen["selection"] != ref(selection) or set(qwen["entries"]) != {pair_key(pair) for pair in pairs}:
        raise ValueError("Original Qwen inputs differ from the locked selection")
    if (original / "native-ready.json").exists() or (original / "sft").exists():
        raise RuntimeError("Original production supervisor has finished Native preparation")
    if not all(available(gpu, 56000) for gpu in (1, 2)):
        raise RuntimeError("Expected GPUs 1 and 2 to each have at least 56,000 MiB free for parallel assembly")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < args.workers * 6:
        raise RuntimeError("Too few CPUs for the requested independent assembly workers")
    work.mkdir()
    lock = (work / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pinned(INIT, INIT_SHA)
    publish_cache(original, work, qwen["entries"])
    prefix = assembled_prefix(original, pairs)
    roots = {pair_key(pair): original for pair in pairs[:prefix]}
    shards = partition_by_note(pairs[prefix:], args.workers)
    scripts = root / "scripts"
    env = {**os.environ, "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
    children = []
    for i, shard in enumerate(shards):
        gpu = 1 + i % 2
        directory = work / f"shard-{i}"
        shard_selection = directory / "selection.json"
        write(shard_selection, {"pairs": shard})
        entries = {pair_key(pair): qwen["entries"][pair_key(pair)] for pair in shard}
        write(directory / "qwen-inputs.json", {**qwen, "selection": ref(shard_selection), "entries": entries})
        roots.update({pair_key(pair): directory for pair in shard})
        cpu_list = ",".join(map(str, cpus[i * 6:i * 6 + 6]))
        with (directory / "native.log").open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(["taskset", "-c", cpu_list, sys.executable, "-u", str(scripts / "native_assets.py"),
                "--root", str(directory), "--selection", str(shard_selection), "--face-models",
                "/home/shangguanrz/project/pic-edit/models/antelopev2", "--appearance-module", str(scripts / "appearance.py")],
                env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=stream, stderr=subprocess.STDOUT)
        children.append((directory, child, gpu, len(shard)))
    write(work / "plan.json", {"pid": os.getpid(), "selection": ref(selection), "borrowed_pairs": prefix,
          "initialization": pinned(INIT, INIT_SHA), "shards": [{"directory": str(d), "pid": p.pid, "gpu": g, "pairs": n}
          for d, p, g, n in children]})
    print(f"Reusing {prefix} complete Native pairs and all {len(pairs)} Qwen cache entries", flush=True)
    last = None
    while any(child.poll() is None for _, child, _, _ in children):
        state = [child.poll() for _, child, _, _ in children]
        if state != last:
            print(f"Native shard exit status: {state}", flush=True)
            last = state
        time.sleep(10)
    failures = [str(directory) for directory, child, _, _ in children if child.returncode != 0]
    if failures:
        raise RuntimeError("Failed Native shards; original process is independent: " + ", ".join(failures))
    statuses = [read(directory / "native-ready.json") for directory, _, _, _ in children]
    publish_native(work, selection, pairs, qwen["entries"], roots, statuses)
    if not handoff(root, original, work):
        print("Original production run already ready; continuing it with the original supervisor", flush=True)
        return
    wait_gpu(1, 79500)
    subprocess.run([sys.executable, "-u", str(scripts / "train_sft.py"), "--root", str(work)],
                   env={**env, "CUDA_VISIBLE_DEVICES": "1"}, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(4, 6, 8), default=8)
    parser.add_argument("--attempt", type=int, required=True)
    main(parser.parse_args())
