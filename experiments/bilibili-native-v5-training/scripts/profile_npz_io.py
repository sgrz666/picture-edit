"""Read one existing Body NPZ and profile lossless serialization entirely in RAM.

The only file created is --report (exclusive creation). No torch/model/CUDA import,
archive replacement, dtype conversion, cache eviction or training is performed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile
import zlib

import numpy as np


def measured(function):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    return value, {"wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_paths(header):
    result = defaultdict(list)

    def visit(node, path):
        if not isinstance(node, dict):
            return
        if "array" in node:
            result[node["array"]].append(path)
        elif "record" in node:
            for key, value in node["fields"].items():
                visit(value, path + "." + key)
        elif "dict" in node:
            for key, value in node["dict"].items():
                visit(value, path + "." + key)
        else:
            for kind in ("list", "tuple"):
                if kind in node:
                    for index, value in enumerate(node[kind]):
                        visit(value, path + "[" + str(index) + "]")
    visit(header["tree"], "sample")
    return dict(result)


def raw_deflate(payload, level):
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
    return compressor.compress(payload) + compressor.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.report.exists():
        raise FileExistsError("Use a new report path; existing reports are never overwritten")
    source = args.input.resolve(strict=True)
    before = source.stat()
    blob, read_time = measured(source.read_bytes)
    digest, archive_sha_time = measured(lambda: sha(blob))
    disk_digest, disk_sha_time = measured(lambda: file_sha(source))
    if digest != disk_digest:
        raise ValueError("Source archive changed while being read")
    print(json.dumps({"event": "read_existing_archive", "bytes": len(blob), "read": read_time}), flush=True)

    arrays, payload_hashes, members = {}, {}, []
    parse_wall = parse_cpu = inflate_wall = inflate_cpu = 0.
    serialization_wall = serialization_cpu = member_sha_wall = member_sha_cpu = 0.
    levels = {str(level): {"wall_s": 0., "cpu_s": 0., "compressed_bytes": 0} for level in (1, 3, 6)}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                raise ValueError("Unexpected non-NPY member: " + info.filename)
            payload, inflate_time = measured(lambda: archive.read(info))
            array, parse_time = measured(lambda: np.load(io.BytesIO(payload), allow_pickle=False))
            key = info.filename[:-4]
            arrays[key] = array
            payload_digest, hash_time = measured(lambda: sha(payload))
            payload_hashes[info.filename] = payload_digest

            def serialize():
                output = io.BytesIO()
                np.lib.format.write_array(output, array, allow_pickle=False)
                return output.getvalue()
            serialized, serialize_time = measured(serialize)
            if serialized != payload:
                raise ValueError("NPY serialization is not byte-identical: " + info.filename)
            compression = {}
            for level in (1, 3, 6):
                compressed, duration = measured(lambda: raw_deflate(serialized, level))
                if zlib.decompress(compressed, -15) != payload:
                    raise ValueError("Lossless DEFLATE round-trip failed")
                compression[str(level)] = {**duration, "bytes": len(compressed)}
                for field in ("wall_s", "cpu_s"):
                    levels[str(level)][field] += duration[field]
                levels[str(level)]["compressed_bytes"] += len(compressed)
            inflate_wall += inflate_time["wall_s"]
            inflate_cpu += inflate_time["cpu_s"]
            parse_wall += parse_time["wall_s"]
            parse_cpu += parse_time["cpu_s"]
            serialization_wall += serialize_time["wall_s"]
            serialization_cpu += serialize_time["cpu_s"]
            member_sha_wall += hash_time["wall_s"]
            member_sha_cpu += hash_time["cpu_s"]
            members.append({"member": info.filename, "shape": list(array.shape), "dtype": str(array.dtype),
                "array_bytes": array.nbytes, "npy_bytes": info.file_size, "compressed_bytes": info.compress_size,
                "zip_method": info.compress_type, "npy_sha256": payload_digest,
                "inflate_from_memory": inflate_time, "parse_npy": parse_time,
                "serialize_npy": serialize_time, "sha_npy": hash_time, "deflate": compression})

    header = json.loads(str(arrays["record_json"]))
    paths = array_paths(header)
    groups = defaultdict(list)
    for member in members:
        key = member["member"][:-4]
        member["record_paths"] = paths.get(key, ["record_json"] if key == "record_json" else [])
        groups[member["npy_sha256"]].append(member["member"])
    print(json.dumps({"event": "members_profiled", "members": len(members), "deflate": levels}), flush=True)

    whole = []
    for iteration in range(args.repeat):
        output = io.BytesIO()
        _, duration = measured(lambda: np.savez_compressed(output, **arrays))
        compressed_blob = output.getvalue()
        with zipfile.ZipFile(io.BytesIO(compressed_blob)) as rebuilt:
            if {item.filename for item in rebuilt.infolist()} != set(payload_hashes):
                raise ValueError("Whole NPZ membership differs")
            for name, expected in payload_hashes.items():
                if sha(rebuilt.read(name)) != expected:
                    raise ValueError("Whole NPZ member SHA differs")
        whole.append({"iteration": iteration + 1, **duration, "bytes": len(compressed_blob),
                      "all_member_sha256_identical": True})

    def load_npz():
        with np.load(io.BytesIO(blob), allow_pickle=False) as existing:
            return {key: existing[key] for key in existing.files}
    loaded, full_load_time = measured(load_npz)
    if any(not np.array_equal(loaded[key], value, equal_nan=True)
           if np.issubdtype(value.dtype, np.number) else not np.array_equal(loaded[key], value)
           for key, value in arrays.items()):
        raise ValueError("Memory NPZ loading changed array contents")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source archive changed during profiling")

    duplicate_groups = [names for names in groups.values() if len(names) > 1]
    redundant_members = {name for names in duplicate_groups for name in names[1:]}
    report = {"schema": "npz_io_readonly_profile_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": digest, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                        "zlib": zlib.ZLIB_RUNTIME_VERSION, "max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "protocol": {"source_archives_loaded": 1, "source_changed": False, "body_files_created": 0,
            "all_recompression_outputs": "BytesIO only", "model_loaded": False, "gpu_used": False,
            "training_started": False, "dtype_changed": False, "fields_removed": False,
            "read_cache_state": "OS page-cache state is uncontrolled; no cold-disk claim",
            "deflate_levels": [1, 3, 6], "repetitions_np_savez_compressed": args.repeat},
        "timings": {"read_archive_bytes": read_time, "sha_archive_memory": archive_sha_time,
            "sha_archive_read_stream": disk_sha_time, "inflate_npy_from_memory": {"wall_s": inflate_wall, "cpu_s": inflate_cpu},
            "parse_npy": {"wall_s": parse_wall, "cpu_s": parse_cpu},
            "serialize_npy": {"wall_s": serialization_wall, "cpu_s": serialization_cpu},
            "sha_all_npy": {"wall_s": member_sha_wall, "cpu_s": member_sha_cpu},
            "np_load_all_from_memory": full_load_time, "np_savez_compressed": whole},
        "totals": {"members": len(members), "array_bytes": sum(m["array_bytes"] for m in members),
            "npy_bytes": sum(m["npy_bytes"] for m in members), "compressed_payload_bytes": sum(m["compressed_bytes"] for m in members),
            "zip_overhead_bytes": len(blob) - sum(m["compressed_bytes"] for m in members)},
        "deflate_levels": levels, "within_archive_duplicate_npy_groups": duplicate_groups,
        "redundant_identical_members": {"count": len(redundant_members),
            "npy_bytes": sum(m["npy_bytes"] for m in members if m["member"] in redundant_members),
            "level6_deflate_wall_s": sum(m["deflate"]["6"]["wall_s"] for m in members if m["member"] in redundant_members)},
        "members": members}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "totals": report["totals"], "timings": report["timings"],
                      "deflate_levels": levels, "duplicates": report["redundant_identical_members"]}), flush=True)


if __name__ == "__main__":
    main()
