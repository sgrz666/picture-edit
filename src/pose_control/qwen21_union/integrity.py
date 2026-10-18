"""Read-only, process-start verification of locked Union workspace inputs.

Successful verification is memoized for a read-only process. Call once before
loading inputs in each new GPU job. A process that intentionally changes any
input must call ``verify_workspace_inputs.cache_clear()`` before verifying again.
This module imports only the Python standard library and never reads caches.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re


def _sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _json_bound(path, expected):
    value = Path(path).read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError(f"integrity metadata changed during verification: {path}")
    return json.loads(value)


def _verify_mapping(mapping, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} SHA256 mapping is missing or invalid")
    for path, expected in mapping.items():
        if not isinstance(path, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"{label} SHA256 entry is invalid: {path}")
    def verify(item):
        path, expected = item
        try:
            actual = _sha256(path)
        except OSError as error:
            raise ValueError(f"{label} missing or unreadable: {path}: {error}") from error
        if actual != expected.lower():
            raise ValueError(f"{label} checksum mismatch: {path}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, sorted(mapping.items())))


def _expected_prepared(data, manifest_hashes):
    assets, count, counts = set(), 0, {}
    for split in ("train", "val", "test"):
        conditions = data / f"{split}_conditions.jsonl"
        supervision = data / f"{split}_supervision.jsonl"
        def rows(path):
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest_hashes[path.name]:
                raise ValueError(f"selected manifest checksum mismatch: {path}")
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        split_rows = rows(conditions)
        counts[split] = len(split_rows)
        count += len(split_rows)
        for row in split_rows:
            assets.update([row["source_rgb"], row["control_image"], *row["reference_pose_images"]])
        for row in rows(supervision):
            assets.update(row["requested_keypoints"])
    return assets, count, counts


@lru_cache(maxsize=4)
def _verify_cached(workspace, lock_sha256, registry_sha256):
    data = Path(workspace) / "data"
    lock = _json_bound(data / "lock.json", lock_sha256)
    registry = _json_bound(data / "prepared_assets_sha256.json", registry_sha256)
    if registry.get("lock_sha256") != lock_sha256:
        raise ValueError("prepared registry lock binding differs from actual data lock")
    if registry.get("protocol") != lock.get("protocol"):
        raise ValueError("prepared registry protocol differs from data lock")
    manifest_hashes = lock.get("selected_manifest_sha256")
    required_names = {f"{split}_{kind}.jsonl" for split in ("train", "val", "test", "overfit6")
                      for kind in ("conditions", "supervision")}
    if not isinstance(manifest_hashes, dict) or not required_names <= manifest_hashes.keys():
        raise ValueError("selected manifest inventory is incomplete")
    if any(Path(name).name != name or name in (".", "..") for name in manifest_hashes):
        raise ValueError("selected manifest inventory contains an unsafe filename")
    _verify_mapping({str(data / name): digest for name, digest in manifest_hashes.items()}, "selected manifest")
    assets, count, counts = _expected_prepared(data, manifest_hashes)
    registered = registry.get("asset_sha256")
    if not isinstance(registered, dict) or set(registered) != assets:
        missing = sorted(assets - set(registered or {}))
        extra = sorted(set(registered or {}) - assets)
        raise ValueError(f"prepared registry inventory differs from selected manifests: missing={missing[:3]}, extra={extra[:3]}")
    if registry.get("asset_count") != len(registered) or registry.get("condition_count") != count:
        raise ValueError("prepared registry inventory counts differ from selected manifests")
    if "counts" in lock and any(counts[split] != lock["counts"].get(split, 0) for split in counts):
        raise ValueError("selected manifest counts differ from data lock")
    originals = lock.get("required_asset_sha256")
    _verify_mapping(originals, "original asset")
    original_manifests = lock.get("manifest_sha256", {})
    _verify_mapping(original_manifests, "original manifest")
    _verify_mapping(registered, "prepared asset")
    # Metadata is checked again after the large read pass so the report cannot
    # combine old input hashes with newly replaced lock/registry files.
    _json_bound(data / "lock.json", lock_sha256)
    _json_bound(data / "prepared_assets_sha256.json", registry_sha256)
    return {"verified": True, "read_only": True, "workspace": workspace,
            "lock_sha256": lock_sha256, "prepared_registry_sha256": registry_sha256,
            "selected_manifest_count": len(manifest_hashes),
            "original_manifest_count": len(original_manifests),
            "original_asset_count": len(originals), "prepared_asset_count": len(registered),
            "condition_count": count, "counts": counts,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "memoization": "successful verification reused only within this read-only process"}


def verify_workspace_inputs(workspace) -> dict:
    """Verify lock, original assets and prepared bytes before a read-only run.

    Only successful checks are memoized. Small lock/registry fingerprints remain
    checked on every call; changing a metadata file invalidates the memoization.
    Asset edits inside the same process require explicit ``cache_clear()``.
    """
    workspace = str(Path(workspace).resolve())
    data = Path(workspace) / "data"
    lock_sha256 = _sha256(data / "lock.json")
    registry_sha256 = _sha256(data / "prepared_assets_sha256.json")
    report = _verify_cached(workspace, lock_sha256, registry_sha256)
    return json.loads(json.dumps(report))


verify_workspace_inputs.cache_clear = _verify_cached.cache_clear
verify_workspace_inputs.cache_info = _verify_cached.cache_info
