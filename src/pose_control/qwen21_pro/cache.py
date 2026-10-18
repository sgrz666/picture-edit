"""Pro 独立的只增缓存；只保存冻结 raw features 或完整 typed condition。"""
from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType

import torch

SCHEMA = "qwen21_pro_raw_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_sha256(value):
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _payload_digest(value):
    def digest(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_sha256": tensor_sha256(item)}
        if isinstance(item, Mapping):
            return {key: digest(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [digest(child) for child in item]
        return item
    return stable_hash(digest(value))


def _record_binding(record):
    # Supervision fields and old target appearance hashes are deliberately absent.
    keys = ("sample_id", "dataset", "split", "group_id", "source_rgb", "reference_pose_images",
            "control_image", "prompt", "person_order", "source_transform", "target_transform", "transforms")
    bound = {key: record[key] for key in keys if key in record}
    metadata = record.get("metadata", {})
    bound["source_and_condition_declarations"] = {key: metadata[key] for key in
             ("original_source_rgb", "original_source_masks", "original_condition_assets") if key in metadata}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, ensure_ascii=False, default=str,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, init=False)
class BoundUnionCondition(Mapping):
    """只由 verified loader 构造的 B payload 快照；禁止给任意 dict 盖章。

    payload 顶层只读，tensor 可供原生 pipeline 使用。prepare/extract 每次仍重算
    source 段及完整 payload 摘要，任何载入后原地修改都会使 binding 失效。
    """
    _payload: dict
    _sample_id: str
    _record_digest: str
    _condition_fingerprint: str
    _prepared_source_sha256: str
    _source_digest: str
    _source_count: int
    _payload_digest: str
    _producer_json: str
    _dtype: str
    _file_payload_sha256: str

    def __init__(self, *args, **kwargs):
        raise TypeError("BoundUnionCondition must be created by load_bound_condition from verified cache files")

    def __getitem__(self, key):
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)

    @property
    def payload(self):
        return MappingProxyType(self._payload)

    @property
    def producer_signature(self):
        """冻结 producer 模型签名与 VAE mean/std/zdim；无 pose/request hash。"""
        return json.loads(self._producer_json)

    @property
    def attestation(self):
        return MappingProxyType({"sample_id": self._sample_id, "condition_fingerprint": self._condition_fingerprint,
                    "payload_sha256": self._file_payload_sha256, "prepared_source_sha256": self._prepared_source_sha256,
                    "source_latent_sha256": self._source_digest, "producer_signature": self.producer_signature})

    def validate_binding(self, condition):
        from ..qwen21_union.cache import condition_fingerprint
        record = condition.to_dict() if hasattr(condition, "to_dict") else dict(condition)
        try:
            if str(record["sample_id"]) != self._sample_id or _record_binding(record) != self._record_digest:
                raise ValueError("verified condition binding mismatch: sample or locked record changed")
            if file_sha256(record["source_rgb"]) != self._prepared_source_sha256:
                raise ValueError("verified condition binding mismatch: prepared source bytes changed")
            expected = condition_fingerprint(record, self.producer_signature, self._dtype)
            if expected != self._condition_fingerprint:
                raise ValueError("verified condition binding mismatch: condition asset bytes changed")
            latent = self._payload["reference_latents"]
            if tensor_sha256(latent[:, :self._source_count]) != self._source_digest:
                raise ValueError("verified condition binding mismatch: loaded source VAE changed")
            if _payload_digest(self._payload) != self._payload_digest:
                raise ValueError("verified condition binding mismatch: loaded payload changed")
        except (OSError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            raise ValueError("verified condition binding mismatch: unreadable or malformed bound inputs") from exc
        return self


def load_bound_condition(condition, cache_root, *, model_signature, vae_config, dtype="torch.bfloat16"):
    """只读旧 B cache，经 expected fingerprint + 原 payload SHA 校验后绑定。

    model_signature 必须由调用者提供冻结的 encoder producer 签名；生产 B 使用
    completed_20261016T184014.json 中的签名。运行时训练 determinism flags 不能
    替代 producer。Task7 负责冻结事件 SHA、核对当前权重，旧 index/payload 不改。
    """
    from ..qwen21_union.cache import (UnionConditionCache, cache_model_signature,
                                      expected_condition_fingerprint, load_condition)
    record = condition.to_dict() if hasattr(condition, "to_dict") else dict(condition)
    dtype = str(dtype)
    expected = expected_condition_fingerprint(record, model_signature, vae_config, dtype)
    cache = UnionConditionCache(cache_root)
    meta = cache.metadata(record["sample_id"])
    if str(meta.get("sample_id")) != str(record["sample_id"]):
        raise ValueError("verified cache sample binding mismatch")
    payload = load_condition(cache_root, record["sample_id"], expected_hash=expected)
    if cache.metadata(record["sample_id"]) != meta:
        raise ValueError("verified cache binding mismatch: index changed during load")
    path = Path(cache_root) / "conditions" / f"{expected}.pt"
    if file_sha256(path) != meta["payload_sha256"]:
        raise ValueError("verified cache payload SHA256 mismatch")
    latent = payload.get("reference_latents")
    hw = payload.get("target_hw")
    if (not isinstance(latent, torch.Tensor) or latent.ndim != 3 or latent.shape[0] != 1
            or latent.shape[-1] != 64 or str(latent.dtype) != dtype
            or not isinstance(hw, (tuple, list)) or len(hw) != 2
            or any(type(value) is not int or value <= 0 for value in hw)):
        raise ValueError("verified source VAE dtype or cached grid is invalid")
    n = hw[0] * hw[1]
    if latent.shape[1] < n:
        raise ValueError("verified source VAE slice is incomplete")
    producer_json = json.dumps(cache_model_signature(model_signature, vae_config), sort_keys=True,
                               ensure_ascii=False, default=str, separators=(",", ":"), allow_nan=False)
    instance = object.__new__(BoundUnionCondition)
    values = {"_payload": payload, "_sample_id": str(record["sample_id"]), "_record_digest": _record_binding(record),
              "_condition_fingerprint": expected, "_prepared_source_sha256": file_sha256(record["source_rgb"]),
              "_source_digest": tensor_sha256(latent[:, :n]), "_source_count": n,
              "_payload_digest": _payload_digest(payload), "_producer_json": producer_json, "_dtype": dtype,
              "_file_payload_sha256": meta["payload_sha256"]}
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance.validate_binding(record)


def ordinary_cpu(value):
    """inference_mode 内也返回可参与后续 autograd 的普通 CPU tensor。"""
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("cache accepts frozen raw tensors only")
        with torch.inference_mode(False):
            return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: ordinary_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(ordinary_cpu(item) for item in value)
    return value


def asset_hashes(assets):
    """角色名作键，不把含 target RGB 路径的旧 metadata 转储到 provenance。"""
    result = {}
    for role, value in sorted(assets.items()):
        if value is None:
            result[role] = None
        elif isinstance(value, (dict, list, tuple, bool, int, float)):
            result[role] = stable_hash(value)
        elif isinstance(value, torch.Tensor):
            result[role] = tensor_sha256(value)
        elif hasattr(value, "tobytes"):
            digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
            digest.update(value.tobytes())
            result[role] = digest.hexdigest()
        else:
            path = Path(value)
            result[role] = file_sha256(path) if path.is_file() else None
    return result


def code_signature():
    root = Path(__file__).resolve().parents[3]
    # Source/target ROI share the parser/threshold/coordinate transform below.
    # Their bytes are algorithm inputs too; stable repository-relative roles
    # keep fingerprints independent of local worktree versus remote snapshot paths.
    dependencies = ("src/pose_control/qwen21_pro/cache.py", "src/pose_control/qwen21_pro/features.py",
                    "src/pose_control/qwen21_pro/preparation.py", "src/data/qwen21_union_dataset.py",
                    "src/pose_control/qwen21_union/contracts.py")
    return {name: file_sha256(root / name) for name in dependencies}


def source_feature_fingerprint(*, source_assets, model_signature, person_order,
                               source_transform, source_latent, extraction=None):
    """只读源资产字节和第一段源 VAE；目标几何、pose refs 不参与此指纹。"""
    return stable_hash({"schema": SCHEMA, "code": code_signature(), "model": model_signature,
                        "assets": asset_hashes(source_assets), "person_order": list(person_order),
                        "source_transform": source_transform, "source_vae": tensor_sha256(source_latent),
                        "extraction": extraction or {"profile": "vae64_baseline"}})


def typed_pack_fingerprint(*, source_fingerprint, geometry_assets, condition, target_hw,
                           config, geometry_codec):
    """目标外观无读取路径；typed pack 另绑定请求几何、transform 和模型配置。"""
    record = condition.to_dict() if hasattr(condition, "to_dict") else condition
    bound = {key: record[key] for key in ("dataset", "prompt", "person_order", "target_transform") if key in record}
    refs = {f"pose_{i}": path for i, path in enumerate(record.get("reference_pose_images", []))}
    if record.get("control_image"):
        refs["control"] = record["control_image"]
    return stable_hash({"schema": SCHEMA, "code": code_signature(), "source": source_fingerprint,
                        "geometry": asset_hashes(geometry_assets), "condition": bound,
                        "request_assets": asset_hashes(refs), "target_hw": list(target_hw),
                        "config": config.to_dict(), "geometry_codec": geometry_codec})


class ProCache:
    """root 必须由 runner 显式给出，例如 /data/.../qwen21_pro_adapter_v1/cache。

    两个 namespace 均与 B lock/conditions/supervision 独立。已有 sample 的配置或
    内容变化需要新 cache root/sample_id，避免把上一轮结果静默冲掉。
    """

    def __init__(self, root, *, namespace):
        root = Path(root).resolve()
        if namespace not in {"raw_source", "typed_condition"}:
            raise ValueError("unknown Pro cache namespace")
        if any(part.lower() in {"datasets", "models", "conditions", "supervision"} for part in root.parts):
            raise ValueError("Pro cache cannot write protected or legacy directories")
        self.namespace = namespace
        self.root = root / "qwen21_pro_v1" / namespace

    def _index(self, sample_id):
        return self.root / "index" / f"{stable_hash(str(sample_id))}.json"

    def metadata(self, sample_id):
        meta = json.loads(self._index(sample_id).read_text(encoding="utf-8"))
        if (meta.get("schema") != SCHEMA or meta.get("namespace") != self.namespace
                or meta.get("sample_id") != str(sample_id)
                or not re.fullmatch("[a-f0-9]{64}", meta.get("hash", ""))):
            raise ValueError("cache index schema or binding mismatch")
        return meta

    def _validate_payload(self, payload, config=None, signature=None):
        if not isinstance(payload, dict):
            raise ValueError("Pro cache payload must be structured raw features or typed condition")
        if any("resampl" in str(key).lower() for key in payload):
            raise ValueError("trainable resampler outputs cannot enter frozen cache")
        if self.namespace == "raw_source":
            from .features import RawSourceFeatures
            expected = {"person_ids", "tokens", "valid", "face_identity", "face_identity_valid", "fingerprint", "provenance"}
            if set(payload) != expected:
                raise ValueError("raw_source cache requires complete RawSourceFeatures schema")
            RawSourceFeatures.from_dict(payload)
        else:
            from .contracts import TaskCondition
            if config is None:
                raise ValueError("typed condition cache requires the actual adapter config")
            if config.signature() != signature:
                raise ValueError("typed condition config signature mismatch")
            TaskCondition.from_dict(payload).validate(config)
        return payload

    def get(self, sample_id, *, expected_hash, signature, config=None):
        meta = self.metadata(sample_id)
        if meta["hash"] != expected_hash:
            raise ValueError("cache fingerprint mismatch (stale inputs)")
        if meta["signature"] != signature:
            raise ValueError("cache signature mismatch (stale configuration)")
        path = self.root / f"{meta['hash']}.pt"
        if file_sha256(path) != meta["payload_sha256"]:
            raise ValueError("cache payload SHA256 mismatch")
        payload = ordinary_cpu(torch.load(path, map_location="cpu", weights_only=True))
        return self._validate_payload(payload, config, signature)

    def put(self, sample_id, payload, fingerprint, *, signature, config=None):
        if not re.fullmatch("[a-f0-9]{64}", fingerprint):
            raise ValueError("fingerprint must be SHA256")
        index = self._index(sample_id)
        if index.exists():
            meta = self.metadata(sample_id)
            if meta["hash"] != fingerprint or meta["signature"] != signature:
                raise FileExistsError("sample already cached; use a new cache root or sample_id")
            self.get(sample_id, expected_hash=fingerprint, signature=signature, config=config)
            return meta
        frozen = ordinary_cpu(payload.to_dict() if hasattr(payload, "to_dict") else payload)
        self._validate_payload(frozen, config, signature)
        stream = io.BytesIO()
        torch.save(frozen, stream)
        data = stream.getvalue()
        payload_sha = hashlib.sha256(data).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{fingerprint}.pt"
        try:
            with path.open("xb") as output:
                output.write(data)
        except FileExistsError:
            if file_sha256(path) != payload_sha:
                raise ValueError("existing cache payload differs; refusing overwrite")
        meta = {"schema": SCHEMA, "namespace": self.namespace, "sample_id": str(sample_id),
                "hash": fingerprint, "signature": signature, "path": str(path), "payload_sha256": payload_sha}
        index.parent.mkdir(parents=True, exist_ok=True)
        with index.open("x", encoding="utf-8") as output:
            output.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        return meta
