"""固定 VideoX-Fun/Union 的惰性加载和局部残差注入范围。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
import subprocess
import sys
import types

VIDEOX_PIN = "13bb3c625e8b1e457d7124d92207b49e00e4a6fe"
UNION_REVISION_PIN = "8a4702014d4dabb5f896fcba917e2ee0a961465f"
UNION_SHA256_PIN = "65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd"
ASSET_ROOT = Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1")
DEFAULT_SOURCE = ASSET_ROOT / "source/VideoX-Fun"
DEFAULT_BASE = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
DEFAULT_UNION = ASSET_ROOT / "models/Qwen-Image-2.1-Fun-Controlnet-Union"
_FILE_HASH_CACHE = {}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_asset_signature(base, components=("vae", "text_encoder", "processor", "transformer", "scheduler")):
    """Hash actual local model/config bytes; reuse reads only while file stats match.

    The fingerprint is independent of its absolute location. No GPU/model imports
    or network access occur. A file changed while hashing is rejected.
    """
    base = Path(base)
    manifest = {}
    for component in components:
        root = base / component
        files = sorted(path for path in root.rglob("*") if path.is_file()
                       and ".cache" not in path.relative_to(root).parts
                       and "__pycache__" not in path.relative_to(root).parts)
        if not files:
            raise FileNotFoundError(f"missing model asset component: {root}")
        entries = []
        for path in files:
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            key = str(path.resolve())
            previous = _FILE_HASH_CACHE.get(key)
            if previous is None or previous[0] != identity:
                digest = sha256_file(path)
                after = path.stat()
                if identity != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                    raise RuntimeError(f"model asset changed while hashing: {path}")
                _FILE_HASH_CACHE[key] = (identity, digest)
            else:
                digest = previous[1]
            entries.append({"path": str(path.relative_to(base)).replace("\\", "/"),
                            "bytes": stat.st_size, "sha256": digest})
        manifest[component] = entries
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {"fingerprint": hashlib.sha256(encoded).hexdigest(), "components": manifest}


def encoder_asset_signature(base):
    return model_asset_signature(base, components=("vae", "text_encoder", "processor"))


def normalize_inference_options(true_cfg_scale, guidance_scale, injection_scope, transformer):
    """Accept the plan's public names while preserving one immutable model scope."""
    if injection_scope is not None:
        if injection_scope not in {"joint", "target"} or injection_scope != getattr(transformer, "injection_scope", "joint"):
            raise ValueError("injection_scope must match the scope chosen when constructing the transformer")
    if guidance_scale is not None:
        if true_cfg_scale != 1.0 and true_cfg_scale != guidance_scale:
            raise ValueError("guidance_scale and true_cfg_scale specify conflicting values")
        true_cfg_scale = guidance_scale
    return true_cfg_scale


def verify_source(source_root):
    root = Path(source_root).resolve()
    probe = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != VIDEOX_PIN:
        raise RuntimeError(f"VideoX-Fun must be at fixed commit {VIDEOX_PIN}: {root}")
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise RuntimeError(f"fixed commit checkout has tracked modifications: {dirty.stdout.strip()}")
    return VIDEOX_PIN


@contextmanager
def _optional_audio_stub():
    """Scope the proven Plan-A librosa import stub to upstream initialization."""
    # Finish dependency availability checks before supplying the unrelated stub.
    import diffusers  # noqa: F401
    import transformers  # noqa: F401

    name = "librosa"
    added = name not in sys.modules and importlib.util.find_spec(name) is None
    if added:
        module = types.ModuleType(name)
        module.__path__ = []
        module.__file__ = "<qwen21-union-unrelated-audio-stub>"
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)

        def blocked(attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            raise ImportError("librosa is unavailable: this scoped stub only skips unrelated VideoX audio models")

        module.__getattr__ = blocked
        sys.modules[name] = module
    try:
        yield
    finally:
        if added:
            sys.modules.pop(name, None)


def _upstream(source_root=None):
    root = Path(source_root or os.environ.get("QWEN21_UNION_SOURCE", DEFAULT_SOURCE)).resolve()
    verify_source(root)
    existing = sys.modules.get("videox_fun")
    if existing is not None and Path(existing.__file__).resolve().parent != root / "videox_fun":
        raise RuntimeError("a different VideoX-Fun checkout is already imported in this process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with _optional_audio_stub():
        models = importlib.import_module("videox_fun.models")
        pipelines = importlib.import_module("videox_fun.pipeline")
        utils = importlib.import_module("videox_fun.utils")
    return root, models, pipelines, utils


def scoped_transformer_class(official_class):
    """Subclass without hooks; mask only official per-block residual outputs."""
    class ScopedUnionTransformer(official_class):
        injection_scope = "joint"

        def forward_control(self, control_joint, x_joint, kwargs):
            hints = super().forward_control(control_joint, x_joint, kwargs)
            if self.injection_scope == "joint":
                return hints
            if self.injection_scope != "target":
                raise ValueError(f"unknown injection scope: {self.injection_scope}")
            mask = kwargs.get("target_token_mask")
            if mask is None or mask.ndim != 1 or mask.numel() != x_joint.shape[1]:
                raise ValueError("target-only requires official target_token_mask metadata")
            keep = mask.to(device=x_joint.device, dtype=x_joint.dtype)[None, :, None]
            return tuple(hint * keep for hint in hints)

    return ScopedUnionTransformer


def _checkpoint(path=None):
    path = Path(path or os.environ.get("QWEN21_UNION_CHECKPOINT", DEFAULT_UNION))
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected one pinned Union safetensors, found {len(candidates)} at {path}; specify exact file")
    return candidates[0]


def _base(path=None):
    path = Path(path or os.environ.get("QWEN21_UNION_BASE", DEFAULT_BASE))
    if not path.is_dir():
        raise FileNotFoundError(f"missing existing Qwen-Image-2.1 model: {path}")
    return path


def build_transformer(scope="joint", device="cpu", dtype=None, *, source_root=None, base_model=None,
                      union_checkpoint=None):
    """Load base, restore all 180 Union tensors, then freeze; allocates no VLM/VAE."""
    if scope not in {"joint", "target"}:
        raise ValueError(f"unknown injection scope: {scope}")
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    root, models, _, _ = _upstream(source_root)
    base = _base(base_model)
    dtype = dtype or torch.bfloat16
    config = root / "config/qwenimage21/qwenimage21_control.yaml"
    extra = OmegaConf.to_container(OmegaConf.load(config)["transformer_additional_kwargs"])
    if extra["control_in_dim"] != 129 or extra["control_layers"] != list(range(0, 32, 2)):
        raise ValueError("pinned Union requires 129 channels and 16 control blocks")
    cls = scoped_transformer_class(models.QwenImage21ControlTransformer2DModel)
    transformer = cls.from_pretrained(str(base), subfolder="transformer", low_cpu_mem_usage=True,
                                      torch_dtype=dtype, transformer_additional_kwargs=extra).to(dtype)
    transformer.injection_scope = scope
    transformer.register_to_config(injection_scope=scope)
    named = dict(transformer.named_parameters())
    control_names = sorted(n for n in named if n.startswith(("control_blocks.", "control_img_in.")))
    zeroed = [n for n in control_names if not bool(named[n].detach().count_nonzero())]
    checkpoint = _checkpoint(union_checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != UNION_SHA256_PIN:
        raise ValueError("Union checkpoint SHA256 differs from the pinned official asset")
    state = load_file(str(checkpoint), device="cpu")
    control_keys = sorted(n for n in state if n.startswith(("control_blocks.", "control_img_in.")))
    if control_keys != control_names or len(control_keys) != 180:
        raise ValueError("Union must restore exactly all 180 control keys, without missing/extra keys")
    for name in control_keys:
        if state[name].shape != named[name].shape:
            raise ValueError(f"Union shape mismatch for {name}")
    incompatible = transformer.load_state_dict(state, strict=False)
    if any(n.startswith(("control_blocks.", "control_img_in.")) for n in incompatible.missing_keys) or incompatible.unexpected_keys:
        raise ValueError("unexpected or missing Union checkpoint keys")
    differing = [n for n in control_keys if not torch.equal(state[n].to(dtype), named[n].detach())]
    if differing or not zeroed:
        raise ValueError(f"Union reload audit failed: differing={differing}, zeroed_before={len(zeroed)}")
    audit = {"backend": "videox-union", "videox_commit": VIDEOX_PIN, "source_root": str(root),
             "union_revision_pin": UNION_REVISION_PIN, "union_checkpoint": str(checkpoint),
             "union_sha256": checkpoint_sha256, "union_bytes": checkpoint.stat().st_size,
             "config_sha256": sha256_file(config), "control_keys_in_checkpoint": len(control_keys),
             "restored_after_union_load": len(control_keys), "zeroed_by_loader_before_union": len(zeroed),
             "bitwise_differing": differing, "loader_missing_keys": len(incompatible.missing_keys),
             "loader_unexpected_keys": list(incompatible.unexpected_keys), "control_in_dim": 129,
             "control_layers": list(transformer.control_layers), "num_control_blocks": 16,
             "injection_scope": scope, "dtype": str(dtype), "base_model": str(base), "gate_pass": True}
    if not transformer.config.causal_condition or getattr(transformer, "sp_world_size", 1) != 1:
        raise ValueError("scoped Union requires causal_condition=True and single-card execution")
    transformer.requires_grad_(False).eval()
    return transformer.to(device), audit


def build_pipeline(kind="union", scope="joint", device="cpu", offload="model_group_offload", dtype=None,
                   *, source_root=None, base_model=None, union_checkpoint=None, transformer_setup=None,
                   cached=False):
    """Reuse pinned Plan-A assets. encoder loads only frozen VLM/VAE, no DiT."""
    if kind not in {"native", "union", "encoder"}:
        raise ValueError(f"unknown pipeline kind: {kind}")
    if scope not in {"joint", "target"}:
        raise ValueError(f"unknown injection scope: {scope}")
    if kind == "encoder" and cached:
        raise ValueError("encoder kind requires VLM; cached=True is for inference")
    import torch
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler

    root, models, pipelines, utils = _upstream(source_root)
    base = _base(base_model)
    dtype = dtype or torch.bfloat16
    audit = {"backend": f"videox-{kind}", "videox_commit": VIDEOX_PIN, "source_root": str(root),
             "base_model": str(base), "dtype": str(dtype), "injection_scope": scope}
    if kind == "union":
        transformer, audit = build_transformer(scope, "cpu", dtype, source_root=root, base_model=base,
                                                union_checkpoint=union_checkpoint)
        pipe_cls = importlib.import_module(f"{__package__}._reference_pipeline").ReferenceUnionPipeline
    elif kind == "native":
        # The baseline is the installed native implementation used by workflow;
        # its unpatched64 and 4-token image-slot contract matches these caches.
        transformer = diffusers.QwenImage21Transformer2DModel.from_pretrained(
            str(base), subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=dtype).to(dtype)
        pipe_cls = diffusers.QwenImage21Pipeline
        audit["backend"] = "diffusers-native"
        audit["diffusers_version"] = diffusers.__version__
        audit["transformer_module"] = transformer.__class__.__module__
    else:
        transformer = None
        pipe_cls = pipelines.QwenImage21ControlPipeline
    vae = models.AutoencoderKLQwenImage21.from_pretrained(str(base), subfolder="vae").to(dtype)
    processor = models.Qwen3VLProcessor.from_pretrained(str(base), subfolder="processor")
    text_encoder = None if cached else models.Qwen3VLForConditionalGeneration.from_pretrained(
        str(base), subfolder="text_encoder", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(base), subfolder="scheduler")
    pipeline = pipe_cls(vae=vae, text_encoder=text_encoder, processor=processor,
                        transformer=transformer, scheduler=scheduler)
    for model in (vae, text_encoder, transformer):
        if model is not None:
            model.requires_grad_(False).eval()
    # LoRA parameter registration must precede group-offload's parameter scan.
    # Freeze original modules first so this step cannot refreeze installed LoRA.
    if transformer_setup is not None:
        if transformer is None:
            raise ValueError("transformer_setup cannot be used with encoder kind")
        audit["transformer_setup"] = transformer_setup(transformer)
    audit["cached_inference"] = bool(cached)
    if str(device).startswith("cuda") and offload and offload != "none":
        if kind == "encoder":
            pipeline.enable_model_cpu_offload(device=device)
            audit["offload"] = "model_cpu_offload"
        else:
            utils.apply_gpu_memory_mode(pipeline, offload, device, dtype,
                                        exclude_module_name=["img_in", "txt_in", "time_text_embed", "modulation"])
            audit["offload"] = offload
    else:
        pipeline = pipeline.to(device)
        audit["offload"] = "none"
    pipeline.union_audit = audit
    return pipeline, audit


def __getattr__(name):
    if name == "ScopedUnionTransformer":
        _, models, _, _ = _upstream()
        value = scoped_transformer_class(models.QwenImage21ControlTransformer2DModel)
    elif name == "ReferenceUnionPipeline":
        _upstream()
        value = importlib.import_module(f"{__package__}._reference_pipeline").ReferenceUnionPipeline
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
