"""方案 B 的独立运行目录、执行状态与 CLI 编排。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
from functools import lru_cache


def new_run_directory(workspace, name):
    if not name or name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("run name must be a single safe path component")
    target = Path(workspace).resolve() / "runs" / name
    target.mkdir(parents=True, exist_ok=False)
    return target


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_progress(workspace):
    work = Path(workspace)
    trained, image_groups, checkpoints = [], {}, {"union_lora_joint": set(), "union_lora_target": set()}
    lock_path = work / "data/lock.json"
    current_lock = file_sha256(lock_path) if lock_path.is_file() else None
    for summary_path in (work / "runs").glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("kind") == "train" and summary.get("complete") and summary.get("optimizer_step") == 512 and summary.get("metadata", {}).get("lock_sha256") == current_lock:
            checkpoint = summary_path.parent / "checkpoint_000512/checkpoint.json"
            if checkpoint.is_file() and summary.get("arm") in checkpoints:
                payload = checkpoint.parent / "state.pt"
                manifest = json.loads(checkpoint.read_text())
                if payload.is_file() and file_sha256(payload) == manifest.get("state_sha256"):
                    trained.append(summary["arm"])
                    checkpoints[summary["arm"]].add(manifest["state_sha256"])
        if summary.get("kind") == "infer" and summary.get("split") == "test":
            if current_lock is not None and summary.get("lock_sha256") != current_lock:
                continue
            signature = summary.get("signature", {})
            group_key = json.dumps(signature, sort_keys=True)
            image_group = image_groups.setdefault(group_key, set())
            records_path = summary_path.parent / "records.jsonl"
            for row in read_jsonl(records_path) if records_path.is_file() else []:
                if row.get("steps") == 40 and row.get("seed") in (42, 43) and row.get("arm") in FORMAL_ARMS and Path(row.get("out", "")).is_file() and file_sha256(row["out"]) == row.get("image_sha256"):
                    image_group.add((row["sample_id"], row["arm"], row["seed"]))
    expected = set()
    manifest = work / "data/test_conditions.jsonl"
    if manifest.is_file():
        expected = {(r["sample_id"], arm, seed) for r in read_jsonl(manifest)
                    for arm in ("native", "union_frozen", "union_lora_joint", "union_lora_target") for seed in (42, 43)}
    complete = False
    for key, images in image_groups.items():
        signature = json.loads(key)
        complete |= bool(expected) and expected <= images and signature.get("joint_checkpoint") in checkpoints["union_lora_joint"] and signature.get("target_checkpoint") in checkpoints["union_lora_target"]
    return dict(completed_training_arms=sorted(set(trained)), test_images=max(map(len, image_groups.values()), default=0), complete=bool(complete))


FORMAL_ARMS = ("native", "union_frozen", "union_lora_joint", "union_lora_target")
DIAGNOSTIC_ARMS = ("union_zero", "union_target_frozen", "wrong_pose")


def validate_inference_protocol(arms, seeds, split, steps, limit):
    arms, seeds = arms.split(","), [int(seed) for seed in seeds.split(",")]
    if len(set(arms)) != len(arms) or any(arm not in FORMAL_ARMS + DIAGNOSTIC_ARMS for arm in arms):
        raise ValueError("unknown or repeated inference arm")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be distinct")
    if split == "test" and (set(arms) != set(FORMAL_ARMS) or set(seeds) != {42, 43} or steps != 40 or limit is not None):
        raise ValueError("formal test requires the locked four arms, two seeds, 40 steps and full sample list")
    return arms, seeds


def formal_trajectory_sha256(workspace):
    from .training import balanced_sample_order
    records = load_records(workspace, "train")
    order = balanced_sample_order({dataset: [r["sample_id"] for r in records if r["dataset"] == dataset]
                                  for dataset in ("iper", "tiktok", "hi4d")})
    return hashlib.sha256(json.dumps(dict(order=order, protocol=formal_training_protocol()), sort_keys=True).encode()).hexdigest()


def validate_formal_inference_checkpoint(path, expected_scope, workspace):
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != "qwen21_union_training_checkpoint_v1":
        raise ValueError("formal checkpoint format differs")
    cursor = manifest.get("cursor", {})
    expected_cursor = dict(optimizer_step=512, accumulation_steps=3, microstep=1536,
                           next_sample_position=1536, loss_log_cursor=1536, gradient_log_cursor=512)
    if cursor != expected_cursor:
        raise ValueError("formal inference requires the completed 512-update boundary")
    metadata = manifest.get("metadata", {})
    if metadata.get("scope") != expected_scope:
        raise ValueError("formal checkpoint scope differs")
    if metadata.get("training_protocol") != formal_training_protocol():
        raise ValueError("formal checkpoint training protocol differs")
    if any(metadata.get(k) != v for k, v in training_binding(workspace).items()):
        raise ValueError("formal checkpoint model/code/initialization binding differs")
    if metadata.get("trajectory_sha256") != formal_trajectory_sha256(workspace):
        raise ValueError("formal checkpoint sampling trajectory differs")
    if not (path / "state.pt").is_file() or file_sha256(path / "state.pt") != manifest.get("state_sha256"):
        raise ValueError("formal checkpoint payload checksum differs")
    return manifest


def load_records(workspace, split, supervision=False):
    from .integrity import verify_workspace_inputs
    if (Path(workspace) / "data/lock.json").is_file():
        verify_workspace_inputs(workspace)
    suffix = "supervision" if supervision else "conditions"
    return read_jsonl(Path(workspace) / "data" / f"{split}_{suffix}.jsonl")


@lru_cache(maxsize=1)
def model_signature():
    import importlib.metadata
    import os
    import torch
    from .pipeline import _base, VIDEOX_PIN, UNION_REVISION_PIN, model_asset_signature
    base = _base()
    paths = [base / "model_index.json", base / "transformer/config.json",
             base / "transformer/diffusion_pytorch_model.safetensors.index.json",
             base / "scheduler/scheduler_config.json", base / "vae/config.json"]
    return dict(videox_commit=VIDEOX_PIN, union_revision=UNION_REVISION_PIN, base_model=str(base), assets=model_asset_signature(base),
                runtime={**{name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")},
                         "torch_module": str(torch.__version__), "cuda": str(torch.version.cuda)},
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                preprocessing_code={name: file_sha256(Path(__file__).with_name(name)) for name in ("cache.py", "conditions.py")},
                configs={str(p): file_sha256(p) for p in paths if p.is_file()})


@lru_cache(maxsize=8)
def locked_conditions(workspace):
    work = Path(workspace)
    lock = json.loads((work / "data/lock.json").read_text(encoding="utf-8"))
    for name, digest in lock["selected_manifest_sha256"].items():
        if file_sha256(work / "data" / name) != digest:
            raise ValueError(f"locked data manifest changed: {name}")
    result = {}
    for split in ("train", "val", "test"):
        result.update({r["sample_id"]: r for r in load_records(work, split)})
    return result


def current_cache_config():
    from .pipeline import _base
    return json.loads((_base() / "vae/config.json").read_text())


def validated_condition(workspace, sample_id):
    from .cache import expected_condition_fingerprint, load_condition
    record = locked_conditions(str(Path(workspace).resolve()))[sample_id]
    expected = expected_condition_fingerprint(record, model_signature(), current_cache_config())
    return load_condition(Path(workspace) / "cache", sample_id, expected_hash=expected)


def command_audit(args):
    import subprocess
    from src.data.qwen21_union_dataset import read_records, audit_records
    from .pipeline import verify_source, DEFAULT_SOURCE
    records = read_records(args.project_root)
    result = audit_records(records)
    prefixes = {split: {r["group_id"].split("_")[0] for r in records if r["dataset"] == "iper" and r["split"] == split} for split in ("train", "val", "test")}
    result["iper_subject_prefix_intersections"] = {a + "__" + b: sorted(prefixes[a] & prefixes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    result["videox_commit"] = verify_source(DEFAULT_SOURCE)
    result["project_head"] = subprocess.check_output(["git", "-C", str(args.project_root), "rev-parse", "HEAD"], text=True).strip()
    directory = Path(args.workspace) / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / (datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_cache(args):
    from .cache import precompute_conditions, precompute_supervision
    from .pipeline import build_pipeline
    pipe, audit = build_pipeline(kind="encoder", device="cuda", offload=None)
    signature = model_signature()
    cache_root = Path(args.workspace) / "cache"
    started = time.monotonic()
    for split in args.conditions.split(","):
        records = load_records(args.workspace, split)
        if args.limit:
            records = records[:args.limit]
        for index, record in enumerate(records):
            precompute_conditions(pipe, [record], cache_root, signature)
            if (index + 1) % 10 == 0 or index == 0:
                print(json.dumps(dict(stage="condition_cache", split=split, completed=index + 1, total=len(records), seconds=round(time.monotonic() - started, 1))), flush=True)
    if args.supervision:
        if args.supervision != "train":
            raise ValueError("only train supervision may be encoded")
        records = load_records(args.workspace, "train", True)
        if args.conditions == "overfit6":
            ids = {r["sample_id"] for r in load_records(args.workspace, "overfit6")}
            records = [r for r in records if r["sample_id"] in ids]
        if args.limit:
            records = records[:args.limit]
        for index, record in enumerate(records):
            precompute_supervision(pipe, [{**record, "split": "train"}], cache_root, signature)
            if (index + 1) % 50 == 0 or index == 0:
                print(json.dumps(dict(stage="supervision_cache", completed=index + 1, total=len(records))), flush=True)
    target = Path(args.workspace) / "cache" / ("completed_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json")
    write_json(target, dict(conditions=args.conditions, supervision=args.supervision, limit=args.limit,
                            model_signature=signature, backend=audit, seconds=time.monotonic() - started))


def _decode(pipe, latents, device):
    import torch
    latents = pipe._unpack_latents(latents, 512, 512, pipe.vae_scale_factor).to(pipe.vae.dtype)
    mean = torch.as_tensor(pipe.vae.config.latents_mean, device=device, dtype=latents.dtype).view(1, 64, 1, 1, 1)
    std = torch.as_tensor(pipe.vae.config.latents_std, device=device, dtype=latents.dtype).view(1, 64, 1, 1, 1)
    pixels = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(pixels, output_type="pil")[0].convert("RGBA")
    from PIL import Image
    result = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(result, image).convert("RGB")


def generate_cached(pipe, condition, seed, *, steps=40, native=False, control_scale=1.0):
    import numpy as np
    import torch
    from .training import transformer_inputs
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, 1, 64, 32, 32), generator=generator, device=device, dtype=dtype).view(1, 64, 1024).transpose(1, 2)
    config = pipe.scheduler.config
    slope = (config.get("max_shift", 1.15) - config.get("base_shift", .5)) / (config.get("max_image_seq_len", 4096) - config.get("base_image_seq_len", 256))
    shift = config.get("base_shift", .5) + (1024 - config.get("base_image_seq_len", 256)) * slope
    pipe.scheduler.set_timesteps(sigmas=np.linspace(1., 1. / steps, steps), device=device, mu=shift)
    pipe.scheduler.set_begin_index(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for timestep in pipe.scheduler.timesteps:
            inputs = transformer_inputs(condition, latents, timestep.reshape(1).to(dtype) / 1000, device=device, dtype=dtype)
            if native:
                inputs.pop("control_context")
                inputs.pop("control_context_scale")
            else:
                inputs["control_context_scale"] = control_scale
            prediction = pipe.transformer(**inputs, return_dict=False)[0][:, -1024:]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("nonfinite transformer prediction")
            latents = pipe.scheduler.step(prediction, timestep, latents, return_dict=False)[0].to(dtype)
        if not torch.isfinite(latents).all():
            raise FloatingPointError("nonfinite generated latents")
        return _decode(pipe, latents, device)


def command_infer(args):
    import gc
    import torch
    from .cache import load_condition
    from .pipeline import build_pipeline
    from .weights import install_control_lora, load_lora_state_dict
    arms, seeds = validate_inference_protocol(args.arms, args.seeds, args.split, args.steps, args.limit)
    for arm, checkpoint in (("union_lora_joint", args.joint_checkpoint), ("union_lora_target", args.target_checkpoint)):
        if arm in arms and checkpoint is None:
            raise ValueError(f"{arm} requires an explicit checkpoint")
    records = load_records(args.workspace, args.split)
    if set(arms) & set(DIAGNOSTIC_ARMS) and args.limit is None:
        if args.split != "val":
            raise ValueError("diagnostic arms use validation samples")
        records = [row for dataset in ("iper", "tiktok", "hi4d") for row in [r for r in records if r["dataset"] == dataset][:4]]
    if args.limit:
        records = records[:args.limit]
    if "wrong_pose" in arms and any(sum(row["dataset"] == dataset for row in records) < 2 for dataset in {r["dataset"] for r in records}):
        raise ValueError("wrong_pose requires two distinct targets per dataset")
    signature = dict(model_fingerprint=model_signature()["assets"]["fingerprint"],
                     lock_sha256=file_sha256(Path(args.workspace) / "data/lock.json"))
    for name, checkpoint in (("joint", args.joint_checkpoint), ("target", args.target_checkpoint)):
        if checkpoint:
            manifest = validate_formal_inference_checkpoint(checkpoint, name, args.workspace) if args.split == "test" else json.loads((Path(checkpoint) / "checkpoint.json").read_text())
            signature[name + "_checkpoint"] = manifest["state_sha256"]
    run = new_run_directory(args.workspace, args.name)
    produced = 0
    for arm in arms:
        native = arm == "native"
        scope = "target" if arm in ("union_lora_target", "union_target_frozen") else "joint"
        setup = None
        if arm.startswith("union_lora_"):
            checkpoint = args.target_checkpoint if scope == "target" else args.joint_checkpoint
            if not checkpoint:
                raise ValueError(f"{arm} requires an explicit checkpoint")
            manifest = json.loads((Path(checkpoint) / "checkpoint.json").read_text())
            if manifest["metadata"]["scope"] != scope or manifest["metadata"]["lock_sha256"] != file_sha256(Path(args.workspace) / "data/lock.json"):
                raise ValueError("inference checkpoint scope/data signature differs")
            from .pipeline import UNION_SHA256_PIN
            if manifest["metadata"]["base_fingerprint"] != model_signature()["assets"]["fingerprint"] or manifest["metadata"]["union_fingerprint"] != UNION_SHA256_PIN:
                raise ValueError("inference checkpoint base/Union weights differ")
            state_path = Path(checkpoint) / "state.pt"
            if file_sha256(state_path) != manifest["state_sha256"]:
                raise ValueError("checkpoint checksum differs")
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("metadata") != manifest["metadata"] or state.get("cursor") != manifest["cursor"]:
                raise ValueError("checkpoint payload metadata/cursor differs from manifest")
            def setup(transformer):
                result = install_control_lora(transformer)
                load_lora_state_dict(transformer, state["lora"])
                return result
        pipe, audit = build_pipeline(kind="native" if native else "union", scope=scope,
                                     device="cuda", offload="model_group_offload", cached=True,
                                     transformer_setup=setup)
        write_json(run / f"backend_{arm}.json", audit)
        for record_index, record in enumerate(records):
            condition = validated_condition(args.workspace, record["sample_id"])
            control_sample_id = record["sample_id"]
            if arm == "wrong_pose":
                from .cache import load_condition
                candidate = next(other for other in records[record_index + 1:] + records[:record_index] if other["dataset"] == record["dataset"] and other["sample_id"] != record["sample_id"])
                condition = {**condition, "control_latents": validated_condition(args.workspace, candidate["sample_id"])["control_latents"]}
                control_sample_id = candidate["sample_id"]
            for seed in seeds:
                output = run / f"{record['sample_id']}_{arm}_{seed}.png"
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                image = generate_cached(pipe, condition, seed, steps=args.steps, native=native,
                                         control_scale=0. if arm == "union_zero" else 1.)
                with output.open("xb") as stream:
                    image.save(stream, format="PNG")
                row = dict(sample_id=record["sample_id"], dataset=record["dataset"], group_id=record["group_id"],
                           split=args.split, arm=arm, seed=seed, steps=args.steps, out=str(output),
                           control_sample_id=control_sample_id,
                           seconds=round(time.monotonic() - started, 3), peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                           image_sha256=file_sha256(output))
                append_jsonl(run / "records.jsonl", row)
                produced += 1
                print(json.dumps(dict(stage="infer", produced=produced, expected=len(records) * len(arms) * len(seeds), **row)), flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    write_json(run / "summary.json", dict(kind="infer", split=args.split, produced=produced,
               expected=len(records) * len(arms) * len(seeds), complete=produced == len(records) * len(arms) * len(seeds), arms=arms, seeds=seeds, signature=signature,
               lock_sha256=file_sha256(Path(args.workspace) / "data/lock.json")))


def load_cached_sample(workspace, sample_id):
    from .cache import load_supervision, expected_supervision_fingerprint
    root = Path(workspace) / "cache"
    truth = next(r for r in load_records(workspace, "train", True) if r["sample_id"] == sample_id)
    expected = expected_supervision_fingerprint({**truth, "split": "train"}, model_signature(), current_cache_config())
    return validated_condition(workspace, sample_id), load_supervision(root, sample_id, expected_hash=expected)


def training_binding(workspace):
    from .cache import PREPROCESS_VERSION
    from .pipeline import VIDEOX_PIN, UNION_REVISION_PIN, UNION_SHA256_PIN, _upstream
    from .weights import control_lora_targets
    from .integrity import verify_workspace_inputs
    inputs = verify_workspace_inputs(workspace)
    root, _, _, _ = _upstream()
    code = Path(__file__).parent
    files = ("runner.py", "weights.py", "training.py", "checkpoint.py", "resume_logs.py", "gates.py", "pipeline.py", "_reference_pipeline.py", "cache.py", "conditions.py", "contracts.py", "integrity.py")
    return dict(base_fingerprint=model_signature()["assets"]["fingerprint"], union_fingerprint=UNION_SHA256_PIN,
        runtime=model_signature()["runtime"],
        videox_commit=VIDEOX_PIN, union_revision=UNION_REVISION_PIN,
        control_config_sha256=file_sha256(root / "config/qwenimage21/qwenimage21_control.yaml"),
        lock_sha256=file_sha256(Path(workspace) / "data/lock.json"), preprocessing_version=PREPROCESS_VERSION,
        prepared_registry_sha256=inputs["prepared_registry_sha256"],
        rank=16, alpha=16, lora_targets=list(control_lora_targets()), dataset_order=["iper", "tiktok", "hi4d"],
        initial_lora_sha256=file_sha256(Path(workspace) / "initialization/lora_rank16_seed3407.pt"),
        training_code_sha256={name: file_sha256(code / name) for name in files})


def formal_training_protocol():
    return dict(train_seed=3407, sampler_seed=20261015, optimizer="AdamW", lr=1e-4, betas=[.9, .999], eps=1e-8,
        weight_decay=.01, warmup_updates=32, lr_after_warmup="constant", gradient_clip=1.,
        accumulation_steps=3, updates=512, canvas_wh=[512, 512], schedule_steps=1000,
        schedule="base_FlowMatchEuler_dynamic_shift_target_tokens_v1", loss="content_coverage_flow_MSE_v1",
        condition_dropout=0., augment=False, vae_encoding="posterior_mode", deterministic_algorithms=True,
        cublas_workspace_config=":4096:8", cudnn_benchmark=False, cudnn_deterministic=True)


def validated_gate_kinds(gates, binding):
    passed = set()
    for gate in gates:
        metadata = gate.get("metadata", {})
        if not gate.get("gate_pass") or any(metadata.get(k) != v for k, v in binding.items()):
            continue
        if gate.get("kind") == "smoke":
            if metadata.get("training_protocol") != formal_training_protocol():
                continue
            resume = gate.get("resume_equivalence", {})
            images = gate.get("image_checks", [])
            if gate.get("probe_microsteps") == 10 and gate.get("frozen_unchanged") and resume.get("passed") and resume.get("continuous_updates") == 4 and resume.get("resumed_updates") == [2, 2] and {r.get("dataset") for r in images} == {"iper", "hi4d"}:
                passed.add("smoke")
        elif gate.get("kind") == "overfit":
            protocol = metadata.get("training_protocol", {})
            if gate.get("optimizer_step") == 200 and gate.get("relative_loss_drop", -1) >= .2 and gate.get("evaluated_positions") == 18 and gate.get("frozen_unchanged") and protocol.get("kind") == "overfit6" and protocol.get("accumulation_steps") == 1 and protocol.get("train_seed") == 3407:
                passed.add("overfit")
    return passed


def initialize_training(workspace, scope, run_dir):
    """Both arms and the gates start from one immutable LoRA initialization."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from .pipeline import build_transformer, _base
    from .weights import install_control_lora, lora_state_dict, load_lora_state_dict, control_lora_targets
    from .training import balanced_sample_order, make_optimizer, make_lr_scheduler, configure_determinism
    from .cache import PREPROCESS_VERSION
    configure_determinism()
    model_signature.cache_clear()
    model, audit = build_transformer(scope=scope, device="cpu")
    torch.manual_seed(3407)
    lora_audit = install_control_lora(model)
    initial_root = Path(workspace) / "initialization"
    initial_root.mkdir(parents=True, exist_ok=True)
    initial_path = initial_root / "lora_rank16_seed3407.pt"
    if initial_path.exists():
        stored = torch.load(initial_path, map_location="cpu", weights_only=True)
        expected = lora_state_dict(model)
        if stored.keys() != expected.keys() or any(not torch.equal(stored[key], expected[key]) for key in expected):
            raise ValueError("immutable common LoRA initialization differs from seed3407")
        load_lora_state_dict(model, stored)
        del stored, expected
    else:
        with initial_path.open("xb") as stream:
            torch.save(lora_state_dict(model), stream)
    model.enable_gradient_checkpointing()
    model = model.to("cuda")
    records = load_records(workspace, "train")
    samples = {dataset: [r["sample_id"] for r in records if r["dataset"] == dataset] for dataset in ("iper", "tiktok", "hi4d")}
    order = balanced_sample_order(samples)
    protocol = formal_training_protocol()
    metadata = dict(**training_binding(workspace), scope=scope, training_protocol=protocol,
                    trajectory_sha256=hashlib.sha256(json.dumps(dict(order=order, protocol=protocol), sort_keys=True).encode()).hexdigest())
    if run_dir is not None:
        write_json(Path(run_dir) / "model_audit.json", dict(model=audit, lora=lora_audit, metadata=metadata))
    optimizer = make_optimizer(model)
    lr = make_lr_scheduler(optimizer)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(_base()), subfolder="scheduler")
    return dict(model=model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler,
                metadata=metadata, sample_order=order, initial_lora=initial_path)


def validation_generation(workspace, model, run_dir, step, arm):
    """Reuse cached conditions; no VLM or validation GT latent enters this path."""
    import gc
    from types import SimpleNamespace
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    from .pipeline import _upstream, _base
    from .cache import load_condition
    _, models, pipelines, _ = _upstream()
    torch.cuda.empty_cache()
    base = _base()
    vae = models.AutoencoderKLQwenImage21.from_pretrained(str(base), subfolder="vae").to(dtype=torch.bfloat16, device="cuda").eval().requires_grad_(False)
    pipe = SimpleNamespace(transformer=model, vae=vae, vae_scale_factor=16,
        image_processor=VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(base), subfolder="scheduler"),
        _unpack_latents=pipelines.QwenImage21ControlPipeline._unpack_latents)
    target = Path(run_dir) / f"validation_{step:06d}"
    target.mkdir(exist_ok=False)
    model.eval()
    for index, record in enumerate(load_records(workspace, "val")):
        image = generate_cached(pipe, validated_condition(workspace, record["sample_id"]), 42)
        output = target / (record["sample_id"] + ".png")
        with output.open("xb") as stream:
            image.save(stream, format="PNG")
        append_jsonl(target / "records.jsonl", dict(sample_id=record["sample_id"], dataset=record["dataset"], group_id=record["group_id"],
                     split="val", arm=arm, optimizer_step=step, seed=42, out=str(output)))
        if (index + 1) % 12 == 0:
            print(json.dumps(dict(stage="validation_generation", optimizer_step=step, completed=index + 1, total=72)), flush=True)
    del pipe, vae
    gc.collect()
    torch.cuda.empty_cache()
    model.train()


def command_train(args):
    import signal
    from dataclasses import asdict
    from .checkpoint import TrainingCursor, save_checkpoint, load_checkpoint
    from .training import run_training_updates, TrainingResourceError
    from .weights import frozen_parameter_digest
    work = Path(args.workspace)
    if args.updates != 512:
        raise ValueError("the formal pilot protocol requires exactly 512 updates")
    gate_files = list((work / "runs").glob("*/gate.json"))
    gates = [json.loads(p.read_text()) for p in gate_files]
    passed = validated_gate_kinds(gates, training_binding(work))
    if not {"smoke", "overfit"} <= passed:
        raise ValueError("formal training requires passed smoke and overfit gates")
    run = new_run_directory(work, args.name)
    scope = "target" if args.arm == "union_lora_target" else "joint"
    state = initialize_training(work, scope, run)
    model, optimizer, lr = state["model"], state["optimizer"], state["lr_scheduler"]
    cursor = TrainingCursor()
    if args.resume:
        cursor = load_checkpoint(args.resume, model, optimizer, lr, expected_metadata=state["metadata"])
        from .resume_logs import prepare_resume_logs
        prepare_resume_logs(run, args.resume, cursor)
    freeze_before = frozen_parameter_digest(model)
    write_json(run / "frozen_before.json", dict(sha256=freeze_before))
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def save(cursor, label=None):
        return save_checkpoint(run / (label or f"checkpoint_{cursor.optimizer_step:06d}"), model, optimizer, lr,
                               metadata=state["metadata"], cursor=cursor)
    save(cursor)
    validation_generation(work, model, run, cursor.optimizer_step, args.arm)
    def on_microstep(event):
        append_jsonl(run / "microsteps.jsonl", event)
    def on_update(event):
        clean = {k: v for k, v in event.items() if k != "cursor"}
        append_jsonl(run / "updates.jsonl", clean)
        print(json.dumps({k: v for k, v in clean.items() if k != "trajectory"}), flush=True)
        if event["optimizer_step"] in (128, 256, 512):
            save(event["cursor"])
        if event["optimizer_step"] in (256, 512):
            validation_generation(work, model, run, event["optimizer_step"], args.arm)
    try:
        result = run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=state["scheduler"],
            sample_order=state["sample_order"], load_sample=lambda sid: load_cached_sample(work, sid),
            updates=args.updates, cursor=cursor, on_microstep=on_microstep, on_update=on_update,
            stop_requested=lambda: stop["requested"])
    except TrainingResourceError as error:
        report = {k: asdict(v) if k == "cursor" else v for k, v in error.report.items() if k != "updates"}
        write_json(run / "resource_failure.json", dict(error=str(error), **report))
        raise
    cursor = result["cursor"]
    if result["stopped_early"]:
        save(cursor, f"interrupted_{cursor.optimizer_step:06d}")
    freeze_after = frozen_parameter_digest(model)
    if freeze_before != freeze_after:
        raise RuntimeError("frozen base/Union parameters changed during training")
    write_json(run / "summary.json", dict(kind="train", arm=args.arm, optimizer_step=cursor.optimizer_step,
         complete=cursor.optimizer_step == args.updates, frozen_unchanged=True,
         elapsed_seconds=result["elapsed_seconds"], max_memory_allocated=result["max_memory_allocated"],
         metadata=state["metadata"]))


def command_smoke(args):
    import gc
    import torch
    from .gates import run_smoke_gates
    from .pipeline import build_pipeline
    run = new_run_directory(args.workspace, args.name)
    result = run_smoke_gates(args.workspace, run)
    gc.collect()
    torch.cuda.empty_cache()
    pipe, audit = build_pipeline(kind="union", scope="joint", device="cuda", cached=True)
    checks = []
    samples = load_records(args.workspace, "overfit6")
    for dataset in ("iper", "hi4d"):
        record = next(row for row in samples if row["dataset"] == dataset)
        image = generate_cached(pipe, validated_condition(args.workspace, record["sample_id"]), 43, steps=2)
        if image.size != (512, 512):
            raise ValueError("smoke image canvas differs")
        output = run / f"{dataset}_two_step.png"
        with output.open("xb") as stream:
            image.save(stream, format="PNG")
        checks.append(dict(dataset=dataset, sample_id=record["sample_id"], out=str(output), width=image.width, height=image.height))
    result["image_checks"] = checks
    result["image_backend"] = audit
    write_json(run / "gate.json", dict(kind="smoke", **result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_overfit(args):
    from .gates import overfit_gate
    run = new_run_directory(args.workspace, args.name)
    result = overfit_gate(args.workspace, run, updates=args.updates)
    write_json(run / "gate.json", dict(kind="overfit", **result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_evaluate(args):
    from .evaluation import evaluate_artifacts
    if not args.run:
        raise ValueError("evaluate requires --run")
    run = Path(args.workspace) / "runs" / args.run
    print(str(evaluate_artifacts(args.workspace, run, args.project_root)))


def command_report(args):
    from PIL import Image, ImageDraw
    if not args.run:
        raise ValueError("report requires --run")
    work = Path(args.workspace)
    run = work / "runs" / args.run
    evaluations = sorted(run.glob("evaluation_*/summary.json"))
    report = run / ("report_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    report.mkdir(exist_ok=False)
    summary = json.loads(evaluations[-1].read_text()) if evaluations else None
    records = read_jsonl(run / "records.jsonl")
    by_sample = {}
    for r in records:
        by_sample.setdefault((r["sample_id"], r["seed"]), {})[r["arm"]] = r
    conditions = {r["sample_id"]: r for split in ("train", "val", "test") for r in load_records(work, split)}
    truths = {r["sample_id"]: r for split in ("train", "val", "test") for r in load_records(work, split, True)}
    from .evaluation import _prepared_target
    panels = report / "panels"
    panels.mkdir()
    for (sid, seed), variants in by_sample.items():
        record = conditions[sid]
        cells = [("SOURCE", Image.open(record["source_rgb"]).convert("RGB")),
                 ("CONTROL", Image.open(record["control_image"]).convert("RGB")),
                 ("GT (EVAL ONLY)", _prepared_target(truths[sid]))]
        wrong = variants.get("wrong_pose")
        if wrong:
            cells.append(("WRONG CONTROL", Image.open(conditions[wrong["control_sample_id"]]["control_image"]).convert("RGB")))
        cells += [(arm, Image.open(row["out"]).convert("RGB")) for arm, row in sorted(variants.items())]
        panel = Image.new("RGB", (256 * len(cells), 282), "white")
        draw = ImageDraw.Draw(panel)
        for index, (label, image) in enumerate(cells):
            panel.paste(image.resize((256, 256)), (index * 256, 26))
            draw.text((index * 256 + 3, 6), label, fill="black")
        panel.save(panels / f"{sid}_{seed}.png")
    progress = summarize_progress(work)
    lines = ["# Qwen 2.1 Union 数据集适配实验", "", f"本报告对应 `{run.name}`，实有 {len(records)} 张输出。", "",
             "## 执行状态", "", "```json", json.dumps(progress, ensure_ascii=False, indent=2), "```", "",
             "## 结果", ""]
    if summary:
        lines += ["| 数据集 | 组别 | 样本 | 分组 | PCK@0.2 组宏平均 | 检测覆盖率 | 姿态基准可用率 | 外观余弦组宏平均 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary["aggregate"]:
            lines.append(f"| {row['dataset']} | {row['arm']} | {row['samples']} | {row['groups']} | {row.get('pck_02_group_macro')} | {row.get('detector_coverage_group_macro')} | {row.get('request_available_group_macro')} | {row.get('appearance_mean_group_macro')} |")
        lines += ["", f"原始指标和失败检测见 `{evaluations[-1].parent}`。"]
    else:
        lines += ["尚未执行指标评估；不能据此给出效果结论。"]
    lines += ["", "## 解释边界", "", "iPER 是 appearance 留出；Hi4D 条件预处理有目标 mask 辅助，测试只有两个人物 pair；3DPW 只作域外测试。DINO 是外观代理，ArcFace 须结合检测覆盖率。", "",
              "所有面板包含模型实际使用的源图和控制图。GT 仅供评估展示。训练有无收益以及 target-only 是否更好，应按每个数据集的配对指标与人物失败案例判断。"]
    (report / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(report / "REPORT.zh-CN.md"))


def parser():
    result = argparse.ArgumentParser(description="Qwen21 Union 方案 B")
    result.add_argument("command", choices=("audit", "select", "cache", "smoke", "overfit", "train", "infer", "evaluate", "report", "status"))
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--project-root", type=Path, default=Path("/home/shangguanrz/project/pic-edit"))
    result.add_argument("--protocol", default="pilot_b_v1")
    result.add_argument("--name")
    result.add_argument("--run")
    result.add_argument("--conditions", default="train,val,test")
    result.add_argument("--supervision", default="train")
    result.add_argument("--limit", type=int)
    result.add_argument("--split", default="test", choices=("train", "val", "test", "overfit6"))
    result.add_argument("--arm", default="union_lora_joint", choices=("union_lora_joint", "union_lora_target"))
    result.add_argument("--arms", default="native,union_frozen,union_lora_joint,union_lora_target")
    result.add_argument("--seeds", default="42,43")
    result.add_argument("--steps", type=int, default=40)
    result.add_argument("--updates", type=int, default=512)
    result.add_argument("--joint-checkpoint", type=Path)
    result.add_argument("--target-checkpoint", type=Path)
    result.add_argument("--resume", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.command in ("smoke", "overfit", "train", "infer") and not args.name:
        raise ValueError("this command requires a unique --name")
    if args.command == "select":
        from src.data.qwen21_union_dataset import build_pilot
        if args.protocol != "pilot_b_v1":
            raise ValueError("unknown protocol")
        print(json.dumps(build_pilot(args.project_root, args.workspace), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(summarize_progress(args.workspace), indent=2))
    else:
        globals()["command_" + args.command](args)
