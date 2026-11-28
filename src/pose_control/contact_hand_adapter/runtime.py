"""Standalone condition/checkpoint I/O and full-canvas or joint hand ROI generation."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.canvas_contract import CanvasSpec, LETTERBOX
from src.pose_edit.common import sha256_file

from .conditions import HandCondition, HandMesh, HandPair
from .geometry import build_contact_graph, pixel_geometry, roi_boxes, support_box
from .integration import bind_hand_condition
from .model import AdapterConfig, ContactHandAdapter


CONDITION_SCHEMA = "contact_hand_condition_v2_1"
CHECKPOINT_SCHEMA = "contact_hand_adapter_v3"
_LEGACY_CONDITION_SCHEMA = "contact_hand_condition_v2"
_CHECKPOINT_SCHEMAS = {2: "contact_hand_adapter_v2", 21: "contact_hand_adapter_v2_1", 3: CHECKPOINT_SCHEMA}
DEFAULT_LOCAL_DILATION = 8


def condition_record(condition):
    condition.validate()
    record = asdict(condition)
    record["canvas"] = condition.canvas.metadata()
    return {"schema": CONDITION_SCHEMA, **record}


def load_condition(path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.pop("schema", None) not in (CONDITION_SCHEMA, _LEGACY_CONDITION_SCHEMA):
        raise ValueError("Unsupported Contact Hand condition schema")
    record["canvas"] = CanvasSpec.from_record(record["canvas"])
    record["person_ids"] = tuple(record["person_ids"])
    if "appearance_sources" in record:
        record["appearance_sources"] = tuple(record["appearance_sources"])
    if "appearance_dino_crop_sha" in record:
        record["appearance_dino_crop_sha"] = tuple(record["appearance_dino_crop_sha"])
    record["pairs"] = tuple(HandPair(**{**pair, "slots": tuple(pair["slots"])}) for pair in record.get("pairs", ()))
    if record.get("mesh") is not None:
        mesh = record["mesh"]
        record["mesh"] = HandMesh(**{**mesh, "camera_ids": tuple(mesh["camera_ids"])})
    return HandCondition(**record).validate()


def base_fingerprint(root):
    """Bind the whole local Qwen pipeline, including the scheduler and preprocessing assets."""
    root = Path(root)
    if not (root / "model_index.json").is_file():
        raise FileNotFoundError(root / "model_index.json")
    components = ("transformer", "text_encoder", "vae", "scheduler", "tokenizer", "processor", "image_processor")
    entries = [root / "model_index.json"]
    for component in components:
        entries.extend(path for path in (root / component).rglob("*")
                       if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts))
    for model in ("transformer", "text_encoder", "vae"):
        if not any(path.relative_to(root).parts[0] == model and path.suffix in (".safetensors", ".bin")
                   for path in entries):
            raise ValueError(f"Cannot fingerprint Qwen without the local {model} weights")
    records = [(path.relative_to(root).as_posix(), sha256_file(path)) for path in sorted(entries)]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _check_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA256 for the frozen Qwen base")


def save_adapter(path, adapter, *, base_sha, step, training=None):
    """Publish a new checkpoint atomically, without replacing an existing checkpoint."""
    _check_sha(base_sha)
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(adapter.config)
    version = adapter.config.architecture_version
    schema = _CHECKPOINT_SCHEMAS[version]
    if version == 2:
        config.pop("architecture_version")
    record = dict(schema=schema, base_sha=base_sha, config=config,
                  step=step, adapter=adapter.state_dict(), training=training)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".hand-checkpoint-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink()
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return sha256_file(path)


def load_adapter(path, *, base_sha):
    _check_sha(base_sha)
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("schema") not in _CHECKPOINT_SCHEMAS.values():
        raise ValueError("Unsupported Contact Hand Adapter checkpoint schema")
    if record.get("base_sha") != base_sha:
        raise ValueError("Contact Hand Adapter checkpoint belongs to a different Qwen base")
    config = {**record["config"], "layers": tuple(record["config"]["layers"])}
    version = next(version for version, schema in _CHECKPOINT_SCHEMAS.items() if schema == record["schema"])
    if version == 2:
        config.setdefault("architecture_version", 2)
    if config.get("architecture_version") != version:
        raise ValueError("Contact Hand Adapter checkpoint architecture does not match its schema")
    adapter = ContactHandAdapter(AdapterConfig(**config))
    adapter.load_state_dict(record["adapter"], strict=True)
    return adapter, {key: record[key] for key in ("step", "base_sha", "training")}


def initialize_adapter(adapter, path, *, base_sha):
    """Warm-start a new stage, optionally adding zero-init Qwen output LoRA."""
    previous, record = load_adapter(path, base_sha=base_sha)
    source, target = previous.config, adapter.config
    adding_lora = not source.hand_lora_rank and bool(target.hand_lora_rank)
    if source != target and not (adding_lora and replace(target, hand_lora_rank=0) == source):
        raise ValueError("Warm initialization needs the same Adapter architecture")
    missing, extra = adapter.load_state_dict(previous.state_dict(), strict=not adding_lora)
    allowed = {key for key in adapter.state_dict() if key.startswith("native_lora.")} if adding_lora else set()
    if extra or set(missing) != allowed:
        raise ValueError("Warm initialization does not match Adapter weights")
    return record["step"]


@dataclass(frozen=True)
class LocalPlan:
    box: tuple[int, int, int, int]  # target canvas (x0, y0, x1, y1)
    slots: tuple[int, ...]
    condition: HandCondition
    mask: Image.Image
    global_hw: tuple[int, int]

    def image(self, base):
        self._check_base(base)
        return self.condition.canvas.image(base.crop(self.box))

    def composite(self, base, generated):
        self._check_base(base)
        canvas = self.condition.canvas
        if generated.size != tuple(reversed(canvas.canvas_hw)):
            raise ValueError("Generated local image does not match the planned Qwen canvas")
        patch = canvas.unpad(generated).resize(tuple(reversed(canvas.original_hw)), Image.Resampling.LANCZOS)
        target = base.convert("RGB").copy()
        target.paste(patch.convert("RGB"), self.box[:2], self.mask)
        return target

    def _check_base(self, base):
        if base.size != tuple(reversed(self.global_hw)):
            raise ValueError("Local context image must be on the declared full target canvas")


def _local_plans(condition, *, size, slots, margin, current_support, dilation, single):
    condition = condition.to("cpu").validate()
    if type(size) is not int or size < 32 or size % 32 or not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("Local canvas size must be a multiple of 32 and margin must be in [0,1]")
    if type(dilation) is not int or dilation < 0:
        raise ValueError("Local mask dilation must be a nonnegative integer")
    _, support = pixel_geometry(condition)
    valid = condition.canvas.valid_mask(dtype=torch.bool)
    edit = support > 0
    if current_support is not None:
        if (not torch.is_tensor(current_support) or current_support.shape != support.shape
                or (current_support.dtype != torch.bool and not current_support.is_floating_point())):
            raise ValueError("Current hand support must be a boolean or float [4,H,W] full-canvas tensor")
        current_support = current_support.detach().cpu()
        if not torch.isfinite(current_support).all() or ((current_support < 0) | (current_support > 1)).any():
            raise ValueError("Current hand support must be finite coverage in [0,1]")
        edit |= (current_support > 0) & valid
    if dilation:
        height, width = condition.canvas.canvas_hw
        radius_y, radius_x = min(dilation, height - 1), min(dilation, width - 1)
        expanded = F.max_pool2d(edit.float(), (1, 2 * radius_x + 1), stride=1, padding=(0, radius_x))
        edit = F.max_pool2d(expanded, (2 * radius_y + 1, 1), stride=1, padding=(radius_y, 0)).bool()
    edit &= valid
    graph = build_contact_graph(condition, support)
    components = [set(members) for members, _ in roi_boxes(support, graph)]
    target_slots = {slot for group in components for slot in group}
    active = set(edit.flatten(1).any(-1).nonzero().flatten().tolist())
    components.extend({slot} for slot in sorted(active - target_slots))
    if not active:
        raise ValueError("No eligible hand pixels for local generation")

    # Distinct target components can share an edit region when a hand moves across another hand.
    # Merge those masks before denoising so each editable pixel is composited at most once.
    merged = []
    for group in components:
        mask = edit[sorted(group)].any(0)
        disjoint = []
        for previous, previous_mask in merged:
            if (mask & previous_mask).any():
                group |= previous
                mask |= previous_mask
            else:
                disjoint.append((previous, previous_mask))
        merged = [*disjoint, (group, mask)]
    groups = sorted((group for group, _ in merged), key=min)
    if slots is not None:
        requested = set(slots)
        if not requested or not requested <= active:
            raise ValueError("Requested local hand slots must have current or target support")
        groups = [group for group in groups if group & requested]
        if single:
            groups = [set.union(*groups)]
    elif single:
        def priority(group):
            members = sorted(group)
            contacts = graph.part_state[members][:, members].eq(1).sum().item()
            return contacts, len(members), support[members].sum().item()

        groups = [max(groups, key=priority)]
    for group in groups:
        yield _crop_local_plan(condition, support, edit, group, size, margin)


def make_local_plan(condition, *, size=512, slots=None, margin=.25, current_support=None,
                    dilation=DEFAULT_LOCAL_DILATION):
    """Return one training ROI for an explicit slot selection or the highest-priority group."""
    return next(_local_plans(condition, size=size, slots=slots, margin=margin,
                             current_support=current_support, dilation=dilation, single=True))


def _crop_local_plan(condition, support, edit, selected, size, margin):
    slots = tuple(sorted(selected))
    edit_mask = edit[list(slots)].any(0)
    x0, y0, x1, y1 = support_box(edit_mask)
    radius = math.ceil(max(x1 - x0, y1 - y0) * margin)
    min_x, min_y, max_x, max_y = condition.canvas.valid_rect
    box = (max(min_x, x0 - radius), max(min_y, y0 - radius),
           min(max_x, x1 + radius), min(max_y, y1 + radius))
    x0, y0, x1, y1 = box
    original = (y1 - y0, x1 - x0)
    scale = size / max(original)
    resized = tuple(max(1, round(axis * scale)) for axis in original)
    offset = ((size - resized[1]) // 2, (size - resized[0]) // 2)
    canvas = CanvasSpec(original, resized, (size, size), offset, LETTERBOX)
    mask_slots = torch.tensor([slot in selected for slot in range(4)])

    def spatial(value, *, nearest=False):
        cropped = value[..., y0:y1, x0:x1]
        return canvas.stack(cropped.float(), nearest=nearest)

    def xy(value):
        return (value - value.new_tensor([x0, y0])) * value.new_tensor(canvas.scale_xy) + value.new_tensor(offset)

    def keep_slots(value):
        if value is None:
            return None
        selected_axes = mask_slots.reshape(4, *([1] * (value.ndim - 1)))
        return value.masked_fill(~selected_axes, 0)

    temp = condition.joints.clone()
    temp[..., :2] = xy(temp[..., :2])
    dense, dense_valid = None, None
    if condition.geometry_maps is not None:
        valid = condition.map_valid & mask_slots[:, None, None, None]
        dense_valid = spatial(valid, nearest=True).bool()
        dense = spatial(torch.where(valid, condition.geometry_maps, 0), nearest=True)
    mesh = condition.mesh
    if mesh is not None:
        mesh = replace(mesh, anchor_xy=xy(mesh.anchor_xy), valid=mesh.valid & mask_slots)
    crop_support = support * mask_slots[:, None, None]
    local = replace(
        condition, canvas=canvas, joints=keep_slots(temp), visible=keep_slots(condition.visible),
        support=spatial(crop_support), pairs=tuple(pair for pair in condition.pairs if set(pair.slots) <= selected),
        geometry_maps=dense, map_valid=dense_valid, mesh=mesh, appearance_valid=keep_slots(condition.appearance_valid),
        appearance_dino_valid=keep_slots(condition.appearance_dino_valid),
        appearance_dino_crop_sha=tuple(sha if slot in selected else None
                                       for slot, sha in enumerate(condition.appearance_dino_crop_sha)),
        joints_3d=keep_slots(condition.joints_3d), joint_3d_valid=keep_slots(condition.joint_3d_valid),
        joint_2d_valid=keep_slots(condition.joint_2d_valid), joint_occluded=keep_slots(condition.joint_occluded),
        hand_present=keep_slots(condition.hand_present), joint_rotation_6d=keep_slots(condition.joint_rotation_6d),
        joint_rotation_valid=keep_slots(condition.joint_rotation_valid), mano_shape=keep_slots(condition.mano_shape),
        mano_wrist_rotation_6d=keep_slots(condition.mano_wrist_rotation_6d),
        mano_palm_scale=keep_slots(condition.mano_palm_scale), mano_valid=keep_slots(condition.mano_valid),
    )
    projected = local.projected_joints()
    occluded = local.joint_occluded
    local = replace(local, visible=local.visible & projected,
                    joint_occluded=None if occluded is None else occluded & projected).validate()
    mask = Image.fromarray(edit_mask[y0:y1, x0:x1].byte().mul(255).numpy())
    return LocalPlan(box, slots, local, mask, condition.canvas.canvas_hw)


def qwen_reference_size(image, resolution):
    """The aspect-ratio and 32-pixel rounding of QwenImage21Pipeline.calculate_dimensions."""
    ratio = image.width / image.height
    width = math.sqrt(resolution**2 * ratio)
    return round(width / 32) * 32, round((width / ratio) / 32) * 32


def qwen_reference_resolution(pipe, references, target_hw):
    """Choose an area that the VLM will keep unchanged after Qwen's per-image rounding."""
    processor = getattr(getattr(pipe, "processor", None), "image_processor", None)
    size = getattr(processor, "size", None)

    def limit(edge, pixels):
        configured = size.get(edge) if isinstance(size, dict) else getattr(size, edge, None)
        return configured or getattr(processor, pixels, None)

    minimum = limit("shortest_edge", "min_pixels") or 0
    maximum = limit("longest_edge", "max_pixels")
    resolution = max(max(target_hw), 32 * math.ceil(math.sqrt(minimum) / 32))
    while True:
        shapes = [qwen_reference_size(image, resolution) for image in references]
        if maximum and any(width * height > maximum for width, height in shapes):
            raise ValueError("Rounded Qwen reference exceeds the configured VLM pixel budget")
        if all(width and height and width * height >= minimum for width, height in shapes):
            return resolution
        resolution += 32


@torch.no_grad()
def generate(pipe, condition, *, prompt, references, seed, steps=40, strength=1., source_hashes=None,
             local_context=None, local_size=512, local_slots=None, current_support=None,
             local_dilation=DEFAULT_LOCAL_DILATION, prior_strength=0.):
    """Generate globally or denoise and composite each selected local hand component."""
    condition.validate()
    images = list(references)
    if not 1 <= len(images) + (local_context is not None) <= 10:
        raise ValueError("Contact Hand editing requires 1 to 10 ordered source/context references")

    def render(target, inputs):
        height, width = target.canvas.canvas_hw
        reference_resolution = qwen_reference_resolution(pipe, inputs, (height, width))
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
        with bind_hand_condition(pipe.transformer, target, strength=strength, source_hashes=source_hashes,
                                 prior_strength=prior_strength):
            return pipe(prompt=prompt, image=inputs, width=width, height=height, output_resolution=reference_resolution,
                        generator=generator, num_inference_steps=steps, true_cfg_scale=1., use_kv_cache=False).images[0]

    if local_context is None:
        if current_support is not None:
            raise ValueError("Current hand support requires a local context image")
        return render(condition, images).convert("RGB")

    plans = _local_plans(condition, size=local_size, slots=local_slots, margin=.25,
                         current_support=current_support, dilation=local_dilation, single=False)
    for plan in plans:
        generated = render(plan.condition, [*images, plan.image(local_context)])
        local_context = plan.composite(local_context, generated)
    return local_context
