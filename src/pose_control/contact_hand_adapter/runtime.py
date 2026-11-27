"""Standalone condition/checkpoint I/O and full-canvas or joint hand ROI generation."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile

import torch
from PIL import Image, ImageFilter

from src.data.canvas_contract import CanvasSpec, LETTERBOX
from src.pose_edit.common import sha256_file

from .conditions import HandCondition, HandMesh, HandPair
from .geometry import build_contact_graph, pixel_geometry, roi_boxes, support_box
from .integration import bind_hand_condition
from .model import AdapterConfig, ContactHandAdapter


CONDITION_SCHEMA = "contact_hand_condition_v2"
CHECKPOINT_SCHEMA = "contact_hand_adapter_v2"


def condition_record(condition):
    condition.validate()
    record = asdict(condition)
    record["canvas"] = condition.canvas.metadata()
    return {"schema": CONDITION_SCHEMA, **record}


def load_condition(path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.pop("schema", None) != CONDITION_SCHEMA:
        raise ValueError("Unsupported Contact Hand condition schema")
    record["canvas"] = CanvasSpec.from_record(record["canvas"])
    record["person_ids"] = tuple(record["person_ids"])
    if "appearance_sources" in record:
        record["appearance_sources"] = tuple(record["appearance_sources"])
    record["pairs"] = tuple(HandPair(**{**pair, "slots": tuple(pair["slots"])}) for pair in record["pairs"])
    if record["mesh"] is not None:
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
    record = dict(schema=CHECKPOINT_SCHEMA, base_sha=base_sha, config=asdict(adapter.config),
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
    if not isinstance(record, dict) or record.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported Contact Hand Adapter checkpoint schema")
    if record.get("base_sha") != base_sha:
        raise ValueError("Contact Hand Adapter checkpoint belongs to a different Qwen base")
    config = {**record["config"], "layers": tuple(record["config"]["layers"])}
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


def make_local_plan(condition, *, size=512, slots=None, margin=.25):
    """Select a connected hand group, remap conditions, and make a real high-resolution canvas."""
    condition = condition.to("cpu").validate()
    if type(size) is not int or size < 32 or size % 32 or not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("Local canvas size must be a multiple of 32 and margin must be in [0,1]")
    _, support = pixel_geometry(condition)
    graph = build_contact_graph(condition, support)
    components = [set(members) for members, _ in roi_boxes(support, graph)]
    if not components:
        raise ValueError("No eligible hand pixels for local generation")
    if slots is None:
        def priority(group):
            members = sorted(group)
            contacts = graph.part_state[members][:, members].eq(1).sum().item()
            return contacts, len(members), support[members].sum().item()

        selected = max(components, key=priority)
    else:
        requested = set(slots)
        active = set.union(*components)
        if not requested or not requested <= active:
            raise ValueError("Requested local hand slots must be visible in the target")
        selected = set.union(*(group for group in components if group & requested))
    slots = tuple(sorted(selected))
    x0, y0, x1, y1 = support_box(support[list(slots)].amax(0))
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
    local = replace(condition, canvas=canvas, joints=temp, visible=condition.visible & mask_slots[:, None],
                    support=spatial(crop_support), pairs=tuple(pair for pair in condition.pairs
                                                             if set(pair.slots) <= selected),
                    geometry_maps=dense, map_valid=dense_valid, mesh=mesh)
    local = replace(local, visible=local.active_joints()).validate()
    support_mask = crop_support.amax(0)[y0:y1, x0:x1]
    hard = Image.fromarray((support_mask * 255).round().byte().numpy())
    soft = hard.filter(ImageFilter.GaussianBlur(radius=min(4., max(original) * .025)))
    from PIL import ImageChops

    mask = ImageChops.lighter(hard, soft)
    return LocalPlan(box, slots, local, mask, condition.canvas.canvas_hw)


@torch.no_grad()
def generate(pipe, condition, *, prompt, references, seed, steps=40, strength=1., source_hashes=None,
             local_context=None, local_size=512, local_slots=None):
    """Generate on the specified canvas; with context, denoise a larger joint ROI and composite."""
    condition.validate()
    plan = None
    images = list(references)
    if local_context is not None:
        plan = make_local_plan(condition, size=local_size, slots=local_slots)
        condition = plan.condition
        images.append(plan.image(local_context))
    if not 1 <= len(images) <= 10:
        raise ValueError("Contact Hand editing requires 1 to 10 ordered source/context references")
    h, w = condition.canvas.canvas_hw
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    with bind_hand_condition(pipe.transformer, condition, strength=strength, source_hashes=source_hashes):
        result = pipe(prompt=prompt, image=images, width=w, height=h, output_resolution=max(h, w),
                      generator=generator, num_inference_steps=steps, true_cfg_scale=1., use_kv_cache=False).images[0]
    return result.convert("RGB") if plan is None else plan.composite(local_context, result)
