"""Standalone rectified-flow training and optional differentiable RGB supervision."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from src.pose_edit.common import sha256_file

from .augmentation import distort_source_context, swap_condition_people, swap_hand_slots
from .conditions import HandCondition, verify_appearance_sources
from .geometry import pixel_geometry
from .integration import bind_hand_condition
from .runtime import load_condition, make_local_plan


REGION_WEIGHTS = {"contact": 4., "hand": 3., "wrist": 1.5, "other": .5}
RGB_OBJECTIVES = ("joint", "part", "contact", "anatomy")
LOCAL_CONTEXT_KINDS = ("source", "coarse", "synthetic")


@dataclass(frozen=True)
class FlowBatch:
    condition: HandCondition
    target: torch.Tensor  # [1,N_target,64], supervision only
    references: torch.Tensor  # [1,N_reference,64]
    prompt_embeds: torch.Tensor
    prompt_mask: torch.Tensor | None
    img_mask: torch.Tensor
    img_shapes: tuple[tuple[int, int, int], ...]
    source_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionSupervision:
    hands: torch.Tensor  # [4,H,W], target hand instances
    wrists: torch.Tensor | None = None  # [4,H,W]
    contacts: torch.Tensor | None = None  # [num_contact_pairs,H,W], only confirmed positives


@dataclass(frozen=True)
class TrainingPair:
    condition: HandCondition
    references: tuple[Image.Image, ...]
    target: Image.Image
    prompt: str
    regions: RegionSupervision
    rgb_labels: dict | None
    source_hashes: tuple[str, ...]
    context_kind: str | None = None
    person_swap_status: str = "not_requested"


def swap_training_pair(pair, *, prompt_safe=None, swapped_prompt=None):
    """Swap actor slots and ordered identities, keeping uncertain supervision intact."""
    if pair.rgb_labels:
        return replace(pair, person_swap_status="rgb_labels")
    if prompt_safe is False:
        return replace(pair, person_swap_status="ordered_prompt")
    condition = pair.condition
    people = sum(actor is not None for actor in condition.person_ids)
    if not people:
        return replace(pair, person_swap_status="no_actor")
    independent = not condition.shared_appearance_source and len(pair.source_hashes) >= 2
    # An existing B-only pack can explicitly bind its crop to reference two.
    sole_b_uses_second = (people == 1 and condition.person_ids[0] is None
                          and condition.appearance_valid is not None and condition.appearance_valid[2:].any())
    reorder = independent and (people == 2 or sole_b_uses_second)
    if reorder and prompt_safe is not True and swapped_prompt is None:
        return replace(pair, person_swap_status="ordered_prompt")
    if any(value is not None and value.shape[0] != 4 for value in (pair.regions.hands, pair.regions.wrists)):
        return replace(pair, person_swap_status="region_layout")
    references, source_hashes = pair.references, pair.source_hashes
    if reorder:
        references = (references[1], references[0], *references[2:])
        source_hashes = (source_hashes[1], source_hashes[0], *source_hashes[2:])
    condition = swap_condition_people(condition, shared_appearance_source=people == 1 or not independent)
    verify_appearance_sources(condition, source_hashes)
    regions = replace(pair.regions, hands=swap_hand_slots(pair.regions.hands),
                      wrists=swap_hand_slots(pair.regions.wrists))
    return replace(pair, condition=condition, references=references, source_hashes=source_hashes,
                   prompt=pair.prompt if swapped_prompt is None else swapped_prompt, regions=regions,
                   person_swap_status="swapped")


class LocalContextMixture:
    """Draw an explicit context kind, then a uniform row of that kind."""

    def __init__(self, local_rows, weights):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        if (self.weights.shape != (len(LOCAL_CONTEXT_KINDS),) or not torch.isfinite(self.weights).all()
                or (self.weights < 0).any() or not (self.weights > 0).any()):
            raise ValueError("Local context mixing requires three nonnegative weights with a positive total")
        self.weights = self.weights / self.weights.max()
        self.pools = tuple(tuple(row for row in local_rows if row["context_kind"] == kind)
                           for kind in LOCAL_CONTEXT_KINDS)
        for kind, weight, pool in zip(LOCAL_CONTEXT_KINDS, self.weights, self.pools):
            if weight and not pool:
                raise ValueError(f"Local context mix has positive weight but no {kind} rows")

    def sample(self, rng):
        pool = self.pools[int(torch.multinomial(self.weights, 1, generator=rng))]
        return pool[int(torch.randint(len(pool), (), generator=rng))]


def load_manifest(path):
    """Resolve and check every asset before launching a hand training run."""
    path = Path(path).resolve(strict=True)
    records = []
    ids, groups, images, hashes = set(), {}, {}, {}

    def asset(entry):
        location = (path.parent / entry["path"]).resolve(strict=True)
        if location not in hashes:
            hashes[location] = sha256_file(location)
        if hashes[location] != entry["sha256"]:
            raise ValueError(f"Asset SHA256 differs from the hand manifest: {location}")
        return location

    for line_no, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        row = json.loads(text)
        pair_id, group, split, mode = (row[key] for key in ("id", "group", "split", "mode"))
        if (not pair_id or pair_id in ids or not group or split not in ("train", "val", "test")
                or mode not in ("global", "local") or not row["prompt"].strip()):
            raise ValueError(f"Invalid hand manifest row {line_no}")
        ids.add(pair_id)
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Training group crosses splits: {group}")
        refs = row["source_refs"]
        if not refs or len(refs) + (row.get("context") is not None) > 10:
            raise ValueError("Expected 1 to 10 ordered Qwen source/context references")
        if (row.get("context") is not None) != (mode == "local"):
            raise ValueError("Local training requires target-canvas context; global training omits context")
        context_kind = row.get("context_kind")
        support = row.get("current_support")
        legacy_local_crop = mode == "local" and context_kind is None and support is None
        if ((context_kind is not None and context_kind not in LOCAL_CONTEXT_KINDS)
                or (mode == "global" and (context_kind is not None or support is not None))):
            raise ValueError("Local context_kind must be source, coarse or synthetic and applies only to local rows")
        source_hashes = tuple(entry["sha256"] for entry in refs)
        if mode == "local":
            source_context = row["context"]["sha256"] in source_hashes
            if context_kind == "synthetic" and (support is None or not source_context):
                raise ValueError("Synthetic local context requires current_support and context bound to a source reference")
            if context_kind is None:
                context_kind = "source" if source_context else "unspecified"
        prompt_safe, swapped_prompt = row.get("person_swap_safe"), row.get("prompt_on_person_swap")
        if ((prompt_safe is not None and type(prompt_safe) is not bool)
                or (swapped_prompt is not None and
                    (not isinstance(swapped_prompt, str) or not swapped_prompt.strip() or prompt_safe is False))):
            raise ValueError("Invalid person_swap_safe or prompt_on_person_swap in hand manifest")
        inputs = [*refs] if row.get("context") is None else [*refs, row["context"]]
        target_sha = row["target"]["sha256"]
        if any(entry["sha256"] == target_sha for entry in inputs):
            raise ValueError("The target RGB must be separate from all source and context references")
        for entry in [*inputs, row["target"]]:
            if images.setdefault(entry["sha256"], split) != split:
                raise ValueError("A source or target image crosses training splits")
        record = {**row, "source_refs": tuple(asset(entry) for entry in refs),
                  "source_hashes": source_hashes, "context_kind": context_kind, "legacy_local_crop": legacy_local_crop}
        for key in ("target", "condition", "context", "current_support", "regions", "rgb_labels"):
            record[key] = asset(row[key]) if row.get(key) is not None else None
        if record["condition"] is None:
            raise ValueError("Every hand pair requires a target geometry pack")
        records.append(record)
    if not records:
        raise ValueError("Hand training manifest is empty")
    return records


def open_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def load_training_pair(record, *, local_size=512, swap_people=False, rng=None):
    condition = load_condition(record["condition"])
    verify_appearance_sources(condition, record["source_hashes"])
    references = tuple(open_rgb(path) for path in record["source_refs"])
    target = open_rgb(record["target"])
    canvas = condition.canvas
    if target.size != tuple(reversed(canvas.canvas_hw)):
        target = canvas.image(target)
    context_kind = record.get("context_kind")
    if record["mode"] == "local":
        current_support = None
        if record.get("current_support") is not None:
            support_record = torch.load(record["current_support"], map_location="cpu", weights_only=True)
            if (not isinstance(support_record, dict) or support_record.get("canvas_sha") != canvas.fingerprint()
                    or not torch.is_tensor(support_record.get("support"))):
                raise ValueError("Current hand support must be authored in the full target canvas")
            current_support = support_record["support"]
        # Legacy authored masks were fingerprinted before edit-mask dilation changed the crop.
        crop_options = {"dilation": 0} if record.get("legacy_local_crop") else {}
        plan = make_local_plan(condition, size=local_size, slots=record.get("slots"),
                               current_support=current_support, **crop_options)
        condition, target = plan.condition, plan.image(target)
        if record["context"] is not None:
            context = open_rgb(record["context"])
            if context_kind == "synthetic" and context.size != tuple(reversed(canvas.canvas_hw)):
                context = canvas.image(context)
            context = plan.image(context)
            if context_kind == "synthetic":
                if current_support is None:
                    raise ValueError("Synthetic local context requires declared current hand support")
                context = distort_source_context(context, plan, current_support,
                                                 rng if rng is not None else torch.Generator().manual_seed(0))
            references += (context,)
    heatmaps, hands = pixel_geometry(condition)
    if not hands.any():
        raise ValueError("A hand training row requires usable target hand geometry")
    if record["regions"] is None:
        wrists = (heatmaps[:, 0] > .16).float()
        regions = RegionSupervision(hands, wrists)
    else:
        region = torch.load(record["regions"], map_location="cpu", weights_only=True)
        if region.pop("canvas_sha", None) != condition.canvas.fingerprint():
            raise ValueError("Loss regions must be authored in the final target canvas")
        regions = RegionSupervision(**region)
    masks = token_regions(condition.canvas, regions, "cpu")
    if not masks["hand"].any() and not masks["contact"].any():
        raise ValueError("A hand training row needs eligible target hand or contact supervision")
    labels = None
    if record["rgb_labels"] is not None:
        labels = torch.load(record["rgb_labels"], map_location="cpu", weights_only=True)
        if labels.pop("canvas_sha", None) != condition.canvas.fingerprint():
            raise ValueError("RGB supervision must be authored in the final target canvas")
        eligible_output_labels(labels)
    pair = TrainingPair(condition, references, target, record["prompt"], regions, labels, record["source_hashes"],
                        context_kind=context_kind)
    if swap_people:
        pair = swap_training_pair(pair, prompt_safe=record.get("person_swap_safe"),
                                  swapped_prompt=record.get("prompt_on_person_swap"))
    return pair


def predict_flow(transformer, batch, sigma, noise):
    if (sigma.shape != (1,) or not torch.isfinite(sigma).all() or not ((sigma >= 0) & (sigma <= 1)).all()
            or noise.shape != batch.target.shape):
        raise ValueError("Flow step requires a normalized sigma and noise on the target grid")
    time = sigma[:, None, None].float()
    target = batch.target.float()
    noisy = (1 - time) * target + time * noise.float()
    hidden = torch.cat((batch.references, noisy.to(batch.references.dtype)), 1)
    with bind_hand_condition(transformer, batch.condition, source_hashes=batch.source_hashes):
        output = transformer(
            hidden_states=hidden, timestep=sigma, encoder_hidden_states=batch.prompt_embeds,
            encoder_hidden_states_mask=batch.prompt_mask, img_shapes=[list(batch.img_shapes)],
            img_mask=batch.img_mask, return_dict=False,
        )[0]
    predicted = output[:, -target.shape[1]:].float()
    return predicted, noise.float() - target, noisy - time * predicted


def token_regions(canvas, supervision, device):
    """Clip and resolve pixel ownership before pooling to the target grid."""
    valid = canvas.valid_mask(device=device)
    remaining = valid
    regions = []
    for masks in (supervision.contacts, supervision.hands, supervision.wrists):
        value = remaining[:0] if masks is None else masks.to(device=device, dtype=torch.float32)
        if (value.ndim != 3 or value.shape[-2:] != remaining.shape[-2:] or not torch.isfinite(value).all()
                or ((value < 0) | (value > 1)).any()):
            raise ValueError("Region supervision must be finite canvas masks in [0,1]")
        value = value / value.sum(0).clamp_min(1) * remaining
        remaining = remaining - value.sum(0, keepdim=True)
        regions.append(value)
    regions.append(remaining.clamp_min(0))
    return {name: F.avg_pool2d(value[:, None].clamp_min(0), canvas.stride).flatten(1)
            for name, value in zip(REGION_WEIGHTS, regions)}


def region_flow_objective(predicted, target, canvas, supervision, *, denominators=None):
    if (predicted.ndim != 3 or predicted.shape != target.shape
            or predicted.shape[:2] != (1, canvas.token_count)):
        raise ValueError("Flow velocity shapes must match the target canvas")
    error = (predicted.float() - target.float()).square().mean(-1).squeeze(0)
    masks = token_regions(canvas, supervision, error.device)
    terms = {}
    for name, mask in masks.items():
        area = mask.sum(1)
        eligible = area > 0
        per_region = (mask * error).sum(1) / area.clamp_min(.5)
        term = per_region.sum() / eligible.sum().clamp_min(1)
        terms[name] = term if denominators is None else term / max(1, denominators[name])
    loss = sum(terms[name] * weight for name, weight in REGION_WEIGHTS.items())
    return loss, terms


def eligible_output_labels(labels):
    eligible = []
    for name, (target, valid) in labels.items():
        if (name not in RGB_OBJECTIVES or not torch.is_tensor(target) or not torch.is_tensor(valid)
                or valid.dtype != torch.bool or valid.shape != target.shape):
            raise ValueError("Invalid RGB supervision objective or target/valid array")
        known = target[valid]
        if not torch.isfinite(known).all():
            raise ValueError("Eligible RGB labels must be finite")
        if name in ("contact", "anatomy") and not ((known >= 0) & (known <= 1)).all():
            raise ValueError("Binary RGB labels must be in [0,1]; mask out unknown entries")
        if name == "part" and (known.is_floating_point() or (known < 0).any()):
            raise ValueError("Part labels must be nonnegative integer classes")
        if known.numel():
            eligible.append(name)
    return tuple(eligible)


def output_supervision(predictions, labels):
    """Losses on frozen-head predictions made from decoded RGB; unknown labels are masked."""
    terms = {}
    for name, (target, valid) in labels.items():
        if name not in RGB_OBJECTIVES:
            raise ValueError(f"Unknown RGB hand objective {name!r}")
        prediction = predictions[name].float()
        valid = valid.to(device=prediction.device, dtype=torch.bool)
        target = target.to(prediction.device)
        if valid.shape != target.shape:
            raise ValueError("RGB labels and their eligibility mask must match")
        if name == "part":
            scores = prediction.movedim(1, -1)
            if scores.shape[:-1] != target.shape:
                raise ValueError("Part logits must have a channel axis over the label canvas")
            values = scores[valid]
        else:
            if prediction.shape != target.shape:
                raise ValueError(f"RGB {name} prediction and label shapes disagree")
            values = prediction[valid]
        target = target[valid]
        if not values.numel():
            terms[name] = values.sum()
        elif name == "joint":
            terms[name] = F.smooth_l1_loss(values, target.float())
        elif name == "part":
            terms[name] = F.cross_entropy(values, target.long())
        else:
            terms[name] = F.binary_cross_entropy_with_logits(values, target.float())
    return terms


def check_output_gradients(clean, losses):
    """A newly enabled RGB head must provide a usable gradient through the frozen VAE."""
    for name, loss in losses.items():
        gradient = (torch.autograd.grad(loss, clean, retain_graph=True, allow_unused=True)[0]
                    if loss.requires_grad else None)
        if gradient is None or not torch.isfinite(gradient).all() or not gradient.any():
            raise ValueError(f"RGB {name} head must supply a finite nonzero gradient to the clean latent")


def decode_clean_latent(pipe, clean, canvas):
    """Decode with the frozen VAE while keeping gradients to the velocity prediction."""
    h, w = canvas.token_hw
    if clean.ndim != 3 or clean.shape[1] != h * w or clean.shape[2] != pipe.vae.config.z_dim:
        raise ValueError("Clean latent shape must match the Qwen VAE and target canvas")
    latent = clean.transpose(1, 2).reshape(clean.shape[0], clean.shape[2], 1, h, w).to(pipe.vae.dtype)
    mean = latent.new_tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latent.new_tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    decoded = pipe.vae.decode(latent * std + mean, return_dict=False)[0]
    return (decoded[:, :3, 0].float() * .5 + .5).clamp(0, 1)


@torch.no_grad()
def encode_pair(pipe, *, condition, references, target, prompt, source_hashes=()):
    """Build Qwen VLM and VAE inputs using the same resize and packing as inference."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    canvas = condition.canvas
    h, w = canvas.canvas_hw
    if not references or target.size != (w, h):
        raise ValueError("Expected ordered source RGB and a target on the declared canvas")
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(0)

    def encode(image, width, height):
        pixels = pipe.image_processor.preprocess(image.convert("RGBA"), width=width, height=height).unsqueeze(2)
        latent = pipe._encode_vae_image(pixels.to(device=device, dtype=pipe.vae.dtype), generator)
        return pipe._pack_latents(latent, 1, latent.shape[1], latent.shape[-2], latent.shape[-1])

    resized, ref_latents, grids = [], [], []
    for image in references:
        ref_width, ref_height, _ = calculate_dimensions(max(h, w)**2, image.width / image.height)
        resized.append(pipe.image_processor.resize(image.convert("RGBA"), width=ref_width, height=ref_height))
        ref_latents.append(encode(image, ref_width, ref_height))
        grids.append((1, ref_height // canvas.stride, ref_width // canvas.stride))
    prompt_embeds, prompt_mask, prefix_mask = pipe.encode_prompt(prompt=prompt, image=resized, device=device)
    target_slots = prefix_mask.new_ones(1, canvas.token_count // 4)
    return FlowBatch(condition, encode(target, w, h), torch.cat(ref_latents, 1).to(prompt_embeds.dtype),
                     prompt_embeds, prompt_mask, torch.cat((prefix_mask, target_slots), 1),
                     (*grids, (1, *canvas.token_hw)), source_hashes)
