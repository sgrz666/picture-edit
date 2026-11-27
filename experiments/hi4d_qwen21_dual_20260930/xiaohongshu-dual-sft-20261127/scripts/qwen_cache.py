"""Prepare and shard the frozen four-reference Qwen cache independently of Native assets."""
import argparse
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

from select_pairs import read, ref, write

DATA = Path("/data/shangguanrz")
MODEL = DATA / "pic-edit-models/Qwen-Image-2.1"
MODEL_MANIFEST = (DATA / "pic-edit-experiments/one/qwen21_full_native_v1"
                  / "tiktok-fixedpair-v5_20261025_v1/source/qwen_model_files.json")
MODEL_SHA = "a8c34c15ce9632313dec2f1d96ff2629315d6a8d94b657b4539c196b2f6bcebb"
EXPERIMENT = DATA / "pic-edit-experiments/two/qwen21_hi4d_dual"
INIT = EXPERIMENT / "native-v7-singlepair-overfit-20261117-v1/initialization/bilibili20k-v7-dual192new.pt"
INIT_SHA = "17b23faec7c39c1c8e0b99f4bd05b82bbfe646eb3695a9336459d44796c72331"
EVAL = DATA / "pic-edit-models/repo-models/face_eval/identity_encoder.ts"
EVAL_SHA = "bf059ec89614024682639ef3eaffecd3ef0933aae9c114df4097636d800269d6"
PROMPT = ("Edit the two people in <image1> to match the shared pose in <image2>. "
          "Use <image3> for the person on the left in <image1> and <image4> for the person "
          "on the right in <image1>. Preserve each person's identity, clothing and the background of <image1>.")


def pinned(path, digest):
    value = ref(path)
    if value["sha256"] != digest:
        raise ValueError(f"Frozen asset differs: {path}")
    return value


def target_guide_slots(pair):
    first = min((0, 1), key=lambda slot: pair["source"]["face_centers"][slot][0])
    return [pair["source_indices"].index(first), pair["source_indices"].index(1 - first)]


def prepare(root, selection, long_side):
    from PIL import Image
    from src.pose_control.qwen21_full.native_v5.conditioning import SCHEMA, read_condition
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for

    @lru_cache(maxsize=None)
    def guide(path, digest, hw):
        path = Path(path)
        if ref(path)["sha256"] != digest:
            raise ValueError(f"SDPose guide changed: {path}")
        canvas = canvas_for(hw, long_side)
        out = root / "guides" / f"{digest}.png"
        out.parent.mkdir(exist_ok=True)
        if not out.exists():
            with Image.open(path) as image, out.open("xb") as stream:
                canvas.image(image.convert("RGB"), fill=0).save(stream, format="PNG")
        return ref(out)

    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    entries = {}
    for pair in read(selection)["pairs"]:
        pid = pair["id"]
        source, target = pair["source"], pair["target"]
        canvases = [canvas_for(frame["hw"], long_side) for frame in (source, target)]
        source_canvas, target_canvas = canvases
        if source["rgb"] != ref(source["rgb"]["path"]) or target["rgb"] != ref(target["rgb"]["path"]):
            raise ValueError(f"Locked RGB differs: {pid}")
        bank = root / "prompts" / f"{pid}.json"
        write(bank, {"schema": "xhs_dual_4ref_prompt_v1", "pair_id": pid,
                     "compiled_prompt_inference": PROMPT, "compiled_prompt_inference_sha256": prompt_sha,
                     "target_slots_by_source_side": target_guide_slots(pair)})
        images = [{"role": "source", "file": source["rgb"], "canvas": source_canvas.metadata(),
                   "canvas_id": source_canvas.fingerprint(), "space": "original"}]
        a, b = target_guide_slots(pair)
        for role, slot in (("combined_pose", "combined_pose"), ("person_0_pose", f"p{a}"), ("person_1_pose", f"p{b}")):
            bound = target["files"][slot]
            images.append({"role": role, "file": guide(bound["path"], bound["sha256"], tuple(target["hw"])),
                           "canvas": target_canvas.metadata(), "canvas_id": target_canvas.fingerprint(),
                           "space": "canvas"})
        path = root / "conditions" / f"{pid}.json"
        write(path, {"schema": SCHEMA, "profile": "dual_4ref", "pair_id": pid, "primary_source_index": 0,
                     "target_canvas_id": target_canvas.fingerprint(), "target_pose_backend": "sdpose",
                     "images": images, "prompt": {"text": PROMPT, "sha256": prompt_sha,
                     "field": "compiled_prompt_inference", "bank": ref(bank)}})
        condition = ref(path)
        read_condition(condition)
        entries[pair["split"] + "\0" + pid] = {
            "condition": condition, "source_rgb_sha256": source["rgb"]["sha256"],
            "driver_rgb_sha256": target["rgb"]["sha256"], "target_rgb": target["rgb"],
            "source_canvas_id": source_canvas.fingerprint(), "target_canvas_id": target_canvas.fingerprint(),
            "source_canvas": source_canvas.metadata(), "target_canvas": target_canvas.metadata()}
    write(root / "qwen-inputs.json", {"selection": ref(selection), "entries": entries, "long_side": long_side})
    print(f"Declared {len(entries)} verified Qwen conditions", flush=True)


def encode(root, shard, shards):
    import torch
    from diffusers import QwenImage21Pipeline
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_full_data import _rgb
    from src.pose_control.qwen21_full.native_v62.conditioning import encode_pair, read_condition
    from src.pose_control.qwen21_full.native_v62.training.backend import model_identity

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    manifest = pinned(MODEL_MANIFEST, MODEL_SHA)
    model_identity(SimpleNamespace(model_path=str(MODEL), model_manifest=manifest))
    inputs = read(root / "qwen-inputs.json")["entries"]
    items = list(inputs.items())[shard::shards]
    destination = root / "cache" / f"shard-{shard}"
    destination.mkdir(parents=True, exist_ok=False)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16, local_files_only=True)
    pipe.text_encoder.requires_grad_(False).eval().to("cuda")
    pipe.vae.requires_grad_(False).eval().to("cuda")
    entries = {}
    for i, (key, item) in enumerate(items, 1):
        source, target = (CanvasSpec.from_record(item[role + "_canvas"]) for role in ("source", "target"))
        spec = read_condition(item["condition"])
        sample = {"target_token_hw": target.token_hw, "source_token_hw": source.token_hw,
                  "tgt_image": _rgb(item["target_rgb"], target)}
        pair = encode_pair(pipe, sample, spec)
        pair["condition_sha256"] = item["condition"]["sha256"]
        out = destination / (key.split("\0")[1] + ".pt")
        with out.open("xb") as stream:
            torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in pair.items()}, stream)
        entries[key] = {"file": ref(out), "condition_sha256": pair["condition_sha256"],
                        **{k: item[k] for k in ("source_rgb_sha256", "driver_rgb_sha256",
                                              "source_canvas_id", "target_canvas_id")}}
        if i == 1 or i % 25 == 0 or i == len(items):
            print(f"Frozen Qwen shard {shard} {i}/{len(items)}", flush=True)
    write(destination / "complete.json", entries)


def finalize(root, shards):
    from src.pose_control.qwen21_full.native_v62.conditioning import CACHE_SCHEMA
    inputs = read(root / "qwen-inputs.json")["entries"]
    entries = {}
    for shard in range(shards):
        current = read(root / "cache" / f"shard-{shard}" / "complete.json")
        if set(entries) & set(current):
            raise ValueError("Frozen cache shards overlap")
        entries.update(current)
    bindings = {key: row["condition"]["sha256"] for key, row in inputs.items()}
    if bindings != {key: row["condition_sha256"] for key, row in entries.items()}:
        raise ValueError("Frozen cache coverage differs from declared conditions")
    write(root / "cache/index.json", {"schema": CACHE_SCHEMA, "profile": "dual_4ref", "entries": entries,
          "model_manifest_sha256": MODEL_SHA, "condition_bindings": bindings})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "encode", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.selection, args.long_side)
    elif args.action == "encode":
        encode(args.root, args.shard, args.shards)
    else:
        finalize(args.root, args.shards)
