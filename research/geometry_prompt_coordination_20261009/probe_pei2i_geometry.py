"""Research-only PE-I2I geometry reading probe; no diffusion or Adapter training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I"))
    parser.add_argument("--audit", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_full/native_value_audit_20261008"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise SystemExit("Existing results found; use another output directory.")
    free, _ = torch.cuda.mem_get_info()
    if free < 30 * 1024**3:
        raise SystemExit("Less than 30 GiB GPU memory free; refusing to compete with another job.")
    rows = [json.loads(line) for line in (args.audit / "data/pairs.jsonl").read_text().splitlines()]
    cases = []
    for case_id in ("tiktok_03", "tiktok_05"):
        row = next(row for row in rows if row["pair_id"] == case_id)
        for mode in ("skeleton", "skeleton_normal_depth"):
            roles = [("source_A", row["source_rgb"]), ("target_dwpose", row["target_dwpose"])]
            instruction = (
                "Edit only the posture and head orientation of the existing person in <image1>. "
                "<image1> is the source photograph and the sole source of identity, hairstyle, clothing, "
                "accessories and background. <image2> is a target keypoint skeleton guide, not another "
                "photograph or a source of appearance. "
            )
            if mode != "skeleton":
                roles += [("target_normal", row["target_normal"]), ("target_depth_vis", row["target_depth_vis"])]
                instruction += (
                    "<image3> is the target SMPL-X surface-normal visualization: its RGB colors encode "
                    "surface orientation, not skin or clothing colors. <image4> is the target visualized "
                    "depth map, not the appearance of a person. These three guides describe the same "
                    "target pose. "
                )
            instruction += (
                "Write an actionable pose edit based only on geometry visible in the guides. Use the "
                "viewer's image-left/image-right for spatial descriptions. Preserve the source person's "
                "identity and outfit while allowing their posture and head direction to change. Preserve "
                "the source scene and framing. Transfer no guide colors or guide lines into the output. "
                "Do not invent objects, emotional expressions, or hidden joint positions. Omit pose "
                "details you cannot determine. The output is one photograph of the same person."
            )
            paths = [Path(path) for _, path in roles]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            cases.append({
                "case_id": case_id, "mode": mode, "instruction": instruction,
                "images": [{"role": role, "path": str(path), "sha256": sha256(path),
                            "size": list(Image.open(path).size)} for (role, _), path in zip(roles, paths)],
                "reviewed_pose_for_posthoc_comparison_only": row["action_description"],
            })
    system = (args.model / "system_prompt.txt").read_text().strip()
    manifest = {
        "scope": "prompt rewriting only; no diffusion image generation or Adapter training",
        "model": str(args.model), "system_prompt_sha256": sha256(args.model / "system_prompt.txt"),
        "thinking": True, "seed": 42, "sampling": {"max_new_tokens": 24000, "do_sample": True,
        "temperature": 1.0, "top_p": .95, "top_k": 20}, "source_images_resized_by_probe": False,
        "target_B_rgb_given_to_model": False, "comparison_caption_given_to_model": False,
        "cases": cases,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    processor = AutoProcessor.from_pretrained(args.model)
    print("Loading existing PE-I2I checkpoint", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    results = []
    for case in cases:
        set_seed(42)
        content = [{"type": "image", "image": Image.open(item["path"]).convert("RGB")} for item in case["images"]]
        content.append({"type": "text", "text": case["instruction"]})
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", enable_thinking=True).to(model.device)
        start = time.monotonic()
        print("START", case["case_id"], case["mode"], "input_tokens", inputs["input_ids"].shape[1], flush=True)
        with torch.inference_mode():
            output = model.generate(**inputs, **manifest["sampling"])
        generated = output[0, inputs["input_ids"].shape[1]:]
        raw = processor.tokenizer.decode(generated, skip_special_tokens=True)
        _, separator, answer = raw.partition("</think>")
        answer = answer.strip() if separator else raw.strip()
        try:
            parsed = json.loads(answer)
            parse_error = None
        except Exception as error:
            parsed = None
            parse_error = str(error)
        result = {"case_id": case["case_id"], "mode": case["mode"], "seconds": time.monotonic() - start,
                  "output_tokens": len(generated), "parse_ok": isinstance(parsed, dict),
                  "parse_error": parse_error, "parsed": parsed, "raw": raw}
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print("DONE", result["case_id"], result["mode"], round(result["seconds"], 1),
              "seconds", result["output_tokens"], "tokens", "parse_ok", result["parse_ok"], flush=True)
        if parsed:
            print(json.dumps(parsed, ensure_ascii=False), flush=True)
        del inputs, output
    print("Completed four research probes", flush=True)


if __name__ == "__main__":
    main()
