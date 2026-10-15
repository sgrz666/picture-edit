#!/usr/bin/env python3
"""Native Qwen-Image-2.1 cross-body-type pose retargeting probe.

Designed to run safely alongside the running Qwen21 training:
  - if free VRAM is tight, the pipeline uses model CPU offload (GPU footprint ~2-4 GiB)
  - a VRAM guard waits for headroom before every generation
  - outputs, timings and peak VRAM are recorded per sample

Inputs per pair:  A = appearance source (large build), B = pose reference (slim).
Goal: keep A's identity / clothing / body build, adopt B's pose.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw

ROOT = Path('/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test')
SET = ROOT / 'data/testset'
MODEL = Path('/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1')

GiB = 2 ** 30

# ---------------------------------------------------------------- 提示词
# 统一约定：image 1 = 外貌来源（A）；image 2 = 姿态参考（B）。
# 每对都显式描述 B 的肢体位置，并禁止把 B 的体型套到 A 身上。

BASE = (
    "Image 1 shows the person whose appearance must be kept. "
    "Image 2 shows the pose to adopt. "
)

# 冒烟测试发现：模型会保住体型，但衣着/面部会向 image 2 泄漏。
# 因此统一追加显式的「保留衣着 + 忽略杂人」条款。
KEEP = (
    " Strictly keep the outfit, accessories and personal items that the person in image 1 is wearing, "
    "exactly as they are — do not copy the clothing worn by the person in image 2. "
    "If image 1 contains other people or bystanders, ignore them completely and render only the main subject."
)

PROMPTS = {
    'hard_01': (
        BASE +
        "Produce a full-length studio photograph of the woman from image 1, keeping her exact face, "
        "hairstyle, skin tone, clothing and above all her own full-figured body shape. "
        "Place her in the standing pose of the man in image 2: upright and square to the camera, "
        "both forearms folded horizontally across the chest with the elbows held out, "
        "shoulders relaxed, head level and facing straight at the viewer, "
        "both feet planted firmly on the floor about shoulder-width apart. "
        "Use the plain light-grey seamless studio backdrop and the head-to-toe framing of image 2. "
        "She must remain herself — do not slim, stretch or replace her body with the man's build."
    ),
    'hard_02': (
        BASE +
        "Produce a full-length studio photograph of the woman from image 1, preserving her exact face, "
        "hair, skin tone, outfit and her own fuller figure. "
        "Apply the pose of the man in image 2: standing upright and facing the camera, "
        "arms crossed over the chest with the forearms stacked and the hands tucked near the opposite elbows, "
        "legs straight with the feet close together and pointing slightly outward, "
        "head facing forward with a calm expression. "
        "Keep the light neutral-grey studio background and the whole-body framing of image 2. "
        "Do not change her body proportions to match the slimmer man."
    ),
    'hard_03': (
        BASE +
        "Produce a full-length photograph of the woman from image 1, keeping her face, hair, "
        "skin tone, clothing and her own rounded body shape unchanged. "
        "Place her in the pose of the woman in image 2: the body turned roughly thirty-five degrees "
        "to her own right, the left hand lifted so the fingers touch her chin, "
        "the right arm folded across the waist, head tilted slightly down and to one side, "
        "the right leg stepped a little forward with the weight settled on the back leg, feet together. "
        "Reproduce the bright-white interior with the large pink wall artwork behind her, "
        "in the same full-length framing as image 2. "
        "She must keep her own figure — do not give her the slimmer woman's body."
    ),
    'hard_04': (
        BASE +
        "Produce a full-length photograph of the person from image 1, preserving her face, hair, "
        "skin tone, clothing and above all her own heavier build. "
        "Apply the pose of the woman in image 2: standing with both arms raised above shoulder height, "
        "elbows bent outward and hands lifted to about head level, chest open and shoulders back, "
        "legs set in a wide confident stance with both feet planted well apart, "
        "head straight and looking at the camera. "
        "Keep the rustic brick archway setting and the full-length framing of image 2. "
        "Do not slim her body down or copy the other woman's physique."
    ),
    'hard_05': (
        BASE +
        "Produce a full-length photograph of the woman from image 1, keeping her exact face, hair, "
        "skin tone, costume and her own full-figured body shape. "
        "Place her in the pose of the woman in image 2: standing with her back turned to the camera, "
        "torso upright and shoulders back, the head rotated back over her right shoulder so she looks "
        "straight at the viewer, the right arm lifted slightly away from the body, "
        "the legs straight and close together. "
        "Keep the ornate doorway setting and the head-to-toe framing of image 2. "
        "Preserve her costume exactly and do not replace her body with the slimmer woman's figure."
    ),
    'medium_01': (
        BASE +
        "Produce a full-length photograph of the large-built person from image 1, keeping their face, "
        "hair, skin tone, clothing and their own heavy build. "
        "Apply the standing pose of the man in image 2: standing upright and facing the camera, "
        "both hands slipped into the front trouser pockets, shoulders relaxed and slightly forward, "
        "legs straight with the feet planted comfortably apart. "
        "Keep the stone-wall outdoor setting and the full-length framing of image 2. "
        "Do not change their body proportions to the other man's."
    ),
    'medium_02': (
        BASE +
        "Produce a full-length photograph of the woman from image 1, preserving her face, hair, "
        "skin tone, clothing and her own rounded figure. "
        "Place her in the pose of the woman in image 2: standing squarely facing the camera, "
        "the left hand resting on her hip with the elbow out, the right arm hanging relaxed with a "
        "slight bend, the right leg stepped forward and slightly bent with the weight carried on the "
        "back leg, head straight with the chin level. "
        "Keep the dark leafy setting and the full-length framing of image 2. "
        "She must remain full-figured — do not adopt the slimmer woman's body."
    ),
}

# 消融：看「显式禁止改变体型」这句话到底有没有用
ABLATION = {
    'hard_05__minimal': (
        'hard_05',
        "Change the pose of the person in image 1 to match the pose of the person in image 2."
    ),
    'medium_01__minimal': (
        'medium_01',
        "Change the pose of the person in image 1 to match the pose of the person in image 2."
    ),
}


def vram():
    free, total = torch.cuda.mem_get_info()
    return free / GiB, total / GiB


def apply_vram_cap(reserve_gib: float = 14.0, max_allow_gib: float = 24.0) -> float:
    """Hard-cap this process's torch allocator so it can never crowd out concurrent training.

    Returns the allowed GiB. Excess requests raise OOM inside THIS process only;
    the training process keeps its own 42.5 GiB and its growth headroom untouched.
    """
    free, total = vram()
    allow = max(6.0, min(max_allow_gib, free - reserve_gib))
    torch.cuda.set_per_process_memory_fraction(allow * GiB / (total * GiB), 0)
    print(f'[mem] gpu free {free:.2f} GiB / total {total:.2f} GiB', flush=True)
    print(f'[mem] per-process cap = {allow:.2f} GiB '
          f'(reserve {reserve_gib:.1f} GiB untouched for training)', flush=True)
    return allow


def wait_for_headroom(need_gib: float = 2.5, timeout_s: float = 600.0) -> None:
    began = time.perf_counter()
    while time.perf_counter() - began < timeout_s:
        free, _ = vram()
        if free >= need_gib:
            return
        torch.cuda.empty_cache()
        print(f'    [guard] free {free:.2f} GiB < {need_gib} GiB, waiting…', flush=True)
        time.sleep(15)
    raise RuntimeError('no VRAM headroom within timeout')


def load_pipeline():
    free, total = vram()
    print(f'[mem] free {free:.2f} GiB / total {total:.2f} GiB', flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16)
    strategy = 'full_cuda' if free > 50 else 'model_cpu_offload'
    if strategy == 'full_cuda':
        pipe.to('cuda')
    else:
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    print(f'[mem] strategy = {strategy}', flush=True)
    free2, _ = vram()
    print(f'[mem] free after load {free2:.2f} GiB', flush=True)
    return pipe, strategy


def strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    h = 512
    tiles = []
    for im in images:
        im = im.convert('RGB')
        w = max(1, round(im.width * h / im.height))
        tiles.append(im.resize((w, h), Image.LANCZOS))
    W = sum(t.width for t in tiles) + 8 * (len(tiles) + 1)
    canvas = Image.new('RGB', (W, h + 26), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 8
    for t, label in zip(tiles, labels):
        canvas.paste(t, (x, 26))
        draw.text((x + 4, 8), label, fill=(240, 240, 240))
        x += t.width + 8
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'experiment/results/gen_v1'))
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--output-resolution', type=int, default=1024)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--only', default=None, help='逗号分隔的 pair id，用于冒烟')
    ap.add_argument('--ablation', action='store_true')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--vram-reserve-gib', type=float, default=14.0,
                    help='留给并行训练的显存余量，推理进程永远不会占用')
    args = ap.parse_args()

    apply_vram_cap(reserve_gib=args.vram_reserve_gib)

    man = json.loads((SET / 'manifest.json').read_text())
    pairs = {p['pair']: p for p in man['pairs']}

    jobs = []
    only = set(args.only.split(',')) if args.only else None
    for pid, prompt in PROMPTS.items():
        if only and pid not in only:
            continue
        jobs.append((pid, prompt + KEEP))
    if args.ablation:
        for key, (base_id, prompt) in ABLATION.items():
            if only and base_id not in only:
                continue
            jobs.append((key, prompt))

    out_dir = Path(args.out)
    imgs = out_dir / 'images'
    imgs.mkdir(parents=True, exist_ok=True)

    pipe, strategy = load_pipeline()
    rows = []
    for pid, prompt in jobs:
        base_id = pid.split('__')[0]
        p = pairs[base_id]
        A = Image.open(p['A_path']).convert('RGB')
        B = Image.open(p['B_path']).convert('RGB')
        for rep in range(args.repeats):
            tag = pid if args.repeats == 1 else f'{pid}_r{rep}'
            wait_for_headroom()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            began = time.perf_counter()
            gen = torch.Generator('cpu').manual_seed(args.seed + rep)
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=[A, B],
                    width=args.width,
                    height=args.height,
                    output_resolution=args.output_resolution,
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.cfg,
                    use_kv_cache=True,
                    generator=gen,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            peak = torch.cuda.max_memory_allocated() / GiB
            image = result.images[0]
            image.save(imgs / f'{tag}.png')
            strip([A, B, image], ['A appearance (heavy)', 'B pose (slim)', f'Qwen-Image-2.1  {tag}']
                  ).save(imgs / f'{tag}_strip.png')
            row = {'id': tag, 'base_pair': base_id, 'prompt': prompt,
                   'seconds': round(elapsed, 2), 'peak_cuda_gib': round(peak, 3),
                   'output_size': list(image.size), 'strategy': strategy,
                   'steps': args.steps, 'seed': args.seed + rep,
                   'width': args.width, 'height': args.height,
                   'output_resolution': args.output_resolution, 'cfg': args.cfg}
            rows.append(row)
            (out_dir / 'results.json').write_text(json.dumps(rows, indent=2))
            free, _ = vram()
            print(f'[{len(rows)}/{len(jobs) * args.repeats}] {tag}  {elapsed:.1f}s  '
                  f'peak {peak:.2f} GiB  out {image.size}  free {free:.2f} GiB', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
