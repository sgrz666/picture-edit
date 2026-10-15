#!/usr/bin/env python3
"""Hi4D two-person interaction probe on native Qwen-Image-2.1.

Task = the project's own MAIN_N2C definition:
    source frame : same identity pair, same camera, two people NOT in contact  -> appearance/identity
    goal         : the same two people, now performing the named interaction (contact)

Three arms per sample:
    detailed  : source only + a per-interaction prompt that spells out limb positions
    minimal   : source only + a one-line generic prompt            (ablation)
    w_ref     : source + the target frame as image 2               (reference upper bound)

Runs alongside training under a hard per-process VRAM cap.
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
SET = ROOT / 'data/interaction/testset'
MODEL = Path('/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1')

GiB = 2 ** 30

PEOPLE = ("Image 1 is a studio photograph of two people standing apart without touching each other. "
          "They are the two subjects you must render.")

KEEP = (" Keep both people exactly as they appear in image 1 - the same two faces, hairstyles, skin tones, "
        "body sizes and the same clothing on each of them - and keep the plain green studio background, "
        "the camera angle and the full-length framing unchanged. Do not add, remove or replace either person.")

# 每种交互：肢体位置写死（依据 Hi4D 目标帧实拍）
INTERACTION = {
    'embrace': (
        "The two people step together and hold each other in a warm embrace. "
        "The person who was on the left slides in beside the other, their torsos pressed together and "
        "turned about thirty degrees away from the camera. The left person's right arm reaches over the "
        "other's far shoulder with the forearm resting across the upper back, while the right person's "
        "left arm wraps around the first person's waist. Their heads lean towards each other, and both "
        "keep their feet planted on the floor."
    ),
    'taichi partner push-hands': (
        "The two people square off into a taichi push-hands stance. Both drop into a slight crouch with "
        "knees bent and feet a shoulder-width apart, facing each other about one step away. Both raise "
        "their arms to chest height with the elbows bent and the palms open, so that their hands meet and "
        "press against each other in the space between them."
    ),
    'partner yoga': (
        "The two people perform a supported partner yoga pose. They stand very close, back to side, with "
        "the taller person upright and the other leaning backwards against them. Both raise their arms "
        "overhead and join hands above their heads, and their bodies touch along the side of the torso "
        "and hips while both keep their feet on the floor."
    ),
    'embrace from behind': (
        "The person standing behind steps in and embraces the other from behind. The front person stands "
        "relaxed facing the camera, and the person behind presses against their back and wraps both arms "
        "around them, with one arm crossing the front person's chest and the other resting on the front "
        "person's waist. Their heads are level and close together and both stay on their feet."
    ),
    'partner dance': (
        "The two people take up a closed partner-dance hold. They face each other closely, bodies turned "
        "slightly open to the camera. Each extends one arm to hold the partner's hand at chest height "
        "between them, and each places the other hand on the partner's back or shoulder. They lean "
        "slightly towards each other and stand on their own feet."
    ),
    'partner pose': (
        "The two people stand shoulder to shoulder and strike a mirrored pose together. They move in close "
        "so their shoulders touch, both raise their arms straight overhead and bring their hands together "
        "above their heads, and both stand upright with their weight evenly on both feet."
    ),
}

MINIMAL = ("Make the two people in this image interact: they must physically touch each other.")


def vram():
    free, total = torch.cuda.mem_get_info()
    return free / GiB, total / GiB


def apply_vram_cap(reserve_gib=14.0, max_allow_gib=24.0):
    free, total = vram()
    allow = max(6.0, min(max_allow_gib, free - reserve_gib))
    torch.cuda.set_per_process_memory_fraction(allow * GiB / (total * GiB), 0)
    print(f'[mem] gpu free {free:.2f} GiB / total {total:.2f} GiB', flush=True)
    print(f'[mem] per-process cap = {allow:.2f} GiB (reserve {reserve_gib:.1f} GiB for training)', flush=True)
    return allow


def wait_for_headroom(need_gib=2.5, timeout_s=600.0):
    began = time.perf_counter()
    while time.perf_counter() - began < timeout_s:
        free, _ = vram()
        if free >= need_gib:
            return
        torch.cuda.empty_cache()
        time.sleep(15)
    raise RuntimeError('no VRAM headroom')


def load_pipeline():
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16)
    free, _ = vram()
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
    return pipe, strategy


def strip(images, labels):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'experiment/results/interaction_v1'))
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--output-resolution', type=int, default=1024)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--per-type', type=int, default=1, help='每种交互取几个样本')
    ap.add_argument('--arms', default='detailed,minimal,w_ref')
    ap.add_argument('--vram-reserve-gib', type=float, default=14.0)
    args = ap.parse_args()

    apply_vram_cap(reserve_gib=args.vram_reserve_gib)

    samples = json.loads((SET / 'manifest.json').read_text())['samples']
    by_type = {}
    for s in samples:
        by_type.setdefault(s['interaction'], []).append(s)
    chosen = []
    for name, group in by_type.items():
        chosen.extend(group[:args.per_type])

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    jobs = []
    for s in chosen:
        body = INTERACTION.get(s['interaction'])
        if body is None:
            continue
        for arm in arms:
            if arm == 'detailed':
                prompt, refs = PEOPLE + ' ' + body + KEEP, [s['source_path']]
            elif arm == 'minimal':
                prompt, refs = PEOPLE + ' ' + MINIMAL + KEEP, [s['source_path']]
            elif arm == 'w_ref':
                prompt, refs = (PEOPLE + ' Image 2 shows the same two people already performing the '
                                'interaction to adopt. ' + body + KEEP), [s['source_path'], s['target_path']]
            else:
                continue
            jobs.append((f"{s['id']}__{arm}", s, prompt, refs))

    out_dir = Path(args.out)
    imgs = out_dir / 'images'
    imgs.mkdir(parents=True, exist_ok=True)

    pipe, strategy = load_pipeline()
    rows = []
    for jid, s, prompt, refs in jobs:
        source = Image.open(s['source_path']).convert('RGB')
        target = Image.open(s['target_path']).convert('RGB')
        pil_refs = [Image.open(p).convert('RGB') for p in refs]
        wait_for_headroom()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        gen = torch.Generator('cpu').manual_seed(args.seed)
        with torch.no_grad():
            result = pipe(prompt=prompt, image=pil_refs, width=args.width, height=args.height,
                          output_resolution=args.output_resolution,
                          num_inference_steps=args.steps, true_cfg_scale=args.cfg,
                          use_kv_cache=True, generator=gen)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        peak = torch.cuda.max_memory_allocated() / GiB
        image = result.images[0]
        image.save(imgs / f'{jid}.png')
        strip([source, target, image],
              ['SOURCE (non-contact)', 'TARGET (contact ref)', jid]).save(imgs / f'{jid}_strip.png')
        rows.append({'id': jid, 'sample': s['id'], 'pair': s['pair'],
                     'interaction': s['interaction'], 'arm': jid.split('__')[-1],
                     'source_action': s['source_action'], 'target_action': s['target_action'],
                     'seconds': round(elapsed, 2), 'peak_cuda_gib': round(peak, 3),
                     'output_size': list(image.size), 'n_refs': len(pil_refs),
                     'prompt': prompt})
        (out_dir / 'results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        free, _ = vram()
        print(f'[{len(rows)}/{len(jobs)}] {jid}  {elapsed:.1f}s  peak {peak:.2f} GiB  '
              f'out {image.size}  free {free:.2f} GiB', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
