#!/usr/bin/env python3
"""Generate with the native Qwen backbone from the source photo plus the two-person guides."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import time

from PIL import Image
import torch
from diffusers import QwenImage21Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_prompts import GENERATOR_ROLES
from src.pose_edit.common import atomic_json, finish_stage, read_json, runtime_paths, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    settings, paths = request['identity']['settings'], runtime_paths(request)
    arm = settings['generator_references']
    if settings['adapter'] or settings['lora']:
        raise ValueError('Generation is fixed to native Qwen without adapter or LoRA')
    stage_dir = Path(paths['run_dir']) / 'generation'
    stage_dir.mkdir(parents=True, exist_ok=True)
    images, references = [], []
    for index, role in enumerate(GENERATOR_ROLES[arm], start=1):
        with Image.open(paths[role]) as image:
            images.append(image.convert('RGB'))
        references.append({'index': index, 'role': role, 'path': paths[role],
            'size': list(images[-1].size), 'sha256': sha256_file(paths[role])})
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError('All generation references must share the source canvas; the geometry '
            'stage letterboxes the target photograph and both guides onto it')
    module = importlib.import_module(QwenImage21Pipeline.__module__)
    for reference in references:
        condition = module.calculate_dimensions(settings['resolution'] ** 2,
            reference['size'][0] / reference['size'][1])[:2]
        if condition != (width, height):
            raise ValueError(f'The pipeline would resize reference {reference["role"]} to '
                f'{condition}; canvas protocol must be updated before generating')
    prompt = Path(paths['generation_prompt']).read_text(encoding='utf-8').strip()
    prompt_source = (read_json(Path(paths['generation_prompt_provenance']))
        if paths.get('generation_prompt_provenance') else None)
    started = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(settings['qwen_model'], dtype=torch.bfloat16).to('cuda')
    if any(getattr(component, 'peft_config', {}) for component in
            (pipe.transformer, pipe.text_encoder)):
        raise RuntimeError('Unexpected LoRA/PEFT configuration in native pipeline')
    for component in (pipe.transformer, pipe.text_encoder, pipe.vae):
        component.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result = pipe(prompt=prompt, image=images, width=width, height=height,
            output_resolution=settings['resolution'], num_inference_steps=settings['steps'],
            true_cfg_scale=settings['true_cfg_scale'], use_kv_cache=settings['use_kv_cache'],
            generator=torch.Generator('cuda').manual_seed(settings['seed'])).images[0]
    canvas = Image.alpha_composite(Image.new('RGBA', result.size, 'white'),
        result.convert('RGBA')).convert('RGB')
    if canvas.size != (width, height):
        raise RuntimeError(f'Unexpected generated canvas {canvas.size}')
    canvas.save(stage_dir / 'final_canvas.png')
    preprocess = read_json(paths['source_preprocess'])
    final = canvas.crop(tuple(preprocess['A']['content_box']))
    final.save(stage_dir / 'final_image.png')
    info = {'reference_images': references, 'generator_references': arm,
        'generation_mode': 'Qwen backbone only', 'adapter_loaded': False, 'lora_loaded': False,
        'persons_requested': 2, 'resolution': settings['resolution'],
        'canvas_size': [width, height], 'final_size': list(final.size),
        'source_padding_crop': preprocess['A']['content_box'], 'steps': settings['steps'],
        'seed': settings['seed'], 'true_cfg_scale': settings['true_cfg_scale'],
        'use_kv_cache': settings['use_kv_cache'],
        'generation_prompt': paths['generation_prompt'], 'prompt_source': prompt_source,
        'prompt_sha256': sha256_file(paths['generation_prompt']),
        'seconds': time.perf_counter() - started,
        'peak_gpu_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'torch_version': torch.__version__}
    atomic_json(stage_dir / 'generation.json', info)
    finish_stage(stage_dir / 'stage.json', request,
        {'final_image': stage_dir / 'final_image.png',
         'final_canvas': stage_dir / 'final_canvas.png',
         'generation_info': stage_dir / 'generation.json'}, info, Path(paths['run_dir']))
    print(f'GENERATION_DUAL_COMPLETE {stage_dir / "final_image.png"}', flush=True)


if __name__ == '__main__':
    main()
