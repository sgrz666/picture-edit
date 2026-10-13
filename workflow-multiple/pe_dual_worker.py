#!/usr/bin/env python3
"""Run the official prompt engineer on the source photo, the target photo and both guides."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_prompts import (DUAL_ROLE_SYSTEM_SUFFIX, GENERATOR_ROLES, PE_ROLES, PE_SAMPLING,
    citation_rule, compile_prompt, generator_allowed, parse_pe_answer, pe_instruction)
from src.pose_edit.common import atomic_json, finish_stage, read_json, runtime_paths, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    settings, paths = request['identity']['settings'], runtime_paths(request)
    arm = settings['generator_references']
    run_dir = Path(paths['run_dir'])
    stage_dir = run_dir / 'prompt'
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(settings['pe_model'])
    system_prompt = (model_dir / 'system_prompt.txt').read_text(encoding='utf-8')
    effective_system = system_prompt + DUAL_ROLE_SYSTEM_SUFFIX
    (stage_dir / 'effective_system_prompt.txt').write_text(effective_system, encoding='utf-8')
    instruction = pe_instruction(arm, settings['prop_policy'])
    citation = citation_rule(arm)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16).to('cuda').eval()
    images, inputs_meta = [], []
    for index, role in enumerate(PE_ROLES, start=1):
        with Image.open(paths[role]) as image:
            images.append(image.convert('RGB'))
        inputs_meta.append({'index': index, 'role': role, 'path': paths[role],
            'sha256': sha256_file(paths[role]), 'size': list(images[-1].size),
            'given_to_generator': role in GENERATOR_ROLES[arm]})
    torch.cuda.reset_peak_memory_stats()
    attempts = []
    for attempt in range(2):
        seed = settings['pe_seed'] + attempt
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        retry = '' if not attempts else (
            '\nYour previous answer was rejected: ' + attempts[-1]['error'] +
            '. Write the two body actions and their relationship in explicit words, and keep the '
            'image numbers exactly as you received them as stated above. Keep the official JSON '
            'format and the source ratio contract.')
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': effective_system}]},
            {'role': 'user', 'content': [*[{'type': 'image', 'image': image} for image in images],
                {'type': 'text', 'text': instruction + ' ' + citation + retry}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_dict=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = model.generate(**inputs, **PE_SAMPLING)
        tokens = generated[0, inputs['input_ids'].shape[1]:]
        raw = processor.decode(tokens, skip_special_tokens=True)
        raw_path = stage_dir / f'pe_attempt_{attempt + 1}.txt'
        raw_path.write_text(raw, encoding='utf-8')
        evidence = {'attempt': attempt + 1, 'seed': seed, 'raw_path': str(raw_path),
            'raw_sha256': sha256_file(raw_path), 'output_tokens': len(tokens)}
        try:
            result = parse_pe_answer(raw)
            compiled = compile_prompt(result['rewritten_prompt'], arm)
        except ValueError as error:
            evidence.update(status='rejected', error=str(error))
            attempts.append(evidence)
            atomic_json(stage_dir / 'pe_attempts.json', attempts)
            print(f'PE_RETRY {error}', flush=True)
            continue
        evidence['status'] = 'accepted'
        attempts.append(evidence)
        atomic_json(stage_dir / 'pe_attempts.json', attempts)
        break
    else:
        raise RuntimeError('PE failed the JSON/image-role contract twice; see pe_attempts.json')
    (stage_dir / 'pe_raw.txt').write_text(raw, encoding='utf-8')
    (stage_dir / 'pe_compiled_prompt.txt').write_text(compiled + '\n', encoding='utf-8')
    (stage_dir / 'rewritten_prompt_raw.txt').write_text(result['rewritten_prompt'] + '\n',
        encoding='utf-8')
    atomic_json(stage_dir / 'pe_result.json', {'official_result': result,
        'compiled_prompt': compiled, 'generator_references': arm,
        'input_roles': inputs_meta,
        'generator_roles': {index: role for index, role in enumerate(
            [role for role in PE_ROLES if role in GENERATOR_ROLES[arm]], start=1)},
        'instruction': instruction, 'citation_rule': citation,
        'system_prompt_sha256': sha256_file(model_dir / 'system_prompt.txt'),
        'system_role_suffix': DUAL_ROLE_SYSTEM_SUFFIX,
        'effective_system_prompt_sha256': hashlib.sha256(effective_system.encode('utf-8')).hexdigest(),
        'enable_thinking': True, 'sampling': PE_SAMPLING, 'seed': seed, 'attempts': attempts,
        'output_tokens': len(tokens), 'seconds': time.perf_counter() - started,
        'peak_gpu_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'semantic_review': 'Model-generated candidate; role checks do not guarantee the two '
            'body actions or the contact between them were understood correctly'})
    finish_stage(stage_dir / 'stage.json', request,
        {'pe_compiled_prompt': stage_dir / 'pe_compiled_prompt.txt',
         'pe_result': stage_dir / 'pe_result.json', 'pe_raw': stage_dir / 'pe_raw.txt',
         'rewritten_prompt_raw': stage_dir / 'rewritten_prompt_raw.txt',
         'pe_attempts': stage_dir / 'pe_attempts.json',
         'effective_system_prompt': stage_dir / 'effective_system_prompt.txt',
         **{f'pe_attempt_{item["attempt"]}': item['raw_path'] for item in attempts}},
        {'seconds': time.perf_counter() - started, 'output_tokens': len(tokens),
         'generator_references': arm}, run_dir)
    print(f'PE_DUAL_COMPLETE {stage_dir}', flush=True)


if __name__ == '__main__':
    main()
