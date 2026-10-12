#!/usr/bin/env python3
"""Qualitative A/B different-person check: B supplies geometry, never identity pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw

from eval_iper_four_arm import IdentityMetric
from eval_qwen21_face_references import ROLE_PROMPT, composite
from eval_iper_qwen21_face import source_crop
from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.face_adapter import Qwen21FaceAdapter
from src.pose_control.qwen21.runtime import PROMPT, dataset_image_to_pil
from train_iper_qwen21_face import FaceFeatureStore


ROOT = Path(__file__).resolve().parent
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')
SAMPLED = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cross-pairs', type=Path, default=FACE_ROOT / 'cross_person_8_pairs.jsonl')
    parser.add_argument('--val-pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--features', type=Path, default=FACE_ROOT / 'feature_cache')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--body-checkpoint', type=Path, default=ROOT / 'experiments/qwen21_iper_body/full_20k/checkpoint_last.pt')
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'cross_person_eval')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = arguments()
    cross = [json.loads(line) for line in args.cross_pairs.read_text().splitlines() if line.strip()]
    val = IPERPoseDataset(str(args.val_pairs), str(SAMPLED),
        str(ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast'), 512, False)
    by_target = {(row['appearance'], row['target']['stem']): i for i, row in enumerate(val.samples)}
    store = FaceFeatureStore(args.features)
    pipe = QwenImage21Pipeline.from_pretrained(str(ROOT / 'models/Qwen-Image-2.1'), dtype=torch.bfloat16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    body = Qwen21BodyAdapter().to('cuda', dtype=torch.float32).eval()
    body.load_state_dict(torch.load(args.body_checkpoint, map_location='cpu', weights_only=False)['adapter_state_dict'])
    face = Qwen21FaceAdapter().to('cuda', dtype=torch.float32).eval()
    face.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['face_state_dict'])
    identity = IdentityMetric()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(cross):
        if item['source_appearance'].split('_')[0] == item['target_appearance'].split('_')[0]:
            raise AssertionError('cross-person case uses the same iPER person')
        target_index = by_target[item['target_appearance'], item['target']['stem']]
        target_sample = val[target_index]
        source_path = SAMPLED / item['source_appearance'] / item['source']['rgb_1024']
        with Image.open(source_path) as image:
            source = image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
        target = dataset_image_to_pil(target_sample['tgt_image'])
        source_feature = store.app(item['source_appearance'])['source'][item['source']['stem']]
        geometry_feature = store.app(item['target_appearance'])['target_geometry'][item['target']['stem']]
        visible = store.app(item['target_appearance'])['target_visibility'][item['target']['stem']]['visible']
        valid = bool(source_feature['valid'] and geometry_feature['geometry_valid'] and visible)
        state = face.prepare(source_feature['embedding'][None], source_feature['crop'][None],
                             torch.tensor([geometry_feature['box']]), torch.tensor([valid]))
        batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                 for key, value in target_sample.items()}
        with torch.no_grad():
            geometry = body.encode_geometry(batch, (32, 32))
        face_reference = (source_crop(source_path, source_feature['detection']['bbox'])
                          if source_feature['valid'] else None)
        generated = {}
        for mode in ('body', 'body_face', 'body_face_crop'):
            generator = torch.Generator(device='cuda').manual_seed(args.seed)
            with Qwen21ControlContext(pipe.transformer, body, geometry,
                                      face_adapter=face if mode != 'body' else None,
                                      face_state=state if mode != 'body' else None,
                                      face_strength=1 if mode != 'body' else 0):
                references = [source, face_reference] if mode == 'body_face_crop' and face_reference else [source]
                result = pipe(prompt=ROLE_PROMPT if len(references) == 2 else PROMPT,
                              image=references, width=512, height=512, output_resolution=512,
                              num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=True,
                              generator=generator)
            generated[mode] = composite(result.images[0])
            generated[mode].save(args.output / f'{index:03d}_{mode}.png')
        source_embedding = identity.embedding(source)
        target_embedding = identity.embedding(target)
        scores = {}
        for mode, image in generated.items():
            embedding = identity.embedding(image)
            scores[mode] = {'source_cosine': identity.cosine(source_embedding, embedding),
                            'B_face_cosine': identity.cosine(target_embedding, embedding)}
        sheet = Image.new('RGB', (512 * 5, 512 + 24), 'white')
        for j, (label, image) in enumerate((('A', source), ('B pose', target),
                                             ('Body', generated['body']), ('Body+Face', generated['body_face']),
                                             ('Face+crop', generated['body_face_crop']))):
            sheet.paste(image, (j * 512, 24))
            ImageDraw.Draw(sheet).text((j * 512 + 5, 5), label, fill='black')
        sheet.save(args.output / f'{index:03d}_comparison.png')
        row = {'pair_idx': index, 'source_appearance': item['source_appearance'],
               'target_appearance': item['target_appearance'], 'face_valid': valid, 'scores': scores}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / 'per_sample.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
