"""独立 PromptHMR 四图推理；只读现有模型与自动框，独占输出。"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image
from contracts import fit_record, validate_boxes, project, model_to_original, write_json_exclusive, sha256_file, safe_child

BODY12_OPENPOSE = [5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze_request(config, output):
    records = {row['id']: row for row in read_json(config['fit_records'])['records']}
    selected = {row['id']: row for row in read_json(config['selection'])['records']}
    with Path(config['multihmr_predictions']).open('rb') as stream:
        predictions = pickle.load(stream)
    frozen = []
    for sample in config['sample_ids']:
        row = fit_record(records[sample])
        if sha256_file(row['image']) != row['image_sha256']:
            raise ValueError(f'Image hash changed: {sample}')
        selection = selected[sample]
        if selection['status'] != 'ok' or len(set(selection['candidate_indices'])) != 2:
            raise ValueError(f'No two distinct automatic candidates: {sample}')
        indices = selection['candidate_indices']
        boxes = [predictions[sample]['predictions'][i]['bbox_orig'] for i in indices]
        row.update(boxes=validate_boxes(boxes, row['size']).tolist(), candidate_indices=indices)
        frozen.append(row)
    request = {'schema': config['schema'], 'config': config, 'records': frozen,
               'source_sha256': {name: sha256_file(config[name]) for name in
                                 ['fit_records', 'selection', 'multihmr_predictions']}}
    write_json_exclusive(output, request)
    return request


def source_hashes(source):
    names = ['prompt_hmr/__init__.py', 'prompt_hmr/models/__init__.py',
             'prompt_hmr/models/phmr.py', 'prompt_hmr/models/inference.py',
             'prompt_hmr/models/components/clip_encoder.py',
             'prompt_hmr/models/components/twoway_transformer.py']
    return {name: sha256_file(source/name) for name in names}


def load_model(config):
    import torch
    import open_clip
    import functools
    import xformers.ops
    from xformers.ops.fmha import cutlass
    if config['attention_backend'] != 'xformers-cutlass':
        raise ValueError('This audited run requires the validated CUTLASS backend')
    xformers.ops.memory_efficient_attention = functools.partial(
        xformers.ops.memory_efficient_attention, op=(cutlass.FwOp, cutlass.BwOp))
    source = Path(config['source_root'])
    sys.path.insert(0, str(source))
    os.chdir(source)
    from prompt_hmr.core.config import parse_args
    from prompt_hmr.models import build_phmr
    import prompt_hmr.models.phmr as phmr_module
    phmr_module.SMPL_MODEL_DIR = config['smpl_model_dir']
    weights = Path(config['weights_dir'])
    checkpoint = weights/'checkpoint.ckpt'
    cfg = parse_args(['--cfg', str(weights/'config.yaml')])
    original_create = open_clip.create_model

    def architecture_only(*args, **kwargs):
        # Complete released checkpoint contains both DINOv2 and CLIP text weights.
        # Skip only a redundant construction-time CLIP download; strict loading
        # below must fill every model key, including the text encoder.
        kwargs['pretrained'] = None
        return original_create(*args, **kwargs)

    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['state_dict']
    if not any(k.startswith('image_encoder.backbone.encoder.') for k in state):
        raise ValueError('Checkpoint lacks image encoder weights')
    if not any(k.startswith('prompt_encoder.clip_encoder.encoder.') for k in state):
        raise ValueError('Checkpoint lacks text encoder weights')
    with patch('open_clip.create_model', architecture_only):
        model = build_phmr(cfg)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(str(incompatible))
    model = model.cuda().eval()
    model.is_train = False
    metadata = {'checkpoint_sha256': sha256_file(checkpoint),
                'model_config_sha256': sha256_file(weights/'config.yaml'),
                'model_config_text': (weights/'config.yaml').read_text(),
                'numeric_smpl_sha256': sha256_file(Path(config['smpl_model_dir'])/'SMPL_NEUTRAL.pkl'),
                'checkpoint_keys': len(state), 'strict_load': True,
                'missing_keys': list(incompatible.missing_keys),
                'unexpected_keys': list(incompatible.unexpected_keys),
                'construction_only_clip_pretrained_disabled': True,
                'source_git_head': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': source_hashes(source),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'attention_backend': config['attention_backend']}
    del state
    print(json.dumps({'model_loaded': metadata}, ensure_ascii=False), flush=True)
    return model, metadata


def numpy(value):
    return value.detach().float().cpu().numpy()


def run_frame(model, record, interaction, config, destination, metadata):
    import torch
    from prompt_hmr.models.inference import prepare_batch
    image = np.asarray(Image.open(record['image']).convert('RGB'))
    if [image.shape[1], image.shape[0]] != record['size']:
        raise ValueError('Image dimensions changed')
    if sha256_file(record['image']) != record['image_sha256']:
        raise ValueError('Image SHA256 changed')
    inputs = [{'image_cv': image,
               'boxes': torch.tensor(record['boxes'], dtype=torch.float32),
               'cam_int': torch.tensor(record['K'], dtype=torch.float32)[None],
               'text': None, 'masks': None}]
    batch = prepare_batch(inputs, img_size=config['image_size'], interaction=interaction)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        output = model(batch, use_mean_hands=config['use_mean_hands'])[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    arrays = {key: numpy(output[key]) for key in
              ['pose', 'betas', 'transl', 'rotmat', 'vertices', 'body_joints', 'body_joints2d', 'cam_int']}
    if arrays['vertices'].shape != (2, 10475, 3):
        raise ValueError(f'Unexpected vertices shape: {arrays["vertices"].shape}')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Non-finite prediction')
    if np.min(arrays['vertices'][..., 2]) <= 0:
        raise ValueError('Non-positive mesh depth')
    scale = float(batch[0]['image_scale'])
    offset = np.asarray(batch[0]['image_offset'], float)
    returned_xy = model_to_original(arrays['body_joints2d'], scale, offset)
    direct_xy = project(arrays['body_joints'], record['K'])
    model_xy = project(arrays['body_joints'], arrays['cam_int'][0])
    xy = model_to_original(model_xy, scale, offset)
    error = float(np.max(np.abs(xy-direct_xy)))
    if error > .01:
        raise ValueError(f'Original camera projection mismatch: {error} px')
    left_hand = numpy(model.smplx.hands_mean[:, :15]).repeat(2, axis=0)
    right_hand = numpy(model.smplx.hands_mean[:, 15:]).repeat(2, axis=0)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
        rebuilt = model.smplx(
            global_orient=torch.as_tensor(arrays['rotmat'][:, :1], device='cuda'),
            body_pose=torch.as_tensor(arrays['rotmat'][:, 1:], device='cuda'),
            betas=torch.as_tensor(arrays['betas'], device='cuda'),
            transl=torch.as_tensor(arrays['transl'], device='cuda'),
            left_hand_pose=torch.as_tensor(left_hand, device='cuda'),
            right_hand_pose=torch.as_tensor(right_hand, device='cuda')).vertices
    mesh_roundtrip = float(np.max(np.abs(numpy(rebuilt)-arrays['vertices'])))
    if mesh_roundtrip > 1e-5:
        raise ValueError(f'Exported parameters do not reconstruct vertices: {mesh_roundtrip} m')
    arrays.update(faces=np.asarray(model.smplx.faces, dtype=np.int32),
                  original_K=np.asarray(record['K']), boxes=np.asarray(record['boxes']),
                  image_scale=np.asarray(scale), image_offset=offset,
                  left_hand_rotmat=left_hand, right_hand_rotmat=right_hand,
                  body_joints2d_original=direct_xy,
                  body_joints2d_original_model_output=returned_xy,
                  body12_xy_original=direct_xy[:, BODY12_OPENPOSE],
                  body12_xyz=arrays['body_joints'][:, BODY12_OPENPOSE])
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/'geometry.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    result = {'id': record['id'], 'status': 'ok', 'interaction': interaction,
              'person_count': 2, 'image': record['image'], 'image_sha256': record['image_sha256'],
              'category': record.get('category'), 'split': record.get('split'),
              'box_source': config['box_source'], 'candidate_indices': record['candidate_indices'],
              'body12_xy_original': arrays['body12_xy_original'].tolist(),
              'projection_max_error_px': error,
              'model_returned_projection_error_px': float(np.max(np.abs(returned_xy-direct_xy))),
              'mesh_roundtrip_max_error_m': mesh_roundtrip,
              'mesh_min_depth_m': float(arrays['vertices'][..., 2].min()),
              'elapsed_sec': elapsed, 'geometry_sha256': sha256_file(destination/'geometry.npz'),
              'checkpoint_sha256': metadata['checkpoint_sha256']}
    write_json_exclusive(destination/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--load-only', action='store_true')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {'.', '..'}:
        parser.error('--name must be one directory name')
    config = read_json(args.config)
    config_path = args.config.resolve()
    out = safe_child(config['output_root'], args.name)
    out.mkdir(parents=True, exist_ok=False)
    request = freeze_request(config, out/'request.json')
    import torch
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    model, metadata = load_model(config)
    metadata.update(config_sha256=sha256_file(config_path),
                    runner_sha256=sha256_file(Path(__file__).resolve()),
                    contracts_sha256=sha256_file(Path(__file__).with_name('contracts.py').resolve()),
                    runtime_python=sys.executable,
                    peak_allocated_bytes_before=torch.cuda.max_memory_allocated())
    write_json_exclusive(out/'model.json', metadata)
    if args.load_only:
        return
    records = request['records'][:args.limit] if args.limit else request['records']
    rows = []
    for arm, interaction in config['arms'].items():
        for record in records:
            try:
                row = run_frame(model, record, interaction, config, safe_child(out,arm,record['id']), metadata)
            except Exception as exc:
                row = {'id': record['id'], 'arm': arm, 'status': 'error',
                       'error': str(exc), 'traceback': traceback.format_exc()}
                safe_child(out,arm).mkdir(exist_ok=True, parents=True)
                write_json_exclusive(safe_child(out,arm,record['id']+'_error.json'), row)
                print(json.dumps(row), flush=True)
            row['arm'] = arm
            rows.append(row)
    summary = {'requested_per_arm': len(records), 'arms': list(config['arms']),
               'success': sum(r['status']=='ok' for r in rows), 'total': len(rows),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'source_unchanged': source_hashes(Path(config['source_root']))==metadata['source_sha256'],
               'rows': rows}
    write_json_exclusive(out/'summary.json', summary)
    print(json.dumps({'summary': summary}, ensure_ascii=False), flush=True)
    if summary['success'] != summary['total'] or not summary['source_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
