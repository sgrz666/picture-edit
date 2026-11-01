"""CPU-only identity proof for grounded Hi4D Qwen reference images.

The SHA-bound bank is authoritative. Native guides are independently derived
from its official world joints and camera, never from a condition's identity map.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from src.data.canvas_contract import CanvasSpec, native_pad
from ..native_full_assets import sha256
from .person_order import measure_source_order

SCHEMA = 'hi4d_pei2i_q4_plan_a_v5'
PARENTS = (-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)
COLORS = {'A': (238,90,90), 'B': (72,145,239)}


def verified(bound):
    if not isinstance(bound, dict) or not bound.get('path') or not bound.get('sha256'):
        raise ValueError('Grounded Hi4D requires an exact file reference')
    path = Path(bound['path'])
    if not path.is_absolute() or sha256(path) != bound['sha256']:
        raise ValueError('Grounded Hi4D file SHA differs: ' + str(path))
    return path


@lru_cache(maxsize=64)
def bank_record(path, bank_sha, pair_id):
    """Bounded row cache; callers verify the whole bank SHA before every lookup."""
    found = []
    has_v5 = False
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            has_v5 |= row.get('schema') == SCHEMA
            if row.get('pair_id') == pair_id:
                found.append(row)
    if len(found) > 1:
        raise ValueError('Duplicate pair_id in the bound prompt bank')
    return (found[0] if found else None), has_v5


def _official_refs(row):
    refs = row['grounding']['official_inputs']
    names = ('official_smpl', 'official_camera_file', 'sequence_meta')
    return tuple((str(verified(refs[name])), refs[name]['sha256']) for name in names)


@lru_cache(maxsize=32)
def _official_projection(refs, camera_id, pair, action, target_frame):
    smpl_path, camera_path, meta_path = (Path(item[0]) for item in refs)
    base = smpl_path.parent.parent
    if (smpl_path.parent.name != 'smpl' or base.name != action or base.parent.name != pair
            or smpl_path.stem != f'{target_frame:06d}' or camera_path != base/'cameras/rgb_cameras.npz'):
        raise ValueError('Official Hi4D target sequence/frame binding differs')
    meta = json.loads(meta_path.read_text(encoding='utf8'))
    if (meta.get('pair'), meta.get('action'), int(meta.get('camera_id', -1))) != (pair, action, camera_id):
        raise ValueError('Official Hi4D sequence camera identity differs')
    with np.load(camera_path, allow_pickle=False) as raw:
        indices = np.flatnonzero(raw['ids'] == camera_id)
        if len(indices) != 1:
            raise ValueError('Official Hi4D camera id is missing or duplicated')
        index = int(indices[0])
        k = raw['intrinsics'][index].astype(np.float64)
        rt = raw['extrinsics'][index].astype(np.float64)
        distortion = raw['dist_coeffs'][index]
    if rt.shape != (3,4) or k.shape != (3,3) or np.any(np.abs(distortion) > 1e-10):
        raise ValueError('Unsupported official Hi4D camera contract')
    affine = np.asarray(meta['A_original_to_512'], dtype=np.float64)
    if (not np.allclose(rt[:,:3], meta['R_world_to_camera'], atol=1e-8, rtol=0)
            or not np.allclose(rt[:,3], meta['t_world_to_camera'], atol=1e-8, rtol=0)
            or not np.allclose(k, meta['original_K'], atol=1e-8, rtol=0)
            or not np.allclose(affine @ k, meta['K_512'], atol=1e-7, rtol=0)):
        raise ValueError('Hi4D sequence transform differs from official raw camera')
    with np.load(smpl_path, allow_pickle=False) as raw:
        joints = raw['joints_3d'].astype(np.float64)
    if joints.shape != (2,24,3) or not np.isfinite(joints).all():
        raise ValueError('Official Hi4D world joints must be finite (2,24,3)')
    camera = joints @ rt[:,:3].T + rt[:,3]
    if np.min(camera[...,2]) <= 0:
        raise ValueError('Official Hi4D joints are behind the camera')
    homogeneous = camera @ np.asarray(meta['K_512'], dtype=np.float64).T
    # The exported SMPL24 projection contract stores float32 before inverse affine.
    # Reproduce that quantization from official joints; do not trust exported npy bytes.
    xy512 = (homogeneous[...,:2] / homogeneous[...,2:3]).astype(np.float32).astype(np.float64)
    original = np.concatenate((xy512, np.ones((2,24,1))), axis=-1) @ np.linalg.inv(affine).T
    return original[...,:2], str(base)


def official_native_guides(row, canvas):
    """Derive both identities and their combined guide from official CPU geometry."""
    refs = _official_refs(row)
    points, _ = _official_projection(refs, int(row['camera_id']), row['pair'], row['action'], int(row['target_frame_id']))
    images = {}
    width = max(2, round(canvas.canvas_hw[0]/256))
    for actor, identity in enumerate(('A','B')):
        xy = canvas.pixel_points(points[actor])
        image = Image.new('RGB', (canvas.canvas_hw[1], canvas.canvas_hw[0]))
        draw = ImageDraw.Draw(image)
        for child, parent in enumerate(PARENTS):
            if parent >= 0:
                draw.line([tuple(xy[parent]), tuple(xy[child])], fill=COLORS[identity], width=width)
        for x, y in xy:
            draw.ellipse((x-width,y-width,x+width,y+width), fill=COLORS[identity])
        images[identity] = image
    images['combined'] = ImageChops.lighter(images['A'], images['B'])
    return images


@lru_cache(maxsize=32)
def _native_pixel_hashes(row_json, canvas_json):
    row = json.loads(row_json)
    canvas = CanvasSpec.from_record(json.loads(canvas_json))
    return {key: hashlib.sha256(image.tobytes()).hexdigest() for key, image in official_native_guides(row, canvas).items()}


def verify_grounded_hi4d(spec, sample=None):
    """Require authoritative V5 bindings; historical non-V5 banks retain their reader."""
    prompt = spec['prompt']
    bound = prompt['bank']
    bank = verified(bound)
    row, has_v5 = bank_record(str(bank), bound['sha256'], spec['pair_id'])
    if prompt.get('schema') != SCHEMA and not has_v5:
        return
    if (row is None or row.get('schema') != SCHEMA or prompt.get('schema') != SCHEMA
            or prompt.get('pair_id') != row['pair_id'] or spec['profile'] != 'dual_4ref'):
        raise ValueError('Grounded Hi4D bank row/schema/pair binding differs')
    if (prompt['text'] != row['compiled_prompt_inference']
            or prompt['sha256'] != row['compiled_prompt_inference_sha256']):
        raise ValueError('Grounded Hi4D prompt differs from its actual bank row')
    order = spec.get('person_order')
    guides = spec.get('guide_identity_files')
    if not isinstance(order, dict) or not isinstance(guides, dict) or set(guides) != {'A','B'}:
        raise ValueError('Grounded Hi4D requires person_order and guide_identity_files')
    g = row['grounding']
    masks = [verified(g['source_masks'][identity]) for identity in ('A','B')]
    measured = measure_source_order(*masks)
    left = measured['left_actor']
    right = 'B' if left == 'A' else 'A'
    if (not measured['resolvable'] or row.get('left_actor') != left or row.get('right_actor') != right
            or row['per_person']['person_1']['identity'] != left or row['per_person']['person_2']['identity'] != right
            or order.get('slot0_identity') != left or order.get('slot1_identity') != right
            or (order.get('evidence') or {}).get('left_actor') != left):
        raise ValueError('Grounded Hi4D identity differs from measured bank source-left binding')
    images = spec['images']
    for identity in ('A','B'):
        verified(guides[identity])
    if images[2]['file'] != guides[left] or images[3]['file'] != guides[right]:
        raise ValueError('Grounded Hi4D guide identity differs from Qwen image3/image4')
    if all(entry['space'] == 'original' for entry in images[1:]):
        if (images[0]['file'] != g['source_rgb'] or guides != g['target_guides']
                or images[1]['file'] != g['target_combined_guide']):
            raise ValueError('Grounded Hi4D original guide/source identity differs from bank files')
        for entry in (g['source_rgb'], g['target_combined_guide'], *g['target_guides'].values()):
            verified(entry)
        for entry in images:
            with Image.open(entry['file']['path']) as image:
                expected_canvas = native_pad((image.height, image.width))
            if entry['canvas_id'] != expected_canvas.fingerprint():
                raise ValueError('Grounded Hi4D original image canvas differs from bank pixels')
        return
    if not all(entry['space'] == 'canvas' for entry in images[1:]):
        raise ValueError('Grounded Hi4D native guides must share explicit canvas space')
    canvas = CanvasSpec.from_record(images[1]['canvas'])
    refs = _official_refs(row)  # verify bytes on every call, including cached rendering.
    _, base = _official_projection(refs, int(row['camera_id']), row['pair'], row['action'], int(row['target_frame_id']))
    source_path = Path(base)/'images'/str(row['camera_id'])/f"{int(row['source_frame_id']):06d}.jpg"
    target_path = Path(base)/'images'/str(row['camera_id'])/f"{int(row['target_frame_id']):06d}.jpg"
    if images[0]['file'] != {'path': str(source_path), 'sha256': sha256(source_path)}:
        raise ValueError('Grounded Hi4D native source RGB differs from bank sequence/frame')
    with Image.open(source_path) as image:
        source_canvas = native_pad((image.height, image.width))
    if images[0]['canvas_id'] != source_canvas.fingerprint():
        raise ValueError('Grounded Hi4D native source canvas differs from official RGB')
    with Image.open(target_path) as image:
        expected_canvas = native_pad((image.height, image.width))
    if canvas.fingerprint() != expected_canvas.fingerprint():
        raise ValueError('Grounded Hi4D native target canvas differs from official RGB')
    if sample is not None and canvas.fingerprint() != CanvasSpec.from_record(sample['target_canvas']).fingerprint():
        raise ValueError('Grounded Hi4D native guide differs from actual sample canvas')
    row_json = json.dumps(row, sort_keys=True, separators=(',',':'))
    expected = _native_pixel_hashes(row_json, json.dumps(canvas.metadata(), sort_keys=True))
    for identity, image_ref in (('A', guides['A']), ('B', guides['B']), ('combined', images[1]['file'])):
        with Image.open(verified(image_ref)) as image:
            if (image.height, image.width) != canvas.canvas_hw or hashlib.sha256(image.convert('RGB').tobytes()).hexdigest() != expected[identity]:
                raise ValueError('Grounded Hi4D native guide identity differs from official world/camera projection: ' + identity)
