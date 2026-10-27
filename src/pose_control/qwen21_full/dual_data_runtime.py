"""Versioned dual dataset: target geometry only, source-only identity, no online renderer."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.data.canvas_contract import CanvasSpec, aspect_preserving, letterbox
from .body_pose import JOINT_NAMES, BODY25_BY_NAME, PARENTS_22, ROTATION_CONVENTION
from .conditions import ContactRelationBatch, TaskType
from .pose_cache import sha256
from .dual_hi4d_geometry import CACHE_SCHEMA
from .dual_depth_geometry import build_dual_condition, relative11
from .depth_geometry import onto_canvas, full_frame_affine
from .dual_depth_condition import DualDepthConditionV3, NativeDualIdentityCondition, IDENTITY_SCHEMA
from .native_dual_depth import NativeDualConfig

PAIR_SCHEMA = 'native_dual_pair_v3'


def pair_canvases(row, source, target, max_pixels, reference_side):
    declared = ('source_canvas' in row, 'target_canvas' in row)
    if any(declared):
        if not all(declared):raise ValueError('Declare both independent Source/target canvases')
        s=CanvasSpec.from_record(row['source_canvas']);t=CanvasSpec.from_record(row['target_canvas'])
        if s.original_hw!=tuple(source['original_hw']) or t.original_hw!=tuple(target['original_hw']):
            raise ValueError('Declared native Canvas differs from original geometry')
        return s,t
    return letterbox(source['original_hw'],reference_side),aspect_preserving(target['original_hw'],max_pixels=max_pixels,pad_multiple=32)


def _rgb(path, spec):
    with Image.open(path) as image:
        if (image.height, image.width) != spec.original_hw:
            raise ValueError('Raw RGB dimensions differ from the fingerprinted geometry')
        array = np.asarray(spec.image(image.convert('RGB'))).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float()/127.5-1


def _box(mask):
    ys, xs = mask[0].nonzero(as_tuple=True)
    if not xs.numel():
        raise ValueError('A declared valid person has no independent projected surface')
    h, w = mask.shape[-2:]
    return torch.tensor([xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h], dtype=torch.float32)


def _uv(joints, k):
    projected = joints @ k.T
    z = projected[:, 2]
    xy = projected[:, :2]/torch.where(z.abs() > 1e-8, z, torch.ones_like(z))[:, None]
    return xy, z > 1e-8


class NativeDualDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, *, config=None, max_pixels=512*512, reference_side=512, load_target=True):
        self.path = Path(manifest)
        self.rows = [json.loads(s) for s in self.path.read_text(encoding='utf8').splitlines() if s.strip()]
        self.config, self.max_pixels, self.load_target = config or NativeDualConfig(), max_pixels, load_target
        if reference_side <= 0 or reference_side % 32:
            raise ValueError('Reference side must be a positive multiple of 32')
        self.reference_side = reference_side
        self._verified = {}
        ids = [r.get('pair_id') for r in self.rows]
        if not ids or any(not isinstance(k, str) or not k for k in ids) or len(set(ids)) != len(ids):
            raise ValueError('Manifest requires unique nonempty pair IDs')
        for row in self.rows:
            if (row.get('schema') != PAIR_SCHEMA or row.get('kind') not in ('camera_z', 'relative_png')
                    or row.get('pairing') != 'same_subject_pair_target_frame'):
                raise ValueError('Explicit V3 pair schema/modality/paired supervision required')
            mapping = row.get('source_indices')
            if mapping is None or sorted(mapping) != [0, 1]:
                raise ValueError('Explicit A/B source correspondence required')
            if ([row['source_person_ids'][j] for j in mapping] != row['target_person_ids']
                    or len(set(row['target_person_ids'])) != 2):
                raise ValueError('Source/target identities differ; no retargeted RGB supervision')

    def __len__(self):
        return len(self.rows)

    def verify(self, record):
        if set(record) != {'path', 'sha256'}:
            raise ValueError('Asset references require path + SHA256 only')
        path = Path(record['path'])
        path = path if path.is_absolute() else self.path.parent/path
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, record['sha256'])
        if self._verified.get(str(path)) != signature:
            if sha256(path) != record['sha256']:
                raise ValueError('Dual asset fingerprint changed: '+str(path))
            self._verified[str(path)] = signature
        return path

    def geometry_record(self, record):
        path = self.verify(record)
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k].copy() for k in f.files}
        metadata = json.loads(str(data.pop('metadata_json')))
        if (metadata.get('schema') != CACHE_SCHEMA or metadata.get('body_source') != 'paired_target_frame'
                or metadata.get('normal_basis') != 'camera_cv' or metadata.get('contact_encoding') != 'one_based_zero_no_record'
                or metadata.get('part_partition') != 'argmax_face_sum_lbs_14'
                or metadata.get('joint_order') != ['pelvis', *JOINT_NAMES]
                or metadata.get('parents_22') != list(PARENTS_22) or metadata.get('rotation_convention') != ROTATION_CONVENTION
                or metadata.get('distance_backend') != 'python-fcl_triangle_bvh'):
            raise ValueError('Legacy or incompatible geometry cache cannot be used as V3')
        for field in ('mesh_errors', 'joint_errors', 'camera_mesh_errors'):
            values = metadata.get(field)
            if not isinstance(values, list) or len(values) != 2 or any(not np.isfinite(v) or not 0 <= v <= 1e-4 for v in values):
                raise ValueError('Missing/failed family-specific FK reconstruction gate')
        for source in metadata['sources'].values():
            self.verify(source)
        ah, aw = metadata['asset_hw']
        shapes = {'body_pose': (2, 63), 'joints_cam': (2, 22, 3), 'global26': (2, 26),
            'root_rotation': (2, 3, 3), 'body_height': (2,), 'K': (3, 3), 'z': (2, ah, aw),
            'valid': (2, ah, aw), 'normal': (2, ah, aw, 3), 'part': (2, ah, aw),
            'part_distance': (14, 14), 'part_distance_valid': (14, 14), 'closest_points': (14, 14, 2, 3),
            'closest_points_valid': (14, 14), 'part_contact': (14, 14, 4), 'contact_raster': (2, ah, aw)}
        for name, shape in shapes.items():
            if name not in data or data[name].shape != shape or not np.isfinite(data[name]).all():
                raise ValueError('Invalid raw cache field: '+name)
        for name in ('valid', 'part_distance_valid', 'closest_points_valid'):
            if data[name].dtype != np.bool_:
                raise ValueError('Geometry validity must be independent bool assets')
        if (data['body_height'] <= 0).any() or (data['z'][data['valid']] <= 0).any() or (data['z'][~data['valid']] != 0).any():
            raise ValueError('Invalid physical geometry/scale')
        full_frame_affine((ah, aw), metadata['original_hw'], metadata['asset_to_original'])
        for person, family in enumerate(metadata['model_family_indices']):
            if family not in (0, 1) or metadata['vertex_counts'][person] != (10475 if family == 0 else 6890):
                raise ValueError('Unknown model family')
        indices, distances = data['contact_indices'], data['contact_distance']
        if indices.ndim != 2 or indices.shape[1] != 4 or indices.dtype != np.int64 or distances.shape != (len(indices),):
            raise ValueError('Malformed original directed contact records')
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError('Invalid original contact distance')
        for p, vertex, q, other in indices:
            if (p not in (0, 1) or q != 1-p or not 0 <= vertex < metadata['vertex_counts'][p]
                    or not 0 <= other < metadata['vertex_counts'][q]):
                raise ValueError('Contact correspondence outside its original model topology')
        metadata['record_sha256'] = record['sha256']
        return data, metadata

    def __getitem__(self, index):
        row = self.rows[index]
        target, tm = self.geometry_record(row['target_geometry'])
        source, sm = self.geometry_record(row['source_geometry'])
        if tm['person_ids'] != row['target_person_ids'] or sm['person_ids'] != row['source_person_ids']:
            raise ValueError('Mesh records and pair IDs disagree')
        if row['source_rgb'] != sm['rgb'] or row['target_rgb'] != tm['rgb']:
            raise ValueError('Paired RGB references must match the exact geometry frames')
        source_spec,target_spec=pair_canvases(row,sm,tm,self.max_pixels,self.reference_side)
        tm = {**tm, 'pair_id': row['pair_id']}
        png, png_valid = None, None
        if row['kind'] == 'relative_png':
            with Image.open(self.verify(row['scene_png'])) as image:
                array = np.asarray(image.convert('RGB')).copy()
            if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(array[..., 0], array[..., 2]):
                raise ValueError('Scene PNG must have equal grayscale channels')
            png = torch.from_numpy(array[..., 0]).float()[None, None]/255
            with Image.open(self.verify(row['scene_png_valid'])) as image:
                png_valid = torch.from_numpy(np.asarray(image.convert('L')).copy()>0)[None, None]
            if tuple(png.shape[-2:]) != tuple(tm['asset_hw']) or png_valid.shape != png.shape or row.get('polarity') not in ('near_one', 'near_zero'):
                raise ValueError('Explicit PNG grid/validity/polarity required')
            if row['polarity'] == 'near_zero':
                png = 1-png
            png = torch.where(png_valid, png, 0.)
        dual = build_dual_condition(target, tm, target_spec, self.config,
            mode='full_geometry' if row['kind'] == 'camera_z' else 'scene_only', scene_png=png, scene_png_valid=png_valid)
        sample = {'dual_depth': dual, 'src_image': _rgb(self.verify(row['source_rgb']), source_spec),
            'source_canvas': source_spec.metadata(), 'target_canvas': target_spec.metadata(),
            'source_token_hw': source_spec.token_hw, 'target_token_hw': target_spec.token_hw,
            'valid_image_mask': target_spec.valid_mask(), 'person_b_valid': torch.tensor(True),
            'task_id': torch.tensor(int(TaskType.HUG if tm['action'].lower().startswith('hug') else TaskType.DUAL)),
            'source_indices': torch.tensor(row['source_indices'], dtype=torch.long),
            'pair_metadata': {'pair_id': row['pair_id'], 'cross_person': False, 'dataset_type': 'hi4d_native_dual_v3'},
            'identity_metadata': {'schema': IDENTITY_SCHEMA, 'source_canvas_id': source_spec.fingerprint(),
                'source_rgb_sha256': row['source_rgb']['sha256'], 'source_person_ids': row['source_person_ids'],
                'source_geometry_sha256': row['source_geometry']['sha256'],
                'source_mask_sha256': [m['sha256'] for m in row['source_masks']],
                'source_indices': row['source_indices'], 'head_source': 'source_mesh_head_intersect_visible_source_instance',
                'target_person_ids': row['target_person_ids']}}
        masks, heads = [], []
        for p, record in enumerate(row['source_masks']):
            with Image.open(self.verify(record)) as image:
                if (image.height, image.width) != source_spec.original_hw:
                    raise ValueError('Source mask must match the original reference grid')
                mask = torch.from_numpy(np.asarray(source_spec.image(image.convert('L'), nearest=True)).copy()>0)[None]
            head = onto_canvas(torch.from_numpy((source['part'][p] == 1).copy())[None, None].float(), source_spec)[0] > .5
            masks.append(mask); heads.append(head & mask)
        sample['source_person_mask'], sample['source_head_mask'] = torch.stack(masks), torch.stack(heads)
        k = torch.from_numpy(target_spec.camera(target['K'])).float()
        joints = torch.from_numpy(target['joints_cam']).float()
        boxes = []
        yy, xx = torch.meshgrid(torch.arange(target_spec.canvas_hw[0]), torch.arange(target_spec.canvas_hw[1]), indexing='ij')
        for p, condition in enumerate((dual.person_a, dual.person_b)):
            suffix = '' if p == 0 else '_b'
            box = _box(condition.context.smpl_mask[0]); boxes.append(box)
            uv, projected = _uv(joints[p], k)
            x0, y0, x1, y1 = target_spec.valid_rect
            projected &= (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
            pose = torch.zeros(25, *target_spec.canvas_hw)
            mapping = {**{name: body25 for name, body25 in BODY25_BY_NAME.items()}, 'pelvis': 8}
            for name, body25 in mapping.items():
                joint = 0 if name == 'pelvis' else JOINT_NAMES.index(name)+1
                if projected[joint]:
                    pose[body25] = torch.exp(-((xx+.5-uv[joint, 0])**2+(yy+.5-uv[joint, 1])**2)/50)
            fields = {'normal': condition.context.normal_cam[0]*torch.tensor([1., -1., -1.])[:, None, None],
                'pose_heatmap': pose, 'part_onehot': condition.context.part_onehot[0],
                'human_mask': condition.context.smpl_mask[0].float(), 'smplx_global': target_spec.global_camera(torch.from_numpy(target['global26'][p]).float()),
                'body_pose': torch.from_numpy(target['body_pose'][p]).float(), 'joints_cam': joints[p],
                'joint_uv': uv[1:], 'fk_valid': torch.ones(22, dtype=torch.bool), 'projection_valid': projected[1:],
                'body_height': torch.tensor(target['body_height'][p], dtype=torch.float32),
                'smpl_bbox_height': (box[3]-box[1])*target_spec.canvas_hw[0],
                'pose_provenance': {'record_id': tm['record_id']+f':{p}', 'record_sha256': tm['record_sha256'],
                                    'person_id': tm['person_ids'][p], 'canvas_id': target_spec.fingerprint()}}
            sample.update({key+suffix: value for key, value in fields.items()})
        relative, orientation_valid = relative11(joints, torch.from_numpy(target['root_rotation']).float(),
                                                  torch.from_numpy(target['body_height']).float(), torch.stack(boxes))
        sample['relative_geometry'] = relative
        sample['relative_orientation_valid'] = torch.tensor(orientation_valid)
        sample['contact_raster'] = onto_canvas(torch.from_numpy(target['contact_raster']).float()[None], target_spec)[0]
        # Legacy eight-token contact bridge; all 196 relations and raw vertex pairs
        # remain in the new typed branch, so this truncation loses no V3 records.
        contact = torch.zeros(8, dtype=torch.long)
        relations = ContactRelationBatch(contact.clone()[None], contact.clone()[None], contact.clone()[None],
            contact.clone()[None], contact.clone()[None], torch.zeros(1, 8, 1), torch.zeros(1, 8, dtype=torch.bool))
        pairs = [(i, j) for i in range(14) for j in range(14)
                 if target['part_contact'][i, j, 0] > 0 and target['part_distance_valid'][i, j]] if row['kind'] == 'camera_z' else []
        pairs.sort(key=lambda ij: float(target['part_distance'][ij]))
        for slot, (i, j) in enumerate(pairs[:8]):
            relations.src_part[0, slot], relations.dst_part[0, slot], relations.dst_person[0, slot] = i, j, 1
            relations.distance[0, slot, 0] = float(target['part_distance'][i, j])
            relations.valid_mask[0, slot] = True
        sample['contact_relations'] = relations
        if row['kind'] != 'camera_z':
            sample['contact_raster'].zero_()
        if self.load_target:
            sample['tgt_image'] = _rgb(self.verify(row['target_rgb']), target_spec)
        return sample


def collate_native_dual(samples):
    if not samples:
        raise ValueError('Empty dual batch')
    keys = set(samples[0])
    if any(set(s) != keys for s in samples):
        raise ValueError('Mixed training/inference samples are not a batch')
    grids = ('target_token_hw', 'source_token_hw')
    for key in grids:
        if any(s[key] != samples[0][key] for s in samples):
            raise ValueError('Bucket both reference and target by exact canvas grids')
    result = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if key == 'dual_depth':
            result[key] = DualDepthConditionV3.stack(values)
        elif key == 'contact_relations':
            result[key] = ContactRelationBatch(**{name: torch.cat([getattr(v, name) for v in values])
                                                for name in ContactRelationBatch.__dataclass_fields__})
        elif torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        elif key in grids:
            result[key] = values[0]
        else:
            result[key] = deepcopy(values)
    return result


def identity_from_reference(batch, source_latent, source_token_hw):
    b, n, channels = source_latent.shape
    if channels != 64 or n != source_token_hw[0]*source_token_hw[1] or tuple(batch['source_token_hw']) != tuple(source_token_hw):
        raise ValueError('Source identity must reuse the exact native Qwen reference grid and 64 channels')
    d = source_latent.device
    person = batch['source_person_mask'].to(d)
    return NativeDualIdentityCondition(source_latent.transpose(1, 2).reshape(b, 64, *source_token_hw),
        person, batch['source_head_mask'].to(d), batch['source_indices'].to(d), person.flatten(2).any(2),
        tuple(deepcopy(batch['identity_metadata']))).validate()
