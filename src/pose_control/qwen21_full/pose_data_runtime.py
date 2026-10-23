"""Native v2 paired data. Legacy loaders/caches remain explicit independent paths."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.data.canvas_contract import aspect_preserving, native_pad
from src.data.tiktok_dataset import _pose_heatmaps
from .data_runtime import QwenFullDataset, read_pairs, pair_metadata, SAMPLED, ASSETS
from .pose_cache import BodyPoseRecords, record_id


def _pil(path, mode='RGB'):
    with Image.open(path) as image:
        return image.convert(mode)


def _rgb(image, spec):
    array = np.asarray(spec.image(image, fill=(128, 128, 128)), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1


class NativePoseDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, pose_manifest, *, canvas_policy='aspect_preserving', max_pixels=512*512,
                 sampled_root=SAMPLED, assets_root=ASSETS, **tiktok_options):
        if canvas_policy not in ('native', 'aspect_preserving'):
            raise ValueError('Body21 requires a canonical native/aspect_preserving canvas')
        self.policy, self.max_pixels = canvas_policy, max_pixels
        self.records = BodyPoseRecords(pose_manifest)
        self.rows = read_pairs(pairs)
        if not self.rows:
            raise ValueError('Empty pairs')
        kinds = {'tiktok' if 'sequence_id' in row else 'iper' for row in self.rows}
        if len(kinds) != 1:
            raise ValueError('Use one dataset per manifest')
        self.dataset_type = kinds.pop()
        self.sampled_root, self.assets_root = Path(sampled_root), Path(assets_root)
        self.base = None
        if self.dataset_type == 'tiktok':
            self.base = QwenFullDataset(pairs, resolution='native', dataset_type='tiktok_v3',
                                        align_mode=canvas_policy, max_pixels=max_pixels, **tiktok_options)
            self.rows = self.base.rows

    def __len__(self):
        return len(self.rows)

    @property
    def samples(self):
        return self.rows

    def _spec(self, image):
        hw = (image.height, image.width)
        return native_pad(hw) if self.policy == 'native' else aspect_preserving(hw, max_pixels=self.max_pixels)

    def __getitem__(self, index):
        row = self.rows[index]
        provenance = pair_metadata(row, self.sampled_root)
        if provenance['cross_person'] or row.get('geometry_retargeted', False):
            raise ValueError('Body21 paired RGB requires target-frame shape, pose and camera; cross-person retargeting is separate')
        role = row.get('target', {}).get('role', 'target') if self.dataset_type == 'iper' else 'target'
        key = record_id(self.dataset_type, provenance['target_appearance'], role, provenance['target_stem'])
        data = self.records.raw(key)
        meta = data['metadata']
        if Path(meta['sources']['rgb']['path']).resolve() != Path(provenance['target_rgb']).resolve():
            raise ValueError('Pose cache frame/RGB pair differs')
        if self.base is not None:
            sample = self.base[index]
            # The baseline TikTok spatial maps are kept. Bind their actual sources.
            for name, path in (('normal', row['target_normal']), ('human_mask', row['target_person_mask'])):
                if Path(meta['sources'][name]['path']).resolve() != Path(path).resolve():
                    raise ValueError('Pose cache and existing spatial condition sources differ')
        else:
            source, target = _pil(provenance['source_rgb']), _pil(provenance['target_rgb'])
            a, b = self._spec(source), self._spec(target)
            def raster(name, mode='RGB', nearest=False):
                image = _pil(meta['sources'][name]['path'], mode)
                transform = meta['condition_to_original'][name]
                if (image.height, image.width) != tuple(transform['source_hw']):
                    raise ValueError('Condition asset shape changed')
                # Asset -> original normalised frame -> one shared target canvas.
                if (image.height, image.width) != b.original_hw:
                    image = image.resize((b.original_hw[1], b.original_hw[0]), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
                array = np.asarray(b.image(image, nearest=nearest), dtype=np.float32).copy()
                value = torch.from_numpy(array)
                return value[None] / 255 if mode == 'L' else value.permute(2, 0, 1) / 255
            mask = raster('smpl_mask', mode='L', nearest=True)
            normal = raster('normal')
            labels = (raster('part', mode='L', nearest=True)[0] * 255).round().long()
            if (labels < 0).any() or (labels > 14).any() or not (mask > 0).any():
                raise ValueError('Invalid canonical part/mask')
            part = F.one_hot(labels, num_classes=15)[..., 1:].permute(2, 0, 1).float()
            legacy = torch.load(meta['sources']['legacy_conditions']['path'], map_location='cpu', weights_only=False)
            points = legacy['kps_25_coords'][legacy['stem_to_idx'][provenance['target_stem']]].float().clone()
            if points.shape != (25, 3) or not torch.isfinite(points).all():
                raise ValueError('Body25 heatmap cache missing or invalid')
            points = b.points(points)
            depth = raster('depth_vis', mode='L')
            sample = {'src_image': _rgb(source, a), 'tgt_image': _rgb(target, b),
                      'normal': (normal * 2 - 1) * mask, 'part_onehot': part, 'human_mask': mask,
                      'pose_heatmap': _pose_heatmaps(points, resolution=b.canvas_hw, sigma=4.) * b.valid_mask(),
                      'depth': depth, 'control_map': torch.cat((normal, depth, torch.zeros_like(normal), mask)),
                      'valid_image_mask': b.valid_mask(), 'source_canvas': a.metadata(), 'target_canvas': b.metadata(),
                      'source_token_hw': a.token_hw, 'target_token_hw': b.token_hw,
                      'smplx_global': data['smplx_global'], 'task_id': torch.tensor(0, dtype=torch.long),
                      'appearance': provenance['source_appearance'], 'source_stem': provenance['source_stem'],
                      'target_stem': provenance['target_stem'], 'pair_metadata': provenance,
                      'depth_metadata': {'metric': False, 'source': 'depth_vis'},
                      'camera_metadata': {'original_smplx_global': data['smplx_global']}}
        canonical = self.records.read(key, sample['target_canvas'])
        sample.update(canonical)
        sample['camera_metadata'] = {'convention': 'canonical Global26; same-source camera, both translation slots synchronised',
                                     'original_smplx_global': data['smplx_global'],
                                     'canvas_smplx_global': canonical['smplx_global']}
        from src.data.geometry_pair_dataset_v2 import assert_pair_contract
        assert_pair_contract(sample, where='Native Body21 paired geometry')
        return sample
