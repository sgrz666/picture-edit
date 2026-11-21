"""The dataset switch must not leak notes or duplicate images across splits."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / 'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/prepare.py'
    spec = importlib.util.spec_from_file_location('xhs_region_prepare', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def inputs():
    pairs = []; frames = {}
    for i in range(20):
        note = f'note{i:02d}'
        for suffix in ('a', 'b'):
            frames[note + suffix] = {'clip_id': note, 'rgb_sha256': note + suffix}
        pairs.append({'pair_id': note, 'note_id': note, 'source_frame_id': note + 'a',
                      'target_frame_id': note + 'b', 'disk_status': 'complete'})
    return pairs, frames


def test_note_split_is_deterministic_and_groups_duplicate_rgb_content():
    m = module(); pairs, frames = inputs()
    frames['note01a']['rgb_sha256'] = frames['note00b']['rgb_sha256']
    first = m.split_pairs(pairs, frames, 42)
    assert first == m.split_pairs(list(reversed(pairs)), frames, 42)
    assert first['note_splits']['note00'] == first['note_splits']['note01']
    hashes = {}
    for split in ('train', 'val', 'test'):
        hashes[split] = {frames[p[key]]['rgb_sha256'] for p in first[split]
                         for key in ('source_frame_id', 'target_frame_id')}
    assert not hashes['train'] & hashes['val']
    assert not hashes['train'] & hashes['test']
    assert not hashes['val'] & hashes['test']


def test_pilot_requires_visible_hands_and_uses_one_pair_per_note():
    m = module(); pairs, _ = inputs()
    pose = {p['target_frame_id']: {'valid_mask134': [True] * 134} for p in pairs}
    pose[pairs[0]['target_frame_id']]['valid_mask134'][92:] = [False] * 42
    chosen = m.select_pilot(pairs, pose, 10, 42)
    assert len(chosen) == len({p['note_id'] for p in chosen}) == 10
    assert pairs[0] not in chosen
    with pytest.raises(ValueError, match='qualified'):
        m.select_pilot(pairs, pose, 20, 42)


def test_pair_cannot_claim_a_different_note_than_its_rgb():
    m = module(); pairs, frames = inputs()
    pairs[0]['target_frame_id'] = pairs[1]['target_frame_id']
    with pytest.raises(ValueError, match='note'):
        m.split_pairs(pairs, frames, 42)


def test_estimator_crop_binding_only_replaces_the_bbox_detector(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/geometry.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('xhs_geometry', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    source = 'def main():\n    yolo_bbox = detector.predict(image).boxes.numpy()\n    output = model(image)\n'
    rewritten = m.bind_sdpose_boxes(source, tmp_path/'boxes.npy')
    assert 'detector.predict' not in rewritten
    assert 'output = model(image)' in rewritten
    assert 'allow_pickle=False' in rewritten and 'frame-1:frame' in rewritten
    with pytest.raises(ValueError, match='detector'):
        m.bind_sdpose_boxes('output = model(image)\n', tmp_path/'boxes.npy')


@pytest.mark.parametrize('score', [.9, 1.2])
def test_sdpose_hands_keep_anatomical_slots_and_intersect_sam_before_pooling(monkeypatch,score):
    import numpy as np
    from src.data.canvas_contract import native_pad
    path = Path(__file__).parents[1] / 'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/assets.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('xhs_assets', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    canvas = native_pad((64, 96)); body = np.zeros((64,96), np.uint8); body[5:60,5:90] = 1
    points = np.full((134,2), [45.,30.]); points[7] = [25,35]; points[4] = [70,35]
    for start,x in ((92,25),(113,70)):
        points[start:start+21] = [[x+(i%5)*2,35+(i//5)*2] for i in range(21)]
    sd = {'points134_raw': points.tolist(), 'scores134_raw': [score]*134, 'valid_mask134': [True]*134,
          'bbox_xyxy': [5,5,90,60], 'height':64, 'width':96}
    arrays,meta = m.build_regions(sd,body,canvas,['person','__inactive__'],'a'*64,'b'*64,.8)
    assert arrays['hands'][0].sum()>0 and arrays['hands'][1].sum()>0
    assert not np.array_equal(arrays['hands'][0],arrays['hands'][1])
    assert not arrays['hands'][2:].any() and not arrays['person'][1].any()
    assert np.all(arrays['hands'] <= body[None])
    assert np.array_equal(arrays['background'][0],canvas.valid_mask()[0].numpy()-body)
    assert meta['hand_slot_order']==['A-left','A-right','B-left','B-right']
    assert arrays['joint_quality'].max()<=1 and arrays['hands_quality'].max()<=1
