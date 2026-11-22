import importlib.util
from pathlib import Path
import pytest
import torch


def driver():
    path=Path(__file__).parents[1]/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts/rl.py'
    spec=importlib.util.spec_from_file_location('bilibili_region_rl_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_reference_weight_context_restores_parameter_objects_on_failure():
    from src.pose_control.qwen21_full.native_rl.runner import FrozenReference

    net = torch.nn.Linear(2, 1)
    reference = FrozenReference(net.parameters(), storage_device='cpu')
    with torch.no_grad():
        net.weight.add_(1.)
    current = net.weight.detach().clone()
    ids=[id(p) for p in net.parameters()]
    with torch.no_grad(), pytest.raises(RuntimeError, match='controlled'):
        with reference.use():
            assert torch.equal(net.weight, reference.weights[0])
            raise RuntimeError('controlled reference-probe failure')
    assert [id(p) for p in net.parameters()]==ids
    assert torch.equal(net.weight, current)


def test_regional_reward_columns_keep_generation_misses_and_reject_invalid_measurements():
    score = driver().score_columns
    applicable = {'body': True, 'hands': False, 'identity': True, 'background': False}
    row = {'pose': {'body': {'reward': .5}, 'hands': {'reward': None}},
           'identity': {'identity_reward': 0.}, 'pixel': {'background_reward': None}}
    assert score([row], applicable) == {'body': [.5], 'identity': [0.]}
    row['identity']['identity_reward'] = None
    with pytest.raises(ValueError, match='fixed target eligibility'):
        score([row], applicable)


def test_background_scores_source_on_verified_support_and_disables_unknown_alignment(tmp_path):
    import numpy as np
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_rl.background import prepare_background
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha

    canvas = native_pad((64, 64))
    src = np.zeros((64, 64, 3), dtype=np.uint8)
    gt = np.full_like(src, 190)
    gt[25:35, 28:35] = 255
    source_mask = src[:, :, 0].copy()
    source_mask[25:35, 18:25] = 255
    refs = {}
    for name, image in (('source_rgb', src), ('target_rgb', gt), ('source_mask', source_mask)):
        path = tmp_path / f'{name}.png'
        Image.fromarray(image).save(path)
        refs[name] = {'path': str(path), 'sha256': file_sha(path)}
    pair = {**refs, 'source_canvas': canvas.metadata(), 'target_canvas': canvas.metadata()}
    target_mask = gt[:, :, 0] > 190
    assert prepare_background({**pair, 'source_mask': None}, Image.fromarray(src), target_mask)[0] is None
    pair['background_alignment'] = {'method': 'identity', 'source_rgb_sha256': refs['source_rgb']['sha256'],
                                    'target_rgb_sha256': refs['target_rgb']['sha256'], 'canvas_id': canvas.fingerprint()}
    bg, reason = prepare_background(pair, Image.fromarray(src), target_mask)
    assert reason is None and bg.score(src)['background_reward'] == pytest.approx(1.)
    assert bg.score(src)['background_reward'] > bg.score(gt)['background_reward']
    assert bg.score(np.full_like(src, 255))['background_reward'] < .001
    assert prepare_background({**pair, 'source_mask_accepted': False}, Image.fromarray(src), target_mask) == (
        None, 'source_foreground_unqualified')
