import pytest


def test_spatial_score_uses_two_strict_subscores():
    from src.pose_control.qwen21_full.native_rl.spatial_reward import parse_scores, spatial_score

    semantic = parse_scores('```json\n{"edit_region": [], "score": [20, 15]}\n```')
    quality = parse_scores('{"score": [25, 20]}')
    result = spatial_score(semantic, quality, 'official')
    assert result['reward'] == pytest.approx((18 / 25)**.8 * (22.5 / 25)**.2)
    assert spatial_score(semantic, quality, 'complementary')['reward'] == pytest.approx(.6**.8 * .9**.2)
    assert parse_scores('<think>\n</think>\n{"naturalness": 18, "artifacts": 15}', stream='pq') == [18, 15]
    with pytest.raises(ValueError, match='SpatialReward'):
        parse_scores('{"naturalness": 18, "artifacts": 15}', stream='sc')
    for invalid in ('[20, 21]', '{"score": [25, true]}', '{"score": [4, 26]}', '{"score": [8, NaN]}', '{"score": [20]}'):
        with pytest.raises(ValueError, match='SpatialReward'):
            parse_scores(invalid)


@pytest.mark.parametrize('profile,roles', [
    ('single_2ref', ['source', 'target_pose']),
    ('dual_4ref', ['source', 'combined_pose', 'person_0_pose', 'person_1_pose']),
    ('dual_source_5ref', ['source_a', 'source_b', 'person_0_pose', 'person_1_pose', 'combined_pose']),
])
def test_spatial_references_keep_native_order_and_exclude_target_rgb(profile, roles, tmp_path):
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_rl.spatial_reward import spatial_target
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha

    canvas = native_pad((32, 32))
    assets = []
    for role in roles:
        path = tmp_path / f'{role}.png'
        Image.new('RGB', (32, 32)).save(path)
        assets.append({'role': role, 'space': 'canvas', 'file': {'path': str(path), 'sha256': file_sha(path)},
                       'canvas': canvas.metadata(), 'canvas_id': canvas.fingerprint()})
    spec = {'profile': profile, 'primary_source_index': 0, 'images': assets, 'target_canvas_id': canvas.fingerprint()}
    slots = [1, 0] if profile == 'dual_source_5ref' else None
    guided = spatial_target(spec, 'guides', slot_sources=slots)
    assert guided.roles == tuple(roles)
    if slots:
        assert 'Slot 0 belongs to Source B' in guided.instruction
    rgb = spatial_target(spec, 'rgb')
    assert rgb.roles == (('source_a', 'source_b') if profile == 'dual_source_5ref' else ('source',))
    assert guided.prompt_sha256 != rgb.prompt_sha256
    with pytest.raises(ValueError, match='references'):
        spatial_target({**spec, 'images': assets + [{**assets[0], 'role': 'target_rgb'}]}, 'guides')


def test_spatial_worker_validates_candidate_identity_and_weight_is_opt_in():
    from src.pose_control.qwen21_full.native_rl.cli import _reward_weights
    from src.pose_control.qwen21_full.native_rl.spatial_worker import ordered_results

    a, b = {'id': 0, 'sc': [20, 18], 'pq': [24, 21]}, {'id': 1, 'sc': [10, 9], 'pq': [24, 21]}
    assert [row['id'] for row in ordered_results([b, a], 2, 'official')] == [0, 1]
    with pytest.raises(ValueError, match='candidate'):
        ordered_results([a, a], 2, 'official')
    with pytest.raises(ValueError, match='candidate'):
        ordered_results([a], 2, 'official')
    base = _reward_weights('dual', pose_only=False)
    mixed = _reward_weights('dual', pose_only=False, spatial_weight=.1)
    assert mixed['spatial'] == .1 and 'spatial' not in base
    assert mixed['binding'] == pytest.approx(.9 * base['binding'])
    assert sum(mixed.values()) == pytest.approx(1.)


def test_spatial_failure_disables_the_whole_group(tmp_path):
    import json
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.native_rl.cli import _add_spatial, _regressions

    def fail(*_):
        raise RuntimeError('worker timed out')

    group = [{'reward': {'pose': .5}}, {'reward': {'pose': .7}}]
    bank = SimpleNamespace(spatial=SimpleNamespace(score_group=fail))
    with pytest.raises(RuntimeError, match='timed out'):
        _add_spatial(bank, SimpleNamespace(spatial=object()), ['a.png', 'b.png'], group, tmp_path, ('train', 2))
    assert all('spatial' not in row['reward'] for row in group)
    assert json.loads((tmp_path / 'evaluation-failed.json').read_text())['evaluator'] == 'spatialreward'
    args = SimpleNamespace(spatial_model='model', pose_drop=.02, id_drop=.02, background_drop=.02, binding_drop=.02)
    assert _regressions({'pose': .8, 'background': .4, 'binding': .5},
                        {'pose': .7, 'background': .6, 'binding': .6}, args) == ['background', 'binding']
