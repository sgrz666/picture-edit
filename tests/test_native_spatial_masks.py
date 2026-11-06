"""Offline supervision geometry, not a pose-estimator accuracy benchmark."""
from copy import deepcopy
import numpy as np
import pytest


def frame():
    points = np.tile([48., 80.], (134, 1))
    points[2:14] = [[30, 40], [25, 60], [20, 80], [60, 40], [65, 60], [70, 80],
                     [35, 90], [35, 115], [35, 140], [55, 90], [55, 115], [55, 140]]
    hand = [[20 + (i % 5) * 3, 70 + (i // 5) * 3] for i in range(21)]
    return dict(frame_id='test', original_hw=[160, 96], candidate=points.tolist(),
                effective_confidence=[.9] * 134, valid_keypoint_mask=[True] * 134,
                loss_mask=dict(body=True, face=True, hand_a=True, hand_b=False),
                hand_raw_pose_usable=dict(hand_a=True, hand_b=False),
                hand_landmarks21=dict(hand_a=hand, hand_b=hand),
                hand_block_slices=dict(hand_a=[113, 134], hand_b=[92, 113]),
                detection=dict(primary_box=[10, 10, 85, 150]))


def test_canvas_matches_training_portrait_resize_and_padding():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for
    spec = canvas_for((1920, 1080), 512)
    assert spec.resized_hw == (512, 288) and spec.canvas_hw == (512, 288)
    spec = canvas_for((1280, 720), 0)
    assert spec.resized_hw == (1280, 720) and spec.canvas_hw == (1280, 736)


def test_hand_regions_respect_each_existing_gate_and_do_not_mutate_input():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for, regional_maps
    f = frame(); before = deepcopy(f); spec = canvas_for(f['original_hw'], 160)
    body = np.ones(tuple(f['original_hw']), dtype=np.uint8)
    maps, status = regional_maps(f, body, spec)
    assert maps['hand'].max() > .5 and status['hands']['hand_a']['enabled']
    assert not status['hands']['hand_b']['enabled'] and f == before
    f['hand_raw_pose_usable']['hand_a'] = False
    maps, status = regional_maps(f, body, spec)
    assert maps['hand'].max() == 0 and not status['hands']['hand_a']['enabled']


def test_out_of_frame_and_invalid_joints_never_create_border_hotspots():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for, regional_maps
    f = frame(); f['candidate'] = [[-20., -20.]] * 134
    f['loss_mask']['hand_a'] = False; spec = canvas_for(f['original_hw'], 160)
    maps, _ = regional_maps(f, np.ones((160, 96), np.uint8), spec)
    assert maps['joint'].max() == 0 and maps['hand'].max() == 0


def test_tiny_hand_is_soft_and_survives_area_downsample():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for, regional_maps
    from PIL import Image
    f = frame(); spec = canvas_for(f['original_hw'], 160)
    maps, _ = regional_maps(f, np.ones((160, 96), np.uint8), spec)
    low = np.array(Image.fromarray(maps['hand']).resize((6, 10), Image.Resampling.BOX))
    assert low.max() > 0 and ((maps['hand'] > 0) & (maps['hand'] < 1)).any()


def test_pillow_blur_receives_builtin_float_for_numpy_derived_hand_radius(monkeypatch):
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for, regional_maps
    from PIL import ImageFilter
    original = ImageFilter.GaussianBlur
    def compatible_radius(radius):
        assert type(radius) is float, 'Older Pillow treats NumPy scalar radius as an xy pair'
        return original(radius)
    monkeypatch.setattr(ImageFilter, 'GaussianBlur', compatible_radius)
    f=frame()
    f['hand_landmarks21']['hand_a']=[[20+(i%5)*9,70+(i//5)*9] for i in range(21)]
    maps,_=regional_maps(f,np.ones((160,96),np.uint8),canvas_for((160,96),160))
    assert maps['hand'].max()>0


def test_sam_prompts_use_real_frame_box_and_inframe_body_points():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import build_prompts
    f = frame(); f['candidate'][2] = [-100., -100.]
    xy, labels, box = build_prompts(f)
    assert (xy >= 0).all() and labels.shape == (len(xy),)
    assert box[0] <= 10 and box[2] >= 85 and box[3] < 160


def test_wrong_body_canvas_fails_instead_of_stretching():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for, regional_maps
    f = frame()
    with pytest.raises(ValueError, match='Body|body|shape|original'):
        regional_maps(f, np.ones((96, 160), np.uint8), canvas_for((160, 96), 160))


def test_generator_refuses_overwrite_and_preserves_component_values(tmp_path):
    from scripts.data.build_native_spatial_weights import save_npz
    value = np.ones((32, 32), dtype=np.float32)
    path = tmp_path / 'components.npz'
    maps = dict(body=value, joint=value*.5, hand=value*0, valid=value)
    save_npz(path, maps, {'schema': 'native_spatial_components_v1'})
    with np.load(path, allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached['joint'], value*.5)
    with pytest.raises(FileExistsError):
        save_npz(path, maps, {})


def test_pilot_selection_includes_hand_enabled_examples_without_filtering_full_data():
    from scripts.data.build_native_spatial_weights import select_frames
    rows = []
    for i in range(80):
        f = frame(); f['frame_id'] = str(i); f['clip_id'] = str(i//4)
        f['loss_mask']['hand_a'] = i >= 60
        f['hand_raw_pose_usable']['hand_a'] = i >= 60
        rows.append(f)
    assert len(select_frames(rows, 0, 42)) == 80
    selected = select_frames(rows, 20, 42)
    assert len(selected) == 20 and any(f['frame_id'] in [str(i) for i in range(60,80)] for f in selected)
    assert len({f['clip_id'] for f in selected}) >= 10
    assert len(select_frames(rows, 1, 42)) == 1


def test_interrupted_frame_can_resume_without_overwriting_partial_attempt(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from scripts.data.build_native_spatial_weights import process_frame, sha
    f=frame(); rgb=tmp_path/'rgb.png';Image.new('RGB',(96,160),'gray').save(rgb)
    f.update(rgb_path=str(rgb),rgb_sha256=sha(rgb))
    args=SimpleNamespace(output_root=tmp_path/'output',canvas_long_side=160,geometry_method='sapiens_v1',resume=True)
    class Predictor:
        attempts=0
        def __call__(self,image,row):
            self.attempts+=1
            if self.attempts==1:raise RuntimeError('interrupted inference')
            return np.ones((160,96),np.uint8),{'sam_score':1.}
    predictor=Predictor()
    with pytest.raises(RuntimeError,match='interrupted'):
        process_frame(f,args,predictor,{})
    marker=tmp_path/'output'/'frames'/'test'/'ready.json'
    marker.write_text('{"identity":',encoding='utf-8')
    result=process_frame(f,args,predictor,{})
    assert result['metadata']['frame_id']=='test'
    assert len(list((tmp_path/'output'/'frames'/'test').glob('attempt_*')))==2
    assert process_frame(f,args,predictor,{})==result
    assert predictor.attempts==2
    assert marker.read_text()=='{"identity":'


def test_null_person_box_uses_keypoint_extent_consistently():
    from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for,regional_maps,build_prompts
    f=frame();f['detection']['primary_box']=None
    assert build_prompts(f)[2].shape==(4,)
    maps,_=regional_maps(f,np.ones((160,96),np.uint8),canvas_for((160,96),160))
    assert maps['joint'].max()>0


def test_json_recovery_retains_truncated_marker_and_checks_existing_index(tmp_path):
    from scripts.data.build_native_spatial_weights import publish_recoverable
    path=tmp_path/'index.json';path.write_text('{"entries":',encoding='utf-8')
    fixed=publish_recoverable(path,{'entries':{}})
    assert fixed!=path and path.read_text()=='{"entries":'
    assert publish_recoverable(path,{'entries':{}})==fixed
    with pytest.raises(ValueError,match='differs'):
        publish_recoverable(path,{'entries':{'changed':1}})
