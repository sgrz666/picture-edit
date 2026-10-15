from __future__ import annotations

import pytest
import torch

from scripts.prepare_tiktok_qwen21_face import assemble_cache
from scripts.data.prepare_qwen21_prompt_study_faces import requested_frames, prepare_study_face_cache


def test_target_only_frame_has_geometry_but_no_identity():
    geometry = [
        dict(
            face_landmarks=torch.zeros(72, 3),
            face_boxes=torch.tensor([0.2, 0.1, 0.6, 0.4]),
            jaw_pose=torch.ones(3),
            expression=torch.ones(10),
            geometry_valid=True,
        )
        for _ in range(2)
    ]
    source_features = {"0001": (torch.ones(512), torch.ones(256, 1536))}
    cache = assemble_cache(
        ["0001", "0002"],
        geometry,
        source_features,
        {"schema_version": 1, "appearance": "00001"},
    )
    assert cache["valid"].tolist() == [True, False]
    assert cache["geometry_valid"].tolist() == [True, True]
    assert cache["metadata"]["geometry_valid"] == [True, True]
    assert not cache["arcface"][1].any()
    assert not cache["dino_patches"][1].any()
    assert cache["jaw_pose"][1].any()


def test_requested_frames_aggregates_sources_and_targets():
    rows = [
        dict(
            pair_id="tiktok_00",
            sequence_id="00232",
            source_appearance="00130",
            target_appearance="00232",
            source_stem="0032",
            target_stem="0223",
            source_rgb="/a.png",
            target_rgb="/b.png",
        ),
        dict(
            pair_id="tiktok_01",
            sequence_id="00232",
            source_appearance="00130",
            target_appearance="00232",
            source_stem="0032",
            target_stem="0250",
            source_rgb="/a.png",
            target_rgb="/c.png",
        ),
    ]
    groups = requested_frames(rows)
    assert "00130" in groups
    assert "00232" in groups
    assert groups["00130"]["stems"] == {"0032"}
    assert groups["00130"]["sources"]["0032"] == "/a.png"
    assert groups["00232"]["stems"] == {"0223", "0250"}


def test_missing_real_assets_never_fabricates_valid_face_cache(tmp_path):
    import json
    data = tmp_path / "data"
    data.mkdir()
    row = dict(pair_id="p", sequence_id="00001", source_stem="0001",
               target_stem="0002", source_rgb="/missing/a.png", target_rgb="/missing/b.png")
    (data / "train512_pairs.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(FileNotFoundError):
        prepare_study_face_cache(tmp_path, ["train"], assets_root=tmp_path / "assets",
                                 raw_root=tmp_path / "raw")
    assert not (data / "face_cache" / "00001.pt").exists()


def test_missing_split_is_not_silently_skipped(tmp_path):
    with pytest.raises(FileNotFoundError, match="split"):
        prepare_study_face_cache(tmp_path, ["train"])


def test_real_projection_reuses_only_a_features_and_zeros_target_identity(tmp_path, monkeypatch):
    import json
    from PIL import Image
    import scripts.data.prepare_qwen21_prompt_study_faces as module
    from scripts.precompute_iper_face_features_v65 import atomic_save_cache, load_v65_face_cache
    from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
    data, assets, old = tmp_path / 'data', tmp_path / 'assets', tmp_path / 'old'
    data.mkdir(); old.mkdir()
    frame_dir = assets / '00001' / 'smplestx'; frame_dir.mkdir(parents=True)
    a, b = tmp_path / 'a.png', tmp_path / 'b.png'
    Image.new('RGB', (604, 1080), 'white').save(a)
    Image.new('RGB', (604, 1080), 'black').save(b)
    params = {'jaw_pose': torch.tensor([[1., 0., 0.], [2., 0., 0.]]), 'expression': torch.ones(2, 10)}
    torch.save(params, frame_dir / 'params.pt')
    (frame_dir / 'frame_map.json').write_text(json.dumps({'frames': [{'index': 0, 'stem': '0001'}, {'index': 1, 'stem': '0002'}]}))
    points = torch.stack([torch.linspace(.2, .4, 72), torch.linspace(.1, .3, 72), torch.ones(72)], dim=1)
    geometry = [dict(face_landmarks=points, face_boxes=torch.tensor([.15, .05, .45, .35]),
                     jaw_pose=params['jaw_pose'][i], expression=params['expression'][i], geometry_valid=True) for i in range(2)]
    cached = assemble_cache(['0001', '0002'], geometry,
                            {s: (torch.ones(512), torch.ones(256, 1536)) for s in ['0001', '0002']},
                            {'schema_version': 1, 'appearance': '00001', 'source_hashes': {'0001': sha256_file(a), '0002': sha256_file(b)}})
    atomic_save_cache(cached, old / '00001.pt')
    row = dict(pair_id='p', sequence_id='00001', source_stem='0001', target_stem='0002', source_rgb=str(a), target_rgb=str(b))
    (data / 'train512_pairs.jsonl').write_text(json.dumps(row) + '\n')
    observed = []
    def project(actual, **kwargs):
        observed.append(actual['jaw_pose'].clone())
        return points.unsqueeze(0).repeat(2, 1, 1)
    monkeypatch.setattr(module, 'project_tiktok_face72_to_rgb_normalized', project)
    prepare_study_face_cache(tmp_path, ['train'], old, assets, tmp_path / 'raw')
    result = load_v65_face_cache(data / 'face_cache/00001.pt')
    assert len(observed) == 1 and torch.equal(observed[0], params['jaw_pose'])
    assert result['valid'].tolist() == [True, False]
    assert result['jaw_pose'][1, 0] == 2
    assert not result['arcface'][1].any() and not result['dino_patches'][1].any()
    assert result['metadata']['preprocess']['per_source'][0]['reason'] == 'reused_verified_source'
