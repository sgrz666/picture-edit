from __future__ import annotations

import sys
from pathlib import Path
import pytest

from src.pose_control.qwen21_full.prompt_study.sampling import (
    make_order,
    noise_seed,
    validate_trajectory,
)
import scripts.train.train_qwen21_joint_pilot as train_pilot


def test_resume_at_epoch_boundary_uses_locked_permutation():
    order = make_order()
    saved = {"step": 512, "sample_order": order[:512]}
    rows = [
        dict(step=u + 1, pair_index=order[u], noise_seed=noise_seed(u))
        for u in range(512)
    ]
    validate_trajectory(saved, rows, order)
    assert order[511] == 511
    assert order[512] == make_order()[512]
    rows[-1]["noise_seed"] += 1
    with pytest.raises(ValueError, match="trajectory"):
        validate_trajectory(saved, rows, order)


def test_argument_parsing_study_vs_legacy(monkeypatch):
    base_args = [
        "train.py",
        "--train-pairs", "train.jsonl",
        "--val-pairs", "val.jsonl",
        "--features", "features",
        "--face-supervision", "face_sup.json",
        "--cache", "cache",
        "--output", "output",
        "--init-checkpoint", "init.pt",
        "--gradient-only",
    ]

    # Legacy: steps 1024 should fail
    monkeypatch.setattr(sys, "argv", base_args + ["--steps", "1024"])
    with pytest.raises(SystemExit):
        train_pilot.arguments()

    # Legacy: steps 256 succeeds
    monkeypatch.setattr(sys, "argv", base_args + ["--steps", "256"])
    args = train_pilot.arguments()
    assert args.steps == 256

    # Study: steps 256 should fail
    study_args = base_args + [
        "--study-lock", "lock.json",
        "--arm", "qwen_p0",
        "--prompt-manifest", "prompts.jsonl",
        "--resolution", "native",
        "--align-mode", "native",
        "--noise-offset", "256",
        "--seed", "42",
    ]
    monkeypatch.setattr(sys, "argv", study_args + ["--steps", "256"])
    with pytest.raises(SystemExit):
        train_pilot.arguments()

    # Study: steps 1024 succeeds and supports stop-after-step
    monkeypatch.setattr(
        sys,
        "argv",
        study_args + ["--steps", "1024", "--stop-after-step", "128"],
    )
    args = train_pilot.arguments()
    assert args.steps == 1024
    assert args.stop_after_step == 128
    assert args.arm == "qwen_p0"


@pytest.mark.parametrize("flag,value", [
    ("--body-lr-scale", "1"), ("--face-lr-scale", "1"),
    ("--face-weight", "1"), ("--warmup-steps", "40"),
    ("--aux-every", "1"), ("--save-every", "64"), ("--strength", "0.5"),
    ("--hand-lr-scale", "0.5"), ("--hand-weight", "0.2"),
])
def test_study_rejects_changed_fixed_training_parameters(monkeypatch, flag, value):
    argv = ["train.py", "--train-pairs", "train", "--val-pairs", "val",
            "--features", "face", "--face-supervision", "sup", "--cache", "cache",
            "--output", "out", "--init-checkpoint", "J.pt", "--gradient-only",
            "--study-lock", "lock", "--arm", "qwen_p0", "--prompt-manifest", "prompts",
            "--steps", "1024", "--resolution", "native", "--align-mode", "native",
            "--noise-offset", "256", flag, value]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        train_pilot.arguments()


@pytest.mark.parametrize("step", [-1, 1025])
def test_resume_rejects_out_of_bounds_checkpoint_steps(step):
    order = make_order()
    saved = {"step": step, "sample_order": order[:step]}
    rows = [dict(step=u + 1, pair_index=index, noise_seed=noise_seed(u))
            for u, index in enumerate(order[:step])]
    with pytest.raises(ValueError, match="step"):
        validate_trajectory(saved, rows, order)


@pytest.mark.parametrize("field,wrong", [("pair_id", "other"), ("prompt_sha256", "wrong")])
def test_resume_checks_pair_and_prompt_identity(field, wrong):
    order = make_order()
    ids = [f"pair-{i}" for i in range(512)]
    hashes = {pair_id: f"prompt-{pair_id}" for pair_id in ids}
    rows = [dict(step=1, pair_index=0, noise_seed=noise_seed(0), pair_id=ids[0], prompt_sha256=hashes[ids[0]])]
    rows[0][field] = wrong
    with pytest.raises(ValueError, match="trajectory"):
        validate_trajectory({"step": 1, "sample_order": [0]}, rows, order,
                            pair_ids=ids, prompt_hashes=hashes)


@pytest.mark.parametrize("start,stop", [(128, 0), (1025, 1025), (-1, 0)])
def test_stop_cannot_precede_resume_or_exceed_budget(start, stop):
    from src.pose_control.qwen21_full.prompt_study import sampling
    with pytest.raises(ValueError, match="start_step"):
        sampling.validate_update_bounds(start, stop)


def test_training_checks_locked_initialization_before_gpu(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.prompt_study import protocol
    args = SimpleNamespace(study_lock=tmp_path / 'lock.json', train_pairs=tmp_path / 'train',
        val_pairs=tmp_path / 'val', init_checkpoint=tmp_path / 'wrong.pt', resume=None,
        arm='qwen_p0', model=tmp_path / 'model', lora=None)
    args.init_checkpoint.write_bytes(b'wrong initialization')
    lock = {'epoch_order': make_order(), 'splits': {'train': {'path': 'train'}, 'val': {'path': 'val'}},
            'initial_checkpoint_sha256': protocol.INIT_SHA256}
    monkeypatch.setattr(protocol, 'load_study_lock', lambda *a, **kw: lock, raising=False)
    monkeypatch.setattr(protocol, 'validate_study_binding', lambda *a, **kw: None, raising=False)
    with pytest.raises(ValueError, match='initialization'):
        train_pilot.validate_study_training_inputs(args)


def test_training_cache_preflight_checks_targets_without_pipeline(tmp_path):
    from types import SimpleNamespace
    import torch
    from src.pose_control.qwen21_full.data_runtime import validate_study_cache
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache, dataset_image_to_pil
    sample = {'src_image': torch.zeros(3, 64, 64), 'tgt_image': torch.ones(3, 64, 64),
              'pair_metadata': {'pair_id': 'one'}}
    cache = Qwen21IPERCache(tmp_path, resolution='native', preprocessing='native_v1')
    source = dataset_image_to_pil(sample['src_image'])
    torch.save({'reference_latent': torch.zeros(1, 16, 64), 'prompt_embeds': torch.zeros(1, 1, 4096),
                'prompt_mask': torch.ones(1, 1, dtype=torch.bool), 'image_mask': torch.ones(1, 1, dtype=torch.bool),
                'token_hw': (4, 4)}, cache._path(source, 'source', prompt='one'))
    bank = SimpleNamespace(prompt_for=lambda pid: 'one')
    with pytest.raises(FileNotFoundError, match='target'):
        validate_study_cache(cache, [sample], bank, indices=[0], require_target=True)
    validate_study_cache(cache, [sample], bank, indices=[0], require_target=False)


def test_study_rejects_incomplete_native_supervision(tmp_path):
    from types import SimpleNamespace
    from src.pose_control.qwen21_full.data_runtime import validate_study_supervision
    store = SimpleNamespace(data={'align_mode': 'native'},
                            record=lambda row: {'by_resolution': {'512': {'valid': True}}})
    with pytest.raises(ValueError, match='native'):
        validate_study_supervision(store, [{'pair_id': 'one'}])


def test_runtime_signature_covers_imported_face_and_champ_sources():
    import importlib.util
    from src.pose_control.qwen21_full import data_runtime
    from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
    signature = data_runtime.study_code_signature()['runtime_code_sha256']
    for module in (
        'third_party.v65_face', 'third_party.v65_face.visual_persona',
        'third_party.v65_face.stableanimator', 'third_party.v65_face.mcld',
        'third_party.v65_face.xdyna', 'third_party.champ_guidance',
        'third_party.champ_guidance.guidance_encoder',
    ):
        source = Path(importlib.util.find_spec(module).origin)
        key = source.relative_to(data_runtime.ROOT).as_posix()
        assert signature[key] == sha256_file(source)


def test_runtime_signature_covers_nested_sources_and_excludes_cache_and_binaries(monkeypatch, tmp_path):
    from src.pose_control.qwen21_full import data_runtime
    from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
    required = (
        'src/adapter/pose_adapter.py', 'src/data/letterbox.py', 'src/data/tiktok_dataset.py',
        'scripts/precompute_iper_face_features_v65.py')
    nested = ('third_party/v65_face/nested/forward.py', 'third_party/champ_guidance/nested/control.py')
    ignored = ('third_party/v65_face/__pycache__/forward.py', 'third_party/champ_guidance/control.pyc',
               'third_party/v65_face/model.bin')
    for relative in (*required, *nested, *ignored):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture source')
    monkeypatch.setattr(data_runtime, 'ROOT', tmp_path)
    signature = data_runtime.study_code_signature()['runtime_code_sha256']
    for relative in nested:
        assert signature[relative] == sha256_file(tmp_path / relative)
    assert not set(ignored).intersection(signature)
