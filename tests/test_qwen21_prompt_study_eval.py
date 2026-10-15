from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image


@pytest.fixture
def evaluation():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prompt_study_eval", root / "scripts/eval/eval_qwen21_joint_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_generation_keeps_A_and_original_target_size(evaluation):
    a = Image.new("RGB", (608, 1088))
    args = SimpleNamespace(
        resolution="native",
        align_mode="native",
        steps=30,
        use_kv_cache=True,
    )
    kw = evaluation.generation_arguments(
        a,
        args,
        prompt="Raise both arms",
        generator=None,
        target_size=(736, 1024),
    )
    assert kw["image"] == [a]
    assert kw["prompt"] == "Raise both arms"
    assert (kw["width"], kw["height"]) == (736, 1024)
    assert not {"latents", "target", "gt_image"} & kw.keys()


def test_eval_arguments_study_mode(evaluation, monkeypatch):
    import sys

    argv = [
        "eval.py",
        "--checkpoint", "checkpoint.pt",
        "--pairs", "pairs.jsonl",
        "--features", "features",
        "--cache", "cache",
        "--output", "output",
        "--resolution", "native",
        "--align-mode", "native",
        "--lora", "head.safetensors",
        "--study-lock", "lock.json",
        "--arm", "head_pb",
        "--prompt-manifest", "prompts.jsonl",
        "--study-step", "128",
        "--study-split", "val",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = evaluation.arguments()
    assert args.study_lock == Path("lock.json")
    assert args.arm == "head_pb"
    assert args.study_step == 128
    assert args.study_split == "val"
    assert args.label == "head_pb_step_128"


def test_crop_normalized_native_resolutions(evaluation):
    import numpy as np

    # Test 604x1080
    img1 = Image.new("RGB", (604, 1080), color=(100, 150, 200))
    box = [0.2, 0.1, 0.4, 0.3]
    cropped1, pixels1 = evaluation.crop_normalized(img1, box, side=128)
    assert cropped1 is not None
    assert cropped1.size == (128, 128)
    assert pixels1[0] == int(np.floor(0.2 * 604))
    assert pixels1[1] == int(np.floor(0.1 * 1080))
    assert pixels1[2] == int(np.ceil(0.4 * 604))
    assert pixels1[3] == int(np.ceil(0.3 * 1080))

    # Test 736x1024
    img2 = Image.new("RGB", (736, 1024), color=(50, 80, 110))
    cropped2, pixels2 = evaluation.crop_normalized(img2, box, side=128)
    assert cropped2 is not None
    assert pixels2[0] == int(np.floor(0.2 * 736))
    assert pixels2[1] == int(np.floor(0.1 * 1024))
    assert pixels2[2] == int(np.ceil(0.4 * 736))
    assert pixels2[3] == int(np.ceil(0.3 * 1024))


def test_compute_bg_l1_background_preservation(evaluation):
    src = Image.new("RGB", (100, 100), color=(200, 200, 200))
    # Identical output: bg L1 is 0.0
    mask_a = Image.new("L", (100, 100), color=0)  # no foreground
    l1, reason = evaluation.compute_bg_l1(src, src, mask_a=mask_a, mask_b=mask_a)
    assert reason is None
    assert l1 == pytest.approx(0.0)

    # Different background
    gen = Image.new("RGB", (100, 100), color=(100, 100, 100))
    l1_diff, reason = evaluation.compute_bg_l1(src, gen, mask_a=mask_a, mask_b=mask_a)
    assert reason is None
    assert l1_diff == pytest.approx(100.0 / 255.0, abs=1e-3)


@pytest.mark.parametrize("extra", [
    ["--study-step", "129"], ["--resolution", "512"], ["--steps", "29"],
    ["--with-hand"], ["--face-strength", "0"], ["--geometry-strength", "0"],
    ["--stage", "hotload"], ["--variant", "BFSBody"], ["--seed", "44"],
    ["--no-use-kv-cache"], ["--progress-mode", "sigma"],
])
def test_study_eval_rejects_protocol_drift(evaluation, extra):
    argv = ["--checkpoint", "ckpt", "--pairs", "pairs", "--features", "features",
            "--cache", "cache", "--output", "out", "--study-lock", "lock",
            "--arm", "qwen_p0", "--prompt-manifest", "prompts", "--study-step", "128",
            "--study-split", "val", "--resolution", "native", "--align-mode", "native"]
    with pytest.raises(SystemExit):
        evaluation.arguments(argv + extra)


def test_final_test_allows_the_second_locked_seed(evaluation):
    args = evaluation.arguments(['--checkpoint', 'ckpt', '--pairs', 'pairs', '--features', 'features',
        '--cache', 'cache', '--output', 'out', '--study-lock', 'lock', '--arm', 'qwen_p0',
        '--prompt-manifest', 'prompts', '--study-step', '1024', '--study-split', 'test',
        '--resolution', 'native', '--align-mode', 'native', '--seed', '43'])
    assert args.seed == 43


@pytest.mark.parametrize('split,step,seed,index', [('diagnostic', 128, 42, 0), ('test', 512, 42, 0),
                                                 ('train', 128, 42, 1), ('val', 128, 43, 0)])
def test_direct_eval_cannot_expand_the_locked_schedule(evaluation, split, step, seed, index):
    args = SimpleNamespace(arm='qwen_p0', study_split=split, study_step=step, seed=seed)
    with pytest.raises(ValueError, match='schedule'):
        evaluation.validate_study_schedule(args, [index])


def test_direct_final_test_requires_eight_arm_completion(evaluation, monkeypatch, tmp_path):
    from src.pose_control.qwen21_full.prompt_study import protocol
    args = SimpleNamespace(study_lock=tmp_path / 'data/lock.json', study_split='test',
        pairs=tmp_path / 'data/test24_pairs.jsonl', arm='qwen_p0',
        prompt_manifest=tmp_path / 'prompts/p0/test.jsonl', cache=tmp_path / 'cache/p0',
        model=tmp_path / 'model', lora=None)
    def blocked(run):
        raise RuntimeError('Final test blocked')
    monkeypatch.setattr(protocol, 'require_eight_arms_completed', blocked)
    monkeypatch.setattr(protocol, 'load_study_lock', lambda *a, **kw: {'splits': {'test': {'path': 'test24_pairs.jsonl'}}})
    monkeypatch.setattr(protocol, 'validate_study_binding', lambda *a: None)
    monkeypatch.setattr(evaluation, 'PromptBank', lambda *a: None)
    with pytest.raises(RuntimeError, match='Final test blocked'):
        evaluation.validate_study_inputs(args, SimpleNamespace(rows=[]))


def test_artifacts_preserve_both_target_hand_rois(evaluation, tmp_path):
    image = Image.new("RGB", (604, 1080), "blue")
    image.paste("red", (0, 0, 302, 1080))
    files = evaluation.save_case_artifacts(tmp_path, "hands", source=image, target=image,
        generated=image, normal=None, pose=None, source_box=None, target_box=None,
        target_lh_box=[0, 0, .4, 1], target_rh_box=[.6, 0, 1, 1])
    assert Image.open(tmp_path / files["hand_left_output"]).getpixel((128, 128)) == (255, 0, 0)
    assert Image.open(tmp_path / files["hand_right_output"]).getpixel((128, 128)) == (0, 0, 255)


@pytest.mark.parametrize('failed', [False, True])
def test_artifact_hashes_bind_saved_references_and_masks(evaluation, tmp_path, failed):
    image = Image.new('RGB', (60, 60), 'blue')
    mask = Image.new('L', image.size, 17)
    files = evaluation.save_case_artifacts(tmp_path, 'case', source=image, target=image,
        generated=None if failed else image, normal=image, pose=image,
        source_box=[.1, .1, .5, .5], target_box=[.1, .1, .5, .5],
        target_lh_box=[.1, .1, .2, .2], target_rh_box=[.5, .5, .7, .7],
        source_mask=mask, target_mask=mask, failure='generation failed' if failed else None)
    expected = {key: evaluation.digest_file(tmp_path / path) for key, path in files.items()
                if isinstance(path, str) and not key.endswith('_sha256')}
    assert files['artifact_sha256'] == expected
    for key in ('source', 'target', 'source_mask', 'target_mask'):
        assert files[f'{key}_sha256'] == expected[key]
    Image.new('L', image.size, 18).save(tmp_path / files['source_mask'])
    assert evaluation.digest_file(tmp_path / files['source_mask']) != files['artifact_sha256']['source_mask']


@pytest.mark.parametrize('size', [(604, 1080), (736, 1024)])
def test_failed_native_artifacts_use_original_roi_coordinates(evaluation, size):
    from src.data.letterbox import NativePad
    pad = NativePad.from_size(size)
    image = Image.new('RGB', size, 'blue')
    image.paste('red', (0, 0, size[0] // 2, size[1]))
    metadata = {'alignment': {'source': pad.metadata(), 'target': pad.metadata()}}
    source, target, generated, normal, pose, mask = evaluation.unpad_artifact_images(metadata,
        source=pad.image(image), target=pad.image(image), generated=None,
        normal=None, pose=None, target_mask=None)
    assert source.size == target.size == size
    assert generated is None
    crop, _ = evaluation.crop_normalized(target, [.1, .2, .4, .8])
    assert crop.getpixel((128, 128)) == (255, 0, 0)


@pytest.mark.parametrize('size', [(604, 1080), (736, 1024)])
def test_source_mask_uses_original_A_dimensions(evaluation, tmp_path, size):
    from src.data.letterbox import NativePad
    pad = NativePad.from_size(size)
    mask_file = tmp_path / 'source_mask.png'
    mask = Image.new('L', size, 0)
    mask.paste(255, (12, 20, 30, 40))
    mask.save(mask_file)
    loaded = evaluation.load_source_person_mask(
        {'source_person_mask': str(mask_file)}, {'alignment': {'source': pad.metadata()}},
        pad.image(Image.new('RGB', size)))
    assert loaded.size == size
    assert loaded.tobytes() == mask.tobytes()
    Image.new('L', (size[0] + 1, size[1]), 0).save(mask_file)
    with pytest.raises(ValueError, match='Source A mask dimensions'):
        evaluation.load_source_person_mask(
            {'source_person_mask': str(mask_file)}, {'alignment': {'source': pad.metadata()}},
            pad.image(Image.new('RGB', size)))


def test_conditioning_exports_geometry_and_raw_inputs(evaluation, tmp_path):
    import json
    raw = tmp_path / 'raw.json'
    raw.write_text('{"scores": []}')
    depth = tmp_path / 'depth.png'
    Image.new('RGB', (64, 64), 'gray').save(depth)
    paths = evaluation.save_conditioning_artifacts(tmp_path / 'eval', 'case',
        metadata={'target_dwpose_json': str(raw), 'target_depth_vis': str(depth)},
        normal=None, pose=None, body25=[[.5, .5, 1]])
    assert json.loads((tmp_path / 'eval' / paths['conditioning_body25']).read_text()) == [[.5, .5, 1]]
    assert (tmp_path / 'eval' / paths['conditioning_dwpose_json']).read_bytes() == raw.read_bytes()
    assert (tmp_path / 'eval' / paths['conditioning_depth']).read_bytes() == depth.read_bytes()


def test_cross_person_flag_prohibits_target_pixel_metric(evaluation, monkeypatch, tmp_path):
    import json
    import numpy as np
    import src.evaluation.joint_pilot_metrics as metrics
    detector = SimpleNamespace(detect=lambda image: ([], []))
    monkeypatch.setattr(metrics, "IdentityMetric", lambda root: SimpleNamespace(det=detector))
    monkeypatch.setattr(metrics, "PoseMetric", lambda root: SimpleNamespace())
    Image.new("RGB", (64, 64), "white").save(tmp_path / "source.png")
    Image.new("RGB", (64, 64), "black").save(tmp_path / "target.png")
    Image.new("L", (64, 64), 0).save(tmp_path / "mask.png")
    row = dict(pair_index=0, image_stem="case", status="complete", image="source.png",
               source="source.png", target="target.png", target_mask="mask.png", source_mask="mask.png",
               metadata={"cross_person": True}, resolution="native")
    (tmp_path / "per_sample.jsonl").write_text(json.dumps(row) + "\n")
    score = evaluation.score_saved_outputs(tmp_path)[0]
    assert score["pixel_l1_GT"] is None
    assert score["bg_l1_A"] == 0
    assert score['image_sha256'] == evaluation.digest_file(tmp_path / 'source.png')
    scores = __import__('json').loads((tmp_path / 'scores.json').read_text())
    assert scores['per_sample_sha256'] == evaluation.digest_file(tmp_path / 'per_sample.jsonl')


@pytest.mark.parametrize('study', [True, False])
@pytest.mark.parametrize('detector_missing', [True, False])
def test_pose_detection_failures_are_missing_study_metrics(evaluation, monkeypatch, tmp_path, study, detector_missing):
    import json
    import numpy as np
    import src.evaluation.joint_pilot_metrics as metrics
    compare = metrics.PoseMetric.compare
    detected = None if detector_missing else (np.zeros((17, 2)), np.zeros(17))
    monkeypatch.setattr(metrics, 'IdentityMetric', lambda root: SimpleNamespace(
        det=SimpleNamespace(detect=lambda image: ([], []))))
    monkeypatch.setattr(metrics, 'PoseMetric', lambda root: SimpleNamespace(
        detect=lambda image: detected, compare=compare))
    Image.new('RGB', (64, 64), 'blue').save(tmp_path / 'image.png')
    Image.new('L', (64, 64), 0).save(tmp_path / 'mask.png')
    row = dict(pair_index=0, image_stem='case', status='complete', image='image.png',
        source='image.png', target='image.png', source_mask='mask.png', target_mask='mask.png',
        target_body25=[[.5, .5, 1]] * 25, resolution='native')
    if study:
        row.update(arm='qwen_p0', study_split='val')
    (tmp_path / 'per_sample.jsonl').write_text(json.dumps(row) + '\n')
    score = evaluation.score_saved_outputs(tmp_path)[0]
    assert score['pose_matched_joints'] == 0
    assert score['pose_detected'] is (not detector_missing)
    assert score['pose_nme_diagonal'] is None
    for name in ('pose_pck_002', 'pose_pck_005', 'pose_joint_recall'):
        assert score[name] is None if study else score[name] == 0
    if study:
        assert score['pose_failure']
    else:
        assert 'pose_failure' not in score


def test_tiktok_metadata_preserves_geometry_and_video_groups():
    from src.pose_control.qwen21_full.data_runtime import pair_metadata
    row = dict(pair_id="cross", sequence_id="B", source_appearance="A", target_appearance="B",
               source_stem="1", target_stem="2", source_rgb="a.png", target_rgb="b.png",
               target_dwpose_json="b.json", source_video_group="group-a", target_video_group="group-b")
    meta = pair_metadata(row)
    assert meta["target_dwpose_json"] == "b.json"
    assert meta["source_video_group"] == "group-a"
    assert meta["target_video_group"] == "group-b"
    assert meta["sequence_id"] == "B"


@pytest.mark.parametrize('field,wrong', [
    ('arm', 'head_pb'), ('step', 256), ('study_schema', 'other'),
    ('study_lock_sha256', 'other'), ('joint_initialization_sha256', 'other'),
    ('model_revision', 'other'),
])
def test_study_checkpoint_rejects_mismatched_identity(evaluation, monkeypatch, tmp_path, field, wrong):
    from src.pose_control.qwen21_full.prompt_study import protocol
    from src.pose_control.qwen21_full.qwen_runtime import MODEL_REVISION
    args = SimpleNamespace(arm='qwen_p0', study_step=128, study_lock=tmp_path / 'lock.json',
        model=tmp_path / 'model', features=tmp_path / 'data/face_cache', lora=None)
    args.study_lock.write_text('{}')
    signature = dict(schema=protocol.SCHEMA, study_schema=protocol.SCHEMA, arm='qwen_p0', protocol='p0',
        max_updates=1024, initial_history_steps=256, model_revision=MODEL_REVISION,
        model=str(args.model.resolve()), model_manifest_sha256='model-hash', lora_sha256=None,
        strength=1, study_lock_sha256=evaluation.digest_file(args.study_lock),
        instruction_sha256=protocol.text_sha256(protocol.P0_PROMPT), face_assets_sha256='face-hash',
        resolution='native', align_mode='native', preprocessing='native_v1', with_hand=False)
    checkpoint = dict(arm='qwen_p0', step=128, study_schema=protocol.SCHEMA,
        study_lock_sha256=signature['study_lock_sha256'], joint_initialization_sha256=protocol.INIT_SHA256,
        model_revision=MODEL_REVISION, joint_signature=signature, next_update_index=128,
        sample_order=protocol.make_order()[:128] if hasattr(protocol, 'make_order') else list(range(128)))
    checkpoint[field] = wrong
    lock = {'initial_checkpoint_sha256': protocol.INIT_SHA256,
            'artifacts': {'model': {'manifest_sha256': 'model-hash'}}, 'epoch_order': list(range(128))}
    bank = SimpleNamespace(records={'one': {'instruction_sha256': '', 'instruction_version': 'none'}}, protocol='p0')
    monkeypatch.setattr(protocol, 'validate_study_binding', lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(evaluation, 'study_asset_signature', lambda *a, **kw: {'face_assets_sha256': 'face-hash'}, raising=False)
    with pytest.raises(ValueError):
        evaluation.validate_study_checkpoint(args, lock, checkpoint, bank)


@pytest.fixture
def signed_checkpoint(evaluation, monkeypatch, tmp_path):
    from src.pose_control.qwen21_full.prompt_study import protocol
    from src.pose_control.qwen21_full.prompt_study.sampling import make_order
    from src.pose_control.qwen21_full.qwen_runtime import MODEL_REVISION
    run = tmp_path / 'run'
    (run / 'data').mkdir(parents=True)
    (run / 'prompts/p0').mkdir(parents=True)
    (run / 'data/lock.json').write_text('{}')
    (run / 'prompts/p0/train.jsonl').write_text('train-prompt-records')
    (run / 'prompts/instruction_v1.json').write_text('{}')
    args = SimpleNamespace(arm='qwen_p0', study_step=128, study_lock=run / 'data/lock.json',
        model=tmp_path / 'model', features=run / 'data/face_cache', lora=None)
    lock = {'initial_checkpoint_sha256': protocol.INIT_SHA256,
            'artifacts': {'model': {'manifest_sha256': 'model-hash'}}, 'epoch_order': make_order()}
    bank = SimpleNamespace(protocol='p0', records={'one': {'instruction_sha256': '', 'instruction_version': 'none'}})
    signature = dict(schema=protocol.SCHEMA, study_schema=protocol.SCHEMA, arm='qwen_p0', protocol='p0',
        max_updates=1024, initial_history_steps=256, model_revision=MODEL_REVISION,
        model=str(args.model.resolve()), model_manifest_sha256='model-hash', lora_sha256=None, strength=1,
        study_lock_sha256=evaluation.digest_file(args.study_lock), resolution='native', align_mode='native',
        preprocessing='native_v1', with_hand=False, face_assets_sha256='face-hash',
        instruction_sha256=evaluation.digest_file(run / 'prompts/instruction_v1.json'), instruction_version='none',
        prompt_instruction_sha256=protocol.text_sha256(protocol.P0_PROMPT),
        prompt_manifest_sha256=evaluation.digest_file(run / 'prompts/p0/train.jsonl'))
    signature.update(evaluation.study_code_signature())
    checkpoint = dict(arm=args.arm, step=128, study_schema=protocol.SCHEMA,
        study_lock_sha256=signature['study_lock_sha256'], joint_initialization_sha256=protocol.INIT_SHA256,
        model_revision=MODEL_REVISION, joint_signature=signature, next_update_index=128,
        sample_order=make_order()[:128], protocol='p0', resolution='native', align_mode='native',
        max_updates=1024, initial_history_steps=256, branch='joint')
    monkeypatch.setattr(protocol, 'validate_study_binding', lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, 'study_asset_signature', lambda *a, **kw: {'face_assets_sha256': 'face-hash'})
    return args, lock, checkpoint, bank


def test_study_accepts_valid_checkpoint_without_matching_val_and_train_manifest(evaluation, signed_checkpoint):
    args, lock, checkpoint, bank = signed_checkpoint
    args.prompt_manifest = args.study_lock.parent.parent / 'prompts/p0/val.jsonl'
    args.prompt_manifest.write_text('different-val-records')
    evaluation.validate_study_checkpoint(args, lock, checkpoint, bank)


@pytest.mark.parametrize('field,wrong', [
    ('arm', 'head_p0'), ('protocol', 'pb'), ('model_manifest_sha256', 'other'),
    ('lora_sha256', 'other'), ('instruction_sha256', 'other'), ('face_assets_sha256', 'other'),
    ('prompt_manifest_sha256', 'other'), ('runtime_code_sha256', {}), ('with_hand', True),
])
def test_study_rejects_changed_training_signature(evaluation, signed_checkpoint, field, wrong):
    args, lock, checkpoint, bank = signed_checkpoint
    checkpoint['joint_signature'][field] = wrong
    with pytest.raises(ValueError, match='signature|manifest'):
        evaluation.validate_study_checkpoint(args, lock, checkpoint, bank)


@pytest.mark.parametrize('field,wrong', [('protocol', 'pb'), ('resolution', 512), ('max_updates', 256),
                                      ('initial_history_steps', 0), ('branch', 'face')])
def test_study_rejects_changed_checkpoint_header(evaluation, signed_checkpoint, field, wrong):
    args, lock, checkpoint, bank = signed_checkpoint
    checkpoint[field] = wrong
    with pytest.raises(ValueError, match='checkpoint'):
        evaluation.validate_study_checkpoint(args, lock, checkpoint, bank)


def test_study_reuse_rejects_changed_png(evaluation, tmp_path):
    image = tmp_path / 'out.png'
    Image.new('RGB', (64, 64), 'white').save(image)
    meta = dict(pair_id='one', sequence_id='a', source_appearance='a', target_appearance='a',
                source_stem='1', target_stem='2', source_rgb='a.png', target_rgb='b.png')
    row = dict(pair_index=0, pair_id='one', arm='qwen_p0', protocol='p0', study_step=128,
               study_split='val', seed=42, checkpoint_sha256='checkpoint', lora_sha256=None,
               prompt_sha256='prompt', image='out.png', image_size=[64, 64],
               image_sha256='changed', metadata=evaluation.pair_metadata(meta), status='complete')
    args = SimpleNamespace(arm='qwen_p0', study_step=128, study_split='val', seed=42, output=tmp_path)
    config = dict(checkpoint_sha256='checkpoint', lora_sha256=None)
    bank = SimpleNamespace(protocol='p0', records={'one': {'compiled_prompt_sha256': 'prompt'}})
    with pytest.raises(ValueError, match='image'):
        evaluation.validate_study_records(args, config, SimpleNamespace(rows=[meta]), bank, [row], [0])


@pytest.mark.parametrize('status', ['complete', 'failed'])
@pytest.mark.parametrize('field', ['source', 'target', 'source_mask', 'target_mask'])
def test_direct_study_reuse_rejects_changed_scoring_inputs(evaluation, tmp_path, status, field):
    image = Image.new('RGB', (64, 64), 'blue')
    mask = Image.new('L', image.size, 17)
    files = evaluation.save_case_artifacts(tmp_path, 'case', source=image, target=image,
        generated=image if status == 'complete' else None, normal=None, pose=None,
        source_box=None, target_box=None, source_mask=mask, target_mask=mask)
    meta = dict(pair_id='one', sequence_id='a', source_appearance='a', target_appearance='a',
                source_stem='1', target_stem='2', source_rgb='a.png', target_rgb='b.png')
    row = dict(files, pair_index=0, pair_id='one', arm='qwen_p0', protocol='p0', study_step=128,
               study_split='val', seed=42, checkpoint_sha256='checkpoint', lora_sha256=None,
               prompt_sha256='prompt', metadata=evaluation.pair_metadata(meta), status=status)
    if status == 'complete':
        row.update(image_size=[64, 64], image_sha256=evaluation.digest_file(tmp_path / row['image']))
    args = SimpleNamespace(arm='qwen_p0', study_step=128, study_split='val', seed=42, output=tmp_path)
    config = dict(checkpoint_sha256='checkpoint', lora_sha256=None)
    bank = SimpleNamespace(protocol='p0', records={'one': {'compiled_prompt_sha256': 'prompt'}})
    dataset = SimpleNamespace(rows=[meta])
    evaluation.validate_study_records(args, config, dataset, bank, [row], [0])
    Image.new('L' if 'mask' in field else 'RGB', image.size, 18).save(tmp_path / row[field])
    with pytest.raises(ValueError, match='artifact'):
        evaluation.validate_study_records(args, config, dataset, bank, [row], [0])


@pytest.mark.parametrize('generation_fails', [False, True])
def test_study_main_generates_only_from_A_without_training_or_target_encoding(evaluation, monkeypatch, tmp_path, generation_fails):
    import contextlib
    import json
    import torch
    import diffusers
    import src.pose_control.qwen21_full.execution as execution
    import src.pose_control.qwen21_full.frozen_bfs_lora as bfs
    import src.pose_control.qwen21_full.qwen_adapter as adapter_module
    import scripts.precompute_qwen21_prompt_study_cache as cache_module
    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache
    from src.pose_control.qwen21_full.joint_training import run_joint_loss
    from src.pose_control.qwen21_full.prompt_study import protocol
    from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank, make_p0_record
    from src.pose_control.qwen21_full.prompt_study.results import validate_evaluation_artifacts
    from src.data.letterbox import NativePad
    def forbidden(*a, **kw):
        raise AssertionError('Evaluation attempted target encoding or training')
    for name in ('target', 'pair'):
        monkeypatch.setattr(Qwen21IPERCache, name, forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, 'step', forbidden)
    monkeypatch.setattr('src.pose_control.qwen21_full.joint_training.run_joint_loss', forbidden)
    mask_file = tmp_path / 'source_mask.png'
    Image.new('L', (60, 60), 17).save(mask_file)
    meta = dict(pair_id='one', sequence_id='a', source_appearance='a', target_appearance='a',
                source_stem='1', target_stem='2', source_rgb='a.png', target_rgb='b.png',
                source_person_mask=str(mask_file))
    sample = dict(src_image=torch.zeros(3, 64, 64), tgt_image=torch.ones(3, 64, 64),
                  normal=torch.zeros(3, 64, 64), human_mask=torch.ones(1, 64, 64),
                  pose_heatmap=torch.zeros(25, 64, 64), pair_metadata=evaluation.pair_metadata(meta))
    pad = NativePad.from_size((60, 60))
    sample['alignment_meta'] = json.dumps({'source': pad.metadata(), 'target': pad.metadata()})
    class Dataset:
        rows = [meta]
        dataset_type = 'tiktok_v3'
        def __init__(self, *a, **kw): pass
        def __len__(self): return 1
        def __getitem__(self, index): return sample
    class Cache:
        def __init__(self, *a, **kw): pass
        def source(self, pipe, image, *, prompt):
            pipe.encode_prompt(prompt, image=[image])
            return {'reference_latent': torch.zeros(1, 16, 64), 'token_hw': (4, 4)}
        target = pair = forbidden
    class Store:
        def __init__(self, *a, **kw): pass
        def _load(self, appearance):
            return {'stem_to_idx': {'1': 0, '2': 0}, 'valid': torch.ones(1, dtype=torch.bool),
                    'geometry_valid': torch.ones(1, dtype=torch.bool), 'face_boxes': torch.tensor([[.1, .1, .5, .5]])}
    calls = []
    class Pipe:
        def __init__(self):
            self.transformer = torch.nn.Linear(1, 1)
            self.text_encoder = torch.nn.Linear(1, 1)
            self.vae = torch.nn.Linear(1, 1)
        def to(self, device): return self
        def set_progress_bar_config(self, **kw): pass
        def encode_prompt(self, prompt, *, image): calls.append(('encode', prompt, image))
        def __call__(self, **kw):
            calls.append(('generate', kw))
            assert not {'latents', 'target', 'gt_image', 'mask', 'source_mask', 'target_mask'} & kw.keys()
            assert kw['num_inference_steps'] == 30
            if generation_fails:
                raise RuntimeError('Simulated generation failure')
            return SimpleNamespace(images=[kw['image'][0].copy()])
    class Context:
        def __init__(self, *a, **kw):
            self.forward_count = 30
            self.call_counts = {'controller': 30}
            self.output = SimpleNamespace(diagnostics={})
            self.step_diagnostics = []
        def __enter__(self): return self
        def __exit__(self, *a): pass
    pipe = Pipe()
    monkeypatch.setattr(diffusers, 'QwenImage21Pipeline', SimpleNamespace(from_pretrained=lambda *a, **kw: pipe))
    monkeypatch.setattr(execution, 'QwenFullDataset', Dataset)
    monkeypatch.setattr(execution, 'Qwen21IPERCache', Cache)
    monkeypatch.setattr(execution, 'FaceConditionStore', Store)
    monkeypatch.setattr(execution, 'new_adapter', lambda device: torch.nn.Linear(1, 1))
    header = {'step': 0, 'study_schema': protocol.SCHEMA, 'arm': 'qwen_p0', 'protocol': 'p0',
              'max_updates': 1024, 'joint_initialization_sha256': protocol.INIT_SHA256,
              'joint_signature': {'arm': 'qwen_p0', 'max_updates': 1024, 'lora_sha256': None}}
    monkeypatch.setattr(execution, 'load_adapter_checkpoint', lambda *a: header)
    monkeypatch.setattr(execution, 'seed_all', lambda seed: None)
    monkeypatch.setattr(execution, 'prepare', lambda *a, **kw: SimpleNamespace(
        face=SimpleNamespace(face_valid=torch.tensor([[True, False]])), detail=None, hand=None))
    monkeypatch.setattr(bfs, 'load_frozen_bfs', lambda *a, **kw: {'file_sha256': None})
    monkeypatch.setattr(adapter_module, 'Qwen21FullControlContext', Context)
    monkeypatch.setattr(cache_module, 'validate_cache_manifest', lambda *a, **kw: {})
    monkeypatch.setattr(evaluation, 'validate_study_checkpoint', lambda *a: {})
    monkeypatch.setattr(evaluation, 'validate_study_cache', lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, 'target_keypoints', lambda *a: None)
    original_to = torch.Tensor.to
    def cpu_to(tensor, *a, **kw):
        if a and a[0] == 'cuda': a = ('cpu',) + a[1:]
        if kw.get('device') == 'cuda': kw['device'] = 'cpu'
        return original_to(tensor, *a, **kw)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    cpu_generator = torch.Generator
    monkeypatch.setattr(torch, 'Generator', lambda *a, **kw: cpu_generator(device='cpu'))
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext())
    for name in ('synchronize', 'reset_peak_memory_stats', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    run = tmp_path / 'run'
    (run / 'data').mkdir(parents=True)
    (run / 'prompts/p0').mkdir(parents=True)
    pairs, lock = run / 'data/val24_pairs.jsonl', run / 'data/lock.json'
    checkpoint, model = tmp_path / 'checkpoint.pt', tmp_path / 'model'
    prompt_file = run / 'prompts/p0/val.jsonl'
    prompt_file.write_text(json.dumps(make_p0_record('one', 'val')) + '\n')
    train_prompts = run / 'prompts/p0/train.jsonl'
    train_prompts.write_text(json.dumps(make_p0_record('train', 'train')) + '\n')
    instruction = run / 'prompts/instruction_v1.json'
    instruction.write_text('{}')
    bank = PromptBank(prompt_file, 'p0', ['one'])
    monkeypatch.setattr(evaluation, 'validate_study_inputs', lambda *a: ({}, bank))
    pairs.write_text(json.dumps(meta) + '\n')
    lock.write_text('{}')
    header['joint_signature'].update(model=str(model.resolve()), study_lock_sha256=evaluation.digest_file(lock),
        prompt_manifest_sha256=evaluation.digest_file(train_prompts), instruction_sha256=evaluation.digest_file(instruction))
    # Only fixed production data-count/hash validation is injected; the real
    # producer and consumer still validate the complete retained file contract.
    monkeypatch.setattr(protocol, 'load_study_lock', lambda *a, **kw: {
        'artifacts': {'model': {'path': str(model.resolve())}, 'head_lora': {'sha256': 'head'}}})
    torch.save(header, checkpoint)
    (model / 'transformer').mkdir(parents=True)
    (model / 'transformer/config.json').write_text('{}')
    output = tmp_path / 'output'
    evaluation.main(['--study-lock', str(lock), '--arm', 'qwen_p0', '--study-step', '0', '--study-split', 'val',
        '--checkpoint', str(checkpoint), '--pairs', str(pairs), '--features', 'features', '--cache', 'cache',
        '--prompt-manifest', str(prompt_file), '--output', str(output), '--model', str(model), '--resolution', 'native', '--align-mode', 'native'])
    row = json.loads((output / 'per_sample.jsonl').read_text())
    assert row['status'] == ('failed' if generation_fails else 'complete'), row.get('failure')
    assert row['initialization'] == 'fresh_gaussian'
    assert [kind for kind, *rest in calls] == ['encode', 'generate']
    source = calls[0][2][0]
    assert source.getpixel((0, 0)) == (127, 127, 127)
    assert calls[1][1]['image'] == [source]
    assert calls[0][1] == calls[1][1]['prompt'] == protocol.P0_PROMPT
    if not generation_fails:
        assert row['control_forwards'] == 30
    else:
        assert row['failure_phase'] == 'generation'
    with Image.open(output / row['source_mask']) as mask:
        assert mask.size == (60, 60)
        assert mask.getpixel((0, 0)) == 17
    with Image.open(output / row['source']) as saved_source:
        assert saved_source.size == (60, 60)
    # Read the real producer's file through the validator's provenance contract.
    loading = json.loads((output / 'lora_loading.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    assert loading['file_sha256'] == config['lora_sha256']
    assert loading['checkpoint_sha256'] == config['checkpoint_sha256']
    metadata = json.loads((output / 'checkpoint_metadata.json').read_text())
    for key in ('joint_initialization_sha256', 'study_schema', 'arm', 'protocol', 'max_updates'):
        assert metadata[key] == header[key]
    for key in ('source', 'target', 'source_mask', 'target_mask'):
        assert row['artifact_sha256'][key] == evaluation.digest_file(output / row[key])
    retained = validate_evaluation_artifacts(output, run)
    assert retained['rows'] == [row]
    Image.new('L', (60, 60), 18).save(output / row['source_mask'])
    with pytest.raises(ValueError, match='artifact'):
        validate_evaluation_artifacts(output, run)

