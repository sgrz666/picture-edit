"""CPU contract tests using saved real RGB images; no pipeline simulation."""
import importlib.util
import json
import os
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    path = ROOT / relative
    assert path.is_file(), f"Missing implementation: {relative}"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def portrait():
    path = os.environ.get('QWEN_PILOT_REAL_IMAGE')
    if path and Path(path).is_file():
        image_path = Path(path)
    elif (ROOT / "inputs/person1.jpg").is_file():
        image_path = ROOT / "inputs/person1.jpg"
    else:
        return Image.new("RGB", (512, 512), color=(128, 128, 128))
    with Image.open(image_path) as image:
        return image.convert("RGB")


def test_index_selection_preserves_requested_order():
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    assert evaluation.select_indices("4,0,2-3", 6) == [4, 0, 2, 3]
    assert evaluation.select_indices(None, 3) == [0, 1, 2]


@pytest.mark.parametrize("selection", ["1,1", "6", "-1", "3-1", "", "2-7"])
def test_bad_selection_rejected_without_substitution(selection):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    with pytest.raises(ValueError):
        evaluation.select_indices(selection, 6)


def test_case_name_is_portable_and_distinguishes_metadata():
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    meta = {"pair_id": "unsafe/<pair>:α", "source_appearance": "clip/1", "source_stem": "frame:2", "target_stem": "frame3"}
    name = evaluation.case_stem(4, meta)
    assert name.startswith("004_")
    assert all(c.isalnum() or c in "_-" for c in name)
    assert len(name) < 160
    assert name == evaluation.case_stem(4, meta)
    assert name != evaluation.case_stem(4, {**meta, "pair_id": "unsafe/<pair>:β"})


def test_normalized_roi_uses_identical_target_coordinates(portrait):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    box = [.2, .1, .6, .7]
    crop, pixels = evaluation.crop_normalized(portrait, box, side=128)
    expected = portrait.crop(pixels).resize((128, 128), Image.Resampling.LANCZOS)
    assert crop.tobytes() == expected.tobytes()
    assert pixels == (int(portrait.width * .2), int(portrait.height * .1),
                      __import__("math").ceil(portrait.width * .6), __import__("math").ceil(portrait.height * .7))
    assert evaluation.crop_normalized(portrait, None) == (None, None)


@pytest.mark.parametrize("box", [[0, 0, 0, 0], [-.1, 0, .5, 1], [0, 0, float("nan"), 1]])
def test_invalid_face_roi_does_not_turn_into_full_image(portrait, box):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    assert evaluation.crop_normalized(portrait, box) == (None, None)


def test_artifacts_export_raw_images_and_explicit_failed_case(portrait, tmp_path):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    files = evaluation.save_case_artifacts(tmp_path, "004_case", source=portrait, target=portrait,
        generated=None, normal=portrait, pose=portrait, source_box=None, target_box=[.1, .1, .5, .6],
        failure="generation failed")
    assert files["image"] is None
    assert files["source"] == "images/004_case_source.png"
    assert files["target"] == "images/004_case_target.png"
    assert files["face_source"] is None
    assert (tmp_path / files["face_target"]).is_file()
    assert (tmp_path / files["comparison"]).is_file()
    assert (tmp_path / files["face_comparison"]).is_file()


def test_success_raw_output_works_with_saved_image_scorer(portrait, tmp_path):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    files = evaluation.save_case_artifacts(tmp_path, "004_case", source=portrait, target=portrait,
        generated=portrait, normal=None, pose=None, source_box=[.1, .1, .5, .6], target_box=[.1, .1, .5, .6])
    assert files["image"] == "images/004_case_on.png"
    assert Image.open(tmp_path / files["image"]).size == portrait.size
    assert all(not Path(value).is_absolute() for value in files.values() if isinstance(value, str))


def test_cli_body_face_ablation_switches_keep_other_branches_off():
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    args = evaluation.arguments(['--checkpoint', 'checkpoint.pt', '--pairs', 'pairs.jsonl', '--dataset-type', 'tiktok_v3',
        '--cache', 'cache', '--features', 'face_cache', '--output', 'outputs', '--label', 'common_256', '--face-strength', '0'])
    strengths = evaluation.controller_strengths(args)
    assert strengths['geometry_strength'] == 1
    assert strengths['face_strength'] == 0
    assert strengths['interaction_strength'] == 0
    assert strengths['hand_strength'] == 0
    assert strengths['hand_mode'] == 'off'
    assert strengths['detail_strength'] == 1  # Face's legacy gate; prepared shared Detail remains absent.
    assert (args.resolution, args.steps, args.seed) == (512, 30, 42)
    assert evaluation.infer_variant_stage('common_256', None) == ('J', 'common')


def test_generation_arguments_have_only_A_rgb(portrait):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    args = evaluation.arguments(['--checkpoint', 'checkpoint.pt', '--pairs', 'pairs.jsonl', '--dataset-type', 'tiktok_v3',
        '--cache', 'cache', '--features', 'face_cache', '--output', 'outputs', '--label', 'bfs_body_mid128'])
    kwargs = evaluation.generation_arguments(portrait, args, prompt='fixed', generator=None)
    assert kwargs['image'] == [portrait]
    assert kwargs['true_cfg_scale'] == 1
    assert 'latents' not in kwargs
    assert 'target' not in kwargs


def test_saved_image_scoring_requires_no_generation_inputs():
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    args = evaluation.arguments(['--score-only', '--output', 'copied-evaluation'])
    assert args.score_only
    assert args.output == Path('copied-evaluation')


def test_tracked_metric_has_exact_trusted_coco_to_body25_mapping():
    metric = module('joint_metrics', 'src/evaluation/joint_pilot_metrics.py')
    assert metric.COCO_TO_BODY25 == (0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)
    import numpy as np
    target = np.zeros((25, 3), dtype=np.float32)
    target[:, 0] = np.arange(25) / 25
    target[:, 1] = .5
    target[:, 2] = 1
    points = target[np.array(metric.COCO_TO_BODY25), :2] * 512
    result = metric.PoseMetric.compare((points, np.ones(17)), target)
    assert result['pose_pck_002'] == 1
    assert result['pose_matched_joints'] == 17
    assert result['pose_nme_diagonal'] == 0
    assert metric.ROOT == ROOT


@pytest.mark.parametrize('variant,lora', [('Qwen', 'hash'), ('J', 'hash'), ('init', 'hash'), ('BFSBody', None), ('BFSHead', None)])
def test_canonical_variant_rejects_wrong_lora_presence(variant, lora):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    with pytest.raises(ValueError, match='LoRA'):
        evaluation.validate_base_provenance(variant, 'hotload', lora, {})


@pytest.mark.parametrize('stage', ['mid', 'final'])
def test_adapted_bfs_requires_exact_checkpoint_prior(stage):
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    with pytest.raises(ValueError, match='joint_signature'):
        evaluation.validate_base_provenance('BFSBody', stage, 'body-sha', {})
    with pytest.raises(ValueError, match='LoRA'):
        evaluation.validate_base_provenance('BFSBody', stage, 'body-sha', {'joint_signature': {'lora_sha256': 'head-sha'}})
    provenance = evaluation.validate_base_provenance('BFSBody', stage, 'body-sha', {'joint_signature': {'lora_sha256': 'body-sha', 'strength': 1.}})
    assert provenance['base_matches_training']


def test_bfs_hotload_accepts_plain_J_checkpoint_and_records_override():
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    provenance = evaluation.validate_base_provenance('BFSHead', 'hotload', 'head-sha', {'joint_signature': {'lora_sha256': None}})
    assert provenance['hotload_override']
    assert provenance['checkpoint_lora_sha256'] is None
    assert provenance['inference_lora_sha256'] == 'head-sha'


def test_named_diagnostics_and_plain_final_remain_supported():
    evaluation = module("joint_pilot_eval", "scripts/eval/eval_qwen21_joint_pilot.py")
    assert evaluation.validate_base_provenance('JBodyOnly', 'common', None, {})['canonical_variant'] is None
    assert evaluation.validate_base_provenance('Qwen', 'final', None, {'joint_signature': {'lora_sha256': None}})['base_matches_training']


def test_report_rebases_remote_absolute_artifacts_into_copied_tree(portrait, tmp_path):
    reporting = module("joint_pilot_report", "scripts/reports/build_qwen21_joint_pilot_report.py")
    run = write_run(tmp_path, "qwen", portrait, portrait, "complete", "Qwen", "final")
    assert reporting.artifact_path(run, '/home/old/remote-run/images/output.png') == run / 'images/output.png'
    assert reporting.artifact_path(run, 'D:/old/remote-run/images/output.png') == run / 'images/output.png'
    with pytest.raises(ValueError, match='escapes'):
        reporting.artifact_path(run, '../outside.png')


def write_run(root, name, source, target, status, variant, stage, index=4):
    run = root / name
    (run / "images").mkdir(parents=True)
    source.save(run / "images/source.png")
    target.save(run / "images/target.png")
    if status == "complete":
        source.save(run / "images/output.png")
    (run / "config.json").write_text(json.dumps({"label": f"{variant} {stage}", "variant": variant, "stage": stage,
        "dataset_type": "tiktok_v3", "pairs_sha256": "fixed", "indices": [index], "seed": 42, "steps": 30,
        "resolution": 512, "align_mode": "letterbox", "prompt": "same"}))
    row = {"pair_index": index, "image_stem": "004_case", "status": status, "image": "images/output.png" if status == "complete" else None,
        "source": "images/source.png", "target": "images/target.png", "metadata": {"pair_id": "same_pair"},
        "source_face_box": [.1, .1, .5, .6], "target_face_box": [.1, .1, .5, .6], "face_valid": True,
        "failure": "CUDA OOM" if status == "failed" else None}
    (run / "per_sample.jsonl").write_text(json.dumps(row) + "\n")
    return run


def test_local_report_retains_failures_and_uses_relative_links(portrait, tmp_path):
    reporting = module("joint_pilot_report", "scripts/reports/build_qwen21_joint_pilot_report.py")
    write_run(tmp_path, "evaluations/qwen_final", portrait, portrait, "complete", "Qwen", "final")
    write_run(tmp_path, "evaluations/bfs_final", portrait, portrait, "failed", "BFSBody", "final")
    result = reporting.build_report(tmp_path, tmp_path / "report", thumb_side=96)
    assert result["cases"] == 1
    assert result["failed"] == 1
    text = (tmp_path / "report/index.html").read_text(encoding="utf-8")
    assert "offline" in text
    assert "CUDA OOM" in text
    assert "../evaluations/qwen_final/images/output.png" in text
    assert "file://" not in text
    assert str(tmp_path) not in text
    assert list((tmp_path / "report/panels").glob("*.jpg"))
    assert list((tmp_path / "report/face_sheets").glob("*.jpg"))


def test_report_rejects_conflicting_fixed_pair_metadata(portrait, tmp_path):
    reporting = module("joint_pilot_report", "scripts/reports/build_qwen21_joint_pilot_report.py")
    write_run(tmp_path, "qwen", portrait, portrait, "complete", "Qwen", "final")
    run = write_run(tmp_path, "bfs", portrait, portrait, "complete", "BFSBody", "final")
    row = json.loads((run / "per_sample.jsonl").read_text())
    row["metadata"]["source_stem"] = "different-source"
    (run / "per_sample.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="metadata"):
        reporting.build_report(tmp_path, tmp_path / "report", thumb_side=96)


def test_report_preserves_missing_output_as_failure(portrait, tmp_path):
    reporting = module("joint_pilot_report", "scripts/reports/build_qwen21_joint_pilot_report.py")
    run = write_run(tmp_path, "qwen", portrait, portrait, "complete", "Qwen", "final")
    (run / "images/output.png").unlink()
    result = reporting.build_report(tmp_path, tmp_path / "report", thumb_side=96)
    assert result["failed"] == 1
    assert "missing" in (tmp_path / "report/index.html").read_text(encoding="utf-8").lower()


def test_archived_infrastructure_attempt_is_separate_and_auditable(portrait, tmp_path):
    reporting = module("joint_pilot_report", "scripts/reports/build_qwen21_joint_pilot_report.py")
    write_run(tmp_path, 'evaluations/iper_init', portrait, portrait, 'complete', 'init', 'initial')
    archived = write_run(tmp_path, 'diagnostic_failures/init_initial_iper_missing_cache', portrait, portrait, 'failed', 'init', 'initial')
    (archived / 'rerun_reason.json').write_text(json.dumps({'category': 'infrastructure_condition_cache_path',
        'reason': 'missing four Face caches; rerun same fixed cases'}))
    (tmp_path / 'report').mkdir()
    (tmp_path / 'report/summary.md').write_text('summary', encoding='utf-8')
    result = reporting.build_report(tmp_path, tmp_path / 'report', thumb_side=96)
    assert result['runs'] == 1
    assert result['failed'] == 0
    assert result['archived_attempts'][0]['failed'] == 1
    text = (tmp_path / 'report/index.html').read_text(encoding='utf-8')
    assert '../diagnostic_failures/init_initial_iper_missing_cache/per_sample.jsonl' in text
    assert 'infrastructure_condition_cache_path' in text
    assert 'CUDA OOM' in text
    assert 'summary.md' in text


def test_report_cli_prints_archive_counts_without_full_nested_records(portrait, tmp_path, capsys):
    reporting = module("joint_pilot_report", "scripts/reports/build_qwen21_joint_pilot_report.py")
    write_run(tmp_path, 'evaluations/iper_init', portrait, portrait, 'complete', 'init', 'initial')
    archived = write_run(tmp_path, 'diagnostic_failures/init_initial_iper_missing_cache', portrait, portrait, 'failed', 'init', 'initial')
    (archived / 'rerun_reason.json').write_text(json.dumps({'category': 'infrastructure_condition_cache_path'}))
    reporting.main(['--run', str(tmp_path), '--thumb-side', '96'])
    text = capsys.readouterr().out
    assert len(text) < 1000
    result = json.loads(text)
    assert result['archived_attempts_count'] == 1
    assert 'archived_attempts' not in result
    manifest = json.loads((tmp_path / 'report/report_manifest.json').read_text())
    assert manifest['archived_attempts'][0]['records'][0]['failure'] == 'CUDA OOM'
