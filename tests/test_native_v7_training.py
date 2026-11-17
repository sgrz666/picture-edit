"""The V7 training extension must share the existing strict execution engine."""
import pytest


def test_v7_prepare_is_versioned_model_free_and_defaults_to_4ref():
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7, NativeV7Trainer
    cfg = TrainingConfigV7()
    result = NativeV7Trainer(cfg).prepare()
    assert result['schema'] == 'native_v7_training_v1'
    assert cfg.qwen_condition_profile == 'dual_4ref'
    assert not result['model_constructed'] and not result['optimizer_created'] and not result['training_started']
    assert TrainingConfigV7.from_dict(cfg.record()) == cfg
    assert 'evaluators.identity' in result['missing_bindings']


def test_v7_training_reuses_v62_engine_and_rejects_other_schemas():
    from src.pose_control.qwen21_full.native_v62.training.trainer import NativeV62Trainer
    from src.pose_control.qwen21_full.native_v7.training import NativeV7Trainer, TrainingConfigV7
    assert issubclass(NativeV7Trainer, NativeV62Trainer)
    assert NativeV7Trainer.run is NativeV62Trainer.run
    with pytest.raises(ValueError):
        TrainingConfigV7.from_dict({'schema': 'native_v62_training_v1'})


def test_paper_loss_is_part_of_strict_config_signature():
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    old = TrainingConfigV7()
    raw = old.record()
    raw['paper_loss']['contact_fm'] = 0.
    assert TrainingConfigV7.from_dict(raw).signature() != old.signature()


def test_id_request_with_no_qualified_pair_is_rejected():
    from src.pose_control.qwen21_full.native_v7.training import validate_paper_eligibility
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    rows = [{'eligible': True, 'auxiliary_faces': 0}, {'eligible': True, 'auxiliary_faces': 1}]
    with pytest.raises(ValueError, match='contrastive'):
        validate_paper_eligibility(rows, PaperLossConfig())
    assert validate_paper_eligibility(rows + [{'eligible': True, 'auxiliary_faces': 2}], PaperLossConfig())['contrastive_pairs'] == 1


def test_v7_cli_prepare_and_explicit_velocity_ablation(capsys):
    import json
    from src.pose_control.qwen21_full.native_v7.training import main
    main(['prepare', '--diagnostics', '--loss-profile', 'velocity_only'])
    result = json.loads(capsys.readouterr().out)
    assert result['schema'] == 'native_v7_training_v1'
    assert result['config']['qwen_condition_profile'] == 'dual_4ref'
    assert result['config']['paper_loss']['identity_contrastive'] == 0.
    assert result['config']['paper_loss']['person_fm'] == 0.
    assert not result['training_started'] and not result['model_constructed']


def test_v7_entrypoint_help_uses_existing_cli_without_model_loading():
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for folder, script, flag in (('train', 'train_qwen21_native_full_v7.py', '--profile'),
                                 ('run', 'run_qwen21_native_full_v7.py', '--no-spatial-rope'),
                                 ('data', 'migrate_qwen21_native_full_v7.py', '--v62'),
                                 ('audit', 'audit_qwen21_native_full_v7.py', '--manifest')):
        result = subprocess.run([sys.executable, str(root / 'scripts' / folder / script), '--help'], capture_output=True)
        assert result.returncode == 0, result.stderr.decode('utf8', errors='replace')
        assert flag.encode() in result.stdout


def test_versioned_snapshot_reuses_engine_and_rejects_cross_version(tmp_path, monkeypatch):
    import torch
    from dataclasses import replace
    from tests.test_qwen21_native_v5_training import ToyRuntime
    from src.pose_control.qwen21_full.native_v62.training import checkpointing as cp
    from src.pose_control.qwen21_full.native_v62.training.parameters import ParameterCatalog, UpdateScheduler
    from src.pose_control.qwen21_full.native_v62.training.sampling import DeterministicSampler
    from src.pose_control.qwen21_full.native_v62.training.config import SourceSpec
    from src.pose_control.qwen21_full.native_v62.training.io import Journal
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7, stage_preset_v7
    monkeypatch.setattr(cp, 'environment', lambda: {'synthetic_cpu': True})
    cfg = TrainingConfigV7(stage=stage_preset_v7('S1', 'face', budget=8), sources=(SourceSpec('train', weight=1.),))
    model = ToyRuntime()
    catalog = ParameterCatalog(model)
    active = catalog.apply(cfg.stage)
    optimizer = catalog.build_optimizer(cfg.stage.optimizer)
    scheduler = UpdateScheduler(optimizer, cfg.stage.optimizer, 8)
    sampler = DeterministicSampler(cfg, {('face', 'train'): [0, 1]})
    journal = Journal(tmp_path / 'metrics.jsonl')
    journal.append({'update': 0})
    result = cp.save_snapshot(tmp_path / 'checkpoints', model, optimizer, scheduler, sampler, torch.Generator().manual_seed(42),
        config=cfg, identity={'synthetic': True}, catalog=catalog, active=active, global_update=0,
        stage_update=0, attempts=0, journal=journal, micro_position=0, gradient_report={'path': '/unused'}, stage_origin={'sha256': 'synthetic'})
    saved, _, _ = cp.load_committed(result['checkpoint'], schema=cfg.schema)
    assert cp.validate_resume(saved, cfg, {'synthetic': True}, catalog)['active_names'] == sorted(active)
    with pytest.raises(ValueError, match='commit|schema'):
        cp.load_committed(result['checkpoint'])
    with pytest.raises(ValueError, match='config'):
        cp.validate_resume(saved, replace(cfg, paper_loss=replace(cfg.paper_loss, contact_fm=0.)), {'synthetic': True}, catalog)


def test_input_audit_uses_the_existing_typed_reader(tmp_path):
    from tests.test_native_v6_data import _manifest
    from scripts.audit.audit_qwen21_native_full_v7 import audit_manifest
    manifest, _, _ = _manifest(tmp_path)
    report = audit_manifest(manifest)
    assert report['complete'] and report['audited_rows'] == report['manifest_rows'] == 1
    assert report['profiles'] == {'dual_source_5ref': 1}
    assert not report['model_constructed'] and not report['training_started']
    assert report['qualified_contrastive_pairs'] == 0 and report['repair_rows'] == 0


def test_paper_data_status_reports_real_visible_contact_and_forbidden_pairs():
    from types import SimpleNamespace
    import torch
    from src.pose_control.qwen21_full.native_v7.training import paper_sample_status
    parts = torch.zeros(1, 2, 14, 2, 2)
    parts[:, 0, 0] = 1
    parts[:, 1, 1] = 1
    relations = torch.zeros(1, 14, 14, 4)
    relations[:, 0, 1, :2] = 1
    context=SimpleNamespace(part_valid=torch.ones(1,1,2,2))
    depth = SimpleNamespace(part_contact=relations, person_part=parts, owner_weights=torch.ones(1, 2, 1, 2, 2),
                            person_a=SimpleNamespace(context=context),person_b=SimpleNamespace(context=context))
    sample = {'native_scene_kind': 'dual', 'dual_depth': depth,
              'valid_image_mask':torch.ones(1,2,2),
              'local_condition_v4': SimpleNamespace(person_valid=torch.tensor([[True, True]]))}
    status = paper_sample_status(sample)
    assert status['paper_persons'] == 2 and status['paper_positive_contacts'] == 1
    assert status['paper_visible_contact_persons'] == 2
    relations[:, 0, 1, 2:] = 1
    forbidden = paper_sample_status(sample)
    assert forbidden['paper_forbidden_contacts'] == 1
    assert forbidden['paper_positive_contacts'] == 0 and forbidden['paper_visible_contact_persons'] == 0
    relations[:,0,1,2:]=0
    context.part_valid.zero_()
    assert paper_sample_status(sample)['paper_visible_contact_persons']==0


def test_contrastive_only_backend_preloads_identity_and_keeps_gate_cases_distinct():
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7, TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.config import LossSpec
    cfg = TrainingConfigV7()
    stage = replace(cfg.stage, routes=tuple(replace(route, loss=LossSpec()) for route in cfg.stage.routes))
    backend = object.__new__(NativeBackendV7)
    backend.config = replace(cfg, stage=stage)
    assert backend.required_evaluators == {'identity'}
    first = {'auxiliary_faces': 1, 'paper_positive_contacts': 0, 'paper_visible_contact_persons': 0}
    paired = {**first, 'auxiliary_faces': 2}
    contact = {**paired, 'paper_positive_contacts': 1, 'paper_visible_contact_persons': 2}
    assert backend.gradient_case_key(first) != backend.gradient_case_key(paired)
    assert backend.gradient_case_key(contact) != backend.gradient_case_key(paired)


def test_gradient_gate_selects_sigma_inside_requested_auxiliary_interval():
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7, TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.config import RouteSpec, LossSpec
    backend=object.__new__(NativeBackendV7)
    backend.config=TrainingConfigV7()
    route=RouteSpec('poselink','poselink',1.,loss=LossSpec(auxiliary_sigma_max=.1))
    sigma=backend.preflight_sigma(route)
    assert 0<sigma<=.1
    counts=backend.window_counts([{'route':'poselink','sigma':sigma,'status':{'face_area':1.,'hand_area':0.,'auxiliary_faces':2}}],{'poselink':route},4)
    assert counts['paper_identity']==2
    assert backend.preflight_sigma(replace(route,loss=LossSpec()))==.2


def test_cli_keeps_bound_initialization_when_checkpoint_names_same_file(tmp_path,capsys):
    import json
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7,main
    checkpoint=tmp_path/'parent.pt'
    bound={'path':str(checkpoint.resolve()),'sha256':'a'*64}
    config=replace(TrainingConfigV7(),initialization=bound)
    path=tmp_path/'config.json';path.write_text(json.dumps(config.record()),encoding='utf8')
    main(['prepare','--config',str(path),'--checkpoint',str(checkpoint)])
    report=json.loads(capsys.readouterr().out)
    assert report['config']['initialization']==bound
    assert report['config_sha256']==config.signature()
