"""Version/header, preparation and parameter contracts with synthetic CPU weights."""
import pytest
import torch


def test_v62_prepare_is_configuration_only_and_versioned():
    from src.pose_control.qwen21_full.native_v62.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v62.training.trainer import NativeV62Trainer
    cfg=TrainingConfig();report=NativeV62Trainer(cfg).prepare()
    assert report['schema']=='native_v62_training_v1'
    assert not report['model_constructed'] and not report['optimizer_created'] and not report['training_started']
    assert TrainingConfig.from_dict(cfg.record())==cfg


def test_diagnostics_can_be_requested_without_starting_execution(capsys):
    import json
    from dataclasses import replace
    from src.pose_control.qwen21_full.native_v62.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v62.training.cli import main
    cfg=replace(TrainingConfig(),diagnostics=True)
    assert cfg.diagnostics and TrainingConfig.from_dict(cfg.record())==cfg
    main(['prepare','--diagnostics'])
    report=json.loads(capsys.readouterr().out)
    assert report['config']['diagnostics'] is True
    assert not report['model_constructed'] and not report['optimizer_created'] and not report['training_started']
    with pytest.raises(ValueError):replace(cfg,diagnostics='true')


def test_migration_help_works_with_a_legacy_windows_console_encoding():
    import os,subprocess,sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/data/migrate_qwen21_native_full_v62.py'
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,
        env=dict(os.environ,PYTHONIOENCODING='cp1252',PYTHONUTF8='0'))
    assert result.returncode==0,result.stderr.decode('utf8',errors='replace')
    assert b'--v61' in result.stdout


def test_prepare_emits_utf8_json_for_chinese_config_under_legacy_console(tmp_path):
    import json,os,subprocess,sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/train/train_qwen21_native_full_v62.py'
    config=tmp_path/'config.json';config.write_text(json.dumps({'prompt':'保持人物身份，修改动作。'},ensure_ascii=False),encoding='utf8')
    result=subprocess.run([sys.executable,str(script),'prepare','--config',str(config)],capture_output=True,
        env=dict(os.environ,PYTHONIOENCODING='cp1252',PYTHONUTF8='0'))
    assert result.returncode==0,result.stderr.decode('utf8',errors='replace')
    report=json.loads(result.stdout.decode('utf8'))
    assert report['config']['prompt']=='保持人物身份，修改动作。' and not report['model_constructed']


def test_v61_initialization_and_strict_v62_rebuild_do_not_resume_optimizer():
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody,NativeBodyConfig
    from src.pose_control.qwen21_full.native_dual_runtime import NativeDualRuntime
    from src.pose_control.qwen21_full.native_v61.contracts import PoseLinkConfigV61
    from src.pose_control.qwen21_full.native_v61.runtime import NativeFullRuntimeV61
    from src.pose_control.qwen21_full.native_v62.runtime import NativeFullRuntimeV62
    from src.pose_control.qwen21_full.native_v62.training.parameters import ParameterCatalog
    from src.pose_control.qwen21_full.native_v62.training.config import stage_preset
    layers=tuple(range(6))
    cfg=NativeConfigV5(verification_only=True,hidden_dim=32,num_layers=6,injection_layers=layers)
    base=NativeDualRuntime(body=Qwen21NativeBody(NativeBodyConfig(hidden_dim=32,num_layers=6,injection_layers=layers)))
    old=NativeFullRuntimeV61(base,cfg,pose_config=PoseLinkConfigV61(hidden_dim=32,width=32,heads=4,
        injection_layers=layers,max_pose_tokens_per_reference=16,global_token_budget=8,joint_token_budget=4,local_token_budget=4))
    saved=old.checkpoint_payload(metadata={});saved['training_state']={'must_not_resume':True}
    new=NativeFullRuntimeV62.from_v61_checkpoint(saved)
    assert new.pose_config.person_routing and new.pose_config.body_spatial_gate
    assert new.get_extra_state()['architecture']=='native_full_v62'
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items() if torch.is_tensor(v))
    catalog=ParameterCatalog(new);active=catalog.apply(stage_preset('S2','poselink'))
    assert active and all(name.startswith('poselink.') for name in active)
    payload=new.checkpoint_payload(metadata={});assert 'training_state' not in payload
    rebuilt=NativeFullRuntimeV62.from_checkpoint(payload)
    assert rebuilt.get_extra_state()==new.get_extra_state()
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in new.state_dict().items() if torch.is_tensor(v))
    with pytest.raises(ValueError):NativeFullRuntimeV61.from_checkpoint(payload)
    with pytest.raises(ValueError):NativeFullRuntimeV62.from_checkpoint(saved)
    with pytest.raises(ValueError):NativeFullRuntimeV62.from_v6_checkpoint({'architecture':'native_full_v6'})
