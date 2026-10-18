import importlib
import inspect

import pytest
import torch


def runtime():
    try:
        return importlib.import_module('src.pose_control.qwen21_full.dual.runtime')
    except ModuleNotFoundError as exc:
        if exc.name=='src.pose_control.qwen21_full.dual.runtime':
            pytest.fail('dual runtime and flow losses are not implemented')
        raise


def test_flow_target_and_weighted_normalization():
    r=runtime()
    clean=torch.randn(1,4,64)
    noise=torch.randn_like(clean)
    noisy,goal=r.flow_sample(clean,noise,torch.tensor([.25]))
    torch.testing.assert_close(noisy,.75*clean+.25*noise)
    assert torch.equal(goal,noise-clean)
    weights=torch.ones(1,4,1)
    assert r.weighted_flow_loss(goal+2,goal,weights).item()==4
    assert r.weighted_flow_loss(goal+2,goal,weights*7).item()==4


def test_background_teacher_excludes_both_source_and_target_people():
    r=runtime()
    source=torch.zeros(1,2,1,512,512)
    target=torch.zeros_like(source)
    source[:,0,:,32:128,32:128]=1
    target[:,1,:,256:384,256:384]=1
    background=r.background_teacher_mask(source,target,(32,32)).reshape(32,32)
    assert background[4,4]==0 and background[18,18]==0
    assert background[30,30]==1
    assert background[8,5]<1  # excludes an eight-pixel boundary too


def test_public_generate_signature_has_no_truth_argument():
    signature=inspect.signature(runtime().generate)
    assert set(signature.parameters)=={'pipe','generation','adapter','strengths','steps','cfg','use_kv_cache','single_context'}


def test_inference_records_payload_step_and_actual_checkpoint_hash(tmp_path):
    from scripts.infer.infer_qwen21_dual_v8 import checkpoint_record
    from src.data.hi4d_smplx_dataset import file_sha256
    path=tmp_path/'checkpoint.pt'
    path.write_bytes(b'test actual checkpoint')
    record=checkpoint_record(path,{'step':9}, {'step':17})
    assert record['checkpoint_step']==17
    assert record['checkpoint_sha256']==file_sha256(path)
    assert checkpoint_record(None,None,None)['checkpoint_step'] is None


def test_cache_probe_runs_actual_native_and_nonzero_adapter_paths():
    from scripts.train.train_qwen21_dual_v8_pilot import cache_probe
    from test_qwen21_dual_v8_context import fixture
    from test_qwen21_dual_v8_core import adapter,activate_heads
    from test_qwen21_dual_v8_conditions import make_input
    model,inputs,_=fixture()
    controller=adapter();activate_heads(controller)
    result=cache_probe(model,controller,controller.prepare(make_input()),inputs,.6)
    assert result['prefix_exact'] and result['disabled_exact'] and result['cached_close']
    assert result['target_changed'] and result['cached_max_abs']<1e-5


def test_single_missing_b_uses_caller_supplied_original_single_context():
    from test_qwen21_dual_v8_conditions import make_input
    try:
        from src.pose_control.qwen21_full.dual.api import select_control_context
    except ModuleNotFoundError as exc:
        if exc.name=='src.pose_control.qwen21_full.dual.api':pytest.fail('explicit legacy dispatch is not implemented')
        raise
    marker=object()
    context=select_control_context('dual_v8',object(),generation=make_input(with_b=False,contact=False),
                                  single_context_factory=lambda:marker)
    assert context is marker
    with pytest.raises(ValueError,match='original single'):
        select_control_context('dual_v8',object(),generation=make_input(with_b=False,contact=False))
