"""Stable face assignment and Source-only repair for the real V7 pilot."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
import torch

SCRIPTS=Path(os.environ.get('V7_PILOT_SCRIPT_ROOT',str(Path(__file__).resolve().parents[1]/'experiments/hi4d_qwen21_dual_20260930/native-v7-singlepair-overfit-20261117/scripts')))


def module():
    path=SCRIPTS/'prepare_assets.py'
    spec=importlib.util.spec_from_file_location('v7_prepare_assets',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_face_assignment_tracks_geometry_slots_not_detector_order():
    from src.data.canvas_contract import native_pad
    canvas=native_pad((128,96))
    detections=np.array([[62,30,86,60,.99],[10,30,34,60,.99]],dtype=np.float32)
    expected=np.array([[10/96,30/128,34/96,60/128],[62/96,30/128,86/96,60/128]])
    assert module().match_faces(detections,expected,canvas)==[1,0]
    with pytest.raises(ValueError,match='qualified'):
        module().match_faces(detections[:1],expected,canvas)


def test_repair_uses_source_slot_mapping_and_refuses_implicit_canvas_resize():
    from types import SimpleNamespace
    from src.data.canvas_contract import native_pad
    canvas=native_pad((32,32)).metadata()
    masks=torch.zeros(2,1,32,32);masks[0,:,:,:16]=1;masks[1,:,:,16:]=1
    local=SimpleNamespace(source_indices=torch.tensor([[1,0]]),person_valid=torch.tensor([[True,True]]),
        metadata=({'source_rgb_sha256':'source','target_canvas_id':'canvas','target_person_ids':['b','a'],'driver_record_sha256':'driver'},))
    sample={'source_canvas':canvas,'target_canvas':canvas,'source_person_mask':masks,
            'valid_image_mask':torch.ones(1,32,32),'local_condition_v4':local}
    repair=module().source_repair(sample)
    assert torch.equal(repair['mask'][0,0],masks[1,0]) and torch.equal(repair['mask'][0,1],masks[0,0])
    sample['target_canvas']=native_pad((64,32)).metadata()
    with pytest.raises(ValueError,match='Canvas'):
        module().source_repair(sample)


def test_existing_identity_factory_freezes_actual_torchscript(tmp_path):
    path=tmp_path/'identity.ts'
    traced=torch.jit.trace(torch.nn.Linear(3,2).eval(),torch.zeros(1,3))
    torch.jit.save(traced,str(path))
    loaded=module().load_identity(path)
    assert not loaded.training and all(not parameter.requires_grad for parameter in loaded.parameters())


def test_execution_policy_records_official_tiling_in_strict_identity(tmp_path):
    from types import SimpleNamespace
    path=SCRIPTS/'gpu_action.py'
    spec=importlib.util.spec_from_file_location('v7_gpu_action',path)
    action=importlib.util.module_from_spec(spec);spec.loader.exec_module(action)
    class VAE:
        use_tiling=False
        tile_sample_min_height=256;tile_sample_min_width=256
        tile_sample_stride_height=192;tile_sample_stride_width=192
        def enable_tiling(self):self.use_tiling=True
    backend=SimpleNamespace(identity={},pipe=SimpleNamespace(vae=VAE()))
    action.apply_execution_policy(backend,vae_tiling=True,memory_fraction=.5)
    assert backend.pipe.vae.use_tiling
    policy=backend.identity['pilot_execution']
    assert policy['vae_tiling'] is True and policy['memory_fraction']==.5
    assert len(policy['wrapper_sha256'])==64 and len(policy['vae_implementation_sha256'])==64


def test_hidden_cpu_roundtrip_preserves_checkpoint_output_and_gradient():
    if not torch.cuda.is_available():pytest.skip('Real CUDA activation offload acceptance')
    from torch.utils.checkpoint import checkpoint
    path=SCRIPTS/'gpu_action.py'
    spec=importlib.util.spec_from_file_location('v7_gpu_offload',path)
    action=importlib.util.module_from_spec(spec);spec.loader.exec_module(action)
    torch.manual_seed(42)
    projection=torch.nn.Sequential(torch.nn.Linear(4096,64),torch.nn.SiLU(),torch.nn.Linear(64,4096)).cuda().requires_grad_(False)
    original=torch.randn(1,512,4096,device='cuda',requires_grad=True)
    expected=checkpoint(projection,original,use_reentrant=False).tanh()
    expected.square().mean().backward();reference=original.grad.clone();original.grad=None
    telemetry={}
    with torch.autograd.graph.saved_tensors_hooks(*action.hidden_offload_hooks(telemetry)):
        actual=checkpoint(projection,original,use_reentrant=False).tanh()
    actual.square().mean().backward()
    assert torch.equal(actual,expected) and torch.equal(original.grad,reference)
    assert telemetry['offloaded_tensors']>0 and telemetry['offloaded_bytes']>=original.numel()*original.element_size()


def test_released_gpu_allows_explicit_full_budget_with_headroom():
    spec=importlib.util.spec_from_file_location('v7_gpu_budget',SCRIPTS/'gpu_action.py')
    action=importlib.util.module_from_spec(spec);spec.loader.exec_module(action)
    assert action.validate_memory_fraction(.9)==.9
    for value in (0.,1.,float('nan')):
        with pytest.raises(ValueError):action.validate_memory_fraction(value)


def test_path_rebind_requires_identical_bytes_and_keeps_semantic_inputs(tmp_path):
    import hashlib
    spec=importlib.util.spec_from_file_location('v7_paths',SCRIPTS/'rebind_paths.py')
    paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths)
    actual=tmp_path/'actual';actual.mkdir();(actual/'image.dat').write_bytes(b'original bytes')
    reference={'path':str(tmp_path/'removed/image.dat'),'sha256':hashlib.sha256(b'original bytes').hexdigest()}
    result=paths.rebind_ref(reference,[(str(tmp_path/'removed'),str(actual))])
    assert result['sha256']==reference['sha256'] and Path(result['path'])==actual/'image.dat'
    with pytest.raises(ValueError,match='SHA256'):
        paths.rebind_ref({**reference,'sha256':'0'*64},[(str(tmp_path/'removed'),str(actual))])


def test_path_rebind_verifies_old_cache_before_reading_and_rehashing(tmp_path):
    import hashlib
    spec=importlib.util.spec_from_file_location('v7_paths_verified',SCRIPTS/'rebind_paths.py')
    paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths)
    original=tmp_path/'cache.pt';original.write_bytes(b'original latent')
    reference={'path':str(original),'sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
    assert paths.verified_path(reference)==original
    original.write_bytes(b'altered latent')
    with pytest.raises(ValueError,match='SHA256'):
        paths.verified_path(reference)
