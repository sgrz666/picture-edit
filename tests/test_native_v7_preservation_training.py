"""Source anchors extend the V7 training contract without changing old records."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch


def config(**weights):
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    return TrainingConfigV7(preservation=weights)


def test_disabled_preservation_keeps_legacy_serialization_and_signature():
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    from src.pose_control.qwen21_full.native_v62.training.config import digest
    legacy = TrainingConfigV7()
    assert legacy.preservation is None
    record = legacy.record()
    assert 'preservation' not in record and 'preservation_index' not in record
    assert TrainingConfigV7.from_dict(record) == legacy
    record.pop('gradient_report'); record.pop('body_baseline_report')
    assert digest(record) == legacy.signature()


def test_source_config_is_bound_and_requires_actual_evaluator():
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    enabled = config(source_identity=.2, background=.1)
    assert 'preservation_index' in enabled.missing()
    assert 'evaluators.identity' in enabled.missing()
    assert TrainingConfigV7.from_dict(enabled.record()) == enabled
    assert enabled.signature() != TrainingConfigV7().signature()
    with pytest.raises(ValueError):
        replace(TrainingConfigV7(), preservation_index={'path': '/data/a.json', 'sha256': 'a'*64})


def test_explicit_velocity_ablation_turns_off_source_objectives(tmp_path,capsys):
    import json
    from src.pose_control.qwen21_full.native_v7.training import main
    cfg=config(source_identity=.2,background=.1)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg.record()),encoding='utf8')
    main(['prepare','--config',str(path),'--loss-profile','velocity_only'])
    report=json.loads(capsys.readouterr().out)
    assert 'preservation' not in report['config'] and 'preservation_index' not in report['config']


def test_source_only_gates_count_source_gt_intersection_over_window():
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7
    from src.pose_control.qwen21_full.native_v62.training.config import RouteSpec, LossSpec
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    backend = object.__new__(NativeBackendV7)
    backend.config = replace(config(source_identity=.2, background=.1), paper_loss=PaperLossConfig(0.,0.,0.))
    route = RouteSpec('joint', 'joint', 1., loss=LossSpec(auxiliary_every=2, auxiliary_sigma_max=.5))
    rows = [{'route':'joint', 'sigma': sigma, 'status': {'face_area':1., 'hand_area':1.,
             'auxiliary_faces':2, 'preservation_faces':faces, 'preservation_background': bg}}
            for sigma,faces,bg in ((.2,2,True),(.3,1,True),(.7,2,False))]
    assert backend.required_evaluators == {'identity'} and backend.extra_auxiliary
    counts = backend.window_counts(rows, {'joint':route}, 2)
    assert counts['source_identity'] == 3 and counts['source_background'] == 2
    assert backend.window_counts(rows, {'joint':route}, 3)['source_identity'] == 0
    assert backend.preflight_sigma(route) <= .5


def test_source_identity_reuses_one_gt_decode_and_reference_receives_no_gradient(monkeypatch):
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    from src.pose_control.qwen21_full.native_v62.training.config import LossSpec
    from src.pose_control.qwen21_full.native_v5 import losses
    backend = object.__new__(NativeBackendV7)
    backend.config = replace(config(source_identity=.2), paper_loss=PaperLossConfig(0.,0.,0.))
    source = torch.tensor([1.,0.,0.]).repeat(171)[:512].expand(2,-1).clone().requires_grad_()
    backend.preservation_context = {'source_features':source, 'source_valid':torch.tensor([True,True])}
    calls=[]
    def decode(pipe,tokens,hw):
        calls.append(hw)
        return tokens[:,:,:3].mean(1)[:,:,None,None].expand(-1,-1,*hw)
    monkeypatch.setattr(losses,'decode_qwen64',decode)
    class Encoder(torch.nn.Module):
        def forward(self,images):
            return images.mean((-1,-2)).repeat(1,171)[:,:512]
    backend.pipe=object()
    prediction=torch.zeros(1,4,64,requires_grad=True)
    noisy=torch.zeros_like(prediction);noisy[:,:,:3]=torch.tensor([.7,.8,.6])
    points=torch.tensor([[[[8.,8.],[22.,8.],[16.,16.],[10.,24.],[22.,24.]]]*2])
    supervision=SimpleNamespace(auxiliary_valid=torch.ones(1,2,dtype=torch.bool),landmarks5=points,
        paired=torch.ones(1,dtype=torch.bool),weights=torch.ones(1,6,32,32),validate=lambda:None)
    counts=dict(velocity=2,face=0,hand=0,identity=0,perceptual=0,paper_identity=0,source_identity=4,source_background=0)
    loss,parts=backend.compute_loss(prediction,torch.zeros_like(prediction),torch.ones(1,4),supervision,
        noisy,torch.tensor([.2]),torch.zeros(1,3,32,32),LossSpec(auxiliary_every=1),counts,1,(Encoder().eval(),None))
    assert len(calls)==1 and parts['source_identity_eligible']==2
    assert torch.allclose(loss,.2*parts['source_identity']*2/4)
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum()>0
    assert source.grad is None


def test_background_only_uses_source_x0_no_decode_and_image_window_count():
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    from src.pose_control.qwen21_full.native_v62.training.config import LossSpec
    backend=object.__new__(NativeBackendV7)
    backend.config=replace(config(background=.1),paper_loss=PaperLossConfig(0.,0.,0.))
    backend.pipe=object()
    source=torch.zeros(1,4,64,requires_grad=True)
    backend.preservation_context={'source_latent':source,'background_support':torch.tensor([[1.,0.,0.,0.]])}
    pred=torch.zeros(1,4,64,requires_grad=True)
    sup=SimpleNamespace(auxiliary_valid=torch.zeros(1,2,dtype=torch.bool),paired=torch.ones(1,dtype=torch.bool),
        weights=torch.ones(1,6,32,32),validate=lambda:None)
    counts=dict(velocity=2,face=0,hand=0,identity=0,perceptual=0,paper_identity=0,source_identity=0,source_background=2)
    loss,parts=backend.compute_loss(pred,torch.zeros_like(pred),torch.ones(1,4),sup,torch.ones_like(pred),
        torch.tensor([.5]),torch.zeros(1,3,32,32),LossSpec(),counts,1,(None,None))
    assert torch.allclose(loss,torch.tensor(.05))
    assert parts['source_background_area']==1
    loss.backward();assert source.grad is None and pred.grad[0,0].abs().sum()>0
    assert pred.grad[0,1:].abs().sum()==0


def test_data_plan_binds_index_to_identity_and_qualifies_source_intersection(tmp_path,monkeypatch):
    import hashlib
    from src.pose_control.qwen21_full.native_v7.training import DataPlanV7
    from src.pose_control.qwen21_full.native_v62.training.data import DataPlan
    from src.pose_control.qwen21_full.native_v7 import preservation_assets as assets
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    model=tmp_path/'model.ts';model.write_bytes(b'fixed evaluator bytes')
    model_ref={'path':str(model),'sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
    cfg=replace(config(source_identity=.2,background=.1),evaluators={'identity':model_ref},
        preservation_index={'path':str(tmp_path/'index.json'),'sha256':'a'*64},paper_loss=PaperLossConfig(0.,0.,0.))
    sample={'training_supervision_v5':SimpleNamespace(auxiliary_valid=torch.tensor([[True,False]]))}
    class Dataset:
        rows=[{'pair_id':'pair'}]
        def __getitem__(self,index):return sample
        def __len__(self):return 1
    def base_init(self,cfg):
        self.config=cfg;self.datasets={'train':Dataset()};self.spatial={};self.regional={}
        status={'eligible':True,'auxiliary_faces':1}
        self.status={(cfg.stage.routes[0].name,'train',0):status}
        self.records=[dict(status,route=cfg.stage.routes[0].name,role='train',source_id='train',index=0)]
        self.identity={'config_sha256':cfg.signature()}
    monkeypatch.setattr(DataPlan,'__init__',base_init)
    index={'evaluator':model_ref,'entries':{'train\0pair':{'path':str(tmp_path/'record.pt'),'sha256':'b'*64}}}
    monkeypatch.setattr(assets,'read_index',lambda ref:(index,ref))
    record={'source_features':torch.ones(2,512),'source_valid':torch.tensor([True,True]),'background_support':torch.ones(1,4)}
    monkeypatch.setattr(assets,'read_record',lambda ref,sample,sha:record)
    first=DataPlanV7(cfg)
    assert first.records[0]['preservation_faces']==1
    assert first.preservation_records[('train',0)] is record
    assert 'preservation' in first.identity
    second=DataPlanV7(replace(cfg,preservation_index={**cfg.preservation_index,'sha256':'c'*64}))
    assert first.signature!=second.signature
    index['evaluator']={**model_ref,'sha256':'d'*64}
    with pytest.raises(ValueError,match='evaluator'):
        DataPlanV7(cfg)


def test_cross_canvas_zero_background_support_skips_grid_mismatch_in_forward_and_loss(monkeypatch):
    from src.pose_control.qwen21_full.native_v62.training.backend import NativeBackend
    from src.pose_control.qwen21_full.native_v62.training.config import LossSpec
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7

    backend = object.__new__(NativeBackendV7)
    backend.config = replace(config(background=.1), paper_loss=PaperLossConfig(0., 0., 0.))
    backend.device = torch.device('cpu')
    backend.pipe = object()
    pred = torch.zeros(1, 4, 64, requires_grad=True)
    target = torch.ones(1, 4, 64)
    coverage = torch.ones(1, 4)
    noisy = torch.ones(1, 4, 64)
    monkeypatch.setattr(NativeBackend, 'forward', lambda self, *a, **kw: (pred, target, coverage, noisy))

    record = {'source_features': torch.zeros(2, 512), 'source_valid': torch.zeros(2, dtype=torch.bool),
              'background_support': torch.zeros(1, 4)}
    pair = {'source_latent': torch.zeros(1, 6, 64), 'target_latent': torch.zeros(1, 4, 64)}
    out = backend.forward({'source_preservation_v7': record}, pair, None, torch.tensor([.5]), None, None, None)
    assert out[0] is pred

    sup = SimpleNamespace(auxiliary_valid=torch.zeros(1, 2, dtype=torch.bool), paired=torch.ones(1, dtype=torch.bool),
        weights=torch.ones(1, 6, 32, 32), validate=lambda: None)
    counts = dict(velocity=1, face=0, hand=0, identity=0, perceptual=0, paper_identity=0,
                  source_identity=0, source_background=0)
    loss, parts = backend.compute_loss(pred, target, coverage, sup, noisy, torch.tensor([.5]),
        torch.zeros(1, 3, 32, 32), LossSpec(), counts, 1, (None, None))
    assert parts['source_background'].item() == 0.
    assert parts['source_background_area'].item() == 0.
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_cross_canvas_nonzero_background_support_still_rejected(monkeypatch):
    from src.pose_control.qwen21_full.native_v62.training.backend import NativeBackend
    from src.pose_control.qwen21_full.native_v62.training.config import LossSpec
    from src.pose_control.qwen21_full.native_v7.losses import PaperLossConfig
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7

    backend = object.__new__(NativeBackendV7)
    backend.config = replace(config(background=.1), paper_loss=PaperLossConfig(0., 0., 0.))
    backend.device = torch.device('cpu')
    backend.pipe = object()
    pred = torch.zeros(1, 4, 64)
    monkeypatch.setattr(NativeBackend, 'forward', lambda self, *a, **kw: (pred, pred, torch.ones(1, 4), pred))

    record = {'source_features': torch.zeros(2, 512), 'source_valid': torch.zeros(2, dtype=torch.bool),
              'background_support': torch.tensor([[1., 0., 0., 0.]])}
    pair = {'source_latent': torch.zeros(1, 6, 64), 'target_latent': torch.zeros(1, 4, 64)}
    with pytest.raises(ValueError, match='target latent grid'):
        backend.forward({'source_preservation_v7': record}, pair, None, torch.tensor([.5]), None, None, None)

    backend.preservation_context = {'source_latent': torch.zeros(1, 6, 64),
                                    'background_support': torch.tensor([[1., 0., 0., 0.]])}
    sup = SimpleNamespace(auxiliary_valid=torch.zeros(1, 2, dtype=torch.bool), paired=torch.ones(1, dtype=torch.bool),
        weights=torch.ones(1, 6, 32, 32), validate=lambda: None)
    counts = dict(velocity=1, face=0, hand=0, identity=0, perceptual=0, paper_identity=0,
                  source_identity=0, source_background=1)
    with pytest.raises(ValueError, match='source_latent'):
        backend.compute_loss(pred, pred, torch.ones(1, 4), sup, pred, torch.tensor([.5]),
            torch.zeros(1, 3, 32, 32), LossSpec(), counts, 1, (None, None))
