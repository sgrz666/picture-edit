"""Data isolation and matched training contracts for the prompt comparison."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def module():
    try:
        return importlib.import_module('src.pose_control.qwen21_full.dual.prompt_pilot')
    except ModuleNotFoundError as exc:
        if exc.name == 'src.pose_control.qwen21_full.dual.prompt_pilot':
            pytest.fail('matched prompt pilot trainer is not implemented')
        raise


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


APPROVED_TRAIN = [('pair14','talk14',32,'hug14',45,4), ('pair14','talk14',32,'hug14',62,4),
                  ('pair12','highfive12',28,'highfive12',33,4), ('pair12','highfive12',28,'highfive12',82,4)]
APPROVED_VAL = [('pair22','game22',52,'hug22',46,4)]


def dataset_case(tmp_path, *, train_specs=None, val_specs=None):
    from src.data.hi4d_smplx_dataset import file_sha256
    from src.pose_control.qwen21_full.dual.schema import generation_versions, geometry_schema_metadata
    split = tmp_path/'split.json'
    write_json(split, dict(train_pairs=['pair12', 'pair14'], val_pairs=['pair22'], test_pairs=['pair00']))
    entries = {}
    for name, specs in [('train', APPROVED_TRAIN if train_specs is None else train_specs),
                        ('val', APPROVED_VAL if val_specs is None else val_specs)]:
        short, structured, truths, ids = [], [], [], []
        for pair, source_action, source_frame, action, frame, camera in specs:
            sid = f'{pair}_{frame}'
            ids.append(sid)
            source = tmp_path/f'sources/{pair}/cam{camera}/{source_action}/{source_frame:06d}'
            target = tmp_path/f'targets/{pair}/{action}/cam{camera}/{frame:06d}'
            source.mkdir(parents=True, exist_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (32,32), (30,40,50)).save(source/'rgb.jpg')
            Image.new('RGB', (32,32), (70,80,90)).save(target/'rgb.jpg')
            for root, prefix in [(source,'mask'), (target,'gt_mask')]:
                for actor in 'AB':
                    Image.new('L',(32,32),255).save(root/f'{prefix}_{actor}.png')
            features = tmp_path/f'{sid}_source.pt'
            torch.save({'marker': torch.tensor([1.])}, features)
            geometry = tmp_path/f'{pair}/{action}_{frame:06d}_cam{camera}_geometry.npz'
            geometry.parent.mkdir(exist_ok=True)
            uids = [f'{pair}_actor0', f'{pair}_actor1']
            np.savez(geometry, **geometry_schema_metadata(), source_shape_shared=np.array(True), person_uids=np.asarray(uids))
            sidecar = dict(geometry_sha256=file_sha256(geometry), pair=pair, action=action, source_frame=source_frame,
                           target_frame=frame, camera=camera, person_uids=uids)
            write_json(geometry.with_suffix('.json'), sidecar)
            row = dict(source_rgb=str(source/'rgb.jpg'), source_masks=[str(source/f'mask_{x}.png') for x in 'AB'],
                       source_features=str(features), source_features_sha256=file_sha256(features),
                       target_geometry=str(geometry), person_uids=uids, prompt='The two people hug.', seed=42,
                       resolution=512, versions=generation_versions(), target_geometry_known=True)
            truth = dict(schema_version='dual_supervision_v1', target_rgb=str(target/'rgb.jpg'),
                         target_masks=[str(target/f'gt_mask_{x}.png') for x in 'AB'],
                         target_rgb_sha256=file_sha256(target/'rgb.jpg'),
                         target_mask_sha256=[file_sha256(target/f'gt_mask_{x}.png') for x in 'AB'],
                         target_geometry=str(geometry), target_geometry_sha256=file_sha256(geometry),
                         target_geometry_sidecar_sha256=file_sha256(geometry.with_suffix('.json')),
                         pair=pair, action=action, source_frame=source_frame, target_frame=frame, camera=camera, person_uids=uids)
            short.append(row)
            structured.append(dict(row, prompt='A and B hug. Preserve their clothing, identity and background.'))
            truths.append(truth)
        paths = {key: tmp_path/f'{name}_{key}.json' for key in ('short_generation','structured_generation','supervision')}
        for key, rows in zip(paths, [short,structured,truths]):
            write_json(paths[key],rows)
        entries[name] = dict({key:str(path) for key,path in paths.items()},sample_ids=ids)
    # Training must not even stat these deliberately absent test manifests.
    entries['test'] = dict(short_generation=str(tmp_path/'unread_test_short.json'),
                           structured_generation=str(tmp_path/'unread_test_structured.json'),
                           supervision=str(tmp_path/'unread_test_truth.json'),sample_ids=['pair00_1'])
    manifest = dict(schema_version='hi4d_dual_prompt_pilot_v1',split_manifest=str(split),
                    split_manifest_sha256=file_sha256(split),splits=entries)
    write_json(tmp_path/'dataset.json',manifest)
    return manifest


def test_training_loader_reads_only_train_val_and_preserves_inputs(tmp_path, monkeypatch):
    manifest = dataset_case(tmp_path)
    original = (tmp_path/'dataset.json').read_bytes()
    real_open = Path.open
    def checked_open(path,*args,**kwargs):
        assert 'unread_test' not in str(path), 'training touched a test manifest'
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked_open)
    data = module().load_training_data(tmp_path)
    assert set(data.splits) == {'train','val'}
    assert data.splits['train'].sample_ids == tuple(manifest['splits']['train']['sample_ids'])
    assert data.splits['val'].pair_ids == ('pair22',)
    assert (tmp_path/'dataset.json').read_bytes() == original
    data.assert_unchanged()


@pytest.mark.parametrize('damage,match', [
    ('test_as_train','train.*membership'), ('val_as_train','train.*membership'),
    ('split_sha','split.*SHA'), ('one_train','at least two'), ('duplicate_sample','sample IDs'),
    ('variant_source','identical'), ('variant_order','identical'), ('same_prompt','different prompts'),
    ('val_uid','person_uids'), ('supervision_sha','supervision')])
def test_invalid_training_bindings_are_rejected(tmp_path,damage,match):
    manifest = dataset_case(tmp_path)
    train = manifest['splits']['train']
    short = json.loads(Path(train['short_generation']).read_text())
    structured = json.loads(Path(train['structured_generation']).read_text())
    truths = json.loads(Path(train['supervision']).read_text())
    if damage in ('test_as_train','val_as_train'):
        split = Path(manifest['split_manifest'])
        payload = json.loads(split.read_text())
        payload['train_pairs'] = ['pair14']
        payload['test_pairs' if damage=='test_as_train' else 'val_pairs'].append('pair12')
        write_json(split,payload)
        from src.data.hi4d_smplx_dataset import file_sha256
        manifest['split_manifest_sha256'] = file_sha256(split)
    elif damage=='split_sha':
        manifest['split_manifest_sha256'] = '0'*64
    elif damage=='one_train':
        short=short[:1];structured=structured[:1];truths=truths[:1];train['sample_ids']=train['sample_ids'][:1]
    elif damage=='duplicate_sample':
        train['sample_ids'][1]=train['sample_ids'][0]
    elif damage=='variant_source':
        structured[0]['source_rgb'] = short[2]['source_rgb']
    elif damage=='variant_order':
        structured.reverse()
    elif damage=='same_prompt':
        structured[0]['prompt']=short[0]['prompt']
    elif damage=='val_uid':
        val=manifest['splits']['val']; rows=json.loads(Path(val['short_generation']).read_text())
        rows[0]['person_uids']=short[0]['person_uids']
        write_json(Path(val['short_generation']),rows)
        rows[0]['prompt']='Structured changed.'
        write_json(Path(val['structured_generation']),rows)
    elif damage=='supervision_sha':
        truths[0]['target_rgb_sha256']='0'*64
    write_json(Path(train['short_generation']),short)
    write_json(Path(train['structured_generation']),structured)
    write_json(Path(train['supervision']),truths)
    write_json(tmp_path/'dataset.json',manifest)
    with pytest.raises(ValueError,match=match):
        module().load_training_data(tmp_path)


def test_loaded_data_detects_later_prompt_and_source_content_changes(tmp_path):
    manifest=dataset_case(tmp_path)
    data=module().load_training_data(tmp_path)
    path=Path(manifest['splits']['train']['structured_generation'])
    rows=json.loads(path.read_text()); rows[0]['prompt']='changed prompt'
    write_json(path,rows)
    with pytest.raises(ValueError,match='changed'):
        data.assert_unchanged()


def test_loaded_data_detects_in_memory_prompt_binding_changes(tmp_path):
    dataset_case(tmp_path)
    data=module().load_training_data(tmp_path)
    data.splits['train'].rows['short'][0]['prompt']='accidentally reused another prompt'
    with pytest.raises(ValueError,match='changed'):
        data.assert_unchanged()


@pytest.mark.parametrize('change',['train_frame','train_source_action','train_camera','val_frame','missing_train'])
def test_production_loader_rejects_unapproved_rows_inside_allowed_pair_groups(tmp_path,change):
    train=list(APPROVED_TRAIN);val=list(APPROVED_VAL)
    if change=='train_frame':
        train[0]=('pair14','talk14',32,'hug14',46,4)
    elif change=='train_source_action':
        train[0]=('pair14','other14',32,'hug14',45,4)
    elif change=='train_camera':
        train[0]=('pair14','talk14',32,'hug14',45,3)
    elif change=='val_frame':
        val[0]=('pair22','game22',52,'hug22',47,4)
    elif change=='missing_train':
        train=train[:2]
    dataset_case(tmp_path,train_specs=train,val_specs=val)
    with pytest.raises(ValueError,match='approved|exactly four'):
        module().load_training_data(tmp_path)


def test_shared_schedule_and_actual_noise_are_identical_for_variants():
    p=module()
    first=p.shared_schedule(('a','b','c','d'),steps=3,seed=14062)
    second=p.shared_schedule(('a','b','c','d'),steps=3,seed=14062)
    assert first==second
    assert [item.sample_id for item in first[:4]]==['a','b','c','d']
    assert len({item.noise_seed for item in first})==12
    for left,right in zip(first,second):
        noise=torch.randn((1,8,4),generator=torch.Generator().manual_seed(left.noise_seed))
        other=torch.randn((1,8,4),generator=torch.Generator().manual_seed(right.noise_seed))
        torch.testing.assert_close(noise,other,rtol=0,atol=0)
    assert p.schedule_sha256(first)==p.schedule_sha256(second)


def test_source_prompt_cache_key_binds_prompt_source_and_versions(tmp_path):
    manifest=dataset_case(tmp_path)
    rows=json.loads(Path(manifest['splits']['train']['short_generation']).read_text())
    row=rows[0]; original=copy.deepcopy(row)
    key=module().source_prompt_cache_key(row)
    assert row==original
    assert key!=module().source_prompt_cache_key(dict(row,prompt='Structured instruction'))
    assert key!=module().source_prompt_cache_key(dict(row,source_rgb=rows[2]['source_rgb']))
    assert key==module().source_prompt_cache_key(dict(row,target_geometry='unused-target.npz'))


def test_validation_disables_autograd_clears_grads_and_preserves_weights():
    p=module(); adapter=torch.nn.Linear(2,1); backbone=torch.nn.Linear(2,1).requires_grad_(False)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.01)
    adapter(torch.ones(1,2)).sum().backward()
    before=p.state_sha256(adapter)
    observed=[]
    def evaluate(index,seed):
        observed.append((torch.is_grad_enabled(),torch.is_inference_mode_enabled(),seed))
        assert all(value.grad is None for value in adapter.parameters())
        return adapter(torch.ones(1,2)).square().mean()
    score,evidence=p.validation_pass(adapter,backbone,optimizer,evaluate,sample_count=1)
    assert score>=0 and len(observed)==3
    assert all(not grad and inference for grad,inference,_ in observed)
    assert all(value.grad is None for value in adapter.parameters())
    assert p.state_sha256(adapter)==before and adapter.training
    assert evidence['adapter_sha256_before']==evidence['adapter_sha256_after']
    assert evidence['autograd_enabled'] is False and evidence['optimizer_steps']==0


@pytest.mark.parametrize('attempt',['tensor_backward','autograd_backward','autograd_grad','optimizer_step'])
def test_validation_rejects_gradient_or_optimizer_attempts_and_restores_state(attempt):
    p=module(); adapter=torch.nn.Linear(2,1); backbone=torch.nn.Linear(2,1).requires_grad_(False)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.01)
    old_backward=torch.Tensor.backward; old_step=optimizer.step
    def evaluate(index,seed):
        x=adapter(torch.ones(1,2)).sum()
        {'tensor_backward':lambda:x.backward(), 'autograd_backward':lambda:torch.autograd.backward(x),
         'autograd_grad':lambda:torch.autograd.grad(x,tuple(adapter.parameters())),
         'optimizer_step':lambda:optimizer.step()}[attempt]()
        return x
    with pytest.raises(RuntimeError,match='validation.*forbidden'):
        p.validation_pass(adapter,backbone,optimizer,evaluate,sample_count=1)
    assert torch.Tensor.backward is old_backward and optimizer.step==old_step
    assert adapter.training and torch.is_grad_enabled()


@pytest.mark.parametrize('stale_name',['checkpoint_last.pt','training_complete.json'])
def test_fresh_output_rejects_incomplete_or_completed_stale_directories(tmp_path,stale_name):
    p=module(); out=tmp_path/'run'
    p.require_fresh_output(out)
    (out/stale_name).write_bytes(b'stale completed or incomplete run')
    with pytest.raises(ValueError,match='fresh|empty'):
        p.require_fresh_output(out)


def test_update_loop_accumulates_four_and_selects_best_from_validation(tmp_path):
    p=module(); adapter=torch.nn.Linear(1,1,bias=False); backbone=torch.nn.Linear(1,1).requires_grad_(False)
    optimizer=torch.optim.SGD(adapter.parameters(),lr=.1)
    schedule=p.shared_schedule(('a','b','c','d'),steps=2,seed=10)
    batches=[]; saved=[]
    def forward(item,scale):
        batches.append((item.step,item.sample_id,item.noise_seed,scale))
        loss=adapter(torch.ones(1,1)).square().sum()
        (loss/scale).backward()
        return loss,dict(flow=float(loss.detach()))
    def validate(index,seed):
        assert not torch.is_grad_enabled()
        return torch.tensor(5. if len(batches)==4 else 7.)
    result=p.run_variant_updates(adapter,backbone,optimizer,schedule,forward,validate,
        sample_count=1,save_every=1,on_save=lambda step,score,is_best,evidence:saved.append((step,score,is_best)),
        metrics_path=tmp_path/'metrics.jsonl')
    assert len(batches)==8 and all(batch[-1]==4 for batch in batches)
    assert saved==[(1,5.,True),(2,7.,False)]
    assert result['completed_steps']==2 and result['best_step']==1 and result['best_validation']==5.
    records=[json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records]==[1,2]
    assert all(row['validation']['optimizer_steps']==0 for row in records)


def test_fresh_warm_start_is_identical_and_export_uses_generic_loader(tmp_path):
    from src.pose_control.qwen21_full.dual.controller import DualV8Adapter
    from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint,save_checkpoint
    p=module(); config=dict(width=32,hidden_size=64,rank=8,layer_indices=[0])
    reference=DualV8Adapter(**config)
    state={}
    for name,value in reference.state_dict().items():
        if name.startswith('geometry_encoder.spatial_encoder.'):
            state[name.replace('geometry_encoder.spatial_encoder.','condition_injector.spatial_encoder.',1)]=value
        elif name.startswith('reasoner.person_reasoner.'):
            state[name]=value
        elif name=='reasoner.shared_role_offset':
            state['reasoner.role_embedding.weight']=value[None].repeat(2,1)
    warm={'step':19500,'adapter_state_dict':state}
    first,report,sha=p.fresh_adapter(7,warm,adapter_kwargs=config)
    torch.manual_seed(991); torch.rand(19)
    second,_,other=p.fresh_adapter(7,warm,adapter_kwargs=config)
    assert sha==other and p.state_sha256(first)==p.state_sha256(second)
    assert report['source_step_verified'] and report['frozen_coverage']['complete']
    for name,param in first.named_parameters():
        frozen=name.startswith(('geometry_encoder.spatial_encoder.','reasoner.person_reasoner.','reasoner.shared_role_offset'))
        assert param.requires_grad is not frozen
    path=tmp_path/'checkpoint_best.pt'
    save_checkpoint(path,first,dict(model_id='native-test',variant='short',validation_selection='true_val_split_only'),step=50)
    loaded,info,payload=load_checkpoint(path)
    assert payload['step']==50 and info['variant']=='short'
    assert p.state_sha256(loaded)==sha
    with pytest.raises(ValueError,match='19500'):
        p.fresh_adapter(7,dict(warm,step=300),adapter_kwargs=config)


def test_cached_sample_detects_tensor_or_prompt_mutation():
    from test_qwen21_dual_v8_conditions import make_input
    p=module()
    source={'prompt_embeds':torch.ones(1,2,3),'reference_latent':torch.ones(1,4,3)}
    sample=p.CachedSample('a',make_input(),source,torch.ones(1,4,3))
    sample.assert_unchanged()
    source['prompt_embeds'].zero_()  # caller-owned dict was detached from the cache
    sample.assert_unchanged()
    sample.source['prompt_embeds'].zero_()
    with pytest.raises(ValueError,match='cache.*changed'):
        sample.assert_unchanged()


def trainer():
    try:
        return importlib.import_module('scripts.train.train_qwen21_dual_prompt_pilot')
    except ModuleNotFoundError as exc:
        if exc.name=='scripts.train.train_qwen21_dual_prompt_pilot':
            pytest.fail('prompt pilot CLI is not implemented')
        raise


def test_cli_defaults_match_approved_training(tmp_path):
    args=trainer().parse_args(['--data-dir',str(tmp_path),'--output',str(tmp_path/'run')])
    assert args.steps==300 and args.variants==['short','structured']
    assert args.lr==1e-4 and args.save_every==50 and args.seed==14062
    assert '/3.singel man/qwen21_iper_body/' in args.warm_start.as_posix()


@pytest.mark.parametrize('flags',[['--steps','301'],['--steps','0'],['--lr','nan'],['--lr','0'],
                                  ['--save-every','0'],['--variants','short','short'],
                                  ['--steps','2'],['--variants','short'],['--lr','0.0002'],['--save-every','100']])
def test_cli_rejects_unapproved_training_before_loading_native(tmp_path,flags):
    with pytest.raises(ValueError):
        trainer().parse_args(['--data-dir',str(tmp_path),'--output',str(tmp_path/'run'),*flags])


def test_code_closure_includes_new_trainer_and_reused_loss_source():
    trainer()
    project=Path(__file__).resolve().parents[1]
    closure=module().prompt_pilot_code_sha256(project)
    for path in ('src/pose_control/qwen21_full/dual/prompt_pilot.py',
                 'scripts/train/train_qwen21_dual_prompt_pilot.py',
                 'scripts/train/train_qwen21_dual_v8_pilot.py'):
        assert path in closure and len(closure[path])==64


def test_cli_complete_lifecycle_exports_actual_best_and_atomic_completion(tmp_path,monkeypatch):
    """Keep the real data gates/optimizer/checkpoint I/O; substitute GPU encoders."""
    from dataclasses import replace
    from test_qwen21_dual_v8_conditions import make_input
    from src.pose_control.qwen21_full.dual.controller import DualV8Adapter
    from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint
    p=module(); cli=trainer()
    dataset_case(tmp_path)
    config=dict(width=32,hidden_size=64,rank=8,layer_indices=[0])
    reference=DualV8Adapter(**config); body={}
    for name,value in reference.state_dict().items():
        if name.startswith('geometry_encoder.spatial_encoder.'):
            body[name.replace('geometry_encoder.spatial_encoder.','condition_injector.spatial_encoder.',1)]=value
        elif name.startswith('reasoner.person_reasoner.'):
            body[name]=value
        elif name=='reasoner.shared_role_offset':
            body['reasoner.role_embedding.weight']=value[None].repeat(2,1)
    warm=tmp_path/'body.pt';torch.save({'step':19500,'adapter_state_dict':body},warm)
    model=tmp_path/'model';(model/'transformer').mkdir(parents=True)
    write_json(model/'transformer/config.json',{'fixture':True})
    class Native(torch.nn.Linear):
        def enable_gradient_checkpointing(self):
            self.checkpointing=True
    class Pipeline:
        def __init__(self):
            self.transformer=Native(2,1).requires_grad_(False).eval()
            self.vae=torch.nn.Linear(2,1).requires_grad_(False).eval()
            self.text_encoder=torch.nn.Linear(2,1).requires_grad_(False).eval()
    pipe=Pipeline();loads=[];observed=[]
    def load_native(path):
        loads.append(path)
        return pipe
    def caches(pipe,data,output,variants):
        values={};evidence={}
        for variant in variants:
            values[variant]={};evidence[variant]={}
            for name,split in data.splits.items():
                rows=[]
                for index,sid in enumerate(split.sample_ids):
                    generation=replace(make_input(),prompts=(split.rows[variant][index]['prompt'],))
                    rows.append(p.CachedSample(sid,generation,{'sample':torch.tensor([index+1.])},torch.ones(1,4,3)))
                values[variant][name]=rows;evidence[variant][name]=[]
        return values,evidence
    def loss_forward(pipe,adapter,generation,source,clean,generator,*,backward_scale=None,teacher_weight=.1,budget_weight=.01):
        prediction=adapter.query_projection(torch.full((1,64),float(source['sample'][0])))
        loss=prediction.square().mean()
        observed.append(dict(prompt=generation.prompts[0],grad=torch.is_grad_enabled(),seed=generator.initial_seed(),
                             scale=backward_scale,teacher=teacher_weight))
        if backward_scale is not None:
            (loss/backward_scale).backward()
        return loss,{'flow':float(loss.detach())}
    monkeypatch.setattr(cli,'load_native',load_native)
    monkeypatch.setattr(cli,'cache_training_samples',caches)
    monkeypatch.setattr(cli,'fresh_adapter',lambda seed,path,device:p.fresh_adapter(seed,path,device='cpu',adapter_kwargs=config))
    monkeypatch.setattr(cli,'zero_initialization_gate',lambda *args:{'zero_init_exact':True,'optimizer_steps':0})
    monkeypatch.setattr(cli,'loss_forward',loss_forward)
    monkeypatch.setattr(cli,'version',lambda package:'fixture')
    # Other agents can edit their own package files while this local unit test runs.
    monkeypatch.setattr(cli,'prompt_pilot_code_sha256',lambda project:{'fixture':'f'*64})
    out=tmp_path/'run'
    args=cli.parse_args(['--data-dir',str(tmp_path),'--output',str(out),'--model',str(model),'--warm-start',str(warm)])
    args.steps=2;args.save_every=1
    result=cli.run_training(args,smoke=True)
    assert len(loads)==1 and result['status']=='smoke_success' and result['completed_steps']==2
    assert result['schema_version']=='qwen21_dual_prompt_smoke_complete_v1'
    assert result['initialization_sha256_equal'] and result['backbone_unchanged']
    assert result['backbone_sha256_before']==result['backbone_sha256_after']
    assert result['variants']['short']['initial_adapter_sha256']==result['variants']['structured']['initial_adapter_sha256']
    for variant in ('short','structured'):
        entry=result['variants'][variant]
        best=min(entry['validation_records'],key=lambda row:row['loss'])
        assert entry['best_step']==best['step'] and entry['best_validation']==best['loss']
        record=entry['best_checkpoint']
        from src.data.hi4d_smplx_dataset import file_sha256
        assert file_sha256(record['path'])==record['sha256']
        loaded,info,payload=load_checkpoint(record['path'])
        assert payload['step']==record['step']==entry['best_step']
        assert info['validation_selection']=='true_val_split_only'
        assert info['run_contract']['data_content_sha256']
        assert info['production_training'] is False
        assert info['prompt_pilot_schema_version']=='qwen21_dual_prompt_smoke_v1'
        assert json.loads((out/variant/'smoke_complete.json').read_text())==entry
        assert not (out/variant/'training_complete.json').exists()
        del loaded
    assert json.loads((out/'smoke_complete.json').read_text())==result
    assert not (out/'training_complete.json').exists()
    train=[row for row in observed if row['scale'] is not None]
    assert len(train)==16 and all(row['scale']==4 and row['teacher']==.1 for row in train)
    assert [row['seed'] for row in train[:8]]==[row['seed'] for row in train[8:]]
    assert all(not row['grad'] for row in observed if row['scale'] is None)
    with pytest.raises(ValueError,match='fresh'):
        cli.main(['--data-dir',str(tmp_path),'--output',str(out),'--model',str(model),'--warm-start',str(warm)])
    assert len(loads)==1
