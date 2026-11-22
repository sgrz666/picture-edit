"""Keep paired image plans while removing inference from stage transitions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def module(monkeypatch):
    p=Path(__file__).parents[1]/'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/deferred_render.py'
    monkeypatch.syspath_prepend(str(p.parent));spec=importlib.util.spec_from_file_location('xhs_deferred_render',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_image_plan_pairs_all_seeds_and_expands_test_threefold(monkeypatch):
    m=module(monkeypatch);tasks=m.image_tasks({'val_pairs':40,'test_pairs':64})
    assert sum(t['pairs'] for t in tasks if t['split']=='test')==960
    assert sum(t['pairs'] for t in tasks if t['split']=='val')==640
    for label in m.FINAL_LABELS:
        assert {t['seed'] for t in tasks if t['label']==label and t['split']=='test'}=={42,123,2026}


def test_primary_paths_reuse_original_seed42_outputs(monkeypatch,tmp_path):
    m=module(monkeypatch)
    assert m.output_folder(tmp_path,'initial','val',42)==tmp_path/'output/generation/initial'
    assert m.output_folder(tmp_path,'regional_S3','test',42)==tmp_path/'output/generation/test/regional_S3'
    assert m.output_folder(tmp_path,'regional_S3','test',123)!=m.output_folder(tmp_path,'regional_S3','test',42)


def test_active_training_is_adopted_until_report_exists_without_relaunch(tmp_path,monkeypatch):
    p=Path(__file__).parents[1]/'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/fast_run.py'
    monkeypatch.syspath_prepend(str(p.parent));spec=importlib.util.spec_from_file_location('xhs_fast_handoff',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    report=tmp_path/'training.json';calls=[]
    def wait(process,event):calls.append(process);report.write_text('{}')
    adoption={'label':'uniform_S0','process':{'pid':123}}
    assert m.adopt_training_report(report,'uniform_S0',adoption,lambda *a,**kw:None,wait)
    assert calls==[{'pid':123}]
    assert not m.adopt_training_report(report,'uniform_S0',adoption,lambda *a,**kw:None,wait)
    assert len(calls)==1


def test_one_loaded_backend_renders_all_seeds_and_existing_images_are_reused(tmp_path,monkeypatch):
    import json,torch
    from PIL import Image
    from src.pose_control.qwen21_full.native_v7 import training
    from src.pose_control.qwen21_full.native_v62 import generation
    from src.pose_control.qwen21_full.native_v62.training.config import SourceSpec
    m=module(monkeypatch);loads=[]
    cfg=training.TrainingConfigV7(initialization={'path':str(tmp_path/'init.pt'),'sha256':'a'*64},
        sources=(SourceSpec('train',weight=1.),SourceSpec('val',role='validation'),SourceSpec('heldout',role='test')))
    def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    put(tmp_path/'prepared.json',{'val_pairs':2,'test_pairs':1})
    put(tmp_path/'configs/cache-ready/uniform/S0.json',cfg.record())
    put(tmp_path/'data/evaluation.json',[{'pair_id':'val0','split':'val'},{'pair_id':'val1','split':'val'},{'pair_id':'heldout0','split':'test'}])
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda value:None)
    monkeypatch.setattr(training,'DataPlanV7',lambda cfg:SimpleNamespace(datasets={'val':[0,1],'heldout':[0]}))
    class Backend:
        def __init__(self,*a,**kw):loads.append(1);self.runtime=SimpleNamespace(eval=lambda:None);self.pipe=None
        def pair(self,source,index):return {'pair_metadata':{'pair_id':source+str(index)},'native_scene_kind':'single'},{},{}
    monkeypatch.setattr(training,'NativeBackendV7',Backend)
    monkeypatch.setattr(generation,'generate_native',lambda *a,seed,**kw:Image.new('RGB',(16,16),(seed%255,0,0)))
    m.render(tmp_path,Path.cwd(),'initial')
    assert len(loads)==1 and len(list((tmp_path/'output').rglob('*.png')))==7
    m.render(tmp_path,Path.cwd(),'initial')
    assert len(loads)==1 and len(list((tmp_path/'output').rglob('*.png')))==7
