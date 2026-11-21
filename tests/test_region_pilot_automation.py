"""The dependency wait must survive a busy GPU without allocating CUDA."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def waiting():
    path=Path(__file__).parents[1]/'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/waiting.py'
    spec=importlib.util.spec_from_file_location('xhs_waiting',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_process_wait_distinguishes_zombie_and_pid_reuse(tmp_path):
    m=waiting();proc=tmp_path/'proc';(proc/'sys/kernel/random').mkdir(parents=True)
    (proc/'sys/kernel/random/boot_id').write_text('boot')
    p=proc/'123';p.mkdir();(p/'cmdline').write_bytes(b'python\0train.py\0')
    fields=['S','1']+['0']*17+['456']
    (p/'stat').write_text('123 (trainer with ) name) '+' '.join(fields))
    first=m.process_snapshot(123,proc)
    assert first['start_ticks']=='456' and m.dependency_active(first,first)
    assert not m.dependency_active(first,{**first,'start_ticks':'457'})
    assert not m.dependency_active(first,{**first,'state':'Z'})
    assert not m.dependency_active(first,None)
    assert m.process_snapshot(999,proc) is None


def test_wait_retries_read_errors_and_only_finishes_after_exit():
    m=waiting();expected={'pid':123,'boot_id':'boot','start_ticks':'4','state':'S'}
    values=iter([PermissionError('transient'),expected,None]);events=[];sleeps=[]
    def snapshot(pid):
        value=next(values)
        if isinstance(value,Exception):raise value
        return value
    m.wait_for_dependency(expected,lambda kind,**kw:events.append(kind),snapshot=snapshot,sleep=sleeps.append,interval=1)
    assert events[-1]=='dependency_finished' and 'dependency_probe_retry' in events
    assert len(sleeps)==2


def test_nvml_probe_is_context_free_and_transient_failure_is_retryable():
    m=waiting();commands=[]
    def run(command,**kwargs):
        commands.append(command);return SimpleNamespace(returncode=0,stdout='65536, 85040\n',stderr='')
    assert m.query_gpu_memory({},run=run)['free_bytes']==64*2**30
    assert commands[0][0]=='nvidia-smi' and not any('torch' in x for x in commands[0])
    assert m.query_gpu_memory({},run=lambda *a,**kw:SimpleNamespace(returncode=1,stdout='',stderr='busy'))['error']=='busy'


def test_test_generation_selects_the_test_role_instead_of_validation():
    path=Path(__file__).parents[1]/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts/stage.py'
    spec=importlib.util.spec_from_file_location('region_stage_generation',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=SimpleNamespace(sources=[SimpleNamespace(source_id='val',role='validation'),SimpleNamespace(source_id='heldout',role='test')])
    assert m.generation_source_id(cfg,'test')=='heldout'
    assert m.generation_source_id(cfg,'val')=='val'


def test_test_readout_rejects_validation_rows_or_wrong_model(tmp_path):
    import hashlib,json,pytest
    path=Path(__file__).parents[1]/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts/summarize.py'
    spec=importlib.util.spec_from_file_location('region_readout_guard',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    image=tmp_path/'saved.png';image.write_bytes(b'preserved image bytes')
    labels=['initial','uniform_S3','regional_S3','continue_sft','grpo'];expected={label:label+'-sha' for label in labels}
    row={'pair_id':'test-pair','split':'test','path':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
         'target_rgb_used_as_condition':False}
    for label in labels:
        p=tmp_path/f'report/evaluation/test_{label}/metrics.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'records':[{**row,'model_sha256':expected[label]}]}))
    pairs={'test-pair':{'split':'test'},'validation-pair':{'split':'val'}}
    _,images=m.load_evaluations(tmp_path,'test',pairs,expected)
    assert set(images['initial'])=={'test-pair'}
    p=tmp_path/'report/evaluation/test_initial/metrics.json'
    p.write_text(json.dumps({'records':[{**row,'split':'val','model_sha256':expected['initial']}]}))
    with pytest.raises(ValueError,match='split'):m.load_evaluations(tmp_path,'test',pairs,expected)
    p.write_text(json.dumps({'records':[{**row,'model_sha256':'other-checkpoint'}]}))
    with pytest.raises(ValueError,match='checkpoint'):m.load_evaluations(tmp_path,'test',pairs,expected)
