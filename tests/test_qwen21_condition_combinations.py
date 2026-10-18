"""Supplement contracts protect attribution, reference routing and resumption."""
from argparse import Namespace
import hashlib
import json

from PIL import Image
import pytest

from scripts.eval import eval_qwen21_native_value as runner
from scripts.eval.qwen21_condition_combinations import (COMBINATION_METHODS, COMBINATION_REFERENCES,
    NEW_METHODS, combination_prompt, factor_edges, require_protocol_outputs)
from scripts.eval.qwen21_native_value_prompts import build_prompt


def test_each_reference_keeps_its_image_role_and_order(tmp_path):
    names = ('A','B','B_skeleton','B_normal','B_depth')
    images={name:Image.new('RGB',(32,32),(n*40,0,0)) for n,name in enumerate(names)}
    paths={name:str(tmp_path/(name+'.png')) for name in names}
    for method in COMBINATION_METHODS:
        actual,records=runner.references_for_method(method,images,paths)
        recipe=COMBINATION_REFERENCES[method]
        assert actual==[images[x] for x in recipe]
        assert [r['role'] for r in records]==list(recipe)
        assert [r['index'] for r in records]==list(range(1,len(recipe)+1))
    with pytest.raises(ValueError,match='Missing locked reference'):
        runner.references_for_method(NEW_METHODS[-1],{k:v for k,v in images.items() if k!='B_depth'},paths)
    assert len(COMBINATIONS := COMBINATION_REFERENCES[NEW_METHODS[-1]])==5


def test_prompt_only_changes_roles_and_numbers_all_inputs():
    row={'action_description':'Raise both arms and tilt the head right.',
         'source_preserve_description':'A wears glasses and a grey shirt.'}
    base=build_prompt(row,'native_skeleton')
    assert combination_prompt(base,'native_skeleton')==base
    for method in NEW_METHODS:
        prompt=build_prompt(row,method)
        assert prompt.split('\n\n')[1:]==base.split('\n\n')[1:]
        for i,_ in enumerate(COMBINATION_REFERENCES[method]):assert f'Image {i+1} is ' in prompt
        assert f'Image {len(COMBINATION_REFERENCES[method])+1} is ' not in prompt


def test_legacy_defaults_are_not_expanded_by_new_methods(monkeypatch):
    monkeypatch.setattr('sys.argv',['runner','--pairs','pairs','--prompts','prompts','--output','out'])
    assert runner.arguments().methods==list(runner.LEGACY_METHODS[:-1])
    assert not set(runner.arguments().methods)&set(NEW_METHODS)


def test_factor_pairs_change_one_condition_only():
    edges=factor_edges()
    assert len(edges)==12
    for role,left,right in edges:
        assert set(COMBINATION_REFERENCES[right])-set(COMBINATION_REFERENCES[left])=={role}
        assert set(COMBINATION_REFERENCES[left])<=set(COMBINATION_REFERENCES[right])


def test_resume_rejects_prompt_or_reference_changes(tmp_path):
    images={name:Image.new('RGB',(32,32),'red') for name in ('A','B_skeleton')}
    paths={name:str(tmp_path/(name+'.png')) for name in images}
    for name,img in images.items():img.save(paths[name])
    _,refs=runner.references_for_method('native_skeleton',images,paths)
    refs=[{**r,'sha256':runner.sha256(r['path'])} for r in refs]
    prompt='Keep A; match B skeleton. '*20
    args=Namespace(resolution=512,steps=30,seed=42,prompt_variant=None)
    saved=dict(method='native_skeleton',resolution=512,steps=30,seed=42,prompt_variant=None,
               true_cfg_scale=1.,use_kv_cache=False,prompt=prompt,
               prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),references=refs,lora_loaded=False)
    prepared=(images,paths,{}, {})
    runner.validate_completed_record(saved,args,prompt,prepared)
    with pytest.raises(ValueError,match='prompt differs'):
        runner.validate_completed_record(saved,args,prompt+'different',prepared)
    images['B_skeleton']=Image.new('RGB',(32,32),'blue');images['B_skeleton'].save(paths['B_skeleton'])
    with pytest.raises(ValueError,match='reference bytes'):
        runner.validate_completed_record(saved,args,prompt,prepared)


def test_protocol_requires_correct_resolution_and_steps(tmp_path):
    row=dict(pair_id='case',method='native_ab_skeleton',resolution=1024,steps=30)
    path=tmp_path/'protocol.json'
    path.write_text(json.dumps({'architecture':'native_condition_combinations_v1','expected_outputs':[row]}))
    require_protocol_outputs(path,[row])
    with pytest.raises(ValueError,match='coverage mismatch'):
        require_protocol_outputs(path,[{**row,'resolution':512}])
    with pytest.raises(ValueError,match='Duplicate output'):
        require_protocol_outputs(path,[row,row])


def test_native_common_faces_are_not_lost_when_adapter_is_unevaluable():
    from scripts.eval.score_qwen21_native_value import METRICS, summarize
    def fake(method,reliable):
        metrics={m:.3 for m in METRICS};metrics['pose_detected']=True
        selection={'reviewed':True,'reliable':reliable,'formal_reliable':reliable}
        return dict(pair_id='case',dataset='iper',resolution=512,steps=30,method=method,
            metrics=metrics,face_detection={'generated':{'detected':True,'automatic_selected_index':0}},
            face_selection={'source':selection,'generated':selection},
            individually_reliable_identity_sample=reliable,formal_reliable_identity_sample=reliable,
            common_reliable_identity_sample=False,formal_common_reliable_identity_sample=False,
            seconds=20.,peak_cuda_gib=32.)
    rows=[fake(m,True) for m in COMBINATION_METHODS]+[fake('full_body',False)]
    summary=summarize(rows);key='iper_512_steps30'
    assert summary['groups'][key]['native_skeleton']['identity_a_cosine_formal_common']['n']==0
    assert summary['native_combination_groups'][key]['native_skeleton']['identity_a_cosine_formal_common']['n']==1
    assert len(summary['combination_factor_differences'][key])==12
    assert summary['combination_vs_adapters'][key]['full_body_minus_native_skeleton']['metrics']['identity_a_cosine_reliable']['n']==0
