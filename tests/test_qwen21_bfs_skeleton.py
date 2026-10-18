"""Guard the fixed experiment's input inventory and resume checks."""
import hashlib
from PIL import Image
import pytest
from scripts.eval.eval_qwen21_bfs_skeleton import validate_saved, validate_native_reuse, sha256
from scripts.eval.score_qwen21_bfs_skeleton import expected_keys


def test_expected_protocol():
    pairs=[{'pair_id':f'{ds}_{i:02d}'} for ds in ('iper','tiktok') for i in range(6)]
    keys=expected_keys(pairs)
    assert len(keys)==48
    assert sum(k[1]==1024 for k in keys)==12


@pytest.mark.parametrize('field,value',[('prompt','changed'),('lora_strength',.5),('lock_sha256','changed'),
                                       ('use_kv_cache',True),('runtime_versions',{'torch':'changed'})])
def test_resume_rejects_changed_protocol(tmp_path,field,value):
    image=tmp_path/'image.png';Image.new('RGB',(512,512)).save(image)
    expected=dict(variant='bfs_body',resolution=512,prompt='fixed detailed text',steps=30,seed=42,
                  true_cfg_scale=1.,use_kv_cache=False,lora_sha256='fixed',lora_strength=1.,
                  lock_sha256='locked',runtime_versions={'torch':'same'})
    refs=[{'index':1,'role':'A','path':str(image),'sha256':sha256(image)},
          {'index':2,'role':'B_skeleton','path':str(image),'sha256':sha256(image)}]
    saved={**expected,'references':refs,'image':str(image),'image_sha256':sha256(image),
           'prompt_sha256':hashlib.sha256(expected['prompt'].encode()).hexdigest(),'control_forwards':0}
    validate_saved(saved,expected,refs)
    saved[field]=value
    with pytest.raises(ValueError):validate_saved(saved,expected,refs)


def test_native_resume_checks_reference_content_not_just_filename(tmp_path):
    image=tmp_path/'native.png';Image.new('RGB',(512,512)).save(image)
    a=tmp_path/'A.png';Image.new('RGB',(512,512),'white').save(a)
    skeleton=tmp_path/'skeleton.png';Image.new('RGB',(512,512),'black').save(skeleton)
    refs=[{'index':i+1,'role':role,'path':str(p),'sha256':sha256(p)}
          for i,(role,p) in enumerate([('A',a),('B_skeleton',skeleton)])]
    saved=dict(resolution=512,prompt='fixed',prompt_sha256=hashlib.sha256(b'fixed').hexdigest(),
               steps=30,seed=42,true_cfg_scale=1.,use_kv_cache=False,control_forwards=0,
               lora_loaded=False,target_rgb_in_native_references=False,runtime_versions={'test':'same'},
               references=refs,image=str(image),image_sha256=sha256(image))
    validate_native_reuse(saved,'fixed',refs,512,{'test':'same'})
    Image.new('RGB',(512,512),'red').save(skeleton)
    with pytest.raises(ValueError):validate_native_reuse(saved,'fixed',refs,512,{'test':'same'})
