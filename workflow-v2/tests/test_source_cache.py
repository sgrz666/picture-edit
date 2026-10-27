"""Source-only reuse must verify artifacts and explicit A/B image identities."""
from pathlib import Path
import json
import sys
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_source_cache_reorders_by_actual_rgb_and_rejects_mutated_masks(tmp_path):
    from abc_runner import sha256
    from abc_contracts import resolve_source_cache
    run=tmp_path/'old';geometry=run/'geometry';geometry.mkdir(parents=True)
    sources=[]
    for p in range(2):
        folder=geometry/f'source_{p}';folder.mkdir()
        im=folder/'original.png';Image.new('RGB',(32,32),(p*200,10,20)).save(im)
        mask=folder/'mask.png';Image.new('L',(32,32),255).save(mask)
        params=folder/'params.npz';params.write_bytes(b'raw geometry')
        sources.append(dict(person_id='source_'+str('AB'[p]),role='AB'[p],image=str(im),canvas_image=str(im),
            mask=str(mask),mask_original=str(mask),params_npz=str(params),keypoints_npz=str(params),
            fit_checks={},model={},hand_mean_convention='flat_hand_mean=False',canvas={}))
    scene=geometry/'scene.json';scene.write_text(json.dumps({'sources':sources}))
    (geometry/'request.json').write_text(json.dumps({'settings':{'resolution':512}}))
    outputs={str(p.relative_to(geometry)):{'sha256':sha256(p)} for p in geometry.rglob('*') if p.is_file()}
    (geometry/'stage.json').write_text(json.dumps({'fingerprint':'producer','outputs':outputs}))
    records,binding=resolve_source_cache(run,{'source_a':sources[1]['image'],'source_b':sources[0]['image']},{'resolution':512})
    assert records[0]['person_id']=='source_A' and records[0]['params_npz']==sources[1]['params_npz']
    assert binding['producer_fingerprint']=='producer'
    Path(sources[1]['mask']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        resolve_source_cache(run,{'source_a':sources[1]['image'],'source_b':sources[0]['image']},{'resolution':512})
