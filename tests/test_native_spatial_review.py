"""Review sampling covers videos without reusing earlier examples."""
from collections import Counter
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import pytest


def module():
    default=Path(__file__).resolve().parents[1]/'experiments/native-spatial-supervision/scripts/hand_review_batch.py'
    path=Path(os.environ.get('NATIVE_SPATIAL_REVIEW_SCRIPT',str(default)))
    spec=importlib.util.spec_from_file_location('spatial_review_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def rows():
    return [dict(frame_id=f'{clip}_{i}',clip_id=str(clip),original_hw=[160,96],
                 candidate=[[10.,20.]]*134,confidence=[.9]*134,
                 hand_a_status='occluded' if i==2 else 'visible_clear',hand_b_status='visible_blurry',
                 hand_a_present=i!=2,hand_b_present=True,
                 hand_block_slices={'hand_a':[113,134],'hand_b':[92,113]},
                 loss_mask={'hand_a':i==1,'hand_b':False},hand_raw_pose_usable={'hand_a':i==1,'hand_b':False})
            for clip in range(49) for i in range(4)]


def test_100_new_frames_cover_all_49_videos_and_keep_inputs():
    m=module();source=rows();before=deepcopy(source);excluded={f'{i}_0' for i in range(24)}
    selected=m.select_review_frames(source,excluded,100,20261107)
    counts=Counter(r['clip_id'] for r in selected)
    assert len(selected)==len({r['frame_id'] for r in selected})==100
    assert not excluded&{r['frame_id'] for r in selected}
    assert len(counts)==49 and min(counts.values())>=2
    assert any(r['hand_a_status']=='occluded' for r in selected)
    assert selected==m.select_review_frames(source,excluded,100,20261107) and source==before


def test_sampler_refuses_impossible_size():
    with pytest.raises(ValueError,match='count|Count'):
        module().select_review_frames(rows(),set(),300,42)


def test_gallery_links_follow_actual_comparison_directory(tmp_path):
    import json,re
    from types import SimpleNamespace
    from urllib.parse import unquote
    from PIL import Image
    from src.data.canvas_contract import native_pad
    m=module();f=rows()[0];f['original_hw']=[32,32]
    frames=tmp_path/'frames.jsonl';frames.write_text(json.dumps(f)+'\n',encoding='utf8')
    index=tmp_path/'candidate.json';index.write_text(json.dumps({'entries':{'key':{'metadata':{'frame_id':f['frame_id'],'canvas':native_pad((32,32)).metadata()}}}}),encoding='utf8')
    folder=tmp_path/'comparisons verified';folder.mkdir();preview=folder/(f['frame_id']+'.jpg')
    Image.new('RGB',(32*8,32+44),'gray').save(preview)
    hand=dict(old_enabled=False,new_enabled=False,coarse={'enabled':False,'reason':'hand_not_present'},fine={'enabled':False},
              old_isolated_latent_increment_max=0.,new_isolated_latent_increment_max=0.)
    comparison=tmp_path/'comparison.json';comparison.write_text(json.dumps({'candidate':m.ref(index),
        'cases':[{'frame_id':f['frame_id'],'preview':m.ref(preview),'hands':{'hand_a':hand,'hand_b':hand}}]}),encoding='utf8')
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'frames':m.ref(frames),'negative_cases':[]}),encoding='utf8')
    output=tmp_path/'gallery'
    m.gallery(SimpleNamespace(selection=selection,comparison=comparison,output_root=output))
    document=(output/'index.html').read_text(encoding='utf8')
    data=json.loads(re.search(r'const cases=(.*?);let page',document).group(1))
    assert (output/unquote(data[0]['image'])).resolve()==preview.resolve()
    assert '%20' in data[0]['image']
