import importlib.util
from pathlib import Path
import copy


def module():
    path=Path(__file__).parents[1]/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts/prepare.py'
    assert path.is_file(),'The SDPose regional preparation entry must exist'
    spec=importlib.util.spec_from_file_location('bili_region_prepare',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_selection_is_deterministic_video_balanced_and_keeps_inputs():
    m=module();rows=[{'pair_id':str(i)} for i in range(20)];groups={str(i):'video'+str(i//5) for i in range(20)}
    saved=copy.deepcopy(rows)
    a=m.select_rows(rows,groups,2,42)
    assert a==m.select_rows(rows,groups,2,42) and rows==saved
    assert len(a)==8 and all(sum(groups[x['pair_id']]==v for x in a)==2 for v in set(groups.values()))


def test_sdpose_rebinding_changes_only_pose_and_records_backend():
    m=module();old={'profile':'single_2ref','images':[{'role':'source','file':{'path':'A','sha256':'a'*64}},
        {'role':'target_pose','file':{'path':'old_pose','sha256':'b'*64},'canvas_id':'c'*64}],
        'prompt':{'text':'preserve identity','sha256':'d'*64}}
    saved=copy.deepcopy(old);new=m.rebind_condition(old,{'path':'sdpose_canvas.png','sha256':'e'*64})
    assert old==saved and new['images'][0]==old['images'][0] and new['prompt']==old['prompt']
    assert new['images'][1]['file']['sha256']=='e'*64 and new['target_pose_backend']=='sdpose'


def test_anatomical_hands_follow_explicit_wrist_mapping():
    m=module();frame={'hand_body_wrist_indices':{'hand_a':4,'hand_b':7}}
    assert m.hand_key(frame,'left')=='hand_b' and m.hand_key(frame,'right')=='hand_a'
