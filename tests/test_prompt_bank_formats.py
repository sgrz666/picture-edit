import json
import pytest
from src.pose_control.qwen21_full.native_v5.training.io import file_sha
from src.pose_control.qwen21_full.native_v5.hi4d_guide_binding import bank_record,verify_grounded_hi4d,SCHEMA


def test_pretty_bilibili_json_bank_is_not_a_hi4d_record(tmp_path):
    p=tmp_path/'prompt.json';p.write_text(json.dumps({'schema':'bilibili_native_prompt_v1',
        'pair_id':'p','notes':['ordinary string','second string']},indent=2))
    row,has_hi4d=bank_record(str(p),file_sha(p),'p')
    assert row['pair_id']=='p' and has_hi4d is False
    verify_grounded_hi4d({'pair_id':'p','profile':'single_2ref',
        'prompt':{'schema':'bilibili_native_prompt_v1','bank':{'path':str(p),'sha256':file_sha(p)}}})


def test_json_list_cannot_downgrade_grounded_schema(tmp_path):
    p=tmp_path/'bank.json';p.write_text(json.dumps([{'schema':SCHEMA,'pair_id':'p'}],indent=2))
    with pytest.raises(ValueError):verify_grounded_hi4d({'pair_id':'p','profile':'single_2ref',
        'prompt':{'schema':'wrong','bank':{'path':str(p),'sha256':file_sha(p)}}})


def test_duplicate_bank_pairs_remain_rejected(tmp_path):
    p=tmp_path/'bank.json';p.write_text(json.dumps([{'schema':SCHEMA,'pair_id':'p'},{'schema':SCHEMA,'pair_id':'p'}]))
    with pytest.raises(ValueError):bank_record(str(p),file_sha(p),'p')
