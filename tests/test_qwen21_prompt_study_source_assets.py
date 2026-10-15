import json
from pathlib import Path
import pytest
from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file


def fixture(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.source_assets import source_asset_paths, add_source_asset_hashes
    assets = tmp_path/'assets'
    row = dict(pair_id='one', sequence_id='seq', source_appearance='seq', target_appearance='seq', source_stem='1', target_stem='2')
    for key in ('source_rgb','target_rgb','source_person_mask','target_person_mask','target_depth_vis','target_dwpose','target_dwpose_json'):
        row[key] = str(tmp_path/f'{key}.blob')
    row['target_normal'] = str(assets/'seq/normal/2.png')
    for p in source_asset_paths(row).values():
        p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'original')
    records = add_source_asset_hashes([{'pair_id':'one','split':'train'}], [('train',row)])
    p=tmp_path/'assets.jsonl';p.write_text(json.dumps(records[0])+'\n')
    lock={'source_assets_manifest':{'path':'assets.jsonl','sha256':sha256_file(p),'count':1}}
    return row,lock


def test_complete_actual_source_assets_are_bound(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.source_assets import validate_source_asset_manifest
    row,lock=fixture(tmp_path)
    validate_source_asset_manifest(tmp_path,lock,{'train':[row]},verify_files=True)


@pytest.mark.parametrize('field',['source_person_mask','target_person_mask','target_normal','target_part','target_v6_conditions','target_smplestx_params'])
def test_changed_geometry_or_mask_is_rejected(tmp_path,field):
    from src.pose_control.qwen21_full.prompt_study.source_assets import source_asset_paths,validate_source_asset_manifest
    row,lock=fixture(tmp_path)
    source_asset_paths(row)[field].write_bytes(b'changed')
    with pytest.raises(ValueError,match='source asset'):
        validate_source_asset_manifest(tmp_path,lock,{'train':[row]},verify_files=True)


def test_changed_asset_manifest_cannot_rebind_files(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.source_assets import validate_source_asset_manifest
    row,lock=fixture(tmp_path)
    (tmp_path/'assets.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='manifest'):
        validate_source_asset_manifest(tmp_path,lock,{'train':[row]},verify_files=False)


def test_source_asset_record_cannot_omit_a_condition(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.source_assets import validate_source_asset_manifest
    row,lock=fixture(tmp_path)
    p=tmp_path/'assets.jsonl';record=json.loads(p.read_text());del record['source_files']['target_v6_conditions']
    p.write_text(json.dumps(record)+'\n');lock['source_assets_manifest']['sha256']=sha256_file(p)
    with pytest.raises(ValueError,match='coverage'):
        validate_source_asset_manifest(tmp_path,lock,{'train':[row]},verify_files=False)
