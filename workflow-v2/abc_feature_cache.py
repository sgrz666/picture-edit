"""Immutable, content-bound source appearance cache shared by same-run ablations."""
import hashlib
import json
from pathlib import Path
import uuid


def frozen_source_features(root,binding,build):
    from src.pose_control.qwen21_full.native_full_assets import read_record,write_record
    from src.pose_edit.common import atomic_json
    key=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    folder=Path(root)/key;manifest=folder/'index.json'
    if manifest.is_file():
        record=json.loads(manifest.read_text())
        if record['binding']!=binding:raise ValueError('Source appearance cache binding changed')
        return read_record(record['features'],schema='native_local_reference_v4')
    folder.mkdir(parents=True,exist_ok=True)
    features=build()
    # An interrupted build leaves an orphan payload, never an overwritten cache.
    ref=write_record(folder/('features_'+uuid.uuid4().hex+'.npz'),features,schema='native_local_reference_v4')
    atomic_json(manifest,dict(schema='abc_source_feature_cache_v1',binding=binding,features=ref))
    return features
