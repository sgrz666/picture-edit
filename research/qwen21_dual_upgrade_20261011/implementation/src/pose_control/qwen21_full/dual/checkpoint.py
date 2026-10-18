"""Strict dual_v8 checkpoints and scoped import of the step19500 body baseline."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PosixPath
import os
import tempfile
import torch

from .controller import DualV8Adapter
from .schema import (GEOMETRY_SCHEMA_VERSION, SOURCE_SCHEMA_VERSION,
    GEOMETRY_COORDINATE_CONVENTION, ROTATION6D_CONVENTION,
    SMPLX14_PARTITION_SHA256, GEOMETRY_ENCODER_VERSION, APPEARANCE_ENCODER_VERSION)

BODY_WARM_START = "/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body/full_20k/checkpoint_best.pt"


def encoder_provenance():
    return dict(source_version=SOURCE_SCHEMA_VERSION,geometry_version=GEOMETRY_SCHEMA_VERSION,
        geometry_encoder_version=GEOMETRY_ENCODER_VERSION,appearance_encoder_version=APPEARANCE_ENCODER_VERSION,
        partition_sha256=SMPLX14_PARTITION_SHA256,coordinate_convention=GEOMETRY_COORDINATE_CONVENTION,
        rotation6d_convention=ROTATION6D_CONVENTION)


def _load_body_checkpoint(path):
    # The verified old body checkpoint stores a run directory as PosixPath.
    # Restrict this allowance to its safe load; never use unrestricted pickle.
    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path,map_location="cpu",weights_only=True,mmap=True)


def _frozen_coverage(expected,updates,required):
    scopes = {}
    for scope in ("geometry_encoder.spatial_encoder.","reasoner.person_reasoner.",
                  "reasoner.shared_role_offset"):
        names = sorted(name for name in expected if name.startswith(scope))
        matched = [name for name in names if name in updates]
        missing = [name for name in names if name not in updates]
        scopes[scope] = dict(expected_tensor_count=len(names),matched_tensor_count=len(matched),
            expected_numel=sum(expected[name].numel() for name in names),
            matched_numel=sum(expected[name].numel() for name in matched),
            missing=missing,complete=not missing)
    missing = [name for item in scopes.values() for name in item['missing']]
    return dict(required=required,complete=not missing,missing=missing,scopes=scopes,
        expected_tensor_count=sum(item['expected_tensor_count'] for item in scopes.values()),
        matched_tensor_count=sum(item['matched_tensor_count'] for item in scopes.values()),
        expected_numel=sum(item['expected_numel'] for item in scopes.values()),
        matched_numel=sum(item['matched_numel'] for item in scopes.values()))


def warm_start_from_body(adapter,checkpoint=BODY_WARM_START,*,copy_shared_role=True,
                         require_frozen_coverage=False,expected_source_step=None):
    """Validate the entire requested import before changing any adapter tensor.

    Training that freezes the imported geometry must require complete coverage
    and explicitly verify the expected trained source step (19500 for the pilot).
    The default partial import remains useful for isolated compatibility tests.
    """
    if type(require_frozen_coverage) is not bool:
        raise ValueError('require_frozen_coverage must be bool')
    if expected_source_step is not None and (type(expected_source_step) is not int or expected_source_step<0):
        raise ValueError('expected_source_step must be a nonnegative integer')
    payload = _load_body_checkpoint(checkpoint) if isinstance(checkpoint,(str,Path)) else checkpoint
    if not isinstance(payload,Mapping):
        raise ValueError("warm start checkpoint must be a mapping")
    state = payload.get("adapter_state_dict",payload)
    if not isinstance(state,Mapping):
        raise ValueError("body adapter_state_dict must be a mapping")
    expected = adapter.state_dict()
    report = {"matched":[],"skipped":[],"source_step":payload.get("step",payload.get("global_step")),
              "source_path":str(checkpoint) if isinstance(checkpoint,(str,Path)) else None,
              "expected_source_step":expected_source_step,"source_step_verified":False}
    updates = {}
    for name,value in state.items():
        target = None
        candidate = value
        if not isinstance(name,str):
            raise ValueError('body adapter_state_dict keys must be strings')
        if name.startswith("condition_injector.spatial_encoder."):
            target = name.replace("condition_injector.spatial_encoder.","geometry_encoder.spatial_encoder.",1)
        elif name.startswith("reasoner.person_reasoner."):
            target = name
        elif name == "reasoner.role_embedding.weight" and copy_shared_role and torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 2:
            target = "reasoner.shared_role_offset"
            candidate = value[0]
        entry = {"source":name,"target":target,"source_shape":list(value.shape) if torch.is_tensor(value) else None,
                 "target_shape":list(expected[target].shape) if target in expected else None}
        if target not in expected:
            entry["reason"] = "outside allowed warm start scopes" if target is None else "missing destination"
            report["skipped"].append(entry)
        elif not torch.is_tensor(candidate) or candidate.shape != expected[target].shape:
            entry["reason"] = "shape mismatch"
            report["skipped"].append(entry)
        elif not torch.isfinite(candidate).all():
            entry["reason"] = "nonfinite tensor"
            report["skipped"].append(entry)
        else:
            converted = candidate.to(expected[target])
            if not torch.isfinite(converted).all():
                entry["reason"] = "nonfinite tensor after dtype conversion"
                report["skipped"].append(entry)
                continue
            updates[target] = converted
            report["matched"].append(entry)
    report['frozen_coverage'] = _frozen_coverage(expected,updates,require_frozen_coverage)
    if expected_source_step is not None:
        if type(report['source_step']) is not int or report['source_step']!=expected_source_step:
            raise ValueError(f"warm start source step must be {expected_source_step}, got {report['source_step']!r}")
        report['source_step_verified'] = True
    if require_frozen_coverage and not report['frozen_coverage']['complete']:
        missing = report['frozen_coverage']['missing']
        reasons = {entry['target']:entry['reason'] for entry in report['skipped'] if entry['target'] in missing}
        details = '; '.join(f"{name}: {reasons.get(name,'missing source tensor')}" for name in missing)
        raise ValueError(f"warm start frozen geometry coverage is incomplete: {details}")
    adapter.load_state_dict(updates,strict=False)
    adapter.warm_start_report = report
    return report


def save_checkpoint(path,adapter,metadata,optimizer=None,step=0):
    if not isinstance(adapter,DualV8Adapter) or not isinstance(metadata,Mapping):
        raise ValueError("saving requires a DualV8Adapter and metadata mapping")
    if metadata.get("architecture_version", "dual_v8") != "dual_v8":
        raise ValueError("old weights cannot be labeled as dual_v8")
    if not isinstance(metadata.get("model_id"),str) or not metadata["model_id"]:
        raise ValueError("checkpoint requires the native Qwen model_id")
    config = adapter.config
    for name,value in (("config",config),("layer_indices",list(adapter.layer_indices))):
        if name in metadata and metadata[name] != value:
            raise ValueError(f"checkpoint {name} disagrees with the adapter")
    provenance=encoder_provenance()
    for key,value in provenance.items():
        if key in metadata and metadata[key]!=value:
            raise ValueError(f"checkpoint provenance disagrees: {key}")
    loading_report=getattr(adapter,'warm_start_report',None)
    if loading_report is None:
        loading_report=metadata.get('loading_report',dict(matched=[],skipped=[],source_step=None,source_path=None))
    if not isinstance(loading_report,Mapping) or not {'matched','skipped','source_step','source_path'}<=set(loading_report):
        raise ValueError('checkpoint loading_report provenance is incomplete')
    info = dict(metadata,architecture_version="dual_v8",config=config,
                layer_indices=list(adapter.layer_indices),loading_report=dict(loading_report),**provenance)
    payload = {"metadata":info,"adapter_state_dict":{name:value.detach().cpu() for name,value in adapter.state_dict().items()},"step":int(step)}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # A failed save must not destroy the last valid pilot checkpoint.
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        pending=Path(stream.name)
    try:
        torch.save(payload,pending)
        os.replace(pending,path)
    finally:
        pending.unlink(missing_ok=True)
    return payload


def load_checkpoint(path,device="cpu"):
    payload = torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(payload,Mapping) or not isinstance(payload.get("metadata"),Mapping):
        raise ValueError("dual_v8 checkpoint requires metadata")
    metadata = dict(payload["metadata"])
    if metadata.get("architecture_version") != "dual_v8":
        raise ValueError("checkpoint architecture must be dual_v8")
    config = metadata.get("config")
    if not isinstance(config,Mapping) or set(config) != {"width","hidden_size","rank","layer_indices"}:
        raise ValueError("dual_v8 checkpoint config is incomplete or unexpected")
    if not isinstance(metadata.get("model_id"),str) or not metadata["model_id"]:
        raise ValueError("checkpoint native model_id is missing")
    if metadata.get("layer_indices") != config["layer_indices"]:
        raise ValueError("checkpoint layer metadata disagrees with config")
    for key,value in encoder_provenance().items():
        if metadata.get(key)!=value:
            raise ValueError(f"unsupported or missing checkpoint provenance: {key}")
    report=metadata.get('loading_report')
    if not isinstance(report,Mapping) or not {'matched','skipped','source_step','source_path'}<=set(report):
        raise ValueError('missing checkpoint loading_report provenance')
    adapter = DualV8Adapter(**config)
    state = payload.get("adapter_state_dict")
    if not isinstance(state,Mapping):
        raise ValueError("checkpoint adapter_state_dict is missing")
    expected = adapter.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint adapter_state_dict keys do not match dual_v8")
    for name,value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint tensor {name}")
    dtype = state["query_projection.weight"].dtype
    adapter = adapter.to(dtype=dtype)
    adapter.load_state_dict(state,strict=True)
    adapter.warm_start_report=dict(report)
    adapter.to(device=device)
    return adapter,metadata,payload
