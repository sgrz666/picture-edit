"""Native inference conditions must preserve actual geometry and source roles."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_native_target_canvas_does_not_apply_source_letterbox_twice():
    from abc_native_assets import target_canvas
    from src.data.canvas_contract import aspect_preserving
    source=aspect_preserving((1000,600),max_pixels=512**2,pad_multiple=32)
    scene={'camera':{'canvas':source.metadata()}}
    native=target_canvas(scene)
    assert native.original_hw==source.canvas_hw
    np.testing.assert_allclose(native.transform.numpy(),np.eye(3))

def test_source_reference_job_cannot_read_driver_and_keeps_missing_hands_invalid():
    from abc_native_assets import source_reference_job
    from src.data.canvas_contract import native_pad
    spec=native_pad((64,64))
    source=dict(person_id='source_A',image='/source.png',canvas=spec.metadata(),
        params_npz='/source_geometry.npz')
    points=np.zeros((134,2),np.float32);scores=np.zeros(134,np.float32)
    job=source_reference_job(source,{'points':points,'scores':scores},
        np.array([[.1,.1,.4,.4],[.2,.5,.3,.6],[.6,.5,.7,.6]],np.float32),
        {'arcface_model':'/arc.onnx','native_dino_model':'/dino'},hash_file=lambda p:'sha')
    assert job['role']=='source_only' and 'target_rgb' not in job and 'driver_rgb' not in job
    assert job['source_person_ids']==['source_A','__inactive__']
    assert job['references'][0]['people'][0]['hand_texture_valid']==[False,False]
    assert not job['references'][0]['people'][0]['identity_valid']
