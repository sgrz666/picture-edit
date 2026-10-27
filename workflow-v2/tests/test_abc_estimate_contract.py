"""Multi-HMR full53 export and pelvis conversion regression contracts."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.core import decode_full53, pelvis_translation


def test_full53_slot_layout_and_no_eye_slot_confusion():
    rot=np.arange(159,dtype=np.float32).reshape(53,3)
    out=decode_full53(rot,np.zeros(10),np.zeros(10))
    np.testing.assert_array_equal(out['body_pose'],rot[1:22].reshape(63))
    np.testing.assert_array_equal(out['left_hand_pose'],rot[22:37].reshape(45))
    np.testing.assert_array_equal(out['right_hand_pose'],rot[37:52].reshape(45))
    np.testing.assert_array_equal(out['jaw_pose'],rot[52])
    with pytest.raises(ValueError,match='53'):
        decode_full53(np.zeros((55,3)),np.zeros(10),np.zeros(10))


def test_translation_is_exported_camera_pelvis_minus_rest_pelvis():
    # SMPL-X root rotation leaves its rest pelvis fixed; Multi-HMR rotates around it,
    # then recentres on the model's selected person_center (head).
    np.testing.assert_allclose(pelvis_translation(np.array([1,2,3]),np.array([.1,.2,.3])),[.9,1.8,2.7])


def test_interaction_forward_requires_float32_at_the_model_boundary():
    from abc_estimate_worker import interaction_forward
    class Tensor:
        def __init__(self,dtype):self.dtype=dtype
        def float(self):return Tensor('float32')
    def model(image,**kwargs):
        assert image.dtype==kwargs['K'].dtype=='float32'
        assert kwargs['is_training'] is False
        return ['full_precision_geometry']
    assert interaction_forward(model,Tensor('float16'),Tensor('float64'),.2)==['full_precision_geometry']


def test_interaction_exif_is_normalized_before_fitting_and_role_mapping(tmp_path):
    from PIL import Image
    from abc_estimate_worker import normalize_interaction_image
    raw=tmp_path/'phone.jpg';out=tmp_path/'fit';out.mkdir()
    image=Image.new('RGB',(12,8));exif=Image.Exif();exif[274]=6
    image.save(raw,exif=exif)
    normalized=normalize_interaction_image(raw,out)
    with Image.open(normalized) as result:
        assert result.size==(8,12)
        assert result.getexif().get(274,1)==1
    assert raw.is_file()
