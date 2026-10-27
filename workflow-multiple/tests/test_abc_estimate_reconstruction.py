"""CPU real-model regressions; skip only when the licensed local model is absent."""
from pathlib import Path
import os
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retarget.core import decode_full53,pelvis_translation,mix_parameters

MODEL=Path(os.environ.get('ABC_TEST_MODEL_FILE',str(Path(__file__).resolve().parents[2]/
    'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz')))


@unittest.skipUnless(MODEL.is_file(),'requires existing licensed SMPL-X model')
class Full53ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import smplx
        import torch
        cls.torch=torch;torch.set_num_threads(4)
        cls.flat=smplx.SMPLX(str(MODEL),use_pca=False,flat_hand_mean=True,use_face_contour=True).eval()
        cls.mean=smplx.SMPLX(str(MODEL),use_pca=False,flat_hand_mean=False,use_face_contour=True).eval()

    def parameters(self):
        rng=np.random.default_rng(7)
        rot=rng.normal(0,.1,(53,3)).astype(np.float32)
        rot[0]=np.array([.25,-.3,.15])
        return decode_full53(rot,rng.normal(0,.1,10),rng.normal(0,.1,10))

    def output(self,model,p):
        torch=self.torch
        with torch.inference_mode():return model(**{k:torch.from_numpy(np.asarray(v,np.float32).reshape(1,-1)) for k,v in p.items()})

    def test_pelvis_translation_reproduces_upstream_head_centred_transform(self):
        from smplx.lbs import batch_rodrigues
        params=self.parameters();base={k:v.copy() for k,v in params.items()}
        base['global_orient']=np.zeros(3,np.float32);base['transl']=np.zeros(3,np.float32)
        unrotated=self.output(self.flat,base)
        R=batch_rodrigues(self.torch.from_numpy(params['global_orient'][None]))[0].numpy()
        points=unrotated.joints[0].numpy();vertices=unrotated.vertices[0].numpy();pelvis=points[0]
        rotated_joints=(points-pelvis)@R.T;rotated_vertices=(vertices-pelvis)@R.T
        primary=np.array([.1,-.1,4.],np.float32)
        expected_vertices=rotated_vertices-rotated_joints[15]+primary
        expected_joints=rotated_joints-rotated_joints[15]+primary
        params['transl']=pelvis_translation(expected_joints[0],pelvis)
        result=self.output(self.flat,params)
        np.testing.assert_allclose(result.vertices[0].numpy(),expected_vertices,rtol=0,atol=1e-5)
        # Upstream exports127 static joints. The extra17 contour landmarks are
        # selected from head yaw after global rotation and must be recomputed.
        np.testing.assert_allclose(result.joints[0,:127].numpy(),expected_joints[:127],rtol=0,atol=1e-5)

    def test_subtracting_source_hand_mean_preserves_all_driver_fingers(self):
        params=self.parameters();params['transl']=np.array([0,0,4.],np.float32)
        mixed=mix_parameters(params,params,self.mean.left_hand_mean.numpy(),self.mean.right_hand_mean.numpy())
        expected=self.output(self.flat,params);actual=self.output(self.mean,mixed)
        np.testing.assert_allclose(actual.vertices[0].numpy(),expected.vertices[0].numpy(),rtol=0,atol=1e-5)
        np.testing.assert_allclose(actual.joints[0].numpy(),expected.joints[0].numpy(),rtol=0,atol=1e-5)


if __name__=='__main__':unittest.main()
