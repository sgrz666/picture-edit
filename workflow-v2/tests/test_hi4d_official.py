"""Official Hi4D contacts must not be reinterpreted as SMPL-X vertex IDs."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_official_contact_indices_are_one_based_and_roles_are_remapped():
    from hi4d import ordered_contacts
    contact=np.zeros((2,3));contact[0,1]=3
    vertices=np.array([[[0,0,2],[1,0,2],[2,0,2]],[[0,1,2],[1,1,2],[2,1,2]]],np.float32)
    links,delta=ordered_contacts(contact,vertices,[1,0])
    assert links.tolist()==[[0,2,1,1]]
    np.testing.assert_allclose(delta,[[-1,-1,0]])

def test_official_body_does_not_fabricate_finger_pose():
    from retarget.core import mix_parameters
    source={k:np.ones(n,np.float32) for k,n in {'global_orient':3,'body_pose':63,'left_hand_pose':45,
        'right_hand_pose':45,'jaw_pose':3,'expression':10,'betas':10,'transl':3}.items()}
    driver={k:v*2 for k,v in source.items() if k not in ('left_hand_pose','right_hand_pose')}
    driver['hand_pose_available']=False
    result=mix_parameters(source,driver,np.ones(45)*.5,np.ones(45)*.5)
    np.testing.assert_array_equal(result['left_hand_pose'],source['left_hand_pose'])
    np.testing.assert_array_equal(result['body_pose'],driver['body_pose'])

def test_contact_attachments_use_anatomical_surface_not_raw_vertex_number():
    from hi4d import attach_contacts
    joints=np.zeros((22,3),np.float32);joints[:,2]=3
    sources=[];targets=[]
    for p in range(2):
        offset=np.array([p,0,0],np.float32)
        sources.append(dict(vertices_cam=np.array([[.2,0,3]],np.float32)+offset,
            joints_cam=joints+offset,vertex_part=np.array([4]),dominant_joint=np.array([0]),body_height=np.array(1.),
            joint_frames=np.broadcast_to(np.eye(3),(22,3,3)).copy()))
        targets.append(dict(vertices_cam=np.array([[.2,0,3],[.8,0,3],[.2,0,3]],np.float32)+offset,
            joints_cam=joints+offset,part_vertex=np.array([7,4,4])))
    mapped,deltas,diagnostic=attach_contacts(sources,targets,np.array([[0,0,1,0]]))
    assert mapped.tolist()==[[0,2,1,2]]
    assert diagnostic['source_topology']=='SMPL' and diagnostic['target_topology']=='SMPL-X'
    assert diagnostic['correspondence']=='estimated_bone_local_same_part_surface_attachment'
    # Geometric attachment uncertainty must not erase a known official positive
    # relation; it is a separate confidence field.
    shifted=[{**target,'vertices_cam':target['vertices_cam']+np.array([0.,.4,0.])} for target in targets]
    mapped,_,diagnostic=attach_contacts(sources,shifted,np.array([[0,0,1,0]]),max_error=.01)
    assert len(mapped)==1 and diagnostic['low_confidence_links']

def test_stick_contact_renderer_uses_body22_and_explicit_contacts_only():
    from render.abc_scene import stick_contact_image
    joints=np.tile(np.array([[32.,32.]]),(22,1));joints[1]=[16,48]
    image=stick_contact_image([joints],(64,64),endpoints=[np.array([[20.,20.]])])
    assert image.size==(64,64)
    assert np.asarray(image)[20,20].sum()>0

def test_official_body_reconstructs_original_weights_without_chumpy():
    import pytest
    root=Path(__file__).resolve().parents[2]
    frame=root/'datasets/Hi4D_pilot_v1/raw/pair00/hug00'
    if not (frame/'smpl/000084.npz').is_file():pytest.skip('requires licensed local Hi4D fixture')
    from abc_hi4d_worker import official_person
    from src.pose_control.qwen21_pro.geometry import _load_hi4d_smpl_gender_backend
    with np.load(frame/'smpl/000084.npz',allow_pickle=False) as f:params=dict(f)
    with np.load(frame/'cameras/rgb_cameras.npz',allow_pickle=False) as f:cameras=dict(f)
    model=_load_hi4d_smpl_gender_backend(root/'add/tools/SMPLest-X/human_models/human_model_files/smpl/SMPL_FEMALE.pkl','female')
    result=official_person(params,0,model,cameras['extrinsics'][0],cameras['intrinsics'][0],(1280,940))
    assert result['mesh_error']<=1e-4 and result['camera_mesh_error']<=1e-4
    assert result['body_pose'].shape==(63,)
