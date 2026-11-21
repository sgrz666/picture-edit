import numpy as np


def test_sdpose_reward_prefers_aligned_pose_and_counts_missing_points():
    from src.pose_control.qwen21_full.region_rewards import pose_metrics
    gt=np.zeros((134,2));gt[:,0]=np.linspace(20,220,134);gt[:,1]=100
    valid=np.ones(134,bool)
    perfect=pose_metrics(gt,valid,gt,valid,(512,288))
    shifted=pose_metrics(gt,valid,gt+80,valid,(512,288))
    missing=pose_metrics(gt,valid,None,None,(512,288))
    assert perfect['body']['reward']==1 and perfect['hands']['reward']==1
    assert shifted['body']['reward']<perfect['body']['reward']
    assert missing['body']['reward']==0 and missing['body']['pck_002']==0
    assert missing['hands']['gt_points']==42 and missing['hands']['recall']==0


def test_unavailable_gt_hands_are_not_fabricated_rewards():
    from src.pose_control.qwen21_full.region_rewards import pose_metrics
    gt=np.ones((134,2))*50;valid=np.ones(134,bool);valid[92:]=False
    result=pose_metrics(gt,valid,gt,np.ones(134,bool),(512,288))
    assert result['hands']['applicable'] is False and result['hands']['reward'] is None
