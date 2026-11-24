"""SDPose geometry-order metrics; missing predictions retain the GT denominator."""
import math
import numpy as np

BODY17 = [0,16,14,17,15,5,2,6,3,7,4,11,8,12,9,13,10]
ARM6 = [2,3,4,5,6,7]
WRIST2 = [4,7]
LEFT21 = list(range(92,113))
RIGHT21 = list(range(113,134))
HAND_REWARD = 'wrist_relative_target_box_v1'
MIN_HAND_DIAGONAL = 8.
HAND_GESTURE_SCALE = .10


def _region(gt, valid, pred, pred_valid, ids, diagonal, minimum=1):
    ids=np.asarray(ids);visible=valid[ids]&np.isfinite(gt[ids]).all(1)
    count=int(visible.sum())
    if count<minimum:
        return {'applicable':False,'gt_points':count,'matched_points':0,'reward':None,
                'pck_002':None,'pck_005':None,'recall':None,'l2_px':None,'l2_px_all':None,'nme_all':None}
    points=ids[visible]
    errors=np.full(count,float(diagonal));matched=np.zeros(count,bool)
    if pred is not None:
        matched=pred_valid[points]&np.isfinite(pred[points]).all(1)
        errors[matched]=np.linalg.norm(pred[points[matched]]-gt[points[matched]],axis=1)
    reward=np.zeros(count)
    reward[matched]=np.exp(-.5*(errors[matched]/max(.02*diagonal,1e-12))**2)
    return {'applicable':True,'gt_points':count,'matched_points':int(matched.sum()),
        'reward':float(reward.mean()),'pck_002':float((matched&(errors<.02*diagonal)).sum()/count),
        'pck_005':float((matched&(errors<.05*diagonal)).sum()/count),'recall':float(matched.mean()),
        'l2_px':float(errors[matched].mean()) if matched.any() else None,
        'l2_px_all':float(errors.mean()),'nme_all':float(errors.mean()/diagonal)}


def _hand_gesture(gt, valid, pred, pred_valid, ids):
    wrist, *fingers = ids
    fingers = np.asarray(fingers)
    visible = fingers[valid[fingers] & np.isfinite(gt[fingers]).all(axis=1)]
    if len(visible) < 7 or not valid[wrist] or not np.isfinite(gt[wrist]).all():
        return None
    points = gt[[wrist, *visible]]
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(scale) or scale < MIN_HAND_DIAGONAL:
        return None
    reward = np.zeros(len(visible))
    if pred is not None and pred_valid[wrist] and np.isfinite(pred[wrist]).all():
        matched = pred_valid[visible] & np.isfinite(pred[visible]).all(axis=1)
        ids = visible[matched]
        delta = (pred[ids] - pred[wrist]) - (gt[ids] - gt[wrist])
        error = np.linalg.norm(delta, axis=1) / scale
        reward[matched] = np.exp(-.5 * (error / HAND_GESTURE_SCALE)**2)
    return float(reward.mean())


def pose_metrics(target_points,target_valid,predicted_points,predicted_valid,content_hw):
    gt=np.asarray(target_points,float);valid=np.asarray(target_valid,bool)
    if gt.shape!=(134,2) or valid.shape!=(134,):raise ValueError('Explicit geometry-order 134-slot GT required')
    pred=None if predicted_points is None else np.asarray(predicted_points,float)
    pv=np.zeros(134,bool) if predicted_valid is None else np.asarray(predicted_valid,bool)
    if (pred is not None and pred.shape!=(134,2)) or pv.shape!=(134,):raise ValueError('Prediction layout differs')
    diagonal=math.hypot(*content_hw)
    if not math.isfinite(diagonal) or diagonal<=0:raise ValueError('Positive actual content size required')
    result={name:_region(gt,valid,pred,pv,ids,diagonal,minimum) for name,ids,minimum in
        (('body',BODY17,1),('arm',ARM6,1),('wrist',WRIST2,1),('left_hand',LEFT21,8),('right_hand',RIGHT21,8))}
    for side, ids in (('left_hand', LEFT21), ('right_hand', RIGHT21)):
        hand = result[side]
        placement = hand['reward']
        gesture = _hand_gesture(gt, valid, pred, pv, ids) if hand['applicable'] else None
        hand.update(placement_reward=placement, gesture_reward=gesture,
                    training_reward=placement if gesture is None else .5 * (placement + gesture))
    hands=[result[k] for k in ('left_hand','right_hand') if result[k]['applicable']]
    result['hands']={'applicable':bool(hands),'gt_points':sum(x['gt_points'] for x in hands),
        'matched_points':sum(x['matched_points'] for x in hands)}
    for key in ('reward','pck_002','pck_005','recall','l2_px','l2_px_all','nme_all',
                'placement_reward','gesture_reward','training_reward'):
        values=[x[key] for x in hands if x[key] is not None]
        result['hands'][key]=float(np.mean(values)) if values else None
    return result
