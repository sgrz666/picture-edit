"""Canvas-aware evaluation helpers and explicitly heuristic visual review hints."""
import numpy as np


def pose_coverage(persons):
    requested=18*len(persons)
    detected=sum(p.get('body_joints',0) for p in persons)
    correct=sum(p.get('body_joints',0)*p.get('pck@0.2',0.) for p in persons)
    return dict(requested_points=requested,detected_points=detected,
        missing_points=requested-detected,coverage=detected/requested if requested else None,
        pck_02_all_requested=correct/requested if requested else None,
        basis='Requested geometric target body18; missing generated detections are not counted as correct; driver visibility is not ground truth.')


def geometry_consistency(arrays):
    depth,mask=arrays['depth'],arrays['mask'].astype(bool)
    expected=np.zeros_like(mask)
    closest=np.where(mask,depth,np.inf).argmin(0)
    for p in range(2):expected[p]=(closest==p)&mask.any(0)
    if not np.array_equal(expected,arrays['visible_owner']):raise ValueError('Owner and nearest Depth disagree')
    if not np.allclose(arrays['combined_depth'],(depth*expected).sum(0),atol=1e-5):raise ValueError('Combined Depth disagrees with Owner')
    if not np.allclose(arrays['combined_normal_cam'],(arrays['normal_cam']*expected[...,None]).sum(0),atol=1e-5):
        raise ValueError('Combined Normal disagrees with Owner')
    if not np.isfinite(arrays['normal_cam']).all():raise ValueError('Nonfinite camera Normal')
    if not np.isfinite(depth[mask]).all() or (depth[mask]<=0).any():raise ValueError('Invalid CV camera Depth')
    return dict(owner_depth_normal_consistent=True,normal_basis='camera_cv',unit='m',
        overlap_pixels=int(mask.all(0).sum()),visible_pixels=expected.sum((1,2)).tolist())


def background_support(valid_image, source_mask, target_masks, dilation=17):
    import cv2
    union = np.asarray(source_mask, bool) | np.asarray(target_masks, bool).any(0)
    union = cv2.dilate(union.astype(np.uint8), np.ones((dilation, dilation), np.uint8)) > 0
    return np.asarray(valid_image, bool) & ~union


def contact_projection_proxy(arrays, requested, assigned):
    """Observe only attachments near a body landmark; retain all missing evidence."""
    links=[];missing=0;errors=[]
    points=arrays.get('contact_points_cam',np.empty((0,2,3)))
    indices=arrays.get('contact_indices',np.empty((0,4),int))
    # Anatomical wrist/elbow/shoulder/hip joints with unambiguous DWPose slots.
    smpl=np.array([20,21,18,19,16,17,1,2]);dw=np.array([7,4,6,3,5,2,11,8])
    for contact,owners in zip(points,indices):
        endpoints=[]
        for side,p in enumerate((int(owners[0]),int(owners[2]))):
            distances=np.linalg.norm(arrays['joints_137_cam'][p,smpl]-contact[side],axis=1)
            nearest=int(distances.argmin());slot=int(dw[nearest]);person=assigned[p]
            available=bool(distances[nearest]<=.12 and person is not None and person['scores'][slot]>=.5)
            available=available and bool(np.isfinite(person['points'][slot]).all())
            projected=contact[side]@arrays['K'].T;target=projected[:2]/projected[2]
            # Transfer the fixed surface-to-landmark offset; this remains a proxy.
            observed=None if not available else np.asarray(person['points'][slot])+target-requested[p][slot]
            if not available:missing+=1
            endpoints.append(dict(person=p,landmark=slot,attachment_distance_m=float(distances[nearest]),
                target_px=target.tolist(),observed_proxy_px=None if observed is None else observed.tolist()))
        error=None
        if all(e['observed_proxy_px'] is not None for e in endpoints):
            target_delta=np.subtract(endpoints[1]['target_px'],endpoints[0]['target_px'])
            observed_delta=np.subtract(endpoints[1]['observed_proxy_px'],endpoints[0]['observed_proxy_px'])
            error=float(np.linalg.norm(target_delta-observed_delta));errors.append(error)
        links.append(dict(endpoints=endpoints,relative_error_px=error))
    return dict(status='measured_proxy' if errors else 'unavailable',links=links,comparable=len(errors),
        missing_endpoints=missing,mean_relative_error_px=None if not errors else float(np.mean(errors)),
        caveat='2D landmark-offset proxy within 12cm of an attachment; never a verified RGB surface point or 3D contact distance.')


def review_artifacts(image, arrays, out):
    from PIL import Image
    import cv2
    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # A color hint is not a defect verdict: source clothes may have these colors.
    saturated = (hsv[..., 1] > 150) & (hsv[..., 2] > 100)
    guide_colors = saturated & (((hsv[..., 0] >= 40) & (hsv[..., 0] <= 100)) |
                               (hsv[..., 0] < 20) | (hsv[..., 0] > 165))
    crops = []
    h, w = rgb.shape[:2]
    joints = arrays['joints_137_cam']; K = arrays['K']
    for p in range(2):
        projected = joints[p] @ K.T
        uv = projected[:, :2] / projected[:, 2:]
        # SMPL-X canonical joints: head15, left/right wrist20/21.
        for name, joint, fraction in (('face', 15, .11), ('left_hand', 20, .075), ('right_hand', 21, .075)):
            if not np.isfinite(uv[joint]).all():
                continue
            x, y = uv[joint]; radius = max(24, int(min(h, w) * fraction))
            box = [max(0, int(x)-radius), max(0, int(y)-radius), min(w, int(x)+radius), min(h, int(y)+radius)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = image.crop(box); filename = f'person_{p}_{name}.png'
            crop.resize((256, 256)).save(out / filename)
            region = guide_colors[box[1]:box[3], box[0]:box[2]]
            crops.append(dict(person=p, region=name, path=str(out / filename), box=box,
                saturated_guide_color_fraction=float(region.mean()),
                laplacian_variance=float(cv2.Laplacian(cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())))
    return dict(status='heuristic_requires_visual_review', crops=crops,
                caveat='Color and sharpness are hints only; no calibrated hand or skeleton-leakage classifier is used.')
