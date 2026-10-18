"""Fit four SMPL-X bodies with shared actor shapes and calibrated Hi4D constraints.

This is offline pseudo-annotation fitting. It does not train or invoke Qwen.
No official SMPL/SMPL-X deformation correspondence is assumed or synthesized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import smplx
import torch
from geometry import camera_to_world, wholebody134_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/shangguanrz/project/pic-edit')
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--output-name', default='fit_v1')
    parser.add_argument('--steps', type=int, default=500)
    a = parser.parse_args()
    root, work = Path(a.root), Path(a.workdir)
    outdir = work / a.output_name
    if (outdir / 'result.json').exists():
        raise FileExistsError('Choose a new --output-name to preserve the previous fit')
    outdir.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda'
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    model_path = root / 'add/tools/SMPLest-X/human_models/human_model_files'
    model = smplx.create(str(model_path), 'smplx', gender='neutral', num_betas=10,
                         num_expression_coeffs=10, use_pca=False, flat_hand_mean=True,
                         use_face_contour=True, batch_size=4).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cases = [('talk14', 32), ('hug14', 62)]
    raw_data, cams_all, obs, triangles, initial = [], [], [], [], []
    init_sources = []
    started = time.time()
    for action, frame in cases:
        raw = root / 'datasets/Hi4D_pilot_v1/raw/pair14' / action
        raw_data.append(np.load(raw / 'smpl' / f'{frame:06d}.npz'))
        cams = np.load(raw / 'cameras/rgb_cameras.npz')
        cams_all.append(cams)
        triangles.append(np.load(work / 'observations' / f'{action}_{frame:06d}_triangulated.npz'))
        view_observations = [np.load(work / 'observations' / f'{action}_{frame:06d}_cam{cam}.npz') for cam in cams['ids']]
        obs.extend(view_observations)
        for actor in range(2):
            available = []
            for c, camera in enumerate(cams['ids']):
                z = np.load(work / 'initialization' / f'{action}_{frame:06d}_cam{camera}.npz')
                p = int(z['actor_assignment'][actor])
                if p >= 0:
                    available.append((float(z['assignment_error_px'][actor]), c, p, z))
            if not available:
                raise RuntimeError(f'No valid initialization for {action} actor {actor}')
            error, c, p, z = min(available, key=lambda row: row[0])
            pose = z[f'person_{p}_rotvec'].copy()
            root_world = Rotation.from_matrix(cams['extrinsics'][c, :, :3].T @ Rotation.from_rotvec(pose[0]).as_matrix()).as_rotvec()
            best_hands = []
            for hand, region in enumerate([slice(92, 113), slice(113, 134)]):
                choices = [(float(np.mean(view_observations[v]['scores'][actor, region])), v, q, pred) for _, v, q, pred in available]
                quality, v, q, hand_pred = max(choices, key=lambda row: row[0])
                pose[22 + 15 * hand:37 + 15 * hand] = hand_pred[f'person_{q}_rotvec'][22 + 15 * hand:37 + 15 * hand]
                best_hands.append({'camera': int(cams['ids'][v]), 'mean_dwpose_score': quality})
            initial.append({'global_orient': root_world, 'body_pose': pose[1:22].reshape(63),
                            'left_hand_pose': pose[22:37].reshape(45), 'right_hand_pose': pose[37:52].reshape(45),
                            'jaw_pose': pose[52], 'expression': z[f'person_{p}_expression'], 'betas': z[f'person_{p}_shape']})
            init_sources.append({'action': action, 'frame': frame, 'actor_id': actor, 'camera': int(cams['ids'][c]),
                                 'body_assignment_error_px': error, 'hands': best_hands})

    params = {key: torch.nn.Parameter(tensor(np.stack([row[key] for row in initial]))) for key in initial[0] if key != 'betas'}
    # Actor 0/1 retain the same shape in both actions. Betas are fitted, never copied from SMPL.
    params['betas'] = torch.nn.Parameter(tensor(np.stack([np.mean([initial[i]['betas'] for i in [actor, actor + 2]], axis=0) for actor in range(2)])))
    actor_ids = torch.tensor([0, 1, 0, 1], device=device)
    gt_joints = tensor(np.concatenate([z['joints_3d'][:, :22] for z in raw_data]))
    gt_vertices = np.concatenate([z['verts'] for z in raw_data])
    with torch.no_grad():
        zero = model(**{k: v[actor_ids] if k == 'betas' else v for k, v in params.items()})
    params['transl'] = torch.nn.Parameter(gt_joints[:, 0] - zero.joints[:, 0])
    original = {key: value.detach().clone() for key, value in params.items()}

    indices = wholebody134_indices()
    # Fixed 3D face landmarks are fit; the 17 view-dependent contour landmarks are excluded.
    selected = np.r_[np.arange(24), np.arange(41, 134)]
    mapped = torch.tensor(indices[selected], device=device)
    xy = tensor(np.stack([np.stack([v['xy_native'] for v in obs[t * 8:(t + 1) * 8]]) for t in range(2)]).transpose(0, 2, 1, 3, 4).reshape(4, 8, 134, 2))
    scores = tensor(np.stack([np.stack([v['scores'] for v in obs[t * 8:(t + 1) * 8]]) for t in range(2)]).transpose(0, 2, 1, 3).reshape(4, 8, 134))
    Ks = tensor(np.stack([c['intrinsics'] for c in cams_all])).repeat_interleave(2, dim=0)
    Es = tensor(np.stack([c['extrinsics'] for c in cams_all])).repeat_interleave(2, dim=0)
    tri_xyz = tensor(np.nan_to_num(np.concatenate([z['xyz_world'] for z in triangles])))
    tri_valid = tensor(np.concatenate([z['valid'] for z in triangles]))
    tri_count = tensor(np.concatenate([z['inlier_views'] for z in triangles]))
    # Use 3+ inlier cameras in the fitter; retain 2-view triangulations as weaker audit data.
    tri_weight = tri_valid[:, selected] * (tri_count[:, selected] >= 3)
    reproj_weight = scores[:, :, selected].square() * (scores[:, :, selected] >= .5)
    # Gate 2D outliers using the robust triangulated landmarks where they exist.
    with torch.no_grad():
        cam_tri = torch.einsum('bvij,bkj->bvki', Es[:, :, :, :3], tri_xyz[:, selected]) + Es[:, :, None, :, 3]
        hp = torch.einsum('bvij,bvkj->bvki', Ks, cam_tri)
        uv_tri = hp[..., :2] / hp[..., 2:].clamp_min(1e-5)
        residual = (uv_tri - xy[:, :, selected]).norm(dim=-1)
        reproj_weight *= (~(tri_weight[:, None].bool() & (residual > 20))).float()
    # 2D-only unseen hands are weak constraints; body position is anchored to existing 3D.
    region_scale = tensor(np.r_[np.full(24, .3), np.full(51, .35), np.ones(42)])
    reproj_weight *= region_scale[None, None]

    weights = model.lbs_weights.detach().cpu().numpy()
    body_mask = (weights[:, :22].sum(1) > .9) & (weights[:, 15] < .3)
    body_ids = np.flatnonzero(body_mask)[::3]
    body_ids_t = torch.tensor(body_ids, device=device)
    gt_trees = [cKDTree(v) for v in gt_vertices]
    contact_links = []
    # Closest surface pairs from the original two SMPL meshes, preserving original distances.
    for t, z in enumerate(raw_data):
        ids = np.flatnonzero(z['contact'][0] > 0)
        if len(ids):
            ids = ids[::max(1, len(ids) // 160)][:160]
            _, bid = cKDTree(z['verts'][1]).query(z['verts'][0, ids])
            contact_links.append((t, ids, bid, np.linalg.norm(z['verts'][0, ids] - z['verts'][1, bid], axis=-1)))
    correspondences, contacts = None, []
    optimizer = torch.optim.Adam([
        {'params': [params[k] for k in ['global_orient', 'body_pose', 'transl']], 'lr': .012},
        {'params': [params['betas']], 'lr': .025},
        {'params': [params[k] for k in ['left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression']], 'lr': .012},
    ])

    def forward():
        return model(**{k: v[actor_ids] if k == 'betas' else v for k, v in params.items()})

    def measurements(output):
        body_error = (output.joints[:, :22] - gt_joints).norm(dim=-1).mean(-1)
        whole = output.joints[:, mapped]
        camera = torch.einsum('bvij,bkj->bvki', Es[:, :, :, :3], whole) + Es[:, :, None, :, 3]
        hp = torch.einsum('bvij,bvkj->bvki', Ks, camera)
        uv = hp[..., :2] / hp[..., 2:].clamp_min(1e-5)
        return body_error, uv

    with torch.no_grad():
        start_output = forward()
        before = measurements(start_output)[0].cpu().numpy()
        np.savez_compressed(outdir / 'initial_geometry.npz', vertices_world=start_output.vertices.cpu().numpy(), joints_world=start_output.joints.cpu().numpy())
    history = []
    for step in range(a.steps):
        if step == 300:
            for group in optimizer.param_groups:
                group['lr'] *= .25
        optimizer.zero_grad()
        output = forward()
        body_error, uv = measurements(output)
        verts_np = output.vertices.detach().cpu().numpy()
        if step % 25 == 0:
            nearest = np.stack([gt_vertices[i, gt_trees[i].query(v[body_ids])[1]] for i, v in enumerate(verts_np)])
            correspondences = tensor(nearest)
            contacts = []
            for t, aid, bid, distances in contact_links:
                aidx = cKDTree(verts_np[t * 2]).query(gt_vertices[t * 2, aid])[1]
                bidx = cKDTree(verts_np[t * 2 + 1]).query(gt_vertices[t * 2 + 1, bid])[1]
                contacts.append((t, torch.tensor(aidx, device=device), torch.tensor(bidx, device=device), tensor(distances)))
        joint_loss = (output.joints[:, :22] - gt_joints).square().mean()
        surface_loss = (output.vertices[:, body_ids_t] - correspondences).square().mean()
        tri_residual = (output.joints[:, mapped] - tri_xyz[:, selected]).norm(dim=-1)
        tri_loss = (torch.nn.functional.huber_loss(tri_residual, torch.zeros_like(tri_residual), delta=.025, reduction='none') * tri_weight).sum() / tri_weight.sum().clamp_min(1)
        reproj = ((uv - xy[:, :, selected]) / 100).norm(dim=-1)
        reproj_loss = (torch.nn.functional.huber_loss(reproj, torch.zeros_like(reproj), delta=.15, reduction='none') * reproj_weight).sum() / reproj_weight.sum().clamp_min(1)
        contact_loss = torch.zeros((), device=device)
        if contacts:
            contact_loss = torch.stack([(output.vertices[t * 2, ai] - output.vertices[t * 2 + 1, bi]).norm(dim=-1).sub(d).square().mean() for t, ai, bi, d in contacts]).mean()
        hand_prior = sum((params[k] - original[k]).square().mean() for k in ['left_hand_pose', 'right_hand_pose'])
        body_prior = (params['body_pose'] - original['body_pose']).square().mean()
        face_prior = (params['expression'] - original['expression']).square().mean() + (params['jaw_pose'] - original['jaw_pose']).square().mean()
        detail = min(1., max(0., (step - 60) / 100))
        loss = 200 * joint_loss + 25 * surface_loss + 15 * contact_loss + .002 * body_prior
        loss += detail * (35 * tri_loss + .15 * reproj_loss) + .002 * hand_prior + .0005 * face_prior + .0001 * params['betas'].square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite fitting objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 5)
        optimizer.step()
        if step % 25 == 0 or step == a.steps - 1:
            entry = {'step': step, 'loss': float(loss.detach()), 'body_joint_mean_mm': (body_error.detach().cpu().numpy() * 1000).tolist(), 'contact_loss_m2': float(contact_loss.detach())}
            history.append(entry)
            print(json.dumps(entry), flush=True)
    with torch.no_grad():
        result = forward()
        after = measurements(result)[0].cpu().numpy()
    arrays = {key: value.detach().cpu().numpy() for key, value in params.items()}
    arrays['betas'] = arrays['betas'][[0, 1, 0, 1]]
    arrays.update(vertices_world=result.vertices.detach().cpu().numpy(), joints_world=result.joints.detach().cpu().numpy(), faces=model.faces.astype(np.int32), actor_ids=np.array([0, 1, 0, 1]), frame_ids=np.array([32, 32, 62, 62]), actions=np.array(['talk14', 'talk14', 'hug14', 'hug14']), leye_pose=np.zeros((4, 3), np.float32), reye_pose=np.zeros((4, 3), np.float32))
    np.savez_compressed(outdir / 'smplx_world.npz', **arrays)
    all_metrics = []
    for case, (action, frame) in enumerate(cases):
        observed = triangles[case]
        frame_metrics = []
        for actor in range(2):
            i = case * 2 + actor
            dist = gt_trees[i].query(arrays['vertices_world'][i, body_ids])[0]
            hand_status = []
            for region in [slice(92, 113), slice(113, 134)]:
                valid_count = int(observed['valid'][actor, region].sum())
                reliable_count = int((observed['valid'][actor, region] & (observed['inlier_views'][actor, region] >= 3)).sum())
                hand_status.append({'triangulated_points': valid_count, 'points_with_3_views': reliable_count, 'status': 'observed_multiview' if reliable_count >= 8 else 'mostly_model_prior'})
            frame_metrics.append({'actor_id': actor, 'body_joint_before_mm': float(before[i] * 1000), 'body_joint_after_mm': float(after[i] * 1000), 'body_surface_nearest_vertex_mean_mm': float(dist.mean() * 1000), 'hands': hand_status, 'eyes': 'default_zero_not_observed'})
        contacts_report = []
        for t, ai, bi, d in contacts:
            if t == case:
                distances = np.linalg.norm(arrays['vertices_world'][t * 2, ai.cpu().numpy()] - arrays['vertices_world'][t * 2 + 1, bi.cpu().numpy()], axis=-1)
                contacts_report.append({'samples': len(distances), 'original_smpl_mean_mm': float(d.cpu().numpy().mean() * 1000), 'fitted_smplx_mean_mm': float(distances.mean() * 1000), 'distance_error_mean_mm': float(np.abs(distances - d.cpu().numpy()).mean() * 1000), 'mapping': 'estimated nearest-surface correspondences, not official topology correspondence'})
        all_metrics.append({'action': action, 'frame': frame, 'actors': frame_metrics, 'contacts': contacts_report})
    body_model = model_path / 'smplx/SMPLX_NEUTRAL.npz'
    report = {'annotation_type': 'fitted_pseudo_annotation_not_official_ground_truth', 'model_type': 'smplx', 'gender': 'neutral', 'units': 'metres', 'coordinates': 'Hi4D world; Y is up', 'use_pca': False, 'flat_hand_mean': True, 'betas_shared_across_frames': True, 'steps': a.steps, 'elapsed_seconds': time.time() - started, 'body_model_sha256': hashlib.sha256(body_model.read_bytes()).hexdigest(), 'initialization': init_sources, 'metrics': all_metrics, 'loss_history': history, 'constraints': ['Hi4D SMPL first 22 body joints', 'iterated nearest body surface vertices', 'calibrated multiview DWPose with outlier rejection', 'robustly triangulated anatomical points', 'original contact-pair distances', 'network priors for missing details'], 'limitations': ['No official SMPL-to-SMPL-X deformation correspondence', 'Face contours are view dependent and excluded from fitting', 'Occluded hands can remain prior estimates', 'No dedicated signed-distance collision optimizer; inspect meshes before accepting contact quality']}
    (outdir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('FINISHED', outdir, 'body_joint_after_mm', (after * 1000).round(2).tolist(), flush=True)


if __name__ == '__main__':
    main()
