"""Export calibrated SMPL-X meshes, 3D skeletons and DWPose-layout projections."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import smplx
import torch
import nvdiffrast.torch as dr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import project, wholebody134_indices

BODY_EDGES = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(240, 80, 45), (35, 170, 235)]


def skeleton(canvas, xy, score, actor, thickness=3):
    color = COLORS[actor]
    for i, j in BODY_EDGES:
        cv2.line(canvas, tuple(xy[i].astype(int)), tuple(xy[j].astype(int)), color, thickness, cv2.LINE_AA)
    for i in range(18):
        cv2.circle(canvas, tuple(xy[i].astype(int)), 4, color, -1, cv2.LINE_AA)
    for region in [92, 113]:
        for finger in range(5):
            chain = [region] + list(range(region + 1 + finger * 4, region + 5 + finger * 4))
            for i, j in zip(chain[:-1], chain[1:]):
                c = color if min(score[i], score[j]) > 0 else (120, 120, 120)
                cv2.line(canvas, tuple(xy[i].astype(int)), tuple(xy[j].astype(int)), c, max(1, thickness - 1), cv2.LINE_AA)
    for i in range(24, 92):
        cv2.circle(canvas, tuple(xy[i].astype(int)), 1, color, -1, cv2.LINE_AA)
    for i, j in [(13, 18), (18, 19), (13, 20), (10, 21), (21, 22), (10, 23)]:
        cv2.line(canvas, tuple(xy[i].astype(int)), tuple(xy[j].astype(int)), color, max(1, thickness - 1), cv2.LINE_AA)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/shangguanrz/project/pic-edit')
    p.add_argument('--workdir', required=True)
    p.add_argument('--fit-name', default='fit_v1')
    a = p.parse_args()
    root, work = Path(a.root), Path(a.workdir)
    fitdir = work / a.fit_name
    outdir = fitdir / 'visuals'
    outdir.mkdir(exist_ok=True)
    z = np.load(fitdir / 'smplx_world.npz')
    for i in range(4):
        mesh_name = f"{z['actions'][i]}_{z['frame_ids'][i]:06d}_actor{z['actor_ids'][i]}"
        with (outdir / f'{mesh_name}.obj').open('w') as stream:
            stream.write('# Fitted SMPL-X, Hi4D world, metres. Pseudo annotation.\n')
            for vertex in z['vertices_world'][i]:
                stream.write('v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in z['faces'] + 1:
                stream.write('f %d %d %d\n' % tuple(face))
    model = smplx.create(str(root / 'add/tools/SMPLest-X/human_models/human_model_files'), 'smplx', gender='neutral', batch_size=4, use_pca=False, flat_hand_mean=True, use_face_contour=True).cuda().eval()
    names = ['betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'leye_pose', 'reye_pose']
    kwargs = {k: torch.tensor(z[k], device='cuda') for k in names}
    context = dr.RasterizeCudaContext(device=torch.device('cuda'))
    faces = torch.tensor(np.concatenate([z['faces'], z['faces'] + 10475]), device='cuda', dtype=torch.int32)
    vertex_colors = torch.tensor(np.repeat(np.array(COLORS) / 255., 10475, axis=0), dtype=torch.float32, device='cuda')[None]
    checks = []
    cams = [np.load(root / f'datasets/Hi4D_pilot_v1/raw/pair14/{action}/cameras/rgb_cameras.npz') for action in ['talk14', 'hug14']]
    triangles = [np.load(work / f'observations/{action}_{fid:06d}_triangulated.npz') for action, fid in [('talk14', 32), ('hug14', 62)]]
    all_cam_wholebody, all_world_wholebody = [], []
    panels = []
    for c in range(8):
        Es = np.concatenate([np.repeat(cam['extrinsics'][c][None], 2, 0) for cam in cams])
        root_rot = np.stack([Rotation.from_matrix(E[:, :3] @ Rotation.from_rotvec(rot).as_matrix()).as_rotvec() for E, rot in zip(Es, z['global_orient'])])
        camera_kwargs = dict(kwargs, global_orient=torch.tensor(root_rot, dtype=torch.float32, device='cuda'))
        with torch.no_grad():
            zero = model(**camera_kwargs)
            root_camera = np.einsum('bij,bj->bi', Es[:, :, :3], z['joints_world'][:, 0]) + Es[:, :, 3]
            translation = torch.tensor(root_camera, dtype=torch.float32, device='cuda') - zero.joints[:, 0]
            cam_output = model(**dict(camera_kwargs, transl=translation))
        verts_camera = cam_output.vertices.cpu().numpy()
        joints_camera = cam_output.joints.cpu().numpy()
        wholebody_camera = joints_camera[:, wholebody134_indices()]
        expected_camera = np.einsum('bij,bkj->bki', Es[:, :, :3], z['vertices_world']) + Es[:, None, :, 3]
        camera_error = np.abs(verts_camera - expected_camera).max(axis=(1, 2))
        wholebody_world = np.stack([(j - E[:, 3]) @ E[:, :3] for j, E in zip(wholebody_camera, Es)])
        all_cam_wholebody.append(wholebody_camera)
        all_world_wholebody.append(wholebody_world)
        for t, (action, frame) in enumerate([('talk14', 32), ('hug14', 62)]):
            camera = int(cams[t]['ids'][c])
            K = cams[t]['intrinsics'][c]
            image = np.array(Image.open(work / f'inputs/pair14/{action}/images/{camera}/{frame:06d}.jpg').convert('RGB'))
            h, w = image.shape[:2]
            vertices = verts_camera[t * 2:t * 2 + 2].reshape(-1, 3)
            projected = vertices @ K.T
            uv = projected[:, :2] / projected[:, 2:3]
            depth = vertices[:, 2]
            clip = np.stack([(2 * uv[:, 0] / w - 1) * depth, (1 - 2 * uv[:, 1] / h) * depth, (2 * (depth - .1) / 20 - 1) * depth, depth], axis=-1)
            raster, _ = dr.rasterize(context, torch.tensor(clip, device='cuda', dtype=torch.float32)[None], faces, resolution=[h, w])
            interpolated, _ = dr.interpolate(vertex_colors, raster, faces)
            # nvdiffrast scanline origin is bottom-left; image arrays use top-left.
            color = interpolated[0].flip(0).cpu().numpy()
            triangle_ids = raster[0, :, :, 3].flip(0).cpu().numpy()
            mask = triangle_ids > 0
            overlay = image.copy()
            overlay[mask] = (.5 * image[mask] + .5 * 255 * color[mask]).astype(np.uint8)
            prefix = f'{action}_{frame:06d}_cam{camera}'
            Image.fromarray(overlay).save(outdir / f'{prefix}_mesh.png')
            wholebody_uv = np.stack([project(points, K, np.eye(4)[:3]) for points in wholebody_camera[t * 2:t * 2 + 2]])
            black = np.zeros_like(image)
            projected_overlay = image.copy()
            support = (triangles[t]['valid'] & (triangles[t]['inlier_views'] >= 3)).astype(np.float32)
            for actor in range(2):
                skeleton(black, wholebody_uv[actor], support[actor], actor)
                skeleton(projected_overlay, wholebody_uv[actor], support[actor], actor)
                cv2.putText(black, f'{"A" if actor == 0 else "B"}: actor {actor}', (20, 36 + actor * 32), cv2.FONT_HERSHEY_SIMPLEX, .85, COLORS[actor], 2)
            cv2.putText(black, 'Gray hands: model estimate / insufficient 3D observations', (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1)
            Image.fromarray(black).save(outdir / f'{prefix}_skeleton.png')
            Image.fromarray(projected_overlay).save(outdir / f'{prefix}_projection.png')
            np.savez_compressed(outdir / f'{prefix}_keypoints.npz', wholebody134_camera=wholebody_camera[t * 2:t * 2 + 2], wholebody134_world=wholebody_world[t * 2:t * 2 + 2], xy_native=wholebody_uv, observed_multiview=triangles[t]['valid'], inlier_views=triangles[t]['inlier_views'], K=K, extrinsics=cams[t]['extrinsics'][c], actor_ids=[0, 1], contour_reference_camera=camera)
            # Independent raster orientation check against projected vertices, by actor.
            centroids = []
            for actor in range(2):
                region = ((triangle_ids > 0) & (triangle_ids <= len(z['faces']))) if actor == 0 else (triangle_ids > len(z['faces']))
                ys, xs = np.nonzero(region)
                predicted = uv[actor * 10475:(actor + 1) * 10475]
                # Silhouette bounding boxes are less biased by mesh vertex density.
                if len(xs):
                    centroid_error = np.linalg.norm((np.array([xs.min(), ys.min()]) + [xs.max(), ys.max()]) / 2 - (predicted.min(0) + predicted.max(0)) / 2)
                else:
                    centroid_error = float('inf')
                centroids.append(float(centroid_error))
            checks.append({'action': action, 'frame': frame, 'camera': camera, 'mesh_camera_roundtrip_max_m': float(camera_error[t * 2:t * 2 + 2].max()), 'raster_centroid_error_px': max(centroids), 'render_pixels': int(mask.sum())})
            if camera == 4:
                panels.append((action, frame, image, overlay, projected_overlay, black))
    np.savez_compressed(fitdir / 'wholebody134_by_camera.npz', xyz_camera=np.stack(all_cam_wholebody, axis=1), xyz_world=np.stack(all_world_wholebody, axis=1), camera_ids=cams[0]['ids'], observed_multiview=np.concatenate([tr['valid'] for tr in triangles]), inlier_views=np.concatenate([tr['inlier_views'] for tr in triangles]), joint_index_map=wholebody134_indices())
    # Hi4D world Y is vertical. Plot axes are X, Z, Y and explicitly labelled.
    for t, (action, frame) in enumerate([('talk14', 32), ('hug14', 62)]):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for actor in range(2):
            points = all_world_wholebody[0][t * 2 + actor]
            color = np.array(COLORS[actor]) / 255.
            edges = BODY_EDGES.copy()
            for start in [92, 113]:
                for finger in range(5):
                    chain = [start] + list(range(start + 1 + finger * 4, start + 5 + finger * 4))
                    edges.extend(zip(chain[:-1], chain[1:]))
            for i, j in edges:
                pair = points[[i, j]]
                supported = bool(triangles[t]['valid'][actor, i] and triangles[t]['valid'][actor, j] and triangles[t]['inlier_views'][actor, i] >= 3 and triangles[t]['inlier_views'][actor, j] >= 3)
                ax.plot(pair[:, 0], pair[:, 2], pair[:, 1], color=color if i < 92 or supported else '.55', linewidth=2 if i < 92 else 1)
            ax.scatter(points[:18, 0], points[:18, 2], points[:18, 1], color=color, s=16, label=f'{"A" if actor == 0 else "B"} / actor {actor}')
        vertices = z['vertices_world'][t * 2:t * 2 + 2].reshape(-1, 3)
        centers = (vertices.min(0) + vertices.max(0)) / 2
        span = (vertices.max(0) - vertices.min(0)).max() * .55
        ax.set_xlim(centers[0] - span, centers[0] + span)
        ax.set_ylim(centers[2] - span, centers[2] + span)
        ax.set_zlim(centers[1] - span, centers[1] + span)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('World X (m)'); ax.set_ylabel('World Z (m)'); ax.set_zlabel('World Y (m)')
        ax.view_init(elev=15, azim=-65)
        ax.legend()
        ax.set_title(f'{action} / {frame}: fitted SMPL-X 3D skeleton\nGray fingers: model estimate; not official ground truth')
        fig.savefig(outdir / f'{action}_{frame:06d}_3d_skeleton.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    sheet = Image.new('RGB', (4 * 470, 2 * 690), 'white')
    draw = ImageDraw.Draw(sheet)
    labels = ['Original RGB', 'Fitted SMPL-X mesh: A red / B blue', 'SMPL-X projected skeleton', 'DWPose layout from fitted 3D']
    for row, (action, frame, *images) in enumerate(panels):
        for column, (label, image) in enumerate(zip(labels, images)):
            sheet.paste(Image.fromarray(image).resize((470, 640)), (column * 470, row * 690 + 45))
            draw.text((column * 470 + 10, row * 690 + 7), f'{action} {frame:06d}', fill='black')
            draw.text((column * 470 + 10, row * 690 + 24), label, fill='black')
    sheet.save(outdir / 'pilot_contact_sheet.jpg', quality=95)
    (outdir / 'export_checks.json').write_text(json.dumps({'views': checks, 'wholebody_layout': 'body18 + feet6 + face68 + left21 + right21; SMPL-X derived, not native 3D DWPose', 'face_contour': '17 view-dependent points generated separately for each camera'}, indent=2, allow_nan=False))
    print('EXPORTED', outdir, 'max_roundtrip_m', max(row['mesh_camera_roundtrip_max_m'] for row in checks), flush=True)


if __name__ == '__main__':
    main()
