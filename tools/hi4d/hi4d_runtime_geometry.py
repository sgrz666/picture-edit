#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import smplx


PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

SMPL_MODEL_DIR = (
    PROJECT / "models/smpl"
)


# ============================================================
# BASIC
# ============================================================

def flip_y(x):
    return np.flip(
        x,
        axis=0
    ).copy()


def load_camera(
    camera_file,
    camera_id,
):

    with np.load(
        camera_file,
        allow_pickle=True
    ) as d:

        ids = np.asarray(
            d["ids"]
        ).astype(int)

        K_all = np.asarray(
            d["intrinsics"]
        )

        E_all = np.asarray(
            d["extrinsics"]
        )

    idx = np.where(
        ids == int(camera_id)
    )[0]

    if len(idx) != 1:
        raise RuntimeError(
            f"camera {camera_id} "
            f"not found in {ids.tolist()}"
        )

    i = int(idx[0])

    K = K_all[i].astype(
        np.float32
    )

    E = E_all[i].astype(
        np.float32
    )

    # Hi4D validated convention:
    # world -> camera
    R = E[:, :3]
    t = E[:, 3]

    return K, R, t


def scale_intrinsics(
    K,
    native_w,
    native_h,
    out_w,
    out_h,
):

    sx = (
        float(out_w)
        / float(native_w)
    )

    sy = (
        float(out_h)
        / float(native_h)
    )

    S = np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ],
        dtype=np.float32
    )

    return S @ K


# ============================================================
# GEOMETRY
# ============================================================

def vertex_normals(
    verts,
    faces,
):

    tri = verts[
        faces
    ]

    e1 = (
        tri[:, 1]
        - tri[:, 0]
    )

    e2 = (
        tri[:, 2]
        - tri[:, 0]
    )

    fn = torch.cross(
        e1,
        e2,
        dim=1
    )

    vn = torch.zeros_like(
        verts
    )

    for i in range(3):
        vn.index_add_(
            0,
            faces[:, i],
            fn
        )

    return F.normalize(
        vn,
        dim=1,
        eps=1e-8
    )


def camera_to_clip(
    cam,
    K,
    H,
    W,
):

    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]

    zsafe = torch.clamp(
        z,
        min=1e-4
    )

    fx = float(K[0, 0])
    fy = float(K[1, 1])

    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = (
        fx * x / zsafe
        + cx
    )

    v = (
        fy * y / zsafe
        + cy
    )

    x_ndc = (
        2.0 * u
        / max(W - 1, 1)
        - 1.0
    )

    # 与之前已验证的 Hi4D renderer 保持一致，
    # 输出后统一 flip_y
    y_ndc = (
        1.0
        - 2.0 * v
        / max(H - 1, 1)
    )

    near = 0.05
    far = 20.0

    z_ndc = (
        2.0
        * (zsafe - near)
        / (far - near)
        - 1.0
    )

    clip = torch.stack(
        [
            x_ndc * zsafe,
            y_ndc * zsafe,
            z_ndc * zsafe,
            zsafe,
        ],
        dim=1
    )

    return clip.contiguous()


# ============================================================
# RUNTIME RENDERER
# ============================================================

class Hi4DRuntimeGeometry:

    def __init__(
        self,
        device="cuda",
        max_peel_layers=8,
    ):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for nvdiffrast"
            )

        self.device = torch.device(
            device
        )

        self.max_peel_layers = (
            int(max_peel_layers)
        )

        model = smplx.create(
            str(SMPL_MODEL_DIR),
            model_type="smpl",
            gender="neutral",
            ext="pkl",
        )

        faces_np = np.asarray(
            model.faces,
            dtype=np.int64
        )

        self.faces = torch.from_numpy(
            faces_np
        ).to(
            self.device,
            dtype=torch.long
        ).contiguous()

        self.faces_i32 = (
            self.faces
            .to(torch.int32)
            .contiguous()
        )

        self.num_faces = int(
            self.faces.shape[0]
        )

        self.ctx = (
            dr.RasterizeCudaContext(
                device=self.device
            )
        )


    def _camera_vertices(
        self,
        verts_world,
        R,
        t,
    ):

        v = torch.from_numpy(
            np.asarray(
                verts_world,
                dtype=np.float32
            )
        ).to(
            self.device
        )

        R_t = torch.from_numpy(
            np.asarray(
                R,
                dtype=np.float32
            )
        ).to(
            self.device
        )

        t_t = torch.from_numpy(
            np.asarray(
                t,
                dtype=np.float32
            )
        ).to(
            self.device
        )

        return (
            v @ R_t.T
            + t_t[None]
        )


    def _render_person(
        self,
        cam,
        contact_vertex,
        K,
        H,
        W,
    ):

        normals = vertex_normals(
            cam,
            self.faces
        )

        clip = camera_to_clip(
            cam,
            K,
            H,
            W,
        )

        rast, _ = dr.rasterize(
            self.ctx,
            clip[None].contiguous(),
            self.faces_i32,
            resolution=[
                H,
                W
            ],
        )

        rast = rast.contiguous()

        depth_attr = (
            cam[:, 2:3]
            [None]
            .contiguous()
        )

        normal_attr = (
            normals[
                None
            ]
            .contiguous()
        )

        contact_attr = (
            torch.from_numpy(
                np.asarray(
                    contact_vertex,
                    dtype=np.float32
                )
            )
            .to(self.device)
            .reshape(
                1,
                -1,
                1
            )
            .contiguous()
        )

        depth, _ = dr.interpolate(
            depth_attr,
            rast,
            self.faces_i32,
        )

        normal, _ = dr.interpolate(
            normal_attr,
            rast,
            self.faces_i32,
        )

        contact, _ = dr.interpolate(
            contact_attr,
            rast,
            self.faces_i32,
        )

        mask = (
            rast[
                0, :, :, 3
            ] > 0
        )

        depth = (
            depth[
                0, :, :, 0
            ]
        )

        normal = (
            normal[0]
        )

        normal = F.normalize(
            normal,
            dim=-1,
            eps=1e-8
        )

        contact = (
            contact[
                0, :, :, 0
            ]
        )

        depth = torch.where(
            mask,
            depth,
            torch.zeros_like(
                depth
            )
        )

        normal = torch.where(
            mask[..., None],
            normal,
            torch.zeros_like(
                normal
            )
        )

        contact = torch.where(
            mask,
            contact,
            torch.zeros_like(
                contact
            )
        )

        contact = torch.clamp(
            contact,
            0.0,
            1.0
        )

        return {
            "depth":
                flip_y(
                    depth.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ),

            "normal":
                flip_y(
                    normal.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ),

            "mask":
                flip_y(
                    mask.detach()
                    .cpu()
                    .numpy()
                ),

            "contact":
                flip_y(
                    contact.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ),
        }


    def render(
        self,
        smpl_file,
        camera_file,
        camera_id,
        out_h,
        out_w,
        native_h=1280,
        native_w=940,
    ):

        H = int(out_h)
        W = int(out_w)

        with np.load(
            smpl_file,
            allow_pickle=True
        ) as d:

            verts = np.asarray(
                d["verts"],
                dtype=np.float32
            )

            contact_raw = np.asarray(
                d["contact"]
            )

        if verts.shape != (
            2,
            6890,
            3
        ):
            raise RuntimeError(
                f"bad verts {verts.shape}"
            )

        if contact_raw.shape != (
            2,
            6890
        ):
            raise RuntimeError(
                f"bad contact "
                f"{contact_raw.shape}"
            )

        # Hi4D official:
        # positive value means contact correspondence exists
        contact_vertex = (
            contact_raw > 0
        ).astype(
            np.float32
        )

        K0, R, t = load_camera(
            camera_file,
            camera_id
        )

        K = scale_intrinsics(
            K0,
            native_w,
            native_h,
            W,
            H,
        )

        cam_a = (
            self._camera_vertices(
                verts[0],
                R,
                t
            )
        )

        cam_b = (
            self._camera_vertices(
                verts[1],
                R,
                t
            )
        )

        person_a = (
            self._render_person(
                cam_a,
                contact_vertex[0],
                K,
                H,
                W,
            )
        )

        person_b = (
            self._render_person(
                cam_b,
                contact_vertex[1],
                K,
                H,
                W,
            )
        )

        # ============================================
        # combined scene
        # ============================================

        normal_a = vertex_normals(
            cam_a,
            self.faces
        )

        normal_b = vertex_normals(
            cam_b,
            self.faces
        )

        V = int(
            cam_a.shape[0]
        )

        verts_scene = torch.cat(
            [
                cam_a,
                cam_b
            ],
            dim=0
        )

        normals_scene = torch.cat(
            [
                normal_a,
                normal_b
            ],
            dim=0
        )

        faces_scene = torch.cat(
            [
                self.faces,
                self.faces + V
            ],
            dim=0
        ).contiguous()

        faces_scene_i32 = (
            faces_scene
            .to(torch.int32)
            .contiguous()
        )

        clip_scene = camera_to_clip(
            verts_scene,
            K,
            H,
            W,
        )

        rast, _ = dr.rasterize(
            self.ctx,
            clip_scene[
                None
            ].contiguous(),
            faces_scene_i32,
            resolution=[
                H,
                W
            ],
        )

        rast = rast.contiguous()

        depth_attr = (
            verts_scene[
                :, 2:3
            ][None]
            .contiguous()
        )

        normal_attr = (
            normals_scene[
                None
            ]
            .contiguous()
        )

        depth, _ = dr.interpolate(
            depth_attr,
            rast,
            faces_scene_i32,
        )

        normal, _ = dr.interpolate(
            normal_attr,
            rast,
            faces_scene_i32,
        )

        tri_id = (
            rast[
                0, :, :, 3
            ].long()
            - 1
        )

        valid = (
            tri_id >= 0
        )

        owner = torch.zeros(
            (
                H,
                W
            ),
            dtype=torch.uint8,
            device=self.device
        )

        owner[
            valid
            &
            (
                tri_id
                <
                self.num_faces
            )
        ] = 1

        owner[
            valid
            &
            (
                tri_id
                >=
                self.num_faces
            )
        ] = 2

        depth = (
            depth[
                0, :, :, 0
            ]
        )

        normal = (
            normal[0]
        )

        normal = F.normalize(
            normal,
            dim=-1,
            eps=1e-8
        )

        depth = torch.where(
            valid,
            depth,
            torch.zeros_like(
                depth
            )
        )

        normal = torch.where(
            valid[..., None],
            normal,
            torch.zeros_like(
                normal
            )
        )

        scene_depth = flip_y(
            depth.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        scene_normal = flip_y(
            normal.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        front_owner = flip_y(
            owner.detach()
            .cpu()
            .numpy()
        )

        # ============================================
        # PersonaCraft-style multi-layer occlusion
        #
        # M_occ = 1 when projected ray encounters
        # > 2 mesh surfaces.
        # ============================================

        layer_count = torch.zeros(
            (
                H,
                W
            ),
            dtype=torch.int16,
            device=self.device
        )

        with dr.DepthPeeler(
            self.ctx,
            clip_scene[
                None
            ].contiguous(),
            faces_scene_i32,
            resolution=[
                H,
                W
            ],
        ) as peeler:

            for _ in range(
                self.max_peel_layers
            ):

                layer_rast, _ = (
                    peeler
                    .rasterize_next_layer()
                )

                layer_valid = (
                    layer_rast[
                        0, :, :, 3
                    ] > 0
                )

                if not bool(
                    layer_valid.any()
                ):
                    break

                layer_count += (
                    layer_valid
                    .to(torch.int16)
                )

        layer_count_np = flip_y(
            layer_count.detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        occlusion_mask = (
            layer_count_np > 2
        ).astype(
            np.uint8
        )

        # ============================================
        # Directional inter-person occlusion
        # ============================================

        semantic_a = (
            person_a["mask"]
            .astype(np.uint8)
        )

        semantic_b = (
            person_b["mask"]
            .astype(np.uint8)
        )

        interperson_overlap = (
            (semantic_a > 0)
            &
            (semantic_b > 0)
        )

        # front_owner:
        # 1 = A is visible in front
        # 2 = B is visible in front

        A_occluded_by_B = (
            interperson_overlap
            &
            (front_owner == 2)
        ).astype(
            np.uint8
        )

        B_occluded_by_A = (
            interperson_overlap
            &
            (front_owner == 1)
        ).astype(
            np.uint8
        )

        interperson_overlap = (
            interperson_overlap
            .astype(np.uint8)
        )

        # ============================================
        # Metric depth discontinuity
        #
        # Keep this CONTINUOUS.
        # Do not introduce an arbitrary binary
        # threshold during dataset construction.
        # ============================================

        depth_for_grad = (
            scene_depth.copy()
        )

        valid_depth = (
            depth_for_grad > 0
        )

        # Fill outside region locally with zero,
        # then explicitly suppress background edges.
        dx = np.zeros_like(
            depth_for_grad,
            dtype=np.float32
        )

        dy = np.zeros_like(
            depth_for_grad,
            dtype=np.float32
        )

        dx[:, 1:] = np.abs(
            depth_for_grad[:, 1:]
            -
            depth_for_grad[:, :-1]
        )

        dy[1:, :] = np.abs(
            depth_for_grad[1:, :]
            -
            depth_for_grad[:-1, :]
        )

        depth_gradient = (
            dx + dy
        ).astype(
            np.float32
        )

        # Remove image/background boundaries.
        neighbor_valid_x = np.zeros_like(
            valid_depth
        )

        neighbor_valid_y = np.zeros_like(
            valid_depth
        )

        neighbor_valid_x[:, 1:] = (
            valid_depth[:, 1:]
            &
            valid_depth[:, :-1]
        )

        neighbor_valid_y[1:, :] = (
            valid_depth[1:, :]
            &
            valid_depth[:-1, :]
        )

        valid_gradient = (
            neighbor_valid_x
            |
            neighbor_valid_y
        )

        depth_gradient[
            ~valid_gradient
        ] = 0.0

        # PersonaCraft-style:
        # retain depth discontinuity only in
        # occlusion regions.
        occlusion_depth_gradient = (
            depth_gradient
            *
            occlusion_mask.astype(
                np.float32
            )
        )

        return {
            "height":
                H,

            "width":
                W,

            "K_runtime":
                K.astype(
                    np.float32
                ),

            "depth_scene":
                scene_depth,

            "normal_scene":
                scene_normal,

            # SMPL amodal geometry ownership
            "semantic_A":
                person_a[
                    "mask"
                ].astype(
                    np.uint8
                ),

            "semantic_B":
                person_b[
                    "mask"
                ].astype(
                    np.uint8
                ),

            # currently visible front surface owner
            "front_owner":
                front_owner,

            # official Hi4D contact projected from vertices
            "contact_A":
                person_a[
                    "contact"
                ],

            "contact_B":
                person_b[
                    "contact"
                ],

            # PersonaCraft-style
            "surface_layer_count":
                layer_count_np,

            "occlusion_mask":
                occlusion_mask,

            "occlusion_depth_gradient":
                occlusion_depth_gradient,

            # Person-to-person overlap / direction.
            "interperson_overlap":
                interperson_overlap,

            "A_occluded_by_B":
                A_occluded_by_B,

            "B_occluded_by_A":
                B_occluded_by_A,

            # useful auxiliary
            "depth_A":
                person_a[
                    "depth"
                ],

            "depth_B":
                person_b[
                    "depth"
                ],
        }


# ============================================================
# QC
# ============================================================

def depth_vis(x):

    out = np.zeros(
        x.shape,
        dtype=np.uint8
    )

    m = x > 0

    if not np.any(m):
        return out

    vals = x[m]

    lo = np.percentile(
        vals,
        1
    )

    hi = np.percentile(
        vals,
        99
    )

    if hi <= lo:
        hi = lo + 1e-6

    out[m] = (
        np.clip(
            (
                x[m] - lo
            )
            /
            (
                hi - lo
            ),
            0,
            1
        )
        * 255
    ).astype(
        np.uint8
    )

    return out


def normal_vis(
    n,
    mask,
):

    x = (
        (
            np.clip(
                n,
                -1,
                1
            )
            + 1.0
        )
        * 127.5
    ).astype(
        np.uint8
    )

    x[
        ~mask
    ] = 0

    # numpy is RGB-like here;
    # cv2 output below expects BGR.
    return x[
        :, :, ::-1
    ]


def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True
    )

    parser.add_argument(
        "--contact-only",
        action="store_true"
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    rows = read_jsonl(
        args.manifest
    )

    if args.contact_only:
        rows = [
            x for x in rows
            if bool(
                x.get(
                    "is_contact",
                    False
                )
            )
        ]

    if not rows:
        raise RuntimeError(
            "no matching rows"
        )

    row = rows[0]

    renderer = (
        Hi4DRuntimeGeometry()
    )

    g = renderer.render(
        smpl_file=
            row["official_smpl"],

        camera_file=
            row["camera_file"],

        camera_id=
            row["camera_id"],

        out_h=
            args.height,

        out_w=
            args.width,
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        args.out
        / "runtime_geometry.npz",

        **g
    )

    cv2.imwrite(
        str(
            args.out
            / "depth.png"
        ),
        depth_vis(
            g[
                "depth_scene"
            ]
        )
    )

    mask = (
        g[
            "depth_scene"
        ] > 0
    )

    cv2.imwrite(
        str(
            args.out
            / "normal.png"
        ),
        normal_vis(
            g[
                "normal_scene"
            ],
            mask
        )
    )

    cv2.imwrite(
        str(
            args.out
            / "semantic_A.png"
        ),
        g[
            "semantic_A"
        ] * 255
    )

    cv2.imwrite(
        str(
            args.out
            / "semantic_B.png"
        ),
        g[
            "semantic_B"
        ] * 255
    )

    cv2.imwrite(
        str(
            args.out
            / "contact_A.png"
        ),
        (
            np.clip(
                g[
                    "contact_A"
                ],
                0,
                1
            )
            * 255
        ).astype(
            np.uint8
        )
    )

    cv2.imwrite(
        str(
            args.out
            / "contact_B.png"
        ),
        (
            np.clip(
                g[
                    "contact_B"
                ],
                0,
                1
            )
            * 255
        ).astype(
            np.uint8
        )
    )

    cv2.imwrite(
        str(
            args.out
            / "occlusion_mask.png"
        ),
        g[
            "occlusion_mask"
        ] * 255
    )

    # Continuous metric depth-gradient visualization.
    grad = g[
        "occlusion_depth_gradient"
    ]

    grad_vis = np.zeros_like(
        grad,
        dtype=np.uint8
    )

    gm = grad > 0

    if np.any(gm):

        hi = float(
            np.percentile(
                grad[gm],
                99
            )
        )

        if hi <= 0:
            hi = 1e-6

        grad_vis[gm] = (
            np.clip(
                grad[gm] / hi,
                0,
                1
            )
            * 255
        ).astype(
            np.uint8
        )

    cv2.imwrite(
        str(
            args.out
            / "occlusion_depth_gradient.png"
        ),
        grad_vis
    )

    cv2.imwrite(
        str(
            args.out
            / "interperson_overlap.png"
        ),
        g[
            "interperson_overlap"
        ] * 255
    )

    cv2.imwrite(
        str(
            args.out
            / "A_occluded_by_B.png"
        ),
        g[
            "A_occluded_by_B"
        ] * 255
    )

    cv2.imwrite(
        str(
            args.out
            / "B_occluded_by_A.png"
        ),
        g[
            "B_occluded_by_A"
        ] * 255
    )

    cv2.imwrite(
        str(
            args.out
            / "surface_layer_count.png"
        ),
        np.clip(
            g[
                "surface_layer_count"
            ].astype(
                np.int32
            )
            * 32,
            0,
            255
        ).astype(
            np.uint8
        )
    )

    print(
        "frame:",
        row["pair"],
        row["action"],
        row["frame_id"]
    )

    print(
        "runtime size:",
        args.height,
        "x",
        args.width
    )

    print(
        "contact A pixels:",
        int(
            (
                g[
                    "contact_A"
                ] > 0.05
            ).sum()
        )
    )

    print(
        "contact B pixels:",
        int(
            (
                g[
                    "contact_B"
                ] > 0.05
            ).sum()
        )
    )

    print(
        "occlusion pixels:",
        int(
            g[
                "occlusion_mask"
            ].sum()
        )
    )

    print(
        "max surface layers:",
        int(
            g[
                "surface_layer_count"
            ].max()
        )
    )

    print(
        "saved:",
        args.out
    )


if __name__ == "__main__":
    main()
