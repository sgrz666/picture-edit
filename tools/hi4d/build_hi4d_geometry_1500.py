#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from PIL import Image
import smplx


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

MANIFEST = ROOT / "frames_1500.jsonl"

OUT_ROOT = ROOT / "processed_512_1500"

SMPL_MODEL_DIR = Path(
    "/home/shangguanrz/project/pic-edit/models/smpl"
)

RES = 512

# sequence-level bbox 外扩
MARGIN = 0.12

# 1 cm depth difference 才视为明确遮挡
OCC_EPS = 0.01

# SMPL silhouette 与官方 mask 最低可接受 IoU
MIN_IOU = 0.20


# ============================================================
# SMPL 24 joints
# ============================================================

SMPL_EDGES = [
    (0, 1), (0, 2), (0, 3),

    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),

    (3, 6), (6, 9), (9, 12), (12, 15),

    (9, 13), (13, 16),
    (16, 18), (18, 20), (20, 22),

    (9, 14), (14, 17),
    (17, 19), (19, 21), (21, 23),
]


# ============================================================
# BASIC IO
# ============================================================

def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


def mkdir_parent(path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )


def load_gray(path):
    x = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if x is None:
        raise RuntimeError(
            f"Cannot read {path}"
        )

    return x


def load_bgr(path):
    x = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if x is None:
        raise RuntimeError(
            f"Cannot read {path}"
        )

    return x


def save_gray(path, x):
    mkdir_parent(path)

    Image.fromarray(
        x.astype(np.uint8),
        mode="L"
    ).save(path)


def save_rgb(path, x):
    mkdir_parent(path)

    Image.fromarray(
        x.astype(np.uint8),
        mode="RGB"
    ).save(path)


def save_npy(path, x):
    mkdir_parent(path)
    np.save(path, x)


# ============================================================
# CAMERA
# ============================================================

def load_camera(camera_file, camera_id):
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

        dist_all = np.asarray(
            d["dist_coeffs"]
        )

    idx = np.where(
        ids == int(camera_id)
    )[0]

    if len(idx) != 1:
        raise RuntimeError(
            f"camera {camera_id} not found: "
            f"{ids.tolist()}"
        )

    i = int(idx[0])

    return (
        K_all[i].astype(np.float64),
        E_all[i].astype(np.float64),
        dist_all[i].astype(np.float64),
    )


def extrinsic_candidate(E, mode):

    R0 = E[:, :3]
    t0 = E[:, 3]

    if mode == "direct":
        return R0, t0

    if mode == "inverse":
        R = R0.T
        t = -R @ t0

        return R, t

    raise ValueError(mode)


def world_to_camera(x, R, t):
    return x @ R.T + t[None]


def project(
    points_world,
    K,
    R,
    t,
):
    cam = world_to_camera(
        points_world,
        R,
        t
    )

    z = cam[:, 2]

    valid = z > 1e-5

    uv = np.zeros(
        (len(cam), 2),
        dtype=np.float64
    )

    uv[valid, 0] = (
        K[0, 0]
        * cam[valid, 0]
        / z[valid]
        + K[0, 2]
    )

    uv[valid, 1] = (
        K[1, 1]
        * cam[valid, 1]
        / z[valid]
        + K[1, 2]
    )

    return uv, z, valid


# ============================================================
# SEQUENCE FIXED 512 CROP
# ============================================================

def sequence_transform(rows):

    img0 = load_bgr(
        rows[0]["rgb"]
    )

    H, W = img0.shape[:2]

    xs_all = []
    ys_all = []

    for row in rows:

        a = load_gray(
            row["person_mask_A"]
        ) > 0

        b = load_gray(
            row["person_mask_B"]
        ) > 0

        m = a | b

        ys, xs = np.where(m)

        if len(xs):
            xs_all.extend(
                [int(xs.min()), int(xs.max())]
            )

            ys_all.extend(
                [int(ys.min()), int(ys.max())]
            )

    if not xs_all:
        cx = W / 2
        cy = H / 2
        side = max(W, H)

    else:
        xmin = min(xs_all)
        xmax = max(xs_all)

        ymin = min(ys_all)
        ymax = max(ys_all)

        bw = xmax - xmin + 1
        bh = ymax - ymin + 1

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        side = max(
            bw,
            bh
        ) * (1 + 2 * MARGIN)

    side = max(
        side,
        32
    )

    x0 = cx - side / 2
    y0 = cy - side / 2

    scale = RES / side

    A = np.array(
        [
            [
                scale,
                0,
                -x0 * scale
            ],
            [
                0,
                scale,
                -y0 * scale
            ],
            [
                0,
                0,
                1
            ],
        ],
        dtype=np.float64
    )

    return {
        "width": W,
        "height": H,
        "x0": float(x0),
        "y0": float(y0),
        "side": float(side),
        "scale": float(scale),
        "A": A,
    }


def warp_rgb(x, A):
    return cv2.warpAffine(
        x,
        A[:2].astype(np.float32),
        (RES, RES),

        flags=cv2.INTER_AREA,

        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_mask(x, A):
    return cv2.warpAffine(
        x,
        A[:2].astype(np.float32),
        (RES, RES),

        flags=cv2.INTER_NEAREST,

        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# NVDIFFRAST
# ============================================================

def vertex_normals(
    verts,
    faces
):

    tri = verts[faces]

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

    vn = F.normalize(
        vn,
        dim=1,
        eps=1e-8
    )

    return vn


def camera_to_clip(
    cam,
    K
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
        2.0
        * u
        / (RES - 1)
        - 1.0
    )

    y_ndc = (
        1.0
        - 2.0
        * v
        / (RES - 1)
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

    return clip


def render_person(
    ctx,
    verts_world,
    K,
    R_np,
    t_np,
    faces_long,
    faces_i32,
    device,
):

    v = torch.from_numpy(
        verts_world.astype(
            np.float32
        )
    ).to(device)

    R = torch.from_numpy(
        R_np.astype(
            np.float32
        )
    ).to(device)

    t = torch.from_numpy(
        t_np.astype(
            np.float32
        )
    ).to(device)

    cam = (
        v @ R.T
        + t[None]
    )

    positive = float(
        (
            cam[:, 2] > 0
        ).float().mean()
    )

    if positive < 0.8:
        raise RuntimeError(
            f"positive Z ratio "
            f"{positive:.3f}"
        )

    normals = vertex_normals(
        cam,
        faces_long
    )

    clip = camera_to_clip(
        cam,
        K
    )

    rast, _ = dr.rasterize(
        ctx,
        clip[None],
        faces_i32,
        resolution=[
            RES,
            RES
        ]
    )

    # nvdiffrast requires attr / rast / tri to be contiguous
    rast = rast.contiguous()
    faces_i32 = faces_i32.contiguous()

    depth_attr = (
        cam[:, 2:3][None]
        .contiguous()
    )

    normal_attr = (
        normals[None]
        .contiguous()
    )

    depth, _ = dr.interpolate(
        depth_attr,
        rast,
        faces_i32
    )

    normal, _ = dr.interpolate(
        normal_attr,
        rast,
        faces_i32
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

    return {
        "depth":
            depth.detach()
            .cpu()
            .numpy()
            .astype(np.float32),

        "normal":
            normal.detach()
            .cpu()
            .numpy()
            .astype(np.float32),

        "mask":
            mask.detach()
            .cpu()
            .numpy(),
    }


def render_scene(
    ctx,
    va_world,
    vb_world,
    K,
    R_np,
    t_np,
    faces_long,
    device,
):

    va = torch.from_numpy(
        va_world.astype(
            np.float32
        )
    ).to(device)

    vb = torch.from_numpy(
        vb_world.astype(
            np.float32
        )
    ).to(device)

    R = torch.from_numpy(
        R_np.astype(
            np.float32
        )
    ).to(device)

    t = torch.from_numpy(
        t_np.astype(
            np.float32
        )
    ).to(device)

    va = (
        va @ R.T
        + t[None]
    )

    vb = (
        vb @ R.T
        + t[None]
    )

    na = vertex_normals(
        va,
        faces_long
    )

    nb = vertex_normals(
        vb,
        faces_long
    )

    V = va.shape[0]
    NF = faces_long.shape[0]

    verts = torch.cat(
        [va, vb],
        dim=0
    )

    normals = torch.cat(
        [na, nb],
        dim=0
    )

    faces_scene = torch.cat(
        [
            faces_long,
            faces_long + V
        ],
        dim=0
    )

    faces_i32 = faces_scene.to(
        dtype=torch.int32
    )

    clip = camera_to_clip(
        verts,
        K
    )

    rast, _ = dr.rasterize(
        ctx,
        clip[None],
        faces_i32,
        resolution=[
            RES,
            RES
        ]
    )

    # nvdiffrast requires attr / rast / tri to be contiguous
    rast = rast.contiguous()
    faces_i32 = faces_i32.contiguous()

    depth_attr = (
        verts[:, 2:3][None]
        .contiguous()
    )

    normal_attr = (
        normals[None]
        .contiguous()
    )

    depth, _ = dr.interpolate(
        depth_attr,
        rast,
        faces_i32
    )

    normal, _ = dr.interpolate(
        normal_attr,
        rast,
        faces_i32
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

    person_id = torch.zeros(
        (
            RES,
            RES
        ),
        dtype=torch.uint8,
        device=device
    )

    person_id[
        valid
        & (tri_id < NF)
    ] = 1

    person_id[
        valid
        & (tri_id >= NF)
    ] = 2

    mask = valid

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
        mask,
        depth,
        torch.zeros_like(depth)
    )

    normal = torch.where(
        mask[..., None],
        normal,
        torch.zeros_like(normal)
    )

    return {
        "depth":
            depth.detach()
            .cpu()
            .numpy()
            .astype(np.float32),

        "normal":
            normal.detach()
            .cpu()
            .numpy()
            .astype(np.float32),

        "mask":
            mask.detach()
            .cpu()
            .numpy(),

        "person_id":
            person_id.detach()
            .cpu()
            .numpy(),
    }


# ============================================================
# RASTER ORIENTATION
# ============================================================

def orient(x, mode):

    if mode == "none":
        return x

    if mode == "flip_y":
        return np.flip(
            x,
            axis=0
        ).copy()

    if mode == "flip_x":
        return np.flip(
            x,
            axis=1
        ).copy()

    if mode == "flip_xy":
        return np.flip(
            np.flip(
                x,
                axis=0
            ),
            axis=1
        ).copy()

    raise ValueError(mode)


def mask_iou(a, b):

    a = a.astype(bool)
    b = b.astype(bool)

    union = (
        a | b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        (
            a & b
        ).sum()
        / union
    )


# ============================================================
# POSE
# ============================================================

def pose_data(
    joints,
    K,
    R,
    t
):

    uv, z, valid = project(
        joints,
        K,
        R,
        t
    )

    valid = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < RES)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < RES)
    )

    return np.concatenate(
        [
            uv.astype(
                np.float32
            ),

            z[:, None].astype(
                np.float32
            ),

            valid[:, None].astype(
                np.float32
            ),
        ],
        axis=1
    )


def draw_pose(
    pose,
    color
):

    img = np.zeros(
        (
            RES,
            RES,
            3
        ),
        dtype=np.uint8
    )

    valid = (
        pose[:, 3] > 0.5
    )

    for a, b in SMPL_EDGES:

        if not (
            valid[a]
            and valid[b]
        ):
            continue

        pa = tuple(
            np.round(
                pose[a, :2]
            ).astype(int)
        )

        pb = tuple(
            np.round(
                pose[b, :2]
            ).astype(int)
        )

        cv2.line(
            img,
            pa,
            pb,
            color,
            3,
            cv2.LINE_AA
        )

    for i in range(
        len(pose)
    ):

        if not valid[i]:
            continue

        p = tuple(
            np.round(
                pose[i, :2]
            ).astype(int)
        )

        cv2.circle(
            img,
            p,
            4,
            color,
            -1,
            cv2.LINE_AA
        )

    return img


# ============================================================
# VIS
# ============================================================

def normal_png(
    n,
    mask
):

    rgb = (
        (
            np.clip(
                n,
                -1,
                1
            )
            + 1
        )
        * 127.5
    ).astype(
        np.uint8
    )

    rgb[
        ~mask.astype(bool)
    ] = 0

    return rgb


def depth_mm(depth):

    return np.clip(
        depth * 1000.0,
        0,
        65535
    ).astype(
        np.uint16
    )


def depth_vis(depth):

    out = np.zeros(
        depth.shape,
        dtype=np.uint8
    )

    m = depth > 0

    if not np.any(m):
        return out

    v = depth[m]

    lo = np.percentile(
        v,
        1
    )

    hi = np.percentile(
        v,
        99
    )

    if hi <= lo:
        hi = lo + 1e-6

    x = np.clip(
        (depth - lo)
        / (hi - lo),
        0,
        1
    )

    out[m] = (
        x[m] * 255
    ).astype(
        np.uint8
    )

    return out


# ============================================================
# PATHS
# ============================================================

def frame_paths(row):

    base = (
        OUT_ROOT
        / row["pair"]
        / row["action"]
    )

    f = row[
        "frame_stem"
    ]

    return {
        "rgb":
            base / "rgb" /
            f"{f}.png",

        "pose_a":
            base / "pose" /
            "A" /
            f"{f}.png",

        "pose_b":
            base / "pose" /
            "B" /
            f"{f}.png",

        "pose_combined":
            base / "pose" /
            "combined" /
            f"{f}.png",

        "pose_a_npy":
            base / "pose" /
            "A_npy" /
            f"{f}.npy",

        "pose_b_npy":
            base / "pose" /
            "B_npy" /
            f"{f}.npy",

        "depth_a":
            base / "depth" /
            "A_npy" /
            f"{f}.npy",

        "depth_b":
            base / "depth" /
            "B_npy" /
            f"{f}.npy",

        "depth_scene":
            base / "depth" /
            "scene_npy" /
            f"{f}.npy",

        "depth_scene_png":
            base / "depth" /
            "scene_mm" /
            f"{f}.png",

        "depth_vis":
            base / "depth" /
            "scene_vis" /
            f"{f}.png",

        "normal_a":
            base / "normal" /
            "A_npy" /
            f"{f}.npy",

        "normal_b":
            base / "normal" /
            "B_npy" /
            f"{f}.npy",

        "normal_scene":
            base / "normal" /
            "scene_npy" /
            f"{f}.npy",

        "normal_a_png":
            base / "normal" /
            "A" /
            f"{f}.png",

        "normal_b_png":
            base / "normal" /
            "B" /
            f"{f}.png",

        "normal_scene_png":
            base / "normal" /
            "scene" /
            f"{f}.png",

        "person_mask_a":
            base / "person_mask" /
            "A" /
            f"{f}.png",

        "person_mask_b":
            base / "person_mask" /
            "B" /
            f"{f}.png",

        "person_id":
            base / "person_id" /
            f"{f}.png",

        "smpl_mask_a":
            base / "smpl_mask" /
            "A" /
            f"{f}.png",

        "smpl_mask_b":
            base / "smpl_mask" /
            "B" /
            f"{f}.png",

        "scene_person_id":
            base / "scene_person_id" /
            f"{f}.png",

        "occ_a_by_b":
            base / "occlusion" /
            "A_by_B" /
            f"{f}.png",

        "occ_b_by_a":
            base / "occlusion" /
            "B_by_A" /
            f"{f}.png",

        "occ_union":
            base / "occlusion" /
            "union" /
            f"{f}.png",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    print(
        "======================================"
    )

    print(
        "Hi4D GEOMETRY 1500"
    )

    print(
        "======================================"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        "cuda"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # --------------------------------------------------------
    # SMPL topology
    # --------------------------------------------------------

    model = smplx.create(
        str(SMPL_MODEL_DIR),
        model_type="smpl",
        gender="neutral",
        ext="pkl"
    )

    faces_np = np.asarray(
        model.faces,
        dtype=np.int64
    )

    print(
        "SMPL faces:",
        faces_np.shape
    )

    faces_long = torch.from_numpy(
        faces_np
    ).to(
        device=device,
        dtype=torch.long
    )

    faces_i32 = faces_long.to(
        torch.int32
    )

    ctx = dr.RasterizeCudaContext(
        device=device
    )

    rows = read_jsonl(
        MANIFEST
    )

    print(
        "frames:",
        len(rows)
    )

    if len(rows) != 1500:
        raise RuntimeError(
            f"expected 1500, "
            f"got {len(rows)}"
        )

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["split"],
            row["pair"],
            row["action"],
            int(row["camera_id"])
        )

        groups[key].append(
            row
        )

    print(
        "sequences:",
        len(groups)
    )

    if len(groups) != 30:
        raise RuntimeError(
            "expected 30 sequences"
        )

    processed = []
    sequence_meta = []

    # --------------------------------------------------------
    # SEQUENCES
    # --------------------------------------------------------

    for si, (
        key,
        seq_rows
    ) in enumerate(
        sorted(
            groups.items()
        ),
        start=1
    ):

        split, pair, action, cam_id = key

        seq_rows = sorted(
            seq_rows,
            key=lambda x: x[
                "frame_id"
            ]
        )

        print()
        print(
            f"[{si}/30] "
            f"{split} "
            f"{pair}/{action} "
            f"cam={cam_id} "
            f"n={len(seq_rows)}"
        )

        # ----------------------------------------------------
        # fixed transform
        # ----------------------------------------------------

        tf = sequence_transform(
            seq_rows
        )

        A = tf["A"]

        K0, E, dist = load_camera(
            seq_rows[0][
                "camera_file"
            ],
            cam_id
        )

        K = (
            A @ K0
        )

        # ----------------------------------------------------
        # first frame:
        # automatically determine camera convention
        # + raster orientation
        # ----------------------------------------------------

        first = seq_rows[0]

        with np.load(
            first[
                "official_smpl"
            ],
            allow_pickle=True
        ) as d:

            verts0 = np.asarray(
                d["verts"],
                dtype=np.float32
            )

        official_a = (
            warp_mask(
                load_gray(
                    first[
                        "person_mask_A"
                    ]
                ),
                A
            ) > 127
        )

        official_b = (
            warp_mask(
                load_gray(
                    first[
                        "person_mask_B"
                    ]
                ),
                A
            ) > 127
        )

        official_union = (
            official_a
            | official_b
        )

        best = None

        for extr_mode in [
            "direct",
            "inverse"
        ]:

            R, t = (
                extrinsic_candidate(
                    E,
                    extr_mode
                )
            )

            try:
                scene0 = render_scene(
                    ctx,
                    verts0[0],
                    verts0[1],
                    K,
                    R,
                    t,
                    faces_long,
                    device
                )

            except Exception as e:
                print(
                    f"  [RENDER FAILED] "
                    f"extrinsic={extr_mode}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            for ori in [
                "none",
                "flip_y",
                "flip_x",
                "flip_xy"
            ]:

                m = orient(
                    scene0[
                        "mask"
                    ],
                    ori
                )

                score = mask_iou(
                    m,
                    official_union
                )

                candidate = (
                    score,
                    extr_mode,
                    ori,
                    R,
                    t
                )

                if (
                    best is None
                    or score > best[0]
                ):
                    best = candidate

        if best is None:
            raise RuntimeError(
                f"{pair}/{action}: "
                "camera convention failed"
            )

        (
            best_iou,
            extr_mode,
            orientation,
            R,
            t
        ) = best

        print(
            "  extrinsic:",
            extr_mode
        )

        print(
            "  raster:",
            orientation
        )

        print(
            "  mask IoU:",
            round(
                best_iou,
                4
            )
        )

        if best_iou < MIN_IOU:
            raise RuntimeError(
                f"{pair}/{action}: "
                f"alignment IoU "
                f"{best_iou:.3f} "
                f"< {MIN_IOU}"
            )

        seq_meta = {
            "split": split,
            "pair": pair,
            "action": action,

            "camera_id":
                cam_id,

            "num_frames":
                len(seq_rows),

            "extrinsic_mode":
                extr_mode,

            "raster_orientation":
                orientation,

            "alignment_iou":
                float(best_iou),

            "original_K":
                K0.tolist(),

            "K_512":
                K.tolist(),

            "R_world_to_camera":
                R.tolist(),

            "t_world_to_camera":
                t.tolist(),

            "dist_coeffs":
                dist.tolist(),

            "A_original_to_512":
                A.tolist(),

            "crop": {
                "x0":
                    tf["x0"],

                "y0":
                    tf["y0"],

                "side":
                    tf["side"],

                "scale":
                    tf["scale"],
            },

            "normal_space":
                "camera",

            "depth_unit":
                "meter",
        }

        seq_json = (
            OUT_ROOT
            / pair
            / action
            / "sequence_meta.json"
        )

        mkdir_parent(
            seq_json
        )

        with open(
            seq_json,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                seq_meta,
                f,
                indent=2,
                ensure_ascii=False
            )

        sequence_meta.append(
            seq_meta
        )

        # ----------------------------------------------------
        # FRAMES
        # ----------------------------------------------------

        for fi, row in enumerate(
            seq_rows,
            start=1
        ):

            P = frame_paths(
                row
            )

            # resume support
            required = [
                P["rgb"],
                P["pose_combined"],
                P["depth_scene"],
                P["normal_scene"],
                P["occ_union"]
            ]

            if (
                not args.overwrite
                and all(
                    x.exists()
                    for x in required
                )
            ):

                out = dict(row)

                out.update({
                    "processed_rgb":
                        str(P["rgb"]),

                    "pose_A":
                        str(P["pose_a"]),

                    "pose_B":
                        str(P["pose_b"]),

                    "pose_combined":
                        str(
                            P[
                                "pose_combined"
                            ]
                        ),

                    "scene_depth":
                        str(
                            P[
                                "depth_scene"
                            ]
                        ),

                    "scene_normal":
                        str(
                            P[
                                "normal_scene"
                            ]
                        ),

                    "person_mask_A_512":
                        str(
                            P[
                                "person_mask_a"
                            ]
                        ),

                    "person_mask_B_512":
                        str(
                            P[
                                "person_mask_b"
                            ]
                        ),

                    "occlusion":
                        str(
                            P[
                                "occ_union"
                            ]
                        ),
                })

                processed.append(
                    out
                )

                continue

            with np.load(
                row[
                    "official_smpl"
                ],
                allow_pickle=True
            ) as d:

                verts = np.asarray(
                    d["verts"],
                    dtype=np.float32
                )

                joints = np.asarray(
                    d["joints_3d"],
                    dtype=np.float32
                )

            if verts.shape != (
                2,
                6890,
                3
            ):
                raise RuntimeError(
                    f"bad verts: "
                    f"{verts.shape}"
                )

            # ------------------------------------------------
            # render
            # ------------------------------------------------

            pa = render_person(
                ctx,
                verts[0],
                K,
                R,
                t,
                faces_long,
                faces_i32,
                device
            )

            pb = render_person(
                ctx,
                verts[1],
                K,
                R,
                t,
                faces_long,
                faces_i32,
                device
            )

            scene = render_scene(
                ctx,
                verts[0],
                verts[1],
                K,
                R,
                t,
                faces_long,
                device
            )

            for obj in [
                pa,
                pb,
                scene
            ]:

                for name in list(
                    obj.keys()
                ):

                    if name in [
                        "depth",
                        "normal",
                        "mask",
                        "person_id"
                    ]:

                        obj[name] = orient(
                            obj[name],
                            orientation
                        )

            # ------------------------------------------------
            # RGB + official masks
            # ------------------------------------------------

            rgb = warp_rgb(
                load_bgr(
                    row["rgb"]
                ),
                A
            )

            ma = (
                warp_mask(
                    load_gray(
                        row[
                            "person_mask_A"
                        ]
                    ),
                    A
                ) > 127
            )

            mb = (
                warp_mask(
                    load_gray(
                        row[
                            "person_mask_B"
                        ]
                    ),
                    A
                ) > 127
            )

            mkdir_parent(
                P["rgb"]
            )

            cv2.imwrite(
                str(P["rgb"]),
                rgb
            )

            save_gray(
                P[
                    "person_mask_a"
                ],
                ma.astype(
                    np.uint8
                ) * 255
            )

            save_gray(
                P[
                    "person_mask_b"
                ],
                mb.astype(
                    np.uint8
                ) * 255
            )

            pid = np.zeros(
                (
                    RES,
                    RES
                ),
                dtype=np.uint8
            )

            pid[ma] = 1
            pid[mb] = 2

            save_gray(
                P["person_id"],
                pid
            )

            # ------------------------------------------------
            # pose
            # ------------------------------------------------

            pose_a = pose_data(
                joints[0],
                K,
                R,
                t
            )

            pose_b = pose_data(
                joints[1],
                K,
                R,
                t
            )

            save_npy(
                P["pose_a_npy"],
                pose_a
            )

            save_npy(
                P["pose_b_npy"],
                pose_b
            )

            white = (
                255,
                255,
                255
            )

            pose_a_img = draw_pose(
                pose_a,
                white
            )

            pose_b_img = draw_pose(
                pose_b,
                white
            )

            # OpenCV BGR:
            # A red, B blue
            comb = np.maximum(
                draw_pose(
                    pose_a,
                    (
                        0,
                        0,
                        255
                    )
                ),
                draw_pose(
                    pose_b,
                    (
                        255,
                        0,
                        0
                    )
                )
            )

            mkdir_parent(
                P["pose_a"]
            )

            cv2.imwrite(
                str(P["pose_a"]),
                pose_a_img
            )

            mkdir_parent(
                P["pose_b"]
            )

            cv2.imwrite(
                str(P["pose_b"]),
                pose_b_img
            )

            mkdir_parent(
                P["pose_combined"]
            )

            cv2.imwrite(
                str(
                    P[
                        "pose_combined"
                    ]
                ),
                comb
            )

            # ------------------------------------------------
            # depth
            # ------------------------------------------------

            save_npy(
                P["depth_a"],
                pa["depth"]
            )

            save_npy(
                P["depth_b"],
                pb["depth"]
            )

            save_npy(
                P["depth_scene"],
                scene["depth"]
            )

            mkdir_parent(
                P[
                    "depth_scene_png"
                ]
            )

            cv2.imwrite(
                str(
                    P[
                        "depth_scene_png"
                    ]
                ),
                depth_mm(
                    scene["depth"]
                )
            )

            save_gray(
                P["depth_vis"],
                depth_vis(
                    scene["depth"]
                )
            )

            # ------------------------------------------------
            # normal
            # ------------------------------------------------

            save_npy(
                P["normal_a"],
                pa["normal"].astype(
                    np.float16
                )
            )

            save_npy(
                P["normal_b"],
                pb["normal"].astype(
                    np.float16
                )
            )

            save_npy(
                P[
                    "normal_scene"
                ],
                scene[
                    "normal"
                ].astype(
                    np.float16
                )
            )

            save_rgb(
                P["normal_a_png"],
                normal_png(
                    pa["normal"],
                    pa["mask"]
                )
            )

            save_rgb(
                P["normal_b_png"],
                normal_png(
                    pb["normal"],
                    pb["mask"]
                )
            )

            save_rgb(
                P[
                    "normal_scene_png"
                ],
                normal_png(
                    scene["normal"],
                    scene["mask"]
                )
            )

            # ------------------------------------------------
            # SMPL mask / scene ID
            # ------------------------------------------------

            save_gray(
                P["smpl_mask_a"],
                pa["mask"].astype(
                    np.uint8
                ) * 255
            )

            save_gray(
                P["smpl_mask_b"],
                pb["mask"].astype(
                    np.uint8
                ) * 255
            )

            save_gray(
                P[
                    "scene_person_id"
                ],
                scene[
                    "person_id"
                ]
            )

            # ------------------------------------------------
            # occlusion
            # ------------------------------------------------

            da = pa["depth"]
            db = pb["depth"]

            va = da > 0
            vb = db > 0

            overlap = va & vb

            a_by_b = (
                overlap
                & (
                    db
                    < da
                    - OCC_EPS
                )
            )

            b_by_a = (
                overlap
                & (
                    da
                    < db
                    - OCC_EPS
                )
            )

            occ = (
                a_by_b
                | b_by_a
            )

            save_gray(
                P["occ_a_by_b"],
                a_by_b.astype(
                    np.uint8
                ) * 255
            )

            save_gray(
                P["occ_b_by_a"],
                b_by_a.astype(
                    np.uint8
                ) * 255
            )

            save_gray(
                P["occ_union"],
                occ.astype(
                    np.uint8
                ) * 255
            )

            # ------------------------------------------------
            # manifest
            # ------------------------------------------------

            out = dict(row)

            out.update({
                "processed_rgb":
                    str(P["rgb"]),

                "pose_A":
                    str(P["pose_a"]),

                "pose_B":
                    str(P["pose_b"]),

                "pose_A_npy":
                    str(
                        P["pose_a_npy"]
                    ),

                "pose_B_npy":
                    str(
                        P["pose_b_npy"]
                    ),

                "pose_combined":
                    str(
                        P[
                            "pose_combined"
                        ]
                    ),

                "depth_A":
                    str(P["depth_a"]),

                "depth_B":
                    str(P["depth_b"]),

                "scene_depth":
                    str(
                        P["depth_scene"]
                    ),

                "normal_A":
                    str(P["normal_a"]),

                "normal_B":
                    str(P["normal_b"]),

                "scene_normal":
                    str(
                        P[
                            "normal_scene"
                        ]
                    ),

                "person_mask_A_512":
                    str(
                        P[
                            "person_mask_a"
                        ]
                    ),

                "person_mask_B_512":
                    str(
                        P[
                            "person_mask_b"
                        ]
                    ),

                "person_id":
                    str(P["person_id"]),

                "smpl_mask_A":
                    str(
                        P["smpl_mask_a"]
                    ),

                "smpl_mask_B":
                    str(
                        P["smpl_mask_b"]
                    ),

                "scene_person_id":
                    str(
                        P[
                            "scene_person_id"
                        ]
                    ),

                "occluded_A_by_B":
                    str(
                        P[
                            "occ_a_by_b"
                        ]
                    ),

                "occluded_B_by_A":
                    str(
                        P[
                            "occ_b_by_a"
                        ]
                    ),

                "occlusion":
                    str(
                        P["occ_union"]
                    ),

                "resolution":
                    512,

                "depth_unit":
                    "meter",

                "normal_space":
                    "camera",
            })

            processed.append(
                out
            )

            if (
                fi == 1
                or fi % 10 == 0
                or fi == len(
                    seq_rows
                )
            ):

                print(
                    f"  {fi}/"
                    f"{len(seq_rows)}"
                )

    # --------------------------------------------------------
    # FINAL MANIFESTS
    # --------------------------------------------------------

    processed = sorted(
        processed,
        key=lambda x: (
            x["split"],
            x["pair"],
            x["action"],
            x["frame_id"]
        )
    )

    write_jsonl(
        ROOT
        / "processed_frames_1500.jsonl",
        processed
    )

    write_jsonl(
        ROOT
        / "processed_sequences_1500.jsonl",
        sequence_meta
    )

    split_counts = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        rr = [
            x
            for x in processed
            if x["split"] == split
        ]

        split_counts[
            split
        ] = len(rr)

        write_jsonl(
            ROOT
            / (
                "processed_frames_1500_"
                + split
                + ".jsonl"
            ),
            rr
        )

    summary = {
        "total_frames":
            len(processed),

        "total_sequences":
            len(sequence_meta),

        "split_frames":
            split_counts,

        "resolution":
            512,

        "depth_unit":
            "meter",

        "normal_space":
            "camera",

        "person_id": {
            "0": "background",
            "1": "person_A",
            "2": "person_B",
        },

        "output_root":
            str(OUT_ROOT),
    }

    summary_path = (
        ROOT
        / "geometry_1500_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "======================================"
    )

    print(
        "COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
