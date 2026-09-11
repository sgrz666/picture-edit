import argparse
import json
import pickle
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.pose_preprocess import (
    letterbox_image,
    preprocess_image_and_joints,
)
from pose_control.skeleton_renderer import render_openpose18


# ---------------------------------------------------------
# 人工检查过的 tiny-overfit 数据。
#
# /1 = A-pose/source video
# /2 = random-action/target video
#
# 第一阶段故意只保留姿态差异明显的 target，
# 不追求数据量。
# ---------------------------------------------------------

CASES = {
    "008_1": {
        "person": "008",
        "cloth": "1",
        "source_frame": 51,
        "target_frames": [
            377,
            1175,
            503,
            537,
            648,
            730,
        ],
    },
    "014_1": {
        "person": "014",
        "cloth": "1",
        "source_frame": 295,
        "target_frames": [
            673,
            940,
            794,
            877,
            1003,
            712,
        ],
    },
    "030_1": {
        "person": "030",
        "cloth": "1",
        "source_frame": 426,
        "target_frames": [
            226,
            377,
            1138,
            469,
            1171,
        ],
    },
}


def read_video_frame(video_path: Path, frame_id: int) -> Image.Image:
    """
    用 FFmpeg 精确读取指定视频帧。

    不引入 OpenCV，避免给 DeepGen 环境增加无关依赖。
    """
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_id})",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
    )

    if not result.stdout:
        raise RuntimeError(
            f"读取视频帧失败：{video_path} frame={frame_id}"
        )

    return Image.open(
        BytesIO(result.stdout)
    ).convert("RGB")


def load_kps(path: Path) -> np.ndarray:
    """
    读取 iPER kps.pkl。

    已验证格式：
        [T, 19, 2]
    """
    with open(path, "rb") as file:
        data = pickle.load(
            file,
            encoding="latin1",
        )

    kps = np.asarray(
        data["kps"],
        dtype=np.float32,
    )

    if (
        kps.ndim != 3
        or kps.shape[1:] != (19, 2)
    ):
        raise ValueError(
            f"异常 iPER kps shape：{kps.shape}"
        )

    return kps


def normalized_kps_to_pixels(
    kps: np.ndarray,
    width: int,
    height: int,
):
    """
    将 iPER 归一化坐标恢复为 RGB 像素坐标。

    该公式已经通过三组人物的 RGB overlay 验证。
    """
    joints = []

    for x, y in kps:
        px = (
            (float(x) + 1.0)
            * 0.5
            * (width - 1)
        )

        py = (
            (float(y) + 1.0)
            * 0.5
            * (height - 1)
        )

        joints.append(
            [px, py, 1.0]
        )

    return joints


def lsp14_to_openpose18(joints):
    """
    将 iPER/HMR 前14个身体点映射到
    OpenPose-18 的 body 部分。

    iPER 前14点顺序：

      0  right ankle
      1  right knee
      2  right hip
      3  left hip
      4  left knee
      5  left ankle
      6  right wrist
      7  right elbow
      8  right shoulder
      9  left shoulder
     10  left elbow
     11  left wrist
     12  neck
     13  head top

    V1 是 body-only，因此暂时不使用 head top
    和其余5个面部点。

    这样以后可以直接让 SMPL-X 映射到相同的
    OpenPose body condition，而无需修改 Pose Adapter。
    """
    if len(joints) < 14:
        raise ValueError(
            f"至少需要14个身体关键点，实际为 {len(joints)}"
        )

    invalid = [-1.0, -1.0, -1.0]

    openpose = [
        invalid.copy()
        for _ in range(18)
    ]

    # neck
    openpose[1] = list(joints[12])

    # right arm
    openpose[2] = list(joints[8])
    openpose[3] = list(joints[7])
    openpose[4] = list(joints[6])

    # left arm
    openpose[5] = list(joints[9])
    openpose[6] = list(joints[10])
    openpose[7] = list(joints[11])

    # right leg
    openpose[8] = list(joints[2])
    openpose[9] = list(joints[1])
    openpose[10] = list(joints[0])

    # left leg
    openpose[11] = list(joints[3])
    openpose[12] = list(joints[4])
    openpose[13] = list(joints[5])

    return openpose


def save_source(
    video_path: Path,
    frame_id: int,
    output_path: Path,
    resolution: int,
):
    """提取并保存 source RGB。"""
    image = read_video_frame(
        video_path,
        frame_id,
    )

    image, _, _, _ = letterbox_image(
        image=image,
        output_size=(
            resolution,
            resolution,
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)


def save_target_and_pose(
    video_path: Path,
    kps: np.ndarray,
    frame_id: int,
    target_path: Path,
    pose_path: Path,
    resolution: int,
):
    """
    同时生成 target RGB 与对应 pose map，
    保证二者使用完全相同的几何变换。
    """
    if frame_id >= len(kps):
        raise IndexError(
            f"frame={frame_id} 超过 kps 长度 {len(kps)}"
        )

    image = read_video_frame(
        video_path,
        frame_id,
    )

    width, height = image.size

    joints = normalized_kps_to_pixels(
        kps[frame_id],
        width,
        height,
    )

    # RGB 与 joints 同步 resize / letterbox。
    image, joints = preprocess_image_and_joints(
        image=image,
        joints=joints,
        output_size=(
            resolution,
            resolution,
        ),
    )

    openpose = lsp14_to_openpose18(
        joints
    )

    pose = render_openpose18(
        joints=openpose,
        source_width=resolution,
        source_height=resolution,
        output_width=resolution,
        output_height=resolution,
        confidence_threshold=0.05,
        line_width=6,
        joint_radius=4,
    )

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(target_path)
    pose.save(pose_path)


def validate_dataset(
    output_root: Path,
    manifest: dict,
    resolution: int,
):
    """
    做最基本的数据完整性检查。

    不评价模型效果，只保证训练前的数据文件没有坏。
    """
    for sample in manifest["samples"]:
        for key in (
            "source",
            "target",
            "pose",
        ):
            path = (
                output_root
                / sample[key]
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            with Image.open(path) as image:
                if image.size != (
                    resolution,
                    resolution,
                ):
                    raise ValueError(
                        f"{path} 尺寸异常：{image.size}"
                    )

                if (
                    key == "pose"
                    and image.getbbox() is None
                ):
                    raise ValueError(
                        f"空 Pose Map：{path}"
                    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iper-root",
        type=Path,
        default=Path(
            "/home/lvjy/projects/iper_data"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/lvjy/projects/iper_data/"
            "tiny_pose_adapter_v1"
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    iper_root = args.iper_root
    output_root = args.output_dir
    resolution = args.resolution

    samples = []

    for identity, config in CASES.items():
        person = config["person"]
        cloth = config["cloth"]

        source_video = (
            iper_root
            / "candidate_videos"
            / f"{person}_{cloth}_1.mp4"
        )

        target_video = (
            iper_root
            / "candidate_videos"
            / f"{person}_{cloth}_2.mp4"
        )

        kps_path = (
            iper_root
            / "smpls"
            / person
            / cloth
            / "2"
            / "kps.pkl"
        )

        for path in (
            source_video,
            target_video,
            kps_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    path
                )

        identity_dir = (
            output_root
            / identity
        )

        source_path = (
            identity_dir
            / "source.png"
        )

        save_source(
            video_path=source_video,
            frame_id=config["source_frame"],
            output_path=source_path,
            resolution=resolution,
        )

        kps = load_kps(
            kps_path
        )

        for frame_id in config["target_frames"]:
            name = f"frame_{frame_id:06d}.png"

            target_path = (
                identity_dir
                / "targets"
                / name
            )

            pose_path = (
                identity_dir
                / "poses"
                / name
            )

            save_target_and_pose(
                video_path=target_video,
                kps=kps,
                frame_id=frame_id,
                target_path=target_path,
                pose_path=pose_path,
                resolution=resolution,
            )

            samples.append(
                {
                    "identity": identity,
                    "source_frame": config[
                        "source_frame"
                    ],
                    "target_frame": frame_id,
                    "source": source_path.relative_to(
                        output_root
                    ).as_posix(),
                    "target": target_path.relative_to(
                        output_root
                    ).as_posix(),
                    "pose": pose_path.relative_to(
                        output_root
                    ).as_posix(),
                }
            )

    manifest = {
        "version": 1,
        "task": (
            "source_image_plus_target_pose_"
            "to_target_image"
        ),
        "resolution": resolution,
        "pose_format": (
            "openpose18_body_only"
        ),
        "num_identities": len(CASES),
        "num_samples": len(samples),
        "samples": samples,
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_root
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    validate_dataset(
        output_root,
        manifest,
        resolution,
    )

    print(
        "===== iPER TINY DATASET ====="
    )
    print(
        "identities :",
        len(CASES),
    )
    print(
        "samples    :",
        len(samples),
    )
    print(
        "resolution :",
        f"{resolution}x{resolution}",
    )
    print(
        "pose format:",
        manifest["pose_format"],
    )
    print(
        "manifest   :",
        manifest_path,
    )
    print(
        "Dataset validation: PASS"
    )


if __name__ == "__main__":
    main()
