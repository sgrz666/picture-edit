import argparse
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.pose_preprocess import (
    preprocess_image_and_joints,
)
from pose_control.skeleton_renderer import (
    render_coco17,
)


def load_joints(path: Path):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not data or "keypoints" not in data[0]:
        raise ValueError(
            f"无有效关键点：{path}"
        )

    joints = data[0]["keypoints"]

    if len(joints) != 17:
        raise ValueError(
            f"关键点数量应为17，实际为{len(joints)}"
        )

    return joints


def make_overlay(
    image: Image.Image,
    skeleton: Image.Image,
):
    """把纯骨架覆盖到 RGB 上，仅用于检查空间对齐。"""
    mask = skeleton.convert("L").point(
        lambda value: 255 if value > 0 else 0
    )

    return Image.composite(
        skeleton,
        image,
        mask,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)

    rgb_dir = output_dir / "rgb"
    pose_dir = output_dir / "pose"
    overlay_dir = output_dir / "overlay"

    for directory in (
        rgb_dir,
        pose_dir,
        overlay_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    frames = sorted(
        sequence_dir.glob("*.png")
    )

    if not frames:
        raise RuntimeError("没有找到 RGB 帧")

    for frame_path in frames:
        json_path = (
            sequence_dir
            / "openpose_json"
            / f"{frame_path.name}.json"
        )

        image = Image.open(
            frame_path
        ).convert("RGB")

        joints = load_joints(
            json_path
        )

        image, joints = preprocess_image_and_joints(
            image=image,
            joints=joints,
            output_size=(512, 512),
        )

        skeleton = render_coco17(
            coco_joints=joints,
            source_width=512,
            source_height=512,
            output_width=512,
            output_height=512,
            confidence_threshold=0.20,
            line_width=6,
            joint_radius=4,
        )

        overlay = make_overlay(
            image,
            skeleton,
        )

        name = frame_path.name

        image.save(
            rgb_dir / name
        )

        skeleton.save(
            pose_dir / name
        )

        overlay.save(
            overlay_dir / name
        )

        print(name)

    print("Pose preprocess alignment test: PASS")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
