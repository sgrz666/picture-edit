import argparse
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pose_control.skeleton_renderer import render_coco17


def load_coco17(json_path: Path):
    """
    读取当前 DisCo toy dataset 的 JSON。

    当前观察到的数据格式：
        [
            {
                "keypoints": [
                    [x, y, confidence],
                    ...
                ]
            }
        ]
    """

    data = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"JSON 中没有检测到人体：{json_path}"
        )

    keypoints = data[0].get(
        "keypoints"
    )

    if keypoints is None:
        raise ValueError(
            f"JSON 中不存在 keypoints：{json_path}"
        )

    if len(keypoints) != 17:
        raise ValueError(
            f"期望 17 个关键点，实际为 {len(keypoints)}：{json_path}"
        )

    return keypoints


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-dir",
        required=True,
        help="DisCo toy sequence 目录，例如 toy_dataset/000",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出纯 skeleton 图的目录",
    )

    args = parser.parse_args()

    sequence_dir = Path(
        args.sequence_dir
    )

    output_dir = Path(
        args.output_dir
    )

    pose_dir = (
        sequence_dir
        / "openpose_json"
    )

    if not sequence_dir.exists():
        raise FileNotFoundError(
            sequence_dir
        )

    if not pose_dir.exists():
        raise FileNotFoundError(
            pose_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        sequence_dir.glob("*.png")
    )

    if not image_paths:
        raise RuntimeError(
            f"未找到 RGB 帧：{sequence_dir}"
        )

    print(
        "===== SKELETON RENDER TEST ====="
    )

    print(
        "Sequence:",
        sequence_dir
    )

    print(
        "Frames:",
        len(image_paths)
    )

    for image_path in image_paths:
        json_path = (
            pose_dir
            / f"{image_path.name}.json"
        )

        if not json_path.exists():
            raise FileNotFoundError(
                json_path
            )

        with Image.open(
            image_path
        ) as image:
            source_width, source_height = (
                image.size
            )

        keypoints = load_coco17(
            json_path
        )

        skeleton = render_coco17(
            coco_joints=keypoints,
            source_width=source_width,
            source_height=source_height,
            output_width=512,
            output_height=512,
            confidence_threshold=0.05,
            line_width=6,
            joint_radius=4,
        )

        output_path = (
            output_dir
            / f"{image_path.stem}.png"
        )

        skeleton.save(
            output_path
        )

        print(
            image_path.name,
            "->",
            output_path.name,
        )

    print(
        "Renderer test: PASS"
    )

    print(
        "Output:",
        output_dir
    )


if __name__ == "__main__":
    main()
