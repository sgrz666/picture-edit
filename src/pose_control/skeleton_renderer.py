from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


# COCO-17 关键点顺序：
# 0 nose
# 1 left_eye
# 2 right_eye
# 3 left_ear
# 4 right_ear
# 5 left_shoulder
# 6 right_shoulder
# 7 left_elbow
# 8 right_elbow
# 9 left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle


# 转换后的 OpenPose-18 顺序：
# 0 nose
# 1 neck
# 2 right_shoulder
# 3 right_elbow
# 4 right_wrist
# 5 left_shoulder
# 6 left_elbow
# 7 left_wrist
# 8 right_hip
# 9 right_knee
# 10 right_ankle
# 11 left_hip
# 12 left_knee
# 13 left_ankle
# 14 right_eye
# 15 left_eye
# 16 right_ear
# 17 left_ear


# 参考 OpenPose / ControlNet body pose 的连接关系。
OPENPOSE18_LIMBS = [
    (1, 2),   # neck -> right shoulder
    (2, 3),   # right shoulder -> right elbow
    (3, 4),   # right elbow -> right wrist
    (1, 5),   # neck -> left shoulder
    (5, 6),   # left shoulder -> left elbow
    (6, 7),   # left elbow -> left wrist
    (1, 8),   # neck -> right hip
    (8, 9),   # right hip -> right knee
    (9, 10),  # right knee -> right ankle
    (1, 11),  # neck -> left hip
    (11, 12), # left hip -> left knee
    (12, 13), # left knee -> left ankle
    (1, 0),   # neck -> nose
    (0, 14),  # nose -> right eye
    (14, 16), # right eye -> right ear
    (0, 15),  # nose -> left eye
    (15, 17), # left eye -> left ear
]


# 固定 limb 颜色。
# 这里只需要保证训练和推理完全一致，不追求复刻某个可视化工具的所有细节。
LIMB_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
]


def _is_valid_joint(
    joint: Sequence[float],
    confidence_threshold: float,
) -> bool:
    """
    判断关键点是否有效。

    输入格式：
        [x, y, confidence]
    """

    if len(joint) < 3:
        return False

    x, y, confidence = joint[:3]

    if x < 0 or y < 0:
        return False

    if confidence < confidence_threshold:
        return False

    return True


def coco17_to_openpose18(
    coco_joints: Sequence[Sequence[float]],
    confidence_threshold: float = 0.05,
) -> List[List[float]]:
    """
    将 COCO-17 关键点转换为 OpenPose-18 顺序。

    唯一新增的点是 Neck：
        neck = (left_shoulder + right_shoulder) / 2

    如果任意一个肩膀无效，则 Neck 设为无效。
    """

    if len(coco_joints) != 17:
        raise ValueError(
            f"COCO 关键点数量必须为 17，实际为 {len(coco_joints)}"
        )

    left_shoulder = coco_joints[5]
    right_shoulder = coco_joints[6]

    if (
        _is_valid_joint(left_shoulder, confidence_threshold)
        and _is_valid_joint(right_shoulder, confidence_threshold)
    ):
        neck = [
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
            min(left_shoulder[2], right_shoulder[2]),
        ]
    else:
        neck = [-1.0, -1.0, -1.0]

    openpose_joints = [
        list(coco_joints[0]),   # nose
        neck,                   # neck
        list(coco_joints[6]),   # right shoulder
        list(coco_joints[8]),   # right elbow
        list(coco_joints[10]),  # right wrist
        list(coco_joints[5]),   # left shoulder
        list(coco_joints[7]),   # left elbow
        list(coco_joints[9]),   # left wrist
        list(coco_joints[12]),  # right hip
        list(coco_joints[14]),  # right knee
        list(coco_joints[16]),  # right ankle
        list(coco_joints[11]),  # left hip
        list(coco_joints[13]),  # left knee
        list(coco_joints[15]),  # left ankle
        list(coco_joints[2]),   # right eye
        list(coco_joints[1]),   # left eye
        list(coco_joints[4]),   # right ear
        list(coco_joints[3]),   # left ear
    ]

    return openpose_joints


def render_openpose18(
    joints: Sequence[Sequence[float]],
    source_width: int,
    source_height: int,
    output_width: int = 512,
    output_height: int = 512,
    confidence_threshold: float = 0.05,
    line_width: int = 6,
    joint_radius: int = 4,
) -> Image.Image:
    """
    将 OpenPose-18 关键点渲染为纯黑背景 RGB skeleton。

    关键点坐标来自原图尺寸 source_width × source_height，
    会按比例映射到输出尺寸。

    注意：
    这里采用直接按 x/y 分别缩放的方法。
    tiny overfit 阶段 RGB target 也必须采用相同的几何变换，
    这样 pose 与 target 才能保持空间对齐。
    """

    if len(joints) != 18:
        raise ValueError(
            f"OpenPose 关键点数量必须为 18，实际为 {len(joints)}"
        )

    if source_width <= 0 or source_height <= 0:
        raise ValueError("原图尺寸必须大于 0")

    canvas = Image.new(
        "RGB",
        (output_width, output_height),
        (0, 0, 0),
    )

    draw = ImageDraw.Draw(canvas)

    scale_x = output_width / source_width
    scale_y = output_height / source_height

    def convert_point(joint):
        x, y, confidence = joint[:3]

        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            confidence,
        )

    scaled_joints = [
        convert_point(joint)
        if _is_valid_joint(joint, confidence_threshold)
        else (-1, -1, -1)
        for joint in joints
    ]

    # 先画骨骼连接线。
    for index, (joint_a, joint_b) in enumerate(OPENPOSE18_LIMBS):
        a = scaled_joints[joint_a]
        b = scaled_joints[joint_b]

        if a[2] < 0 or b[2] < 0:
            continue

        color = LIMB_COLORS[index % len(LIMB_COLORS)]

        draw.line(
            (a[0], a[1], b[0], b[1]),
            fill=color,
            width=line_width,
        )

    # 再画关键点，避免关节连接处出现明显断裂。
    for joint in scaled_joints:
        x, y, confidence = joint

        if confidence < 0:
            continue

        draw.ellipse(
            (
                x - joint_radius,
                y - joint_radius,
                x + joint_radius,
                y + joint_radius,
            ),
            fill=(255, 255, 255),
        )

    return canvas


def render_coco17(
    coco_joints: Sequence[Sequence[float]],
    source_width: int,
    source_height: int,
    output_width: int = 512,
    output_height: int = 512,
    confidence_threshold: float = 0.05,
    line_width: int = 6,
    joint_radius: int = 4,
) -> Image.Image:
    """
    COCO-17 -> OpenPose-18 -> 纯 skeleton 图。
    """

    openpose_joints = coco17_to_openpose18(
        coco_joints,
        confidence_threshold=confidence_threshold,
    )

    return render_openpose18(
        joints=openpose_joints,
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
        confidence_threshold=confidence_threshold,
        line_width=line_width,
        joint_radius=joint_radius,
    )
