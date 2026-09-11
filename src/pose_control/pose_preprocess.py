from typing import Sequence, Tuple

from PIL import Image


def letterbox_image(
    image: Image.Image,
    output_size: Tuple[int, int] = (512, 512),
    fill=(0, 0, 0),
):
    """
    保持纵横比缩放图像，再居中填充到目标尺寸。

    Returns:
        output: 处理后的图像
        scale: 统一缩放比例
        pad_x: 左侧填充像素
        pad_y: 上侧填充像素
    """
    src_w, src_h = image.size
    out_w, out_h = output_size

    scale = min(
        out_w / src_w,
        out_h / src_h,
    )

    new_w = round(src_w * scale)
    new_h = round(src_h * scale)

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2

    output = Image.new(
        "RGB",
        output_size,
        fill,
    )

    output.paste(
        resized,
        (pad_x, pad_y),
    )

    return output, scale, pad_x, pad_y


def transform_joints(
    joints: Sequence[Sequence[float]],
    scale: float,
    pad_x: int,
    pad_y: int,
):
    """
    对关键点执行与 RGB 图像完全相同的缩放和平移。
    无效关键点保持无效状态。
    """
    transformed = []

    for joint in joints:
        x, y, confidence = joint[:3]

        if x < 0 or y < 0 or confidence < 0:
            transformed.append(
                [-1.0, -1.0, confidence]
            )
            continue

        transformed.append(
            [
                x * scale + pad_x,
                y * scale + pad_y,
                confidence,
            ]
        )

    return transformed


def preprocess_image_and_joints(
    image: Image.Image,
    joints: Sequence[Sequence[float]],
    output_size: Tuple[int, int] = (512, 512),
):
    """
    同步处理 RGB 与人体关键点，保证二者空间严格对齐。
    """
    image, scale, pad_x, pad_y = letterbox_image(
        image=image.convert("RGB"),
        output_size=output_size,
    )

    joints = transform_joints(
        joints=joints,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )

    return image, joints
