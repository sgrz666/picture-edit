# third_party/v65_face/ —— 目录层级与功能说明

> 本文件由 `scripts/gen_folder_docs.py` 依据目录实际内容自动生成（生成时间：2026-10-11 16:23）；如需补充人工说明，请写在本目录 `README.md` 中（生成器不会覆盖 README）。

- **路径**：`third_party/v65_face`　- **层级**：第 3 级　- **上级**：[../目录说明.md](third_party)

## 一、目录定位

V6.5 人脸模块的外部依赖封装（StableAnimator、X-Dyna 等）。

- **本级内容**：文件 6 个，25.2 KB
- **子树规模**：文件 8 个，逻辑体积 36.5 KB

## 二、层级目录

本子目录（1 个，其中符号链接 0 个）：

- [`licenses/`](licenses/目录说明.md) — 人脸区域专项产物目录（子树 2 个文件、11.2 KB）

## 三、文件清单

共 **6** 个文件（不含隐藏文件与自动生成的说明文件）。类型分布：Python 源码×5、Markdown 文档×1

| 文件 | 类型 | 大小 | 内容摘要 |
|---|---|---|---|
| `UPSTREAM_SOURCES.md` | Markdown 文档 | 2.4 KB | 首个标题/正文：V6.5 face upstream source record |
| `__init__.py` | Python 源码 | 852 B | 模块 docstring：Pinned, locally adapted building blocks used by the V6.5 face path. See :mod:`third_party. |
| `mcld.py` | Python 源码 | 5.2 KB | 模块 docstring：2D adaptations of the pinned MCLD face preprocessing and PoseGuider code. Upstream files `；顶层定义 4 个（`FacePoseGuider2D`, `expand_image_region`, `safe_crop_image`, `safe_square_crop_normalized`） |
| `stableanimator.py` | Python 源码 | 6.7 KB | 模块 docstring：Direct adaptations of StableAnimator identity-fusion components. Upstream files ``animatio；顶层定义 3 个（`FacePerceiver`, `FusionFaceId`, `InsightFaceArcFaceExtractor`） |
| `visual_persona.py` | Python 源码 | 8.9 KB | 模块 docstring：Direct adaptations of Visual Persona's face-token attention primitives. Upstream file: ``m；顶层定义 5 个（`FeedForward`, `_reshape_tensor`, `PerceiverAttention`, `Resampler`…） |
| `xdyna.py` | Python 源码 | 1.1 KB | 模块 docstring：Small pipeline utilities directly adapted from pinned X-Dyna control flow. Upstream file `；顶层定义 2 个（`repeat_condition_batch`, `add_optional_face_residual`） |

## 四、构成画像

- **直属类型分布**：Python 源码×5、Markdown 文档×1
- **子树类型构成**：Python 源码×5、纯文本×2、Markdown 文档×1

## 五、导航

- 根目录总览：[`目录总览.md`](../../目录总览.md)
- 根 README：[README.md](../../README.md)
