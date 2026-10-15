# dwpose_onnx/ —— 内置 DWPose ONNX 推理helper（上游 vendored 副本）

本目录是 `src/evaluation/` 自包含的 DWPose ONNX 推理封装所依赖的上游代码副本：从 [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose) 的 `onnx` 分支**原样 vendored**，未做任何修改，同目录 `LICENSE` 覆盖其再分发（Apache-2.0）。

## 里面有什么

- `onnxdet.py` —— 人体检测部分：Numpy 实现的单类 NMS、YOLOX 检测框预处理与后处理。
- `onnxpose.py` —— 姿态估计部分：RTMPose 推理的 `preprocess`（默认输入 192×256）、关键点解码与后处理。
- `__init__.py` —— 仅含模块 docstring，声明上游来源与「未修改」性质。
- `LICENSE` —— 上游 Apache-2.0 许可证。

## 与上下层的关系

- 被 [`src/evaluation/`](../README.md) 的指标模块调用，为 PCK / 关键点一致性评测提供 134 点全身关键点检测能力；权重本体在 [`models/DWPose/`](../../../models/DWPose/README.md)（`dw-ll_ucoco_384.onnx` + `yolox_l.onnx`）。
- 训练侧的 TikTok / Hi4D 几何条件由 `add/tools/` 下的 DWPose 副本产出（数据工程线），本副本专供仓库内评测，两条线互不依赖。

## 注意与坑

- 这是**只读 vendored 副本**：需要升级 DWPose 时整体替换并更新 `LICENSE`，不要就地改算法；改坏了会影响 `src/evaluation/` 的全部姿态指标口径。
- 依赖 `onnxruntime` 与 `opencv`，属评测环境（`envs/qwen21` 或 deepgen 环境均已具备）。
