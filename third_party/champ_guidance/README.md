# champ_guidance/ —— CHAMP GuidanceEncoder 的单帧改编

`third_party` 下有两个显式独立接口：历史 `ChampGuidanceEncoder` 保留卷积mixer、不加载CHAMP权重；新 `spatial_attention.py` 移植官方有效空间Transformer，供Native Depth分支严格加载官方active-path权重。

## 里面有什么

- `spatial_attention.py`：固定commit `4d0cad2…` 的空间Transformer单帧适配，保留GN/proj、self-attention、LayerNorm/GEGLU/残差与权重key；改用PyTorch SDPA，增加可选depth-valid key mask，去掉时间/cross attention和未用模块。Native Depth的官方初始化与数值核对见仓库 `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。

- `__init__.py`：模块 docstring，说明是 CHAMP condition `GuidanceEncoder` 的单帧改编。
- `guidance_encoder.py`：依赖无关的单帧 `GuidanceEncoder` 改编（上游 `github.com/fudan-generative-vision/champ`，commit `4d0cad2…`）。改动范围（文件头已注明）：
  - 膨胀的 3D 卷积降为 2D 卷积（F=1）；
  - 视频 Transformer 依赖替换为残差空间混合器（`_SpatialMixer`）；
  - 最终投影不做零初始化（因为本项目不加载 CHAMP 权重）。
  顶层符号：`_groups`、`_SpatialMixer`、`ChampGuidanceEncoder`。
- `LICENSE`：上游 CHAMP 许可证文本。

## 与上下层的关系

- 被姿态引导相关的推理/训练路径 `import`。
- 许可证见 `LICENSE`；来源与改动见 `guidance_encoder.py` 文件头。
- 相关：`../README.md`、`../../THIRD_PARTY_NOTICES.md`。

## 注意

- 历史 `guidance_encoder.py` 因不加载CHAMP权重而保留非零输出；Native Depth使用独立的新512零输出head，二者不能混用checkpoint。
