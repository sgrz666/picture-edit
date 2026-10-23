# champ_guidance/ —— CHAMP GuidanceEncoder 的单帧改编

`third_party` 下对 CHAMP 姿态引导特征编码器的无依赖单帧改编，供相关引导特征编码使用。历史guidance_encoder.py不加载CHAMP权重；新增空间attention供Native Depth显式初始化。

## 里面有什么

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

- 因为不加载 CHAMP 权重，最终投影未做零初始化；若未来接入权重，需重新评估该改动是否仍合适。

## Native Depth / CHAMP (2026-10-23)

`spatial_attention.py` 在相同固定commit下移植官方有效空间Transformer，保留GN/proj、self-attention、LayerNorm/GEGLU与state key，改用PyTorch SDPA并支持可选valid-key mask。历史卷积mixer入口保持独立；新Native Depth可显式加载官方active权重，见 `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。
