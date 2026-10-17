# patches/ —— 对上游 DeepGen 的最小补丁

对上游 DeepGen 仓库的微侵入式补丁存放区，记录把姿态控制信号注入扩散主干所需的最小改动。

## 里面有什么

- `deepgen_pose_control.patch`：针对 `models/DeepGen-1.0-diffusers/deepgen_pipeline.py` 的补丁。新增 `block_controlnet_hidden_states`（每层 pose-control token 残差）与 `control_scale` 参数，在去噪主循环注入控制状态，并做形状与有限性校验（出现 NaN/Inf 直接报错）。这是 DeepGen-1.0 技术线接入自研 **Unified SMPL-X Adapter V6** 的入口改动。

## 与上下层的关系

- 守护：`tests/test_deepgen_pose_patch.py`（根 `tests/`）回归测试确保补丁仍适用、接口不变。
- 应用目标：`models/DeepGen-1.0-diffusers`。
- 相关：`tests/`、`deepgen/README.md`（基座正稿，勿改）。

## 注意与坑

- 补丁与上游版本强绑定；升级 DeepGen 主干须同步更新补丁并跑通守护测试，否则 `block_controlnet_hidden_states` 接口可能漂移。
