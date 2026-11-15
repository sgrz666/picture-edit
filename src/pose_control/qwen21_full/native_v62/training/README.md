# V6.2 训练接口

默认 `prepare` 输出 `native_v62_training_v1` 配置和缺失绑定，不创建模型、optimizer 或训练任务。仅接受同版本严格 resume；V6.1 通过显式 weight-only 初始化进入 V6.2，旧 optimizer／RNG 不恢复。

默认 S1 `poselink-heads`，S2 `poselink-full`，S3 `joint`；共享 Geometry 仍需显式解冻，Qwen／VAE／文本编码器保持冻结。人物路由与 Gate 是确定性策略，没有新增可学习参数，完整参数目录复用 V6.1。Body／Interaction Gate 开关随模型配置保存；局部仍用原 owner ROI。

输入、Source Bank 和 `pose_repair_v61` 读写契约保持原样，不要求新 Mask 图。Body 没有 repair 时继续旧 coverage 路径，并记录原因。Gate 不改变 Loss；默认 Velocity 和原空间 FM、权重上限、局部 Loss 互斥规则均不改。

```bash
$PY scripts/train/train_qwen21_native_full_v62.py prepare
$PY scripts/train/train_qwen21_native_full_v62.py prepare --profile poselink-full --diagnostics
```

配置可选 `diagnostics: true`，CLI 用 `--diagnostics/--no-diagnostics` 覆盖，默认 false。诊断前向捕获结束后才执行 backward；输出为每个 micro-batch 的标量记录，不改变 Loss 或 RNG，不持有计算图 Tensor。诊断设置也被严格配置签名绑定，不能在 resume 时静默改变。

后续真实使用仍需独立数据／模型／冻结缓存身份、data 和 grad preflight，再显式 train／advance-stage。迁移、推理、完整 CPU 测试与架构限制见上级 README；本轮没有执行真实数据预检、真实迁移、训练、生成或自动阶段切换。
