# V6.1 训练接口

默认 `prepare` 打印 `native_v61_training_v1` 配置与缺失绑定，不创建模型、optimizer 或训练。模型为独立 `native_full_v61`，不使用 V6 optimizer／RNG 恢复。

`poselink-heads` 是 S1，只训练输出 Head；`poselink-full` 是 S2，包含新增 Token 类型 embedding 与其余 PoseLink；Joint 是 S3，共享 Geometry 仍需显式解冻。Qwen／VAE／Qwen3-VL 保持冻结。参数目录复用 V6 的完整登记与固定 optimizer 超集，新增类型 embedding 在迁移报告中标记 `new_initialized`。

训练、验证、观察都通过 V6.1 reader／collator 保留可选 `pose_repair_v61` 原始侧车。它进入 manifest／数据签名；冻结 Qwen 缓存仍只包含 V6 已验证的 latents、Prompt 和图像角色／Canvas。缺失修复区域时只限制目标人体，不从 Training Supervision 构造条件。

Loss 不修改：默认 Velocity，原空间 FM 系数、归一化与上限继续复用；非零空间 FM 与原 Face／Hand 局部 Velocity 的互斥规则不变。Head 首次反向与临时非零 Head 上游探针、前缀／冻结摘要、数据划分、严格恢复与 optimizer 边界沿用 V6，报告 schema 独立。

```bash
$PY scripts/train/train_qwen21_native_full_v61.py prepare
$PY scripts/train/train_qwen21_native_full_v61.py prepare --profile poselink-full
```

完整迁移、配置、CPU 测试与推理命令见上级 README。真实使用仍需另行绑定数据与模型身份、执行 data／grad preflight，再显式训练或切换阶段；本轮没有执行这些操作。
