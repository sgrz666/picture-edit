# Native Full V6.1 implementation plan

用户已确认：只做 P1＋P2；Gate 只约束 PoseLink；原位置修复 Mask 显式提供才纳入；直接在 main 开发并同步本地／GitHub／服务器。无真实迁移、7B、训练、生成、数据提取或缓存构建。

## 结构与约定

新建独立 `native_v61`，保留 V6 状态与入口。`PoseLinkConfigV61` 继承 V6 配置，新增三个开关、global/joint/local=128/64/64 的每参考图预算、支持扩展 16px 与内部羽化 8px。全部开关关闭必须精确走 V6 算法。

`PoseSpatialConditionV61` 保存目标 Canvas 像素级逐人物 key_support [B,2,H,W]、valid_pixels [B,1,H,W]、条件 Gate [B,N]、规范化 joint_uv [B,2,21,2] 与有效性、local_boxes [B,2,5,4] 与有效性（左手、右手、左足、右足、面部）、Canvas IDs 与 provenance。最终输出沿用 V6 coverage 一次；Gate 是有效像素条件均值，不能再乘一次 coverage。

可选 `pose_repair_v61` 只接受原始条件 Mask，不接受 Training Supervision。绑定 schema、role=source_repair、target_canvas_id、source_rgb_sha256s、target_person_ids 和 geometry SHA；拒绝不同来源、尺寸、范围或人物顺序。Mask 不存在时明确记为 missing，不推测跨 Canvas 映射。

## 执行清单

- [x] 更新本地 main 到已合入 V6 的 85c14fe；保留既有未跟踪文件，写入 main 工作约定。
- [x] 基线：运行原 70 项 CPU 回归，记录当前 main 下结果。
- [x] 空间条件：先写 CPU RED 用例，再实现配置／记录转换、原始关节与局部提取、软支持／编辑 Gate、修复来源验证；不运行任何估计器。
- [x] 多尺度：先写 RED 用例，再实现支持归一化池化与稀疏采样；按人物均分、固定解剖顺序轮询分配 local 配额；无效点不补造；保留 xy/person/type 元数据；预算不超过每图 256。
- [x] Runtime／integration：新 Prepared 类型、继承 V6 权重的显式初始化、严格 V6.1 回读、同一原始 Hidden 的 Target-only 残差、非重入 checkpoint 上下文；旧 V6 不改。
- [x] 数据／训练／推理：可选原始 repair 侧车贯通 reader/collator/cache identity，参数目录完整；独立 V6.1 schema、迁移和 CLI；默认 prepare 不构建模型／optimizer。
- [x] CPU 验证：原 70 项＋空间支持、背景污染、Gate、预算与双人公平性、缺失／失配、2／4／5Ref、前缀、梯度、重算、版本／冻结／全部关闭等价。
- [x] 审查与交付：中文 README；独立规格／代码审查；新文件 py_compile；提交 main 并推送；远端源码／index／diff 时间戳快照，定向同步与 main 索引核对，不改训练快照、不全仓 checkout/reset/clean。

## 验证命令

```powershell
python -m pytest tests/test_native_v6_poselink.py tests/test_native_v6_runtime.py tests/test_native_v6_integration.py tests/test_native_v6_data.py tests/test_native_v6_conditioning.py -q
python -m pytest tests/test_native_v61_spatial.py tests/test_native_v61_poselink.py tests/test_native_v61_runtime.py tests/test_native_v61_integration.py -q
python scripts/train/train_qwen21_native_full_v61.py prepare
```

所有测试只用 CPU 合成张量／随机初始化小 Qwen，不使用真实 checkpoint 或数据。权重初始化工具只实现，不实际迁移真实权重。收益结论等待另行实验。

本地结果：原 V6 70 passed；完整 V6.1 65 passed（新增全套一次运行）。已修复 Gate-only 消融额外改变 Attention、Combined Pose 稀疏 Key 人物边界串扰、配置可突破 256 上限及空间 schema 不一致；相应 CPU 回归通过。规格和独立代码质量复审通过。

交付结果：实现提交 `2b0f198` 已推送 main，服务器从 `0d48cfa` 接纳同一 main；37 个任务文件 SHA 相符、33 个 Python 文件编译通过、39 个已有 V6 文件核对后纳入索引。141 个旧 V5／V6／runtime_repo_v7 文件内容不变，原有已跟踪工作区路径不变；未执行 checkout，历史缺失／未跟踪文件保留。服务器 prepare 输出 `native_v61_training_v1`，model_constructed／optimizer_created／training_started 均 false。

带时间戳的源码、索引、差异与交付证据位于 `/data/shangguanrz/code-snapshots/native-full-v61-20261112T200901/`；此验收记录的主线同步另保留同名前缀的 `-final` 快照。
