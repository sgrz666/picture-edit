# Native Full V6.2：人物路由与分支空间控制

本版只实现人物路由和分支 Gate，继承 V6.1 的编码器、多尺度固定预算和 Loss。不增加 Mask 图片输入，不改 Qwen 原生 Self-Attention／RoPE，不增加 Qwen Tokens。默认仍为 512 维、8 heads、六个 PoseLink 注入层（5／10／15／21／26／31），每参考图最多 256 个 Tokens。

## 人物路由与文件

`contracts.py`／`conditions.py` 从已有可见 owner、投影、空间支持及可选修复区推导 Query 归属。可见 owner 优先；权重归一化后最大值达到 `owner_threshold=0.75` 才硬绑定。无可见 owner 时依次参考投影、空间支持和已校验 repair；混合／不确定处保持软角色约束，不以 argmax 强行选人。

人物0／1是绑定稳定 ID 的目标槽位，不等于左右位置，也不等于 Source 图片顺序。Source Bank、Source／Target ID、原始 Driver SHA、Canvas 都由已有记录核对；双人部位图同时核对父记录与两个人物子记录的 Driver SHA。

`tokens.py` 在原有全局预算中按有效人物固定均分 Combined／target_pose 的全局池化。每个人先按自身支持归一化再池化，Key 有明确人物标签；Joint／Local 沿用 V6.1 的固定配额和采样规则。部位信息使用与 Hidden／几何相同的采样范围。原 Qwen Token Layout 不变，所有 Key 仍进入同一次 Attention。

`poselink.py` 对明确归属区域只允许同人物 Key；只有匹配的已知正接触部位对才能放宽跨人读取，已知非接触／避让优先关闭。不确定 Query 保留原软约束。Contact 仍为14部位级，不是顶点接触点；没有部位映射的原始顶点记录只记录数量，不扩大开放范围。`use_contact=false` 同时关闭正接触放宽。无合法 Key 时，已训练的输出 Head bias 也不能产生该处残差。

人物路由需要各自空间支持，因此即使关闭继承的 Geometry Key Mask，开启人物路由时仍保留按人物池化的必要支持；不能把关闭一个开关理解为删除其他仍开启的约束。真实原生 Qwen 路径继续限制为 B1，单人／双人按原有规则分桶。

## 分支 Gate

`prepared.py`／`runtime.py`／`integration.py` 版本化保存路由与 Gate，所有分支仍读取同一份未控制 Block Hidden；FP32 合成后只写 Target Tokens。

|分支|V6.2 策略|
|---|---|
|Body|有合法 `pose_repair_v61`（包括显式空 Mask）才使用目标人体＋原位置修复区的编辑 Gate；缺失则保留旧 coverage 行为，记录 `missing_repair_legacy_coverage`|
|Interaction|使用双人人体支持的条件 Gate，仍遵循原 `relation_valid`；不是只在接触点控制，以保留不接触场景的相对布局能力|
|Face／Hand／Detail|复用现有局部 ROI、可见 owner 和羽化；不再乘通用软 Mask，也不把身份纹理扩展到背景修复区|
|PoseLink|保留 V6.1 Edit Gate，加上人物读取权限|

Gate 是有效像素的条件均值；coverage 最终只应用一次。旧 `local_mode=static` 会先把局部条件融合进 Body 几何；当 Body Gate 改变时，V6.2 用同一 Hidden 下的 full/base Body 对照差保留静态局部贡献，只限制基础 Body 分量。Gate 为1时直接保留旧完整结果，避免无谓数值差；这种 static 对照会额外计算一次 Body 控制器，不是新网络或新权重。

## 配置、检查与使用

新增三个独立配置开关：`person_routing/body_spatial_gate/interaction_spatial_gate`，默认 true。三个都关闭精确恢复同配置 V6.1。局部安全 owner Mask 本来就是既有控制器的一部分；用原有 `--no-face/--no-hand/--no-source-detail` 和模态强度控制局部分支，不取消其必要归属约束。

安全准备命令不会创建模型、optimizer 或训练：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$PY scripts/train/train_qwen21_native_full_v62.py prepare
$PY scripts/train/train_qwen21_native_full_v62.py prepare --diagnostics
```

CPU 契约测试（Windows 在 main 根目录用 `python` 执行同一文件列表）：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m pytest \
  tests/test_native_v62_conditions.py tests/test_native_v62_poselink.py \
  tests/test_native_v62_runtime.py tests/test_native_v62_integration.py \
  tests/test_native_v62_data.py -q
```

显式 V6.1→V6.2 初始化工具已实现，本轮不执行真实迁移：

```bash
$PY scripts/data/migrate_qwen21_native_full_v62.py --v61 /data/shangguanrz/已有V61.pt \
  --output /data/shangguanrz/新目录/native_full_v62.pt \
  --report /data/shangguanrz/新目录/migration.json
```

`--pose-config` 读取完整的 `PoseLinkConfigV62` JSON，可设置三个新开关与继承配置。保持原模型宽度、层和参数形状，严格继承权重并回读保存文件；新增策略没有新可学习权重，PoseLink 按 adapted 记录。不继承 optimizer／RNG，也不允许 V6.1 作为 V6.2 resume。新 header 为 `native_full_v62`，训练 schema 为 `native_v62_training_v1`；V5／V6／V6.1 原入口与严格状态结构不改。

推理入口为 `scripts/run/run_qwen21_native_full_v62.py`，沿用 2／4／5Ref manifest、可选 `pose_repair_v61` 侧车和 `--poselink-strength`。`--diagnostics` 在新输出目录生成 `diagnostics.json`；训练配置同名 bool 默认 false，开启后将前向诊断放入每个 micro-batch 日志。已有原始修复 Mask 的 SHA／Canvas／人物／Driver 仍须正确，缺失不需要额外上传图片。

`observation.DiagnosticsV62().capture()` 可供 API 调用，记录硬／软路由、跨人物许可、无合法 Key、Gate 来源、残差 RMS 和相对原 Hidden 的 L2 幅度。只存 Python 标量和元数据，不存 Tensor，不改变 RNG 或输出。训练 provider 仅捕获前向，退出 capture 后才反向；手动 API 若把 capture 保持到 checkpoint 重算，记录的是实际调用次数，不能当独立样本数。static 局部经过旧 Body 路径，其合成幅度仍记在 Body 记录中。

## 验收界限与关系

V6.2 复用旧 Loss、空间 FM、固定预算、冻结与阶段门禁；不做动态配额、Sigma 调度、geometry-only K/V 或新的背景 Loss。Body 缺少 repair 时不会承诺完整背景 Gate，诊断会明确说明回退。

本轮测试只使用合成 CPU 张量与随机初始化的小尺寸真实 Qwen Blocks，不迁移真实权重，不运行7B、训练或生成实验，不安装依赖。它证明直接 Key 权限、分支残差空间、前缀、梯度与版本契约；Qwen Pose Hidden、原生 Attention 和旧 Reasoner 仍可能混合上下文，不据此宣称最终身份绝不串位或图像质量提高。

2026-11-15 验收：原135项与新增57项在同一完整套件中 **192 passed**；涵盖2／4／5Ref、直接数值／梯度隔离、接触开启与禁止、static／dynamic 局部兼容、部分 coverage、严格回读、checkpoint、诊断和 Windows 编码。规格与独立代码质量复审通过。V5／V6／V6.1 源码和 Loss 文件无修改。
