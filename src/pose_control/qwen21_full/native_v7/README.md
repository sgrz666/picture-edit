# Native Full V7：逐人物外观与几何交互

## 目的与论文对应

V7 面向 4Ref：一张已有双人 Source、Combined Pose、Person 0 Pose、Person 1 Pose，在同一次去噪中改变双方姿态并保留对应身份。继续支持既有 2Ref/5Ref。模型复用 V6.2 的前端、逐人物采样、Source Bank、原生 Qwen 安装器和生成函数；训练直接继承共享 Trainer，不复制训练循环。

|论文|实际接入|与原论文的区别|
|---|---|---|
|[DanceTogether §3.3](https://arxiv.org/html/2505.18078v1)|`pose_fusion.py`：共享参数的逐人物 Pose/Mask 残差；`losses.py`：逐人物区域 FM|复用现有 Key 与 owner/projected/14 部位证据，不重建 PoseNet/SVD，也不增加 Mask 图片|
|[AnyPhoto §3.3](https://arxiv.org/html/2603.14770v1)|`local.py`：用本人物已绑定 Source identity Tokens 调制局部 identity/texture Query|零初始化 scale/shift，复用现有 Face/Hand ROI；不是给 Qwen 主干每层增加 AdaLN，也不宣称全模型身份隔离|
|[PeopleComposer §3.2](https://arxiv.org/html/2605.23178v1)|`poselink.py`：侧 Attention 的 FP32 XY RoPE|使用全目标 Canvas 坐标，不改 Qwen Frame 轴，不联合预测 Pose，不逐人重复生成 RGB|
|[WithEveryone §3.3](https://arxiv.org/html/2608.20336v3)|复用 V5 的 GT 五点位置对齐、低噪声 one-step ID Loss|不在噪声图上重新检测/匹配人脸，不新建身份编码器|
|[WithAnyone §5.1](https://arxiv.org/html/2510.14975v1)|同一次 GT 对齐/解码增加双人身份对比项|正样本为已配对 GT 人物，负样本为同图另一合格人物；不是论文的跨样本大规模负样本库|

这些是适配当前任务的机制借鉴，没有复制论文模型、权重或整套第三方训练框架。

## 模块、开关与约束

`contracts.py` 定义 `PoseLinkConfigV7`，继承原 512 维、8 heads、6 个注入层、每张 Pose 至多 256 Keys 的预算。新增 4 个独立 bool 默认 true：

- `mask_pose_fusion`：对既有逐人物采样 Key 做可学习残差融合。只读取该 Key 的可见 owner、独立投影支持与部位证据；零初始化，不扩大合法 Key 集或配额。
- `spatial_rope`：每个 Head 的 X/Y 使用各自偶数维旋转。Head 维度必须能被 4 整除；不用 ROI 局部坐标替代全 Canvas 坐标。
- `geometry_contact`：普通 Pose 路径取消接触导致的跨人物 Qwen Hidden 读取，独立 Contact K/V 只读取身份融合前的几何、对应人物与部位 Tokens。已知正接触且未被 forbidden 否决、人物/支持有效时才放行；未知 Query 不开放。此项关闭时恢复 V6.2 的 Pose 接触放行；全部接触需另设 `use_contact=false`，它也关闭此桥。
- `identity_modulation`：本人物的 Source identity Tokens 调制局部 Query；身份缺失时沿用旧结果。Hand 从对应人物的 Face identity Tokens 读取，若该身份分支没有被准备，不伪造可用身份。

`runtime.py` 使用独立 header `native_full_v7`，`integration.py` 只是 V6.2 Target-only 安装器的薄包装。Contact 输出头放在既有 `poselink.output_heads` 名称域，进入 S1；融合、Contact 编码器进入 S2/S3；局部身份调制进入相应局部 controller 和 Joint 阶段。零输出头需要先 Warmup；不能凭刚初始化时上游梯度为零判断模块断路。

前三项 Pose 机制需要 `person_routing=true`；非法组合立即拒绝，避免走继承路径时静默跳过新机制。三项全部关闭时，可以关闭人物路由并单独保留局部身份调制。

全部四项关闭时，显式从同一个 V6.2 初始化的 V7 可以数值回退到 V6.2；保留所有权重。训练过 V7 后关闭新增路径属于消融，不等于恢复训练前的 V6.2 权重。原生 Qwen Attention/Pose Hidden 和几何 Reasoner 仍可能传播上下文，因此上述权限不保证最终图像绝不串位。

`training.py` 继承共享 `NativeV62Trainer.run`，使用 `native_v7_training_v1`。`PaperLossConfig` 默认 `person_fm=0.5`、`contact_fm=0.5`、`identity_contrastive=0.1`、`temperature=0.07`；默认 Stage 也启用原 GT ID 权重 0.1。两种 ID 项共用一次 VAE 解码与五点对齐，参考特征冻结，生成侧保留梯度。人像区域按人物归一化后平均；Contact FM 只监督可见的已知接触部位支持，不把完整遮挡投影当作可见纹理，也不声称是接触点/穿模物理 Loss。

ID 项继承 `auxiliary_every=4`、`auxiliary_sigma_max=0.35` 和 `auxiliary_valid`。没有两个合格人物时不造负样本；显式请求对比项而整个训练路由没有合格双人样本时，数据预检失败。梯度预检区分单脸、双脸、有/无有效接触与 repair，评估器在冻结指纹和 RNG 恢复之前加载。

## 如何检查、初始化、训练和推理

在服务器运行，所有新产物使用 `/data/shangguanrz/` 下的新路径：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python

# 配置检查：不创建模型、optimizer，不启动训练。
$PY scripts/train/train_qwen21_native_full_v7.py prepare --diagnostics

# 真实输入资格审计：读取 paired RGB/几何，不加载 Qwen。
$PY scripts/audit/audit_qwen21_native_full_v7.py \
  --manifest /data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-v5-grounded-binding-pilot_20261030_r4/data/typed_v5_pairs.jsonl \
  --data-profile hybrid_hi4d
```

审计默认遍历全部行；`--limit` 明确标记非完整审计；`--report` 只写新的报告。审计不等于绑定模型、冻结缓存、预算与真实梯度门禁。它明确报告身份监督资格、已知接触、原位置修复与原生图像 Token 数量。

显式初始化（以下 checkpoint 和输出目录需换为实际资产，不会自动搜索或启动）：

```bash
$PY scripts/data/migrate_qwen21_native_full_v7.py \
  --v62 /data/shangguanrz/已有V62.pt \
  --output /data/shangguanrz/新V7目录/native_full_v7.pt \
  --report /data/shangguanrz/新V7目录/migration.json
```

`--pose-config` 接受 `PoseLinkConfigV7` JSON。迁移保留旧张量、登记新参数、严格回读实际保存文件，不继承旧 optimizer/RNG，拒绝覆盖源/输出/报告。V6.2 不能作为 V7 resume；没有 V6.2 权重时先走已有显式初始化链，不把 CPU 合成 checkpoint 用于正式训练。

训练 JSON 使用 `schema: native_v7_training_v1`、`qwen_condition_profile: dual_4ref`。在既有 sources、模型 SHA 清单、冻结缓存、初始化、输出和 Stage 预算之外，可添加：

```json
{
  "paper_loss": {
    "person_fm": 0.5,
    "contact_fm": 0.5,
    "identity_contrastive": 0.1,
    "temperature": 0.07
  },
  "diagnostics": true
}
```

这是合并到完整配置的片段；只给这个片段会停在缺失绑定。冻结 ArcFace evaluator 使用现有 TorchScript 路径/SHA 声明，数据必须提供可靠的训练专用五点资格侧车。Target RGB/GT 人脸位置仅用于训练监督，推理不读取它们。

```bash
$PY scripts/train/train_qwen21_native_full_v7.py preflight --level data \
  --config /data/shangguanrz/新V7目录/config.json --report /data/shangguanrz/新V7目录/data-preflight.json
$PY scripts/train/train_qwen21_native_full_v7.py preflight --level grad \
  --config /data/shangguanrz/新V7目录/config.json --report /data/shangguanrz/新V7目录/grad-preflight.json
$PY scripts/train/train_qwen21_native_full_v7.py train --config /data/shangguanrz/新V7目录/config.json
```

沿用 S1 `poselink-heads` → S2 `poselink-full` → S3 `joint` 的显式 `advance-stage`；S2/S3 使用父 Stage checkpoint 和匹配的新梯度报告，不能新建训练冒充恢复。`resume --checkpoint` 只接收 V7 完整提交状态，继续校验数据/代码/缓存/配置/optimizer/RNG。`--loss-profile velocity_only` 是显式纯 Velocity 消融，会关闭新增区域和 ID 对比项；其他 profile 不静默改变 `paper_loss`。

```bash
$PY scripts/run/run_qwen21_native_full_v7.py \
  --checkpoint /data/shangguanrz/新V7目录/已训练V7.pt \
  --manifest /data/shangguanrz/已验证pairs.jsonl --pair-index 0 \
  --diagnostics --output /data/shangguanrz/新V7推理目录
```

推理可用 `--no-mask-pose-fusion`、`--no-spatial-rope`、`--no-geometry-contact`、`--no-identity-modulation` 单项消融；不传则使用 checkpoint 原设置。`request.json` 同时记录 checkpoint 原配置、实际配置与显式覆盖。API 对应 `runtime.set_ablation(spatial_rope=False)`，原地切策略但不重建/修改权重。

CPU 测试（Windows 在仓库根目录使用 `python`）：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m pytest \
  tests/test_native_v7_components.py tests/test_native_v7_runtime.py \
  tests/test_native_v7_integration.py tests/test_native_v7_losses.py tests/test_native_v7_training.py -q
```

若服务器现有工作区缺少 `tests/__init__.py`，第三方同名包会遮蔽本地 tests。可以仅在测试进程内指定本地路径，不恢复/覆盖其他人的文件：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -c 'import sys,types,pathlib,pytest; p=types.ModuleType("tests"); p.__path__=[str(pathlib.Path.cwd()/"tests")]; sys.modules["tests"]=p; raise SystemExit(pytest.main(["tests/test_native_v7_components.py","tests/test_native_v7_runtime.py","tests/test_native_v7_integration.py","tests/test_native_v7_losses.py","tests/test_native_v7_training.py","-q","-p","no:cacheprovider"]))'
```

## 当前结论与上游关系

V7 是 V6.2 的显式新版本，不替换 V5/V6/V6.1/V6.2 的 checkpoint 或入口。新增学习路径复用旧编码器；原版本仅获得默认无状态扩展点，共用训练引擎仍保留原版本 schema。

CPU 验收使用合成条件、可微小解码器与随机初始化的真实小型 Qwen Blocks，覆盖独立人物污染/梯度、接触禁止、无 Key bias、初始化、真实 optimizer 更新、2/4/5Ref 和 checkpoint 反向。它不证明真实 7B 显存、训练速度或图像质量。当前核对的 Hi4D r4 pilot 只有 2 条 typed 样本，两条均无辅助 ID 资格和 repair；需补齐可靠资格侧车后才适合默认 ID 训练，不能只调大 Loss 权重。

2026-11-17 服务器最终相关套件 **296 passed、3 skipped**（299 项，包含旧 V5/V6/V6.1/V6.2 回归与 V7）；跳过的 3 项需要另行绑定真实资产实验。33 个本次 Python 文件编译通过，36 个交付文件 SHA256 一致，V6.2→V7 合成 checkpoint 的 CLI 迁移与实际文件严格读回通过。Windows 单独 V7 套件 57 项通过，随后补充的未路由配置边界 5 项通过；旧 V5 在 Windows 的 POSIX 绝对路径限制可在未修改基线复现，不据此改动旧训练版本。

服务器源码快照、编译记录、JUnit、数据审计和迁移证据位于 `/data/shangguanrz/code-snapshots/native-full-v7-final-20261117T113622Z/`，上一次快照为 `/data/shangguanrz/code-snapshots/native-full-v7-20261117T112800Z/`。合成权重仅用于 CPU 序列化验收，未作为真实训练初始化。

相关规范与计划：`docs/superpowers/specs/2026-11-17-native-full-v7-design.md`、`docs/superpowers/plans/2026-11-17-native-full-v7.md`。姿态效果评测继续遵守服务器现有 SDPose-Wholebody + YOLO11x 规范，不在 qwen21 环境安装评测依赖。
