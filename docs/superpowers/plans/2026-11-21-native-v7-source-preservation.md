# Native V7 Source Preservation Implementation Plan

> **For agentic workers:** 使用superpowers:subagent-driven-development，分任务实现后做规格和质量审查。用户已授权根据实验改进，直接执行，不重复索要批准。

**Goal:** 增加显式Source身份与背景保持目标，保持旧V7训练行为和签名兼容，完成真实数据／梯度验证。

**Architecture:** 复用原GT辅助解码callback，Source特征由离线SHA绑定资产提供；背景直接约束预测x0的Source latent，不增加解码或改动Qwen。

**Tech Stack:** PyTorch、既有TorchScript/SCRFD、原生V7 Trainer、现有4Ref缓存，不安装包。

## Task 1：独立配置与loss kernels

Files：新增`src/pose_control/qwen21_full/native_v7/preservation.py`、`tests/test_native_v7_preservation.py`。

- [x] 先写并运行失败测试：`source_identity_loss`应吸引Source而非GT，swap mapping影响结果，reference无梯度；`source_background_loss`排除foreground/padding，sigma变化下x0公式正确，Source-only／empty有效性有明确计数。
- [x] 实现`PreservationConfig(source_identity,background)`；`source_identity_loss(generated, source, valid)`返回loss/count；`source_background_loss(prediction,noisy,sigma,source_latent,support,coverage)`返回loss/area。有限值、shape、dtype、device严格校验。
- [x] `python -m pytest tests/test_native_v7_preservation.py -q`通过，提供红绿证据；保持独立模块，不修改训练集成文件。59 passed。

## Task 2：资产绑定与准备

Files：新增`src/pose_control/qwen21_full/native_v7/preservation_assets.py`、`tests/test_native_v7_preservation_assets.py`；实验私有prepare脚本由主智能体实现。

- [x] 先写失败测试覆盖记录读写、Source／Canvas／person／evaluator／grid错配拒绝、index SHA闭包以及BG几何mask膨胀与padding。
- [x] 提供`sample_binding(sample,evaluator_sha)`、`background_support(sample,margin_px=16)`、`write_record(path,features,valid,support,metadata)`、`read_record(ref,sample,evaluator_sha)`和`read_index(ref)`，协议见spec。文件只新建，读时校验SHA。
- [x] `python -m pytest tests/test_native_v7_preservation_assets.py -q`通过，Source特征不从Target提取，支持仅从Source Mask／Target geometry计算。31 passed；真实CPU准备及三份data检查通过。

## Task 3：训练闭环集成

Files：修改`native_v7/training.py`，新增`tests/test_native_v7_preservation_training.py`。

- [x] 旧record往返及signature不变、Source-only callback实际一次decode、index换绑改变data identity、window face/image计数与归一化，先失败再实现。
- [x] 新config仅启用时序列化；DataPlan加载／验证sidecar并把闭包纳入identity，记录每条preservation资格。
- [x] Backend.pair给batch挂Source记录，forward保留当前pair的Source latent与记录；compute_loss复用callback返回Source项，背景只使用x0和Source latent。
- [x] `required_evaluators`、`extra_auxiliary`、`preflight_sigma`和真实grad case选择包含Source-only；缺资产／模型SHA不符不绕门禁。
- [x] 运行V7、V5辅助loss、V6/V62训练回归与新测试；做规格审查，再做质量审查，修复后复核。V7 213 passed／1 skip；旧V5 Linux24 passed／1 skip，Windows原POSIX路径限制9失败不改旧模块。Source-only／联合一次decode探针、disabled旧record签名逐值检查及两级审查通过。另修复CLI velocity_only残留Source项，先红后绿。

## Task 4：真实原生试验与交付

Files：`experiments/hi4d_qwen21_dual_20260930/native-v7-source-preservation-20261121/README.md`及其`scripts/prepare_assets.py`、配置准备脚本；更新父README。

- [x] 先确认两侧现有实验目录，创建同族新实验。真实Source重新SCRFD检测、同一TS提取，source_indices映射，写index；使用旧4Ref缓存及typed manifests，不覆盖旧资产。
- [x] 由180步已提交权重准备显式Source／BG小步配置，设置ID每步、sigma≤.5；保留GT ID与原始Pose/Contact项及严格父链。记录新权重／采样／资源策略。
- [x] 远端git log、源码时间戳快照、scp同步、py_compile及SHA；检查GPU后运行真实data/grad preflight。只用当前可用GPU额度，不杀其他进程或修改环境。
- [x] 若资源允许执行有界小步与同seed生成，采用真实检测五点身份和逐检测锁seed的SDPose、同区域BG指标；否则具体报告实际阻碍与复现命令。
- [x] 更新spec／plan／实验README，显式stage任务文件提交main，报告做了什么、实际测试与如何使用，不把新增loss或预检冒充生成效果已改善。

## 已验证结果

最终代码GPU grad-v2通过312.3秒，实际SourceBG峰值分配57.50GiB。两组各12步、相同180父／data／noise seed／optimizer策略均完成，到global192，保存commit并生成四图。SourceBG相对Control的检测五点SourceID：训练[.56429,.69608]→[.62219,.72833]、观察[.62175,.86348]→[.63391,.87256]；背景MAE训练5.14754→5.11929、观察4.85250→4.82460。训练PCK@.2不变95.83%、误差+0.057px；观察PCK@.2为100%。这只支持同一配对短程正向趋势，不证明全面无损或长程泛化。全部本轮GPU／CPU任务已结束，没有新增第三组试验。
