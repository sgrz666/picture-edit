# scripts/audit/ —— 审计与验证工具集

本目录集中存放「只读检查 / 结构冒烟 / 验收」类脚本，是人物图像姿态编辑系统的事实核查层。它们不训练、不生成最终产物、不修改源权重，只在既有产物、缓存、权重与目录组织上做断言与比对，用来证明训练线（DeepGen V6 / Qwen21 各分支）在架构保全、权重溯源、数据质量与条件契约上没有悄悄退化。所有脚本一律把项目根插入 `sys.path`（`parents[2]`），因此从项目外也能以分类实体身份运行；多数的 docstring 明确标注了「CPU-only / 不写权重 / 不改控制模块」等约束。

## 里面有什么

- `audit_qwen21_native_geometry.py` —— CPU-only读取已有Full checkpoint，核对新512维前端与旧encoder/reasoner/bridge数值一致；不构造Qwen/SD3主干，不写权重，JSON输出使用独占新文件。
- `audit_qwen21_native_body.py` —— CPU真实32层小型Qwen结构核验，检查新Body零输出、prefix、条件与checkpoint梯度；可显式读取Full前端权重。
- `verify_qwen21_native_body_runtime.py` —— 有界CUDA核验：现有7B基座、已训练Full前端、生产维度新Body的前向/反向；不采样图片、不更新优化器或写权重。

### 审计类（权重兼容、prompt 泄漏、架构保全、数据质量）

| 文件 | 一句话职责 |
|---|---|
| `audit_qwen21_weight_compatibility.py` | CPU-only 权重溯源、语义映射与内存内模块审计；从不构造 Qwen 基模型或保存 adapter 张量，源 checkpoint 只读 |
| `inventory_qwen21_full_weights.py` | CPU-only 逐张量溯源「保留的 V6/Qwen 初始模型」，描述初始化来源而非声称已训练 |
| `check_qwen21_full_preservation.py` | 对未改动的 V6 拷贝及其遗留架构测试做 CPU 验收，仅在生产实验验收目录下写生成副本 |
| `check_qwen21_native_value_prediction.py` | 固定 native-value 审计下的真实 Qwen 预测对齐（不采样整图、不编码 B 的 RGB 为 latent） |
| `audit_qwen21_face_alignment.py` | CPU-only 对已生成单样例图的裁切/对齐审计，不合成、不训练、不加载 adapter 权重 |
| `audit_hi4d_qwen21_prompt_leak.py` | 在单个 Hi4D source 上检查 target-RGB 隔离与 target-geometry 敏感性，防 prompt 泄漏 |
| `probe_qwen21_face_single_sample.py` | 训练后固定噪声、区域误差与单步重建诊断（依赖 `scripts/run/run_qwen21_face_single_sample.py` 的输出） |
| `audit_tiktok_v3_blur_sample.py` | TikTok V3 模糊样本审计 |
| `audit_tiktok_v3_final_pairs.py` | TikTok V3 终审 pair 数据质量审计（基于 `datasets/TikTokDataset/TikTok_v3/manifests/pairs`） |
| `audit_tiktok_v3_pose_redundancy.py` | TikTok V3 姿态冗余审计 |
| `audit_tiktok_v3_strict_final.py` | TikTok V3 strict 终审审计 |
| `audit_tiktok_v3_visual_quality_v2.py` | TikTok V3 画质审计 v2（CSV/统计汇总） |
| `audit_tiktok_v3_visual_quality_v3.py` | TikTok V3 画质审计 v3（更完整的画质指标） |

### 冒烟 / 验证类（结构正确性与运行时验收）

| 文件 | 一句话职责 |
|---|---|
| `smoke_adapter_v6.py` | 真实 DeepGen 结构冒烟：用合成条件走 ConditionBundle 边界，零初始化 head 必须不改变冻结的 DeepGen 结果 |
| `smoke_adapter_reasoner.py` | 独立 V6.2 Adapter reasoner 的 GPU/CPU 结构冒烟（合成已准备好的 SMPL-X 条件，不训练、不估体） |
| `smoke_deepgen_control_interface.py` | V6.3 动态控制接口的 DeepGen 冒烟（合成条件、本地权重，不训练、不存 checkpoint） |
| `verify_qwen21_full_resume.py` | 诊断单步 CUDA checkpoint 可复现性：每次重载同一 checkpoint 并还原全部 RNG 状态，不延长训练、不覆盖既有 checkpoint |
| `verify_qwen21_full_runtime.py` | 完整 Qwen controller 的有界 GPU 验收：仅做 velocity 预测与每分支一次 backward，从不更新权重 |
| `verify_qwen21_face_single_sample.py` | 真实 Face loss 梯度与 source/geometry 边界检查；不更新 optimizer |
| `verify_qwen21_face_repair.py` | 检查已修复的 Face 候选，不更新权重、不生成图像 |
| `verify_qwen21_condition_combinations.py` | 基于产物的 native-only 补充实验验收（复用 `scripts/eval/qwen21_condition_combinations.py` 引用与协议） |
| `verify_project_organization.py` | **目录组织验收工具**：只读校验 2026-10-03 整理日志（`organization.json`/`before/`），核对结构、软链接、小文件哈希与资产存在性；`--full-assets` 另读七项资产并比对内容哈希，`--strict-source-baseline` 才要求源码与历史基线完全一致 |

## 与上下层的关系

- **上游**：`src/pose_control/`（V6 / qwen21 / qwen21_full 的控制实现）、`src/adapter/`（PoseConditionAdapter）、`models/`（DeepGen-1.0-diffusers、Qwen-Image-2.1）、`experiments/<run>/`（训练/评测产物）、`datasets/`（TikTok_v3、iPER）。
- **下游消费者**：`scripts/reports/` 的 `write_qwen21_weight_audit_report.py`、`write_qwen21_inheritance_report.py` 等会把这里的审计证据写入 Word 报告；`verify_project_organization.py` 的输出是整理交付的闸门。
- 相关索引：父级分类说明见 [`../README.md`](../README.md)；权重溯源脚本与 `scripts/weights/inherit_qwen21_full_weights.py` 同源；评测对照口径由 [`../eval/README.md`](../eval/README.md) 定义。

## 注意与坑

- 这些脚本绝大多数**只做断言、不修数据**；`verify_project_organization.py` 默认只校验结构/链接/小文件哈希/资产是否存在，不要误以为它已核对过大文件内容——需加 `--full-assets`。
- `probe_qwen21_face_single_sample.py` 依赖 `scripts/run/run_qwen21_face_single_sample.py` 里定义的输出常量（`OUT`），两者需同步。
- 多数审计脚本的 docstring 都白纸黑字写了「不训练 / 不保存权重 / 不下载 checkpoint」，跑之前不要臆测它们会改动任何东西。

## Native Depth / CHAMP (2026-10-23)

`audit_qwen21_native_depth.py` 严格迁入官方active path并比较空间attention；`verify_qwen21_native_depth_runtime.py` 使用真实7B和矩形条件核对prefix、冻结基座与Depth/Body梯度，不更新优化器、不采样图片。

## Global Geometry v2

`audit_qwen21_global_pose.py` 使用真实 TikTok/iPER 缓存，核对 21×256 / 512 接口与相同 canonical 输入下的零 Pose 增量。`verify_qwen21_global_pose_runtime.py` 对真实 7B 矩形网格执行有界前反向，覆盖完整 prefix、Pose/Depth/Body 梯度和冻结状态；不创建 optimizer、不保存权重。`test_qwen21_full_preservation.py` 用原迁移 manifest 加五个明确声明的可扩展文件 SHA，继续核验未改动的 v6 及 legacy 数值。详见 `src/pose_control/qwen21_full/GLOBAL_GEOMETRY_V2.md`。

## Native Full V4（工具未执行）

`audit_qwen21_native_full_sources_v4.py`仅用FakeTensor读取已指定checkpoint容器的键、shape、dtype和历史step元数据，不构造模型或执行迁移。本轮仅编译该脚本；Face为3000步候选，Hand训练来源未确认，不能据结构匹配宣称预训练有效。使用见 `src/pose_control/qwen21_full/NATIVE_FULL_V4.md`。

`audit_qwen21_native_full_v5.py`为V5准备显式state容器的FakeTensor键／尺寸清单，分别记录结构、权重证据、运行、控制和质量结论；不构造模型或执行迁移。本轮工具未执行，源码／日志账本和后续顺序见`src/pose_control/qwen21_full/native_v5/README.md`。
