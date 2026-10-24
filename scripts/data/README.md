# scripts/data/ —— 数据构建、过滤、划分与预计算

本目录是人物图像姿态编辑系统的数据准备中枢，负责把原始视频/图像数据集（iPER、TikTok_v3、3DPW、Hi4D）转换成训练与评测所需的配对、条件图与控制特征缓存。它位于 `src/` 上游模块与 `scripts/train/`、`scripts/eval/` 下游之间：读 `datasets/`、`data/` 中的原始资产，产出 `datasets/<name>/manifests/` 下的 JSONL/JSON 清单与 `TikTok_3d_assets_native/`、`experiments/<run>/condition_cache/` 等特征缓存，供训练加载器与评测脚本消费。本目录还有若干指向 `scripts/` 根下同名脚本的**相对软链接**（如 `build_tiktok_v6_conditions.py`），从分类入口执行时 `Path(__file__).resolve()` 会落回原文件，不改变源码身份。

## 里面有什么

### TikTok V3 候选池 / 配对 / 条件构造

| 文件 | 一句话职责 |
|---|---|
| `build_tiktok_v3_candidate_pool.py` | TikTok V3 正式候选池构造：官方序列 → 序列筛选 → 人体区域运动估计 → 自适应时序采样 → 局部最清晰帧选择 |
| `build_tiktok_v3_condition_inventory.py` | 第 6 步：既有条件（RGB、几何等）对齐/复用/补全清单 |
| `build_tiktok_v3_pairs.py` | 构造 TikTok V3 训练/评测 pair |
| `build_tiktok_v3_pairs_strict.py` | 构造严格版（strict）pair，约束更紧 |
| `build_tiktok_v3_strict_visual_pool.py` | strict 可视池构建 |
| `build_tiktok_v3_usable_pool.py` | 可用池统计与筛选 |
| `build_tiktok_v6_conditions.py` | *(软链接→../build_tiktok_v6_conditions.py)* 为 TikTok 生成 V6.3 Native Adapter 条件：14 类 part map（nvdiffrast 渲染）、SMPL-X 全局 26D 参数 + 25 关键点 |
| `run_tiktok_v3_dwpose.py` | 跑 TikTok V3 的 DWPose 骨架估计 |
| `sample_tiktok_v3_dwpose_qc.py` | DWPose 结果抽样质检 |
| `sample_tiktok_v3_geometry_rejects.py` | 几何拒绝样本抽样 |
| `split_tiktok_v3_groupwise.py` | TikTok V3 按组划分（保持 appearance 无重叠） |
| `fill_tiktok_v3_missing_geometry.py` | 补全缺失几何（对应 `scripts/run/run_tiktok_v3_fill_missing_geometry_bg.sh`） |
| `finalize_tiktok_v3_visual_quality_v1.py` | TikTok V3 画质终稿汇总 |

### iPER / 面部 / 手部特征预计算

| 文件 | 一句话职责 |
|---|---|
| `precompute_iper_face_features_v65.py` | 预计算 V6.5 iPER 人脸 identity/appearance/geometry 特征（ArcFace/DINO 须本地存在，不下载） |
| `precompute_iper_hand_features_v66.py` | 从既有 iPER V6 detail 裁切缓存 source-hand 的 DINO patch（ViTPose/HaMeR 输出被视为离线张量） |
| `precompute_qwen21_face_features.py` | 缓存 Qwen21 Face 训练用的 source-only 人脸 identity/texture 与 target geometry |
| `precompute_tiktok_face_features_v65.py` | *(软链接→../precompute_tiktok_face_features_v65.py)* 预计算 V6.5 TikTok 人脸特征（输出 `v65_face_features.pt`） |
| `split_iper_train_val.py` | 把官方 iPER 79 个训练 appearance 拆成 dev-train(69)/dev-val(10)，20 个官方 test appearance 完全不动，保证无 appearance 重叠 |
| `create_debug_samples.py` | 生成调试用样本 |

### Qwen21 / 通用数据准备

| 文件 | 一句话职责 |
|---|---|
| `prepare_qwen21_face_multisample.py` | 冻结 CPU 审计过的 iPER Body128/Face64/val12 实验（不加载 DINO 或生成模型） |
| `prepare_qwen21_face_split.py` | 修正 Face 训练的 appearance split 并抽取 source-only 人脸裁切 |
| `prepare_qwen21_full_data.py` | 为完整 Qwen adapter 准备固定的小型真实数据验收集（仅 A 帧进 ArcFace/DINO，B 只贡献 SMPL-X 几何） |
| `prepare_qwen21_joint_pilot.py` | 锁定既有合格 pair 的平衡子集（仅 CPU/数据操作） |
| `prepare_tiktok_qwen21_face.py` | *(软链接→../prepare_tiktok_qwen21_face.py)* 为 Qwen TikTok 训练准备 source-only ArcFace/DINO 缓存 |
| `prepare_tiktok_qwen21_training.py` | *(软链接→../prepare_tiktok_qwen21_training.py)* 审计 strict TikTok 划分并写出独立的可训练 manifest |
| `convert_face_supervision_to_native.py` | 把既有 `face_supervision.json`（letterbox/resize）转换为 `native_pad_v1` |

### 控制 / 手部输入自检

| 文件 | 一句话职责 |
|---|---|
| `test_control_signals.py` | 用合成 SMPL-X 人体验证 `src/controls/generator` 的相机与控制信号生成 |
| `inspect_hand_inputs.py` | 检查 Qwen21 full 执行的手部输入前缀（`BODY_PREFIXES`/`FACE_ONLY_PREFIXES`/`HAND_ONLY_PREFIXES`） |

## 与上下层的关系

- **上游输入**：`datasets/`（TikTokDataset/TikTok_v3、iPER、3DPW）、`data/`、`add/tools/SMPLest-X` 与 `add/tools/` 下的离线体估工具、`src/controls/`（ControlSignalGenerator）。
- **产出物**：`datasets/<name>/manifests/`（pairs、candidate_pool、condition_inventory 等 JSONL/JSON）、`TikTok_3d_assets_native/<seq>/`（part_512、v6_conditions.pt、v65_face_features.pt）、`experiments/<run>/condition_cache/`。
- **下游消费者**：`scripts/train/`（各训练入口读这些 manifest/缓存）、`scripts/eval/`（评测加载同一份 pair 与条件）、`configs/asset_registry.json`（登记这些缓存的资产路径）。
- 相关索引：父级 [`../README.md`](../README.md)；特征契约由 [`../../tests/README.md`](../../tests/README.md) 的 `test_tiktok_data_contract.py`、`test_iper_v6_data_integrity.py` 等回归测试守护。

## 注意与坑

- 本目录有 4 个**相对软链接**（`build_tiktok_v6_conditions.py`、`precompute_tiktok_face_features_v65.py`、`prepare_tiktok_qwen21_face.py`、`prepare_tiktok_qwen21_training.py`），它们指向 `scripts/` 根下的同名文件；不要按文件名重复维护两份，改原文件即可。
- iPER 的官方划分约束（train 79 / test 20，appearance 无重叠）由 `split_iper_train_val.py` 强制保证，改划分逻辑时必须维持 test appearance 完全不动。
- 多数特征预计算脚本要求 ArcFace/DINO 等重模型**已存在于本地**，脚本本身不下载权重。

## Body21 版本化几何缓存与迁移

- `build_qwen21_body_pose_cache.py`：现有 SMPLest-X 环境从 TikTok/iPER 参数计算 22 FK、Body63 和原图投影，逐帧核对 mesh_cam（门槛 1e-4），只写新输出目录及来源 SHA256。
- `migrate_qwen21_geometry_v2.py`：Full/v1 严格继承到 Body21，或显式 legacy 重建；不建 optimizer，不覆盖原权重。用法见 `src/pose_control/qwen21_full/GLOBAL_GEOMETRY_V2.md`。

## Native Depth V2

- `export_qwen21_depth_v2.py`：读取现有目标mesh_cam和Body21记录，导出版本化Camera-Z/valid NPY或为已有PNG建立V2 manifest；不重跑人体估计，输出新目录。
- `migrate_qwen21_depth_v2.py`：显式继承Geometry、Body和CHAMP Encoder，丢弃旧Depth投影，初始化新Detail/Gate/投影；原版本仍可重建。本轮两个工具均未执行，见 `src/pose_control/qwen21_full/NATIVE_DEPTH_V2.md`。

## Native Dual V3（源码交付，未执行）

`build_qwen21_native_dual_cache.py` 在独立SMPL/FCL环境构建同源双人Body21、独立表面与真实部位距离缓存；`migrate_qwen21_native_dual_v3.py` 仅从Body21＋Depth V2显式升级，输出新文件，不恢复optimizer。输入契约与命令见 `src/pose_control/qwen21_full/NATIVE_DUAL_V3.md`。本轮没有构建缓存、安装FCL或迁移权重。

## Native Full V4（架构交付，尚未验收）

`import_qwen21_native_smplx_v4.py`导入完整模型参数并核对FK；`cache_qwen21_native_local_references_v4.py`仅提取Source ArcFace/DINO；`build_qwen21_native_full_cache_v4.py`构建同源Body21/Depth/局部几何/BPS；`migrate_qwen21_native_full_v4.py`显式继承V3及兼容局部编码器，缺Hand来源需allow_fresh_hand。全部工具本轮未执行。契约与顺序见 `src/pose_control/qwen21_full/NATIVE_FULL_V4.md`。
