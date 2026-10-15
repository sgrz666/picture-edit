# scripts/data/ —— 数据集构建、筛选、划分与特征预计算入口
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/data　|　本级：0 个子目录 / 24 个实体文件 + 4 个符号链接（`find -maxdepth 1 -type f/-type l` 2026-10-12 实测）　|　实体约 0.5 MB

## 一、这一层在干什么
把原始数据集（iPER、TikTok、Hi4D、3DPW）变成训练可消费的资产：配对挖掘、几何补全、质检剔除、train/val/test 划分，以及 ArcFace/DINOv2 等特征缓存的预计算。上游是 `datasets/` 原始数据与 `add/tools/`（SMPLest-X、DWPose）产出，下游被 `scripts/train/`、`scripts/eval/` 直接消费。部分脚本运行时直接写数据（构建类无通用 `--help` 语义），执行前先读脚本 main 与参数，不要批量试跑。

## 二、层级地图（按工作线分类）
| 工作线 | 文件 | 内容与默认路径实证 |
|---|---|---|
| TikTok V3 建池流水线（顺序即阶段） | `build_tiktok_v3_candidate_pool.py` → `run_tiktok_v3_dwpose.py` → `build_tiktok_v3_condition_inventory.py` → `fill_tiktok_v3_missing_geometry.py` → `build_tiktok_v3_usable_pool.py` → `build_tiktok_v3_pairs.py` → `split_tiktok_v3_groupwise.py` → `build_tiktok_v3_strict_visual_pool.py` → `build_tiktok_v3_pairs_strict.py`；QC 抽样 `sample_tiktok_v3_dwpose_qc.py`、`sample_tiktok_v3_geometry_rejects.py`；定稿 `finalize_tiktok_v3_visual_quality_v1.py` | 全部读写 `datasets/TikTokDataset/TikTok_v3/manifests/`（candidate_pool.jsonl、condition_inventory.jsonl、pairs/、pairs_strict_v1/、splits/）与 `TikTok_v3/qc/`、`TikTok_v3/dwpose_v1/`；`fill_*` 写 `fill_missing_geometry_summary.json`。docs 记载严格池 train/val/test=8,574/1,120/1,156 对（`docs/tiktok_qwen21_full_training.md`） |
| iPER 划分与 V6.x 特征缓存 | `split_iper_train_val.py`（79 官方训练身份→69 dev-train/10 dev-val，20 测试身份隔离；默认写 `datasets/iPER/iper_sampled_6src64tgt/splits/`）、`precompute_iper_face_features_v65.py`（arcface [N,512]、dino_patches [N,256,1536] 写入 `iper_assets_512_nvdiffrast/{seq}/v65_face_features.pt`）、`precompute_iper_hand_features_v66.py` | 模型固定用本地 `models/dinov2-giant`、`models/antelopev2`，明确不下载 |
| Qwen21 训练集准备 | `prepare_qwen21_full_data.py`（固定小规模工程验收集，"只有 A 帧进 ArcFace/DINO，B 只供几何"）、`prepare_qwen21_face_split.py`、`prepare_qwen21_face_multisample.py`（冻结 iPER Body128/Face64/val12 实验，两阶段：锁定→经 prepare_qwen21_full_data 提特征）、`prepare_qwen21_joint_pilot.py`（锁定均衡子集，CPU/data only）、`precompute_qwen21_face_features.py` | 输出多为必需的 `--out/--output` JSONL+缓存目录 |
| 条件/监督格式工具 | `convert_face_supervision_to_native.py`（letterbox/resize 的 face_supervision.json → native_pad_v1）、`create_debug_samples.py`（生成 `data/debug_samples/`，供 `train/train_adapter.py` 默认值）、`test_control_signals.py`（合成 SMPL-X 人体，验证 `src/controls/generator.py` 输出 skeleton/depth/normal/contact PNG 到 `demooutput/`） | — |

## 三、符号链接与实体位置（重要）
本目录 4 个链接不是新脚本，实体在 `scripts/` 根级（`ls -la` 实测）：
- `build_tiktok_v6_conditions.py → ../build_tiktok_v6_conditions.py`（TikTok V6.3 原生条件：14 类 part map 512×512 PNG 写入 `TikTok_3d_assets_native/{seq}/part_512/`）
- `precompute_tiktok_face_features_v65.py → ../…`（TikTok 版 v65_face_features.pt）
- `prepare_tiktok_qwen21_face.py → ../…`（source-only ArcFace/DINO 缓存）
- `prepare_tiktok_qwen21_training.py → ../…`（严格池审计 + 训练清单）
判重时勿用 md5 比对链接与实体（同一 inode）。

## 四、注意事项 / 调用路径
- 环境：几何链路用 `add/envs/smplestx_bw/bin/python`，特征预计算用 `add/envs` 或 qwen21 环境（看脚本 import）。
- 复现 TikTok V3 全链：按上表顺序跑，或后台包装 `bash scripts/run/run_tiktok_v3_fill_missing_geometry_bg.sh` —— 该脚本仍按 10-03 前布局 `cd` 仓库根后调 `fill_tiktok_v3_missing_geometry.py`，根目录副本已删除，直接执行会找不到文件，需改为 `python3 scripts/data/fill_tiktok_v3_missing_geometry.py`。
- Hi4D 的配对 manifest（`final_assets_v4_2/main_pairs_*.jsonl`）由 `add/` 工作区产出，不在本目录。
