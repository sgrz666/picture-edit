# scripts/train/ —— 训练入口全集：DeepGen V6 谱系与 Qwen-Image-2.1 谱系两代架构
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/train　|　本级：0 个子目录 / 11 个实体文件（约 0.77 MB，`du -sh --si` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md、`__pycache__`

## 一、这一层在干什么
每条控制线一个训练入口。两代基座并存：V6 线以 DeepGen-1.0（5B SD3 DiT）为基座、`miniconda3/envs/deepgen` 环境；Qwen21 线以 Qwen-Image-2.1 为基座、`/home/shangguanrz/nihaochen/envs/qwen21` 环境。检查点跨架构不可互换（`docs/PROJECT_STRUCTURE_20261003.md`）。上游是 `scripts/data/` 产出的 pairs/特征缓存，产物写入 `experiments/<run>/`（checkpoint_best/last + 日志 + 配置）。

## 二、层级地图（按谱系分类）
| 谱系 | 文件 | 实证（默认输出/关键超参） |
|---|---|---|
| DeepGen V6 原生（单人方案 A） | `train_iper_v6_native.py`（58 KB；训练 Geometry/Appearance/Dynamic Core/Zero Heads，冻结 Interaction 全支路；默认 train/val pairs = `datasets/iPER/iper_sampled_6src64tgt/splits/dev_{train,val}_pairs.jsonl`、输出 `experiments/iper_v6_native_finetune_v1`、缓存复用 `experiments/iper_adapter_finetune_v1/condition_cache`；分 lr 组 `--lr_zero_heads 1e-4 --lr_condition 5e-5 --lr_bridge 5e-5 --lr_control_block 1e-5`） | 完整四阶段计划与实测（训练峰值 16.50 GiB@RTX 6000D、169.89M 可训练/58.09M 冻结）在 `docs/iper_v6_training_plan.md` |
| V6.5/V6.6 独立 Face/Hand | `train_iper_v65_face.py`（iPER 单样本 Face overfit；默认 `experiments/iper_v65_face_overfit`，评测器 `models/face_eval/*.ts`）、`train_tiktok_v65_face.py`（TikTok 镜像版，默认 `experiments/tiktok_v65_face_overfit`）、`train_iper_v66_hand.py`（固定 pair Hand overfit，legacy/off/v66 共享随机态，默认 `experiments/iper_v66_hand_overfit`）、`train_tiktok_overfit_sanity.py`（两阶段接线正确性 sanity，Phase A/B loss-ratio 与生成 PSNR 门禁参数） | 轻量 helper 无 Diffusers/InsightFace 也可 import（CPU 可测） |
| 最早期 | `train_adapter.py`（V1 8 通道 PoseConditionAdapter 教学式入口；默认数据 `data/debug_samples`、输出 `experiments/adapter_v1`） | 已被 V6 谱系取代；`docs/iper_v6_training_plan.md` 第 1.1 节列了它的三宗罪（架构未对齐、验证集污染、增强破坏 26D 参数） |
| Qwen21 简化线 | `train_iper_qwen21_body.py`（冻结 Qwen 主干上的 target-token Body adapter；`--dataset-type` 支持 iper/tiktok，`--v6-checkpoint` 迁移初始化，transferred/new 双 lr）、`train_iper_qwen21_face.py`（冻结 Body 20k 之上训 source-identity Face ROI 支路，core/head 分层 lr）、`train_hi4d_qwen21_dual.py`（双人；两阶段：2000 步冻结继承的单人路径 + 8000 步联合含 10% iPER 重放；有效批量 4；每 500 步在 64 对固定验证子集选型，结束评全量 822 对，见 `docs/qwen21_hi4d_dual.md`） | 参数含 `--control-cache --latent-cache --iper-replay-rate` 等 |
| Qwen21 完整线 | `train_iper_qwen21_full.py`（保留的 V6 完整架构迁移到 Qwen21；branch=body/face、face-supervision、`--verify-resume`、`--stop-at-utc` 有界运行）、`train_qwen21_joint_pilot.py`（Body+Face 联合，可选冻结 BFS LoRA、严格 resume、梯度探针模式 `--gradient-only`） | TikTok 数据经 `--dataset-type tiktok --tiktok-raw-root/--tiktok-assets-root` 接入；链路说明 `docs/tiktok_qwen21_full_training.md`（Face lr 保守起点 1e-6/5e-7/1e-7，因 1e-4 档两步内 loss 从 0.437 恶化到 2.792） |

## 三、关键文件与实证内容
- `train_iper_qwen21_full.py`：与 `scripts/audit/verify_qwen21_full_runtime.py`、`verify_qwen21_full_resume.py` 捆绑——先 GPU 验收、再一步 CUDA 恢复复现，才算这个入口可用。
- `train_hi4d_qwen21_dual.py`：双人线当前唯一在跑的正式入口（另见记忆/文档：dual_v8 分支谱系）。
- TikTok 侧启动封装：根级 `scripts/run_tiktok_qwen21_full.sh`（modes: prepare/smoke/body/face-prepare/face/resume-body/resume-face/eval，run 目录必须新建，如 `bash scripts/run_tiktok_qwen21_full.sh smoke /home/shangguanrz/nihaochen/tiktok_full_my_smoke`）。

## 四、注意事项 / 调用路径
- 环境选择错误会直接 import 失败：Qwen 线 qwen21 env、V6 线 deepgen env；几何工具另用 `add/envs/smplestx_bw`。
- `--resume` 校验配置一致（seed/lr/对齐/清单），改设置必须新输出目录 + `--init-checkpoint`。
- 10-12 之前本目录文件曾是指回仓库根的软链接，现为实体并入 git；历史 runbook 里的 `python train_xxx.py`（根路径）命令属历史记录。
