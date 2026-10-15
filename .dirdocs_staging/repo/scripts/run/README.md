# scripts/run/ —— 单次实验的编排与收尾：从"启动一次 run"到"等它结束并验收"
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/run　|　本级：1 个子目录（`home/`，4 个链接）/ 14 个实体文件（约 0.37 MB，`du -sh --si` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md、`__pycache__`

## 一、这一层在干什么
train/eval 是"一次前向/一批指标"的粒度，本目录是"一次完整实验"的粒度：装配数据+权重+采样+诊断分阶段跑（phase 编排）、等待长训练收尾并自动评测、把远端产物打包回传 Windows。多数入口明确"有界"（bounded）：固定样本数、固定步数、失败即停、不覆盖旧 run。

## 二、层级地图（按实验线分类）
| 实验线 | 文件 | 实证 |
|---|---|---|
| CHAMP 单样本（DeepGen 8 通道控制） | `run_champ_single_overfit.py`（`--validate_only` 本地即可跑；正式训练输出 `experiments/champ_deepgen_formal_v1/`，完整命令与产物清单在 `docs/champ_deepgen_single_sample_runbook.md`）、`run_deepgen_no_control_ablation.py`（同 seed/步数/CFG 的纯 DeepGen 对照，`config.json` 中 `adapter_created` 与 `control_arguments_passed` 必须为 false，产 `diagnostics.json`/`summary.json`） | 依赖 `patches/deepgen_pose_control.patch` 打过补丁的 `models/DeepGen-1.0-diffusers/deepgen_pipeline.py` |
| Qwen21 Face 单样本诊断 | `run_qwen21_face_single_sample.py`（--phase 分阶段；"每个 GPU 动作都用完整控制器"）、`finalize_qwen21_face_single_sample.py`（收结果+显式单例判据核验）、`finalize_qwen21_face_repair.py`（合并修复检查与人工单例） | 与 `scripts/audit/probe|verify_qwen21_face_single_sample.py` 构成跑→验→收尾 |
| BFS 联合试点 | `run_qwen21_joint_pilot.py`（有界 phase 训练；128→256 续训必须先完成 12 个固定监看案例的书面图像审查）、`sync_qwen21_joint_pilot.ps1`（Windows 侧：ssh 到 sg 把 `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008` 的 data/preflight/evaluation/… 打 tar 回传到 `D:/codeplus/pictureedit/research/…`） | 交接结论见 `docs/experiments/` |
| Hi4D 双人长训收尾 | `complete_hi4d_qwen21_dual.py`（`--train-pid` 轮询等待、重启上限 `--max-restarts`，训练完自动对留出测试集评测） | 配 `scripts/train/train_hi4d_qwen21_dual.py` |
| 权重继承对比 | `run_qwen21_weight_inheritance.py`（固定 60 图继承对照，不训练） | 配 `weights/inherit_qwen21_full_weights.py` |
| 条件组合补充实验 | `run_qwen21_condition_combinations.py`（两个锁定正式推理批次 + 持久化 stage 状态） | 配 `eval/prepare_qwen21_condition_combinations.py`、`audit/verify_qwen21_condition_combinations.py` |
| TikTok 冒烟/历史启动器 | `run_tiktok_sample_00001_smoke.py`（00001 序列 Source 0014→Target 0074；输出默认 `experiments/tiktok_smoke_00001/`，初始化 `experiments/iper_v6_native_20k_v1/checkpoint_best.pt`）、`run_tiktok_v3_fill_missing_geometry_bg.sh`、`run_v6_comparative_3000.sh`（V6.3 对照训练 3,000 micro-steps，deepgen 环境，输出 `experiments/iper_v6_comparative_3000/`，该目录现已冷归档到 `/data/shangguanrz/pic-edit-archive/cleanup_20261003/`） | bg.sh 有陈旧路径问题，见下 |
| 演示 | `run_demo_pose_edit.py`（DeepGen 基座姿态编辑演示，输出 `demooutput/`） | — |

## 三、符号链接与实体位置
本目录实体 14 个，无内部链接；但 `scripts/` 根有 2 个兼容链接指向这里：`run_qwen21_face_single_sample.py`、`finalize_qwen21_face_single_sample.py`（历史文档中的旧命令据此仍可解析）。`home/` 全部是指向 `/home/shangguanrz/` 实体的链接，见 `home/README.md`。

## 四、注意事项 / 调用路径
- `run_tiktok_v3_fill_missing_geometry_bg.sh` 在仓库根 `cd` 后调 `fill_tiktok_v3_missing_geometry.py`，该实体现位于 `scripts/data/`，直接执行会 FileNotFound，需改为 `"$PYTHON" -u scripts/data/fill_tiktok_v3_missing_geometry.py`。
- 冒烟 vs 正式：smoke/单例 runner 的结论不可外推为泛化能力（多个 docstring 明说 "not a training-coverage or quality claim"）。
- 启动长任务前先查 GPU 与根分区（历史根分区 96%）；`sync_qwen21_joint_pilot.ps1` 只在 Windows Git Bash/PowerShell 侧运行。
