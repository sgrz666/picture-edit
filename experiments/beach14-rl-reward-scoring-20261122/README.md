# beach14 RL reward 复合分打分（2026-11-22）

## 1. 实验目的

用 GRPO 训练同一套冻结打分内核，给 `2.beach14` seed42 的 4 臂 × 14 模板生成图打 RL reward 复合分，得到与未来双人场景 RL 训练直接可比的排名，并保留全部分项与资格诊断。打分口径按用户 2026-11-22 决定：内核照搬（双人五项权重），姿态后端由 DWPose ONNX 换成项目统一的 SDPose-Wholebody + YOLO11x。

## 2. 里面有什么

- [scripts/score_rl_rewards.py](scripts/score_rl_rewards.py)：打分脚本（远端正式副本在 `sg:/data/shangguanrz/pic-edit-experiments/two/workflow-v2/experiments/2.beach14/scripts/`，SHA256 `71592b7497cdcd445dc24f14b2d780be2f42cafc340f390bbb0b89c5034aaf2a`，两侧一致）。
  - 打分内核：runtime-v3 冻结的 `src/pose_control/qwen21_full/native_rl/rewards.py` 的 `score_candidate` + `DUAL_WEIGHTS`（pose .30 / identity .25 / binding .20 / layout .15 / contact .10，缺资格分项按剩余权重归一化），OKS 位姿分 + 匈牙利身份盲配对。
  - 姿态后端：SDPose-Wholebody + YOLO11x（`src/evaluation/sdpose.py`，score > 0.30、前景 min_area 5000，与 beach14 既有 SDPose 评测口径一致），134 槽位按 `cocowholebody133_to_dwpose134` 反解映射到 COCO17。
  - 身份后端：antelopev2（scrfd_10g_bnkps + glintr100），目标身份取自 `beach.jpg` 左右两人嵌入；目标姿势来自各样本 `B_reference.png` 按 `a_content_box` 裁剪（与 `final_image.png` 同一 669×896 像素空间）。
  - contact 自动推导：`pose_1120` 的 `maps_3d.npz` part/owner 顶点 3D 证据（顶点标签采样要求 owner 为本人，剔除拥抱互遮挡面；贴合阈值 10 mm，对齐原 fcl 网格碰撞语义），nominal 映射 label L → RL 部位组 L-1，per-person 关节投票只作诊断；期望距离为 SDPose 目标姿态对应部位关节均值在打分画布下的距离。
  - 资格语义：目标检测不是恰好 2 人 → 整模板不合格，不进聚合；生成人脸漏检记 `id_missing`，binding 项记 0。
- 远端产物 `sg:.../2.beach14/results/rl_score/`（本目录不落产物）：
  - `rl_score.json`：每模板每臂 reward + details（person_pose / pck_002 / joint_recall / id_cosine / id_swap / id_missing / matched_people）+ composite + contacts 诊断（投票、分歧、每对 3D gap 与期望距离）。
  - `summary.json`：按臂聚合 + manifest（runtime-v3 路径、rewards.py 与 sdpose.py SHA256、SDPose ckpt/OOD 路径、antelopev2 路径、阈值、GPU 等待轮数）。
  - `leaderboard.md`：排名表 + 每模板矩阵；`rl_score_panel.png`：排名条形图 + 每模板 composite 热图。
- 运行命令（SDPose 环境）：`CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21 <SDPose venv python> scripts/score_rl_rewards.py`（默认全量 4 臂 × 14 模板；输出目录已存在即拒绝，防覆盖）。

## 3. 关键结论（12/14 可打分；t05/t10 目标检出 3 人判不合格，与 beach14 既有 SDPose 评测排除原因一致）

| 排名 | 臂 | composite 均值 | 中位数 | pose | identity | binding | layout | contact |
|---|---|---|---|---|---|---|---|---|
| 1 | `#4` with_b+skel+Depth | `0.7014` | `0.7299` | 0.600 | 0.640 | 0.888 | 0.740 | 0.738 |
| 2 | `#7` skel_only | `0.6865` | `0.6980` | 0.534 | 0.672 | 0.900 | 0.676 | 0.809 |
| 3 | `#3` with_b+skel | `0.6813` | `0.6723` | 0.463 | 0.669 | 0.974 | 0.684 | 0.823 |
| 4 | `#8` skel_only+Depth | `0.6766` | `0.6929` | 0.595 | 0.612 | 0.792 | 0.752 | 0.802 |

- **与 beach14 README 3.1 节 SDPose 纯姿态口径排名不同**（那边 `#8` 最优）：RL 复合分 identity/binding 权重大，`#8` 被 `t02` 人脸漏检（composite `0.272`）拖累；`#4` 在 `t04`（`0.827`，双人握手全场最高档）/`t06`/`t08` 的 Depth 收益被放大。两个口径回答的问题不同，各自保留。
- `t04` 的 Depth 增益在 RL 口径下成立（`#7` 0.621 → `#8` 0.664；`#3` 0.589 → `#4` 0.827）；`t14` `#4` layout 崩到 `0.081`，补充 `with_b` 失败模式证据。
- `binding=0.000` 的格子（t02 `#8`、t11 `#4/#7/#8`）经 `details` 核实全部是**生成人脸漏检**（`id_missing=1`），不是身份交换；`t02` `#8` 可见人脸 cos 仅 `0.378`。
- `t06` 接触项 `0.000/0.010/0.019`：目标侧推导出接触对但生成图对应部位关节不可用，属生成质量信号。
- contact 推导自校验：t01（拥抱）给出 4 个躯干/手臂贴合对；t04（握手）给出前臂↔手 16 px 紧接触；t09（无接触姿势）为空列表。

## 4. 与其他实验的关系

- **打分对象**：`2.beach14/results/`（seed42，4 臂 × 14 模板）；multirun 三 seed 与 `1.beach`、`results_s3_20k` 本轮未打（用户圈定范围），脚本 `--templates/--arms` 可扩。
- **内核来源**：`xiaohongshu-region-sdpose-sft-rl-20261121-expanded-v1/source/runtime-v3`（与进行中的小红书 SFT/GRPO 训练同一冻结源码）；姿态后端统一 SDPose（`docs/SDPOSE_POSE_EVAL_SPEC.md`），不向 qwen21/SDPose 环境装任何包。
- **后续**：若在双人场景启动 GRPO，本表即 reward 基线；multirun 三 seed 打分可直接复用本脚本。
