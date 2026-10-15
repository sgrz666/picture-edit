# workflow 无 GT 的定量评测：调研结论与可落地方案

调研日期 2026-10-13。对象 `sg:/home/shangguanrz/project/pic-edit/workflow`（单人动作迁移：A 出身份/背景，B 出骨架，PE-I2I 写提示词，Qwen-Image-2.1 出图）。
本文所有"本机可用性"均为实测结论（`ssh sg` 上跑 `import` 检查与文件统计），不是推测；引用文献见文末。

---

## 一、先把问题说准：不是"没有 GT"，是"只用了一种参照"

现在的工作流并不是没有指标，而是只测了一条轴。

已有实现（`workflow/evaluate.py`，8.9 KB）：用与几何阶段**同一个** SMPLest-X 回估生成图，取 `B_keypoints.npz` 里请求的骨架（减 `source_preprocess.json` 的 content_box 偏移）逐点比对，分 5 组（`body 0:18 / feet 18:24 / face 24:92 / left_hand 92:113 / right_hand 113:134`）输出 `mpjpe_px`、`mpjpe_norm`（按颈—髋中心躯干长归一）、`p95_norm`、`pck@0.2`、`pck@0.5`、`worst_joint_offset_px`，另记 `persons_detected`、`camera_checks`、`mesh_max_error_m`，产物落在 `runs/<name>/evaluation/{eval.json,final_keypoints.npz,final_smplx.pt,eval_overlay.png}`，并有 `stage.json` 缓存（命中打印 `CACHE_HIT`）。

它自己在 `eval.json.limits` 里写明了边界（原文）：*"只回答生成器有没有把身体摆到骨架要求的位置；身份、服装、场景是否保持不在指标内；两侧用同一估计器，其自身偏差同进分子分母，未与人工标注或第二套检测器交叉验证；请求骨架中回估落在画外的关节被排除而不计为错；`persons_detected>1` 时估计器自选的人不保证是 `--subject` 选的那位。"* `workflow/README.md` 第 114 行重复了同一套局限。

实测的评测覆盖缺口：`runs/` 下 17 个 run，只有 `hard_01~05 / hard_05_manual / hard_05_torso / medium_01 / medium_02 / wp1_accept_hard02` 有 `evaluation/`；而**真正拿来比较变量差异的批量 run 一个都没评**——`testset_hard05_fitgeo`（11 张生成图）、`testset_hard05_promptv2`（7 张）、`testset_hard05_768`（2 张）、`testset_medium02_768`（2 张）、`scale_anchor_20261013`、`demo` 均无。`batch.sh` 其实已经串了 `evaluate.py --run <name>` 和收尾 `report.py`，这些 run 只是早于评测阶段落地时留下的历史产物。

所以"没法定量评测"的实际组成是：① 姿态轴可信度不足（同一估计器自证、无外部校验）；② 身份/服装/背景/画质/指令遵循 5 条轴完全没数；③ 多变体对照（fitgeo vs promptv2 vs 768 vs 尺度定标）没有可比的数字；④ 没有统计口径（`config.json` 里 `seed:42`、`steps:30` 单值单次）。

---

## 二、核心思路：无 GT 时用四类"替代参照"

编辑评测文献的共识做法是不找输出 GT，而是把参照换成下面四类，每类都能给出可复核的数字。

| 参照类型 | 依据 | 能测什么 |
|---|---|---|
| **输入即参照** | A 图含身份/服装/场景，B 图含姿态；指令本身含"该改什么" | 保持度（不该变的没变）、遵循度（该变的变了）。Emu Edit 的 CLIPdir/CLIPim 与识别式指标、CVPRW24 的 proxy metrics 都属此类 |
| **物理与解剖先验** | SMPL-X β 短期不变、肢体长度比跨姿态不变、脚地接触、网格可重投影 | 体型保持、肢体畸形、姿态合理性；`evaluate.py` 已有 `mesh_max_error_m` 就是这个思路 |
| **分布与偏好** | 真实图像流形、人偏好模型、无参考 IQA | 画质、伪影、整体可用率；不需要成对 GT，只需一个参照集合 |
| **伪 GT 校准集** | 从 iPER/TikTok 真视频里取真实 (source, target) 帧对，让同一批数据既有 GT 又能走同一 workflow | 用来**验证上面三类指标是否真的与真值/人判断相关**，并定阈值 |

第 4 类是你现在最缺、也最便宜的一环：`datasets/iPER/iper_sampled_6src64tgt/splits/` 里实测有 `baseline_50_eval_pairs.jsonl`（50 对）、`dev_val_pairs.jsonl`（3,840）、`baseline_3000_microstep_pairs.jsonl`（3,000）、`showcase_20k_8pairs.jsonl`（8）；TikTok 侧 `TikTok_v3/manifests/` 有 `test_pairs.jsonl`、`candidate_pool*.jsonl`、`condition_inventory.jsonl` 等。这些对子的 target 就是**真实照片**：把 A 的身份 + B 的姿态喂进 workflow，输出与真实 target 同场景可比，于是可以在这个子集上算 PSNR/LPIPS/真实身份余弦，同时算所有无 GT 指标，检验二者相关性（SRCC/PLCC）。DirectSwap（跨身份、用帧对齐真实视频当 GT 做 full-reference 评测）就是这一套校准逻辑的近期实现，可直接借用其协议叙述。

---

## 三、指标矩阵（6 条轴 × 具体方法 × 本机落点 × 坑）

### 轴 1 姿态遵循：可信度补强（不新造指标，加交叉验证）

1. **第二读者一致性**：机器上已经有独立估计器 —— `models/DWPose/{dw-ll_ucoco_384.onnx, yolox_l.onnx}`，且 `src/evaluation/joint_pilot_metrics.py:22-40` 的 `PoseMetric` 已封装好（top-down、COCO-17、可见阈 0.3、**按图像对角线归一**的 `pose_pck_002/005`、`pose_joint_recall`、`pose_nme_diagonal`）。
   注意口径差异，两者数字不可混用：`evaluate.py` 的 PCK 阈是 0.2/0.5 且按躯干长归一；`PoseMetric` 是 0.02/0.05 且按对角线归一。报告时要写成 `pck_torso@0.2` 与 `pck_diag@0.02` 两个字段。
   用法：同一张生成图同时过 SMPLest-X（3D）与 DWPose（2D），报两套 PCK 的**一致率**与差值；差异大 = 姿态证据不可信（该图进"需人工看"队列）。双人线的经验口径（宽松姿态看 DWPose、缠绕动作看 3D 拟合）在这里可直接沿用。
2. **肢体长度比恒定性**（跨姿态不变量，专抓扩散常见病：腿被拉长、手臂畸形）：从 `final_smplx.pt` 回估参数取 `joints_cam`，算 6 个比值（上臂/前臂、大腿/小腿、肩宽/髋宽、躯干/腿长），与对 **A 图**同样回估得到的比值比，报相对误差与最大偏差关节。零新依赖。
3. **体型一致性 β**（身份/服装体积的 3D 代理）：SMPL-X 的 β 与姿态解耦、短时间内不变（arXiv:2409.17671 明确用这一点做人体身份一致性；SMPL-X 官方定义见 smpl-x.is.tue.mpg.de）。报 `‖β_out − β_A‖₂` 与逐维（先按真实数据 β 协方差做马氏距离更好）。坑：视角、遮挡、衣物体积都会污染 β，所以只当诊断，不当主指标；同一对 A/B 建议同时报"基线漂移"（A 与真实 target 之间的 β 距离）作标尺。
4. **可见性诚实化**：现有实现把画外关节"排除而不计为错"，会系统性抬高 PCK。补两个字段：`joints_excluded_ratio = (guide_valid − both) / guide_valid`，以及一个"排除即 0 分"的严格版 PCK。同一图两个数一起看，才知道高 PCK 是姿态好还是人出画了。
5. **合理性检查**：脚底接触/悬空（踝—地平面距离与左右脚接触相位）、根朝向与躯干倾角相对 A 的变化；`camera_checks`、`mesh_max_error_m` 已有，扩成可比的标量。

### 轴 2 身份保持：现在完全没测

1. **脸部身份余弦**：`src/evaluation/joint_pilot_metrics.py:64-88` 的 `IdentityMetric` 已实现（`scrfd_10g_bnkps.onnx` 检测 + `norm_crop` + `glintr100.onnx` ArcFace-R100 嵌入 + 余弦，全 CPU，权重实测都在：glintr100 260 MB / scrfd 16.9 MB / 1k3d68 143 MB / 2d106det 5 MB / genderage 1.3 MB）。直接调它算 `identity_cosine(A_face, out_face)`。
   坑（关键）：ArcFace 本身随 yaw/表情下降，绝对值不可跨对比较。两种解法：① 用第 4 类的伪 GT 集标定天花板 `cos(GT_target, A)`，报 `Δ = cos(out,A) − ceiling`；② 同种子多次生成的分布区间。另按检出置信度分层报 `n_reliable`（四臂评测里 `identity_cosine` 只有 47/50 可靠，这个口径值得照搬）。
2. **身体/服装外观一致**：`models/antelopev2/dinov2-giant/model.safetensors`（4.55 GB，含 `config.json`/`preprocessor_config.json`）可用 `transformers` 直接加载；在人体掩膜内比较 A 与 out 的 DINOv2 特征余弦（身份+服装的联合代理）。坑：DINOv2 对姿态/视角敏感，必须与轴 1 的姿态误差**一起**看，姿态偏了外观分自然掉。
3. **部件/服装类别一致性**：`add/models/sapiens-seg-0.3b-torchscript/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2`（104 类部件分割）→ 报分割图类别面积占比差、"换装检测"（例如上衣类别从 shirt 变 dress 的占比变化）。这条能抓住"PCK 很好但衣服换了"这一类失败。

### 轴 3 场景与背景保持：实现现成，只差接进 workflow

`scripts/eval/eval_iper_four_arm.py:262` 已把口径写清：`background_mae_source` = *"在 source 与 target 人体掩膜并集之外，与 source 的 RGB MAE（0–255）"*，另有 `human_mae`、`background_ssim_source`、`human_ssim`。把同一函数接进 workflow 即可，但需要一个任意照片可用的掩膜来源（iPER 的掩膜是资产自带的）：用轴 2.3 的 Sapiens 分割，或用 SMPLest-X 回估网格投影成 person mask（pipeline 里已有渲染路径）。对齐要按 `source_preprocess.json` 的 content_box 反变换，否则背景会整体平移、MAE 虚高。
机器上 `lpips` 在两个环境都已安装 → 背景区域再加 `LPIPS_bg`，比 MAE 更接近人眼。
多人图"别人被改动"（`--subject` 相关，README 第 43 行承认没覆盖）：用已装的 `ultralytics 8.4.159` 检人框，A 与 out 的非主体框区域做 IoU 匹配 + 区域 MAE，报 `nonsubject_displacement` 与 `nonsubject_mae`。

### 轴 4 画质与解剖合理性（无参考）

- **NR-IQA 全家桶**：`pyiqa` 未安装（实测 `import` 失败），但它是一个纯 Python 包 + 权重文件，可在 Windows 下 `pip download` 后 `scp` 上去（node01 无外网，这是唯一路子）。里面有 MUSIQ / MANIQA / CLIPIQA+ / LIQE / TOPIQ / Q-Align；AIGC 画质专用还出了 ELIQ（label-free，arXiv:2602.03558）。建议先只接 2 个：`musiq`（快、无需参考）与 `topiq_nr`（对畸形/伪影敏感），避免指标堆砌。
  **2026-10-14 更正**：这句"node01 无外网"**不成立**——实测 `curl` pypi/HF/GitHub 均 200、`git ls-remote --heads origin` rc=0（3.1 s）、`pip install --dry-run torchao` 经 aliyun 镜像真的下到了 wheel。
  `pyiqa` 可以**在 sg 上另建环境直接装**（红线 7 仍然有效：不要塞进 `qwen21`）。证据见 `workflow_PE重写成本_量化与蒸馏调研_20261014.md` §1.5。
- **手脸专项**：手 21 点共线性/指长比异常、回估 face 组重投影残差（已有 `face_mpjpe_norm`）、`persons_detected≠1` 直接标红。
- **分布级 FID/KID**：`pytorch_fid`/`cleanfid` 都没装，且 InceptionV3 权重也需 scp。低成本替代：用已在本地的 DINOv2-giant 特征，对参考集（iPER/TikTok 真帧）算特征均值/协方差的 Fréchet 距离与 MMD，作"分布外程度"参考。注意这是集合级指标，单图无意义，只在批量 run 上报。

### 轴 5 指令遵循（PE 重写后的提示词到底实现没有）

- **CLIP 方向性相似**（Emu Edit 的 CLIPdir / CLIPim；ADIEE 指出 CLIP/DINO 单看输出会忽略文本方向，故必须用"方向"形式）：本机**没有** CLIP（`clip`、`open_clip` 均未安装，全盘 grep `*clip*` 只命中一些 mp4）。要么 scp `CLIP-ViT-L/14` 权重（transformers 已可用 `CLIPModel`），要么用下面这条更省事的本地路线。
- **本地 VQA 评分（推荐，零下载）**：`pic-edit-models/Qwen-Image-2.1/text_encoder/config.json` 实测 `architectures=['Qwen3VLForConditionalGeneration']` 且**带 `vision_config`**，同级 `processor/preprocessor_config.json` 是完整的图像预处理配置 —— 也就是说这 18 GB 的"文本编码器"本身就是一个可用的本地多模态理解模型。据此可离线复刻 VIEScore（arXiv:2312.14867，ACL 2024；代码 TIGER-AI-Lab/VIEScore）的三件套：`SE`（指令遵循，逐项 yes/no 问答后取合规比）、`OQ`（整体画质）、`D`（带解释的对比判断），以及 TIFA 式的 QA 准确率。VIEScore 的核心价值正是不需要 GT、且比 CLIP 更贴人判断（ACL 版给出与人类判断的相关性分析）。
  坑（必须写进报告）：MLLM 打分的 self-preference bias（评分模型与生成模型同源时更明显）、对手部/计数/空间关系不可靠、分数不可跨 prompt 版本直接比。ICCV 2025 的"What Changed?"用 MLLM 做**变更定位**再判遵循度，这条"先定位改了哪里、再判断改得对不对"的流程比整图打分稳，值得照抄其两步式；IE-Critic-R1（arXiv:2511.18055）是同方向的近期工作。
- **偏好奖励模型**：ImageReward（arXiv:2304.05977）、HPSv2（tgxs002/HPSv2）、PickScore 面向文生图；编辑专用有 EditReward（tiger-ai-lab.github.io/EditReward，arXiv:2509.26346）。它们的共同用法是**成对比较**（同一对 A/B 的两个变体哪个更好），比绝对分可信得多，正好适配你"fitgeo vs promptv2 vs 768"这类对照。注意需 scp 权重。

### 轴 6 统计口径：把"一张图"变成"一个结论"

- 现在 `config.json` 是 `seed 42 / steps 30 / resolution 1024`（batch 用 `RESOLUTION=768`）单次运行 → 单点数字没有方差。建议每对至少 3 seed，报中位数与 IQR；`peak_cuda_gib`/`seconds` 也顺带有了。
- 变体比较一律用**配对**统计 + bootstrap CI95：四臂评测里 `summary.json.paired_deltas_second_minus_first`（5,000 次有放回、seed 20260927）已是现成口径，直接复用其字段命名与"只在两侧都有值的样本上算差"的处理。
- 预设主指标（建议三个）：`pck_torso@0.2`（遵循）、`identity_cosine Δ`（身份）、`background_mae_source`（保持）；其余全部标"诊断"，不参与"变好了"的结论，避免事后挑数字。
- 指标有效性证明：抽 40–60 张生成图做 3 轴 Likert 人工打分（姿态对不对/人还是不是同一个人/背景与衣服有没有被改），与各指标算 SRCC。只有相关的指标才留在 `report.py` 主表。

---

## 四、落地路线（三档，按引入成本严格排序）

**T0｜零新依赖，一天内能出数（全部用已验证存在的权重/包）**
1. 把 `evaluate.py` 从"单 run"扩成可 `--all`/`--runs` 遍历，补跑那 6 个无评测的批量 run（历史产物不动，遵守"不覆盖既有产物"）。
2. 同一 `evaluation/` 目录新增 **`eval2.json`**（不覆写 `eval.json`），字段建议：
   `pose`（`pck_torso@0.2/0.5`、`pck_diag@0.02/0.05`(DWPose)、`cross_estimator_agreement`、`joints_excluded_ratio`、`strict_pck`）；
   `shape`（`limb_ratio_max_relerr`、`beta_l2`、`beta_maha`）；
   `identity`（`face_cosine`、`face_n_reliable`、`dino_giant_person_cosine`）；
   `preservation`（`background_mae_source`、`background_lpips`、`human_mae`、`nonsubject_mae`、`seg_class_area_abs_delta`）；
   `plausibility`（`persons_detected`、`mesh_max_error_m`、`foot_contact_dev`、`hand_anomaly`）；
   `run_meta`（seed、steps、resolution、prompt 来源哈希、估计器模型签名 —— 复用 `common.model_signature` 与 `sha256_file`）。
3. 实现位置按 AGENTS.md 走：指标函数进 `src/evaluation/no_gt_metrics.py`（与 `joint_pilot_metrics.py` 同族、同样懒加载 ONNX），workflow 侧只加入口脚本；每条指标都要"同一张图一次前向多处取值"（尤其 SMPLest-X 回估很慢，别每条重跑）。
4. 用 `report.py` 现成的 `stage_facts/run_facts` 结构把 `eval2.json` 并进去，主表只印预设的三个主指标。

**T1｜需要 scp 权重（本地下载 → `scp` 上机，不走外网 pip）**
`pyiqa`（musiq + topiq_nr 两个头）、CLIP ViT-L/14（算 CLIPdir/CLIPim）、可选 `cleanfid`/FID-Inception。预期成本：包与权重各一次传输，代码几十行。

**T2｜需要一点工程投入，但价值最高**
本地 Qwen3VL 做 VIEScore/TIFA 式两步打分（先"定位改了哪里"再"判断遵循与保持"），以及第 4 类**伪 GT 校准集**：取 iPER `baseline_50_eval_pairs.jsonl` 的 50 对 + TikTok `test_pairs.jsonl`，走同一 workflow，用真 target 作参照算 full-reference 指标，再算无 GT 指标与二者的 SRCC —— 这一步把"指标"变成"被证明有效的指标"，论文里也站得住。

---

## 五、明确不建议

1. 拿 CLIPScore 或整图 PSNR 当"效果变好"的证据：姿态编辑里前者对指令方向不敏感（ADIEE 明确指出），后者在无 GT 时根本没有对象。
2. 单一 MLLM 打分当权威结论：self-preference 与手部/计数盲区，必须与轴 1–3 的客观量同时出现。
3. 每条指标各自重跑一遍 SMPLest-X/DWPose：批量 run 会被拖死，一次前向 → 多指标共享。
4. 把 `pck@0.2`（躯干归一）与 `pck@0.02`（对角线归一）写进同一列比较：这是两套不可换算的口径，字段名必须显式带归一方式。

---

## 七、PoC 实测（2026-10-13，零新依赖）

只调用现成的 `src/evaluation/joint_pilot_metrics.py:IdentityMetric`（`scrfd_10g_bnkps` 检脸 + `glintr100` 嵌入），对 15 个 run 各取 `inputs/A_rgb.png`（或 `A_reference.png`）与 `generation/final_image.png` 算脸部身份余弦。全部 15 张两侧都检出了脸，无需 GPU：

| run | id_cos | | run | id_cos |
|---|---|---|---|---|
| hard_01 | 0.5004 | | testset_hard05_768 | 0.8183 |
| hard_02 | 0.6559 | | testset_hard05_fitgeo | **0.3430** |
| hard_03 | **0.2259**（最低） | | testset_hard05_promptv2 | **0.8400**（最高） |
| hard_04 | 0.5406 | | testset_medium02_768 | 0.7648 |
| hard_05 | 0.8183 | | wp1_accept_hard02 | 0.6559 |
| hard_05_manual | 0.3149 | | demo | 0.5098 |
| hard_05_torso | 0.3892 | | medium_01 | 0.2462 |
| medium_02 | 0.7648 | | | |

n=15，mean 0.5592，median 0.5406，min 0.2259，max 0.8400。

这批数已经能回答"变体之间到底有没有差别"：同一对 `hard_05`，PE 编译提示词与那两份手写/legacy 提示词的身份保持差了一倍多（0.8183 / 0.8400 vs 0.3430 / 0.3149 / 0.3892）；而 README 记录的手写提示词在 **姿态** 轴上更好（body PCK 0.667 vs PE 0.389）——两轴方向相反，正是"只看一条轴会得出相反结论"的实例，也说明为什么主指标必须同时锁三个。

PoC 顺带暴露两个必须写进协议的坑：
1. **重复图**：`testset_hard05_768`≡`hard_05`、`testset_medium02_768`≡`medium_02`、`wp1_accept_hard02`≡`hard_02`（余弦逐位相同 = 同一张输出）。批量统计前必须按图像 SHA256 去重，否则同一张图被计多次、均值被拉偏（与 AGENTS.md 第 5 条"副本产物与对照 SHA256 相同即副本没生效"是同一类问题）。
2. **绝对值不可跨对比较**：ArcFace 随 yaw/遮挡下降，`hard_03` 的 0.2259 可能是大侧脸而非身份丢失。所以正式口径是第 2/4 节讲的 `Δ = cos(out,A) − cos(GT_target,A)`（伪 GT 天花板），或至少只在同一对内部比较变体。

复现（远端，只读，不写任何 run 产物）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python - <<'PY'
import sys; from pathlib import Path; import numpy as np; from PIL import Image
ROOT = Path('/home/shangguanrz/project/pic-edit'); sys.path.insert(0, str(ROOT))
from src.evaluation.joint_pilot_metrics import IdentityMetric
im = IdentityMetric()
for run in sorted((ROOT/'workflow/runs').iterdir()):
    gen, src = run/'generation/final_image.png', run/'inputs/A_rgb.png'
    if not (gen.is_file() and src.is_file()): continue
    with Image.open(src) as a, Image.open(gen) as g:
        e1, e2 = im.embedding(a.convert('RGB')), im.embedding(g.convert('RGB'))
    print(run.name, im.cosine(e1, e2))
PY
```

---

## 八、参考（文献与开源）

编辑评测与指标：[Emu Edit: Precise Image Editing via Recognition and Generation Tasks](https://arxiv.org/abs/2311.10089)（CLIPdir/CLIPim、识别式指标）、[Towards Quantitative Evaluation Metrics for Image Editing Approaches (CVPRW 2024 EvGenFM)](https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Hochberg_Towards_Quantitative_Evaluation_Metrics_for_Image_Editing_Approaches_CVPRW_2024_paper.pdf)、[ADIEE: Automatic Dataset Creation and Scorer for Image Editing](https://arxiv.org/html/2507.07317v1)、[Instruction-based image editing: a survey on data, models, evaluation and applications](https://link.springer.com/article/10.1007/s44336-026-00034-3)、[SpotEdit: Evaluating Visually-Guided Image Editing Methods](https://arxiv.org/html/2508.18159v1)、[What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal LLMs (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Baraldi_What_Changed_Detecting_and_Evaluating_Instruction-Guided_Image_Edits_with_Multimodal_ICCV_2025_paper.pdf)、[IE-Critic-R1](https://arxiv.org/html/2511.18055v1)、[Awesome Evaluation of Visual Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation/blob/master/README.md)。
MLLM/偏好打分：[VIEScore](https://arxiv.org/abs/2312.14867)（[代码](https://github.com/TIGER-AI-Lab/VIEScore)、[项目页](https://tiger-ai-lab.github.io/VIEScore/)）、[EditReward](https://tiger-ai-lab.github.io/EditReward/)（[论文](https://huggingface.co/papers/2509.26346)）、[ImageReward](https://arxiv.org/html/2304.05977v4)、[HPSv2](https://github.com/tgxs002/HPSv2)。
无参考 IQA：[IQA-PyTorch / pyiqa](https://github.com/chaofengc/iqa-pytorch)、[ELIQ: label-free AIGC IQA](https://arxiv.org/html/2602.03558v1)、[MANIQA](https://www.researchgate.net/publication/362884664_MANIQA_Multi-dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment)。
人体几何与身份：[SMPL-X](https://smpl-x.is.tue.mpg.de/)、[Leveraging Anthropometric Measurements to Improve …（β 作为不随姿态变化的人体形状/身份量）](https://arxiv.org/html/2409.17671v1)、[Scaling Up Expressive Human Pose and Shape Estimation（SMPLest-X，NeurIPS 2023）](https://proceedings.neurips.cc/paper_files/paper/2023/file/2614947a25d7c435bcd56c51958ddcb1-Supplemental-Datasets_and_Benchmarks.pdf)、[DirectSwap（用帧对齐真实视频构造 full-reference 跨身份评测）](https://arxiv.org/html/2512.09417v2)、[Dense Intrinsic Appearance Flow for Human Pose Transfer](https://mmlab.ie.cuhk.edu.hk/projects/pose-transfer/)、[Weakly Supervised 2D Human Pose Transfer](https://vcc.tech/research/2021/PoseTransfer)、[Imposing temporal consistency on deep monocular body pose and shape estimation（β 时序/身份一致性）](https://ieeexplore.ieee.org/iel8/10750449/10897681/10897690.pdf)。
