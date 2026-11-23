# face — Native V7 Identity Reward 人脸后端对照（B0/E1/E2/E3）

> 服务器实验根：`/data/shangguanrz/pic-edit-experiments/姿态估计/face`。
> 本目录只保留 README、指标汇总与图、脚本；嵌入缓存（249 MB）、对齐脸（96 MB）、
> `.venv`、`build/rpe` 与 `data/scan-v1` 留在服务器，不入仓库。

## 1. 实验目的

Native V7 RL 的 Identity Reward 目前只有 `glintr100`（ArcFace R100，Glint360K）一个后端。
本实验回答三个问题：

1. 换成 LVFace-B（ViT-B，ICCV 2025）或 AdaFace ViT-B KP-RPE（关键点相对位置编码）后，
   **在我们自己的人像数据上**同人/异人区分能力是否真的提升；
2. 双模型融合（`0.5·S_LVFace + 0.5·S_KPRPE`）是否稳定优于单模型；
3. 现有分数的差距有多少来自**检测、五点对齐、侧脸、低清晰度**，而不是识别网络本身。

## 2. 里面有什么

```
face/
├── .venv/                 # --system-site-packages 复用 qwen21 的 torch，只额外装 omegaconf/easydict
├── build/rpe/             # rpe_index_cpp.cpython-312-x86_64-linux-gnu.so（KP-RPE 的 CUDA 扩展，40s 编完）
├── data/scan-v1/          # scan.json（3 464 张图的检测框/五点/质量）+ crops.npz（3 418 张 112×112 对齐脸）
├── data/pairs-v1.json     # 1 474 同人 + 1 474 异人（tier-1）；98 + 98 真实→生成（tier-2）
├── output/emb/            # 10 组嵌入：clean / align_j5 / align_j10 / blur1.5 / lowres48 / lowres32 / jpeg20 / occl / genlike / clean_flip
├── output/results-v1/     # results.json（全部指标）、results-v1-flip/
├── output/binding-check.json, det_sweep.json
├── report/summary.md      # 自动生成的完整表格 + 三张图（figures/）
└── scripts/               # 见下表
```

| 脚本 | 作用 |
|---|---|
| `facekit.py` | 检测（SCRFD 10G bnkps）、5 点 ArcFace 对齐（可加抖动）、降质、四个后端封装 |
| `scan.py` | 采样图片、检测、缓存对齐脸与质量描述子 |
| `build_pairs.py` | 从扫描结果生成同人/异人对（按尺寸、姿态、清晰度分档；跨片段正例优先） |
| `extract.py` | 单条件下四模型批量出嵌入（`--flip-tta` 做原图+镜像平均） |
| `evaluate.py` | AUC / EER / TAR@FAR / d' / 组内 SNR / 分层 / 融合 / 配对 bootstrap |
| `report.py` | 生成 `report/summary.md` 与图 |
| `binding_check.py` | 双人 2×2 绑定矩阵与"最大脸规则"绑错率 |
| `det_sweep.py` | det_size 640 vs 1024 的检出率对照 |
| `build_rpe.sh` | 编译 KP-RPE 的 `rpe_index_cpp` |

复现（`PY=.venv/bin/python`，需 `PYTHONPATH=$PWD/build/rpe`）：

```bash
$PY scripts/scan.py --out data/scan-v1 --tiktok-frames 8 --iper-sources 3 --iper-targets 4
$PY scripts/build_pairs.py --scan data/scan-v1/scan.json --out data/pairs-v1.json \
    --tiktok-positives 4 --iper-positives 3 --iper-cross 6
for C in clean align_j5 align_j10 blur1.5 lowres48 lowres32 jpeg20 occl genlike; do
  $PY scripts/extract.py --crops data/scan-v1/crops.npz --out output/emb/$C.npz --condition $C
done
$PY scripts/evaluate.py --pairs data/pairs-v1.json --emb output/emb \
    --scan data/scan-v1/scan.json --conditions clean align_j5 align_j10 blur1.5 \
    lowres48 lowres32 jpeg20 occl genlike --out output/results-v1
```

### 数据与口径

- **tier-1（统计主力）**：TikTok 340 条序列（一条=一个身份，取 8 帧）+ iPER 28 个 subject
  （103 个 clip，source 3 帧 + target 4 帧），共 **368 个身份、3 418 张对齐脸**。
  正例为同身份不同帧（1 378 对）与不同 clip（96 对，换衣/换光/换机位，最接近 RL 场景）；
  负例为异身份且尽量同尺寸档匹配（1 474 对）。
- **tier-2（跨域检查）**：`workflow-v2/experiments/2.beach14/results_native_v7` 的 56 张 native v7
  生成图 + 源图 `A_reference.png`。按左右顺序绑定 Person 0/1（源图男左女右，生成图布局一致），
  过滤重影/粘连/小脸后剩 49 帧有效。
- 所有模型**共用同一次 SCRFD 检测与同一次 5 点 ArcFace 对齐**，差异只来自识别网络；
  KP-RPE 的五点由同一对齐矩阵变换到 112 画布并除以 112（该模型的 RPE 要求 [0,1] 归一化坐标）。
- 融合在校准身份（368 的一半）上拟合，在另一半身份上评测。

## 3. 关键结论

### 3.1 主结果（tier-1 clean，2 948 对）

| 模型 | AUC | EER | TAR@FAR=1e-3 | 同人cos | 异人cos | d' | 组内SNR | ΔAUC vs glintr100 |
|---|---|---|---|---|---|---|---|---|
| glintr100（当前） | 0.9830 | 4.58% | 0.353 | 0.586 | 0.048 | 3.69 | 2.99 | — |
| w600k_r50 | 0.9884 | 3.60% | 0.341 | 0.565 | 0.033 | 3.86 | 3.07 | +0.0054 [+0.0028,+0.0084] |
| LVFace-B | 0.9833 | 4.07% | 0.478 | 0.494 | 0.019 | 3.36 | 2.54 | +0.0003 [−0.0029,+0.0038] |
| **AdaFace-KPRPE** | **0.9906** | **3.46%** | **0.514** | 0.615 | 0.046 | **4.05** | **3.32** | **+0.0076 [+0.0047,+0.0109]** |

- **LVFace-B 在我们的数据上没有超过 glintr100**：AUC 只差 +0.0003（95% CI 跨 0）。
  它公开的 IJB-C TAR@1e-4=97.70% 不能迁移到这里：IJB-C 是证件/监控正脸，我们是舞蹈视频、
  小脸、动态模糊。按"公开榜单更强"直接换模型是不成立的。
- 唯一值得肯定 LVFace-B 的地方是**低 FAR 尾部**：TAR@1e-3 0.478 明显高于 glintr100 的 0.353
  （AUC 持平但尾部更干净），仍低于 KPRPE 的 0.514。
- **AdaFace-KPRPE 全面第一**：AUC +0.0076、TAR@1e-3 相对提升 46%、d' 4.05（其余 3.36–3.86）。
- `w600k_r50`（R50，WebFace600K）比 `glintr100`（R100，Glint360K）高 0.0054，说明**训练域匹配
  比骨干容量更重要**，不能把"R100 比 R50 大"当成更强。

### 3.2 差距主要来自哪里（分档，clean）

| 分档 | glintr100 | w600k_r50 | LVFace-B | AdaFace-KPRPE |
|---|---|---|---|---|
| 大脸 ≥96px（n=922） | 0.9882 | 0.9895 | 0.9893 | 0.9927 |
| 小脸 48–96px（n=488） | 0.9836 | 0.9928 | 0.9819 | 0.9933 |
| 微脸 <48px（n=64） | **0.8965** | 0.9387 | 0.8957 | 0.9449 |
| 正脸（n=948） | 0.9905 | 0.9929 | 0.9915 | 0.9934 |
| 半侧（n=226） | 0.9900 | 0.9927 | 0.9897 | 0.9950 |
| 侧脸（n=300） | **0.9599** | 0.9778 | 0.9587 | **0.9852** |
| 模糊（n=444） | 0.9702 | 0.9814 | 0.9665 | **0.9863** |
| 清晰（n=1030） | 0.9904 | 0.9929 | 0.9917 | 0.9940 |

- glintr100 的短板集中在**微脸（0.8965）和侧脸（0.9599）**，不是整体偏弱；
  KPRPE 在侧脸档领先 0.025、模糊档领先 0.016，这两档正是生成图最常见的情形。
- LVFace-B 在微脸/侧脸两档与 glintr100 基本同水平（0.8957 / 0.9587），
  说明它的 ViT 主干没解决小脸问题。

### 3.3 鲁棒性消融（tier-1 AUC）

| 条件 | glintr100 | w600k_r50 | LVFace-B | AdaFace-KPRPE |
|---|---|---|---|---|
| clean | 0.9830 | 0.9884 | 0.9833 | 0.9906 |
| 对齐抖动 ±5°/4%/3px | 0.9821 | 0.9877 | 0.9823 | **0.9905** |
| 对齐抖动 ±10°/8%/6px | 0.9798 | 0.9867 | 0.9792 | 0.9901 |
| 高斯模糊 σ=1.5 | 0.9796 | 0.9880 | 0.9836 | 0.9907 |
| 降采样 48px | 0.9813 | 0.9883 | 0.9845 | 0.9906 |
| 降采样 32px | 0.9786 | 0.9877 | 0.9829 | 0.9904 |
| JPEG q=20 | 0.9851 | 0.9867 | 0.9789 | 0.9899 |
| 下半脸遮挡 | 0.9711 | 0.9812 | 0.9711 | 0.9833 |
| 生成式复合降质 | 0.9751 | 0.9845 | 0.9754 | 0.9879 |

- 对齐抖动 ±10° 下 **LVFace-B 掉 0.0041、glintr100 掉 0.0032，KPRPE 只掉 0.0005**，
  与 KP-RPE 的设计目标一致；但它的绝对收益（+0.0076）才是选型理由，抗抖动只是附带优势。
- 下半脸遮挡让四个模型同时掉 0.007–0.012（glintr100/LVFace-B 掉得更多），排名不变
  → 遮挡是共同瓶颈，换模型解决不了。
- 降质（模糊/低清）对 KPRPE 几乎无损，对 glintr100 有 0.002–0.004 的损失。

### 3.4 双模型融合（留出身份）

| 融合 | AUC | TAR@1e-3 |
|---|---|---|
| z: glintr100+LVFace-B | 0.9839 | 0.355 |
| z: LVFace-B+KPRPE | 0.9888 | 0.391 |
| z: 三模型 | 0.9884 | 0.348 |
| lr: LVFace-B+KPRPE | 0.9893 | 0.376 |
| lr: 三模型 | 0.9891 | 0.354 |
| **单模型 AdaFace-KPRPE** | **0.9906** | **0.514** |

**融合没有收益**：最好的融合（0.9893 / 0.376）仍低于单 KPRPE（0.9906 / 0.514）。
把 LVFace-B 拉进来只会稀释——它的同人侧 cosine 只有 0.494（其余三个 0.565–0.615），
是融合里最弱的一路。E3 不成立，不建议上线双模型。

### 3.5 生成图（tier-2）与双人绑定

49 帧有效生成图，2×2 余弦矩阵：

| 模型 | 对角占优 | 同人cos | 异人cos | 间隔 |
|---|---|---|---|---|
| glintr100 | 100% | 0.352 | −0.015 | 0.366 |
| w600k_r50 | 100% | 0.351 | −0.009 | 0.359 |
| LVFace-B | 100% | 0.292 | 0.002 | 0.290 |
| AdaFace-KPRPE | 100% | **0.399** | −0.059 | **0.458** |

- 四个模型都能 100% 把两个人分开，**但源图本身是"男方口罩 + 女方墨镜"的困难样本**，
  同人 cosine 只有 0.29–0.40，远低于真人同人的 0.49–0.62。这一档只有 2 个身份，
  AUC 无判别意义，只能看绝对保留度：KPRPE 保留最多，LVFace-B 最少。
- **按"取面积最大人脸"绑定 Person 0，会在 16.3%（8/49）的生成帧上绑错人**。这是独立于
  识别模型的错误源，也说明"先按人物 Mask／身体框绑定再识别"这条改动比换识别网络更值钱。
- 56 张生成图里 7.1% 未检出两张脸；这些帧既可能是生成重影/粘连，也可能是漏检，
  当前 reward 无法区分，统一按 `(cos+1)/2 = 0` 记零分。

### 3.6 检测

| 数据 | 图数 | det 640 失败率 |
|---|---|---|
| TikTok | 2 714 | 1.77% |
| iPER | 693 | 7.36% |
| beach14 生成 | 56 | 7.14%（未检出两人） |

det_size 640 → 1024 不提升检出（iPER 抽样 94.8% → 93.2%，TikTok 96.4% → 95.2%），
中位脸宽也不变（121px / 66px）。**检测分辨率不是瓶颈**，不需要换 RetinaFace 或加分辨率。

### 3.7 翻转 TTA

对四个模型统一做"原图+镜像平均"：AUC 变化 +0.0002 ~ +0.0006，排名不变。
LVFace 官方 IJB-C 用了镜像特征，但补上这项也不能让它反超——结论不是协议差异造成的。

## 4. 结论与建议

1. **主 Identity Reward 换 AdaFace ViT-B KP-RPE**（`models/face/cvlface_adaface_vit_base_kprpe_webface4m`）：
   AUC 0.9906、TAR@1e-3 0.514、微脸/侧脸/模糊三档全部第一，单次前向 1.5 ms/face。
2. **不做双模型融合**：收益为负，且要多维护一个 455 MB ONNX。
3. **LVFace-B 不作为候选**：AUC 与现有 glintr100 持平（+0.0003，CI 跨 0），权重为"非商用研究"许可。
4. 若出于工程稳妥暂时留在 ArcFace 家族，**`w600k_r50` 是比 `glintr100` 更好的低成本替换**
   （AUC +0.0054，0.5 ms/face，无新依赖），但低 FAR 区间仍不如 KPRPE。
5. **比换模型更该先改的两处代码**：
   - `src/evaluation/joint_pilot_metrics.py:IdentityMetric.embedding` 的"取最大人脸"规则 →
     在双人/复杂背景场景改成按人物 Mask 或身体框绑定（本实验测得最大脸规则 16.3% 绑错）；
   - `src/pose_control/qwen21_full/native_rl/rewards.py:score_candidate` 里缺脸按
     `(cos+1)/2=0` 计入均值 → 应改为按目标姿态的**人脸可见性**预先决定该项是否适用，
     只惩罚"该露脸却没检出"的候选。

### 接入注意

- KP-RPE 需要 `rpe_index_cpp` 扩展（`models/vit_kprpe/RPE/rpe_ops`，CUDA 12.8 + torch 2.8
  编译约 40 s），本实验产物在 `build/rpe/`。接入 qwen21 前需确认扩展随代码一起分发，
  且模型加载要求以 `models/face/cvlface_adaface_vit_base_kprpe_webface4m` 为工作目录。
- 权重许可：CVLface 要求遵循训练集（WebFace4M）许可，LVFace 权重为"非商用研究"。
  商用前需法务确认，本实验不改变现有生产后端的许可状态。
- `evaluator_assets()`（`rewards.py`）目前只登记 `scrfd_10g_bnkps.onnx` + `glintr100.onnx`；
  换后端时必须把新权重与 `rpe_index_cpp` 的 SHA 一并纳入冻结清单，否则 RL 打分不可复现。

## 5. 与其他实验的关系

- 上游：`models/face/README.md`（2026-11-22 注册的 LVFace-B / AdaFace-KPRPE 候选）。
- 下游：Native V7 RL 的 Identity Reward 与双人 Binding Reward 后端选型；结论回填到
  `src/pose_control/qwen21_full/native_rl/rewards.py`。
- 邻居：`姿态估计/four-skeleton-compair`（SDPose 姿态评分口径）、`workflow-v2/experiments/2.beach14`
  （本实验 tier-2 生成图来源，其 `score_rl_rewards.py` 用的仍是 glintr100）。
- 与 `2.beach14` 的 ID_sim 口径差异：那边 `ID_sim` 均值 0.68–0.72 是**全身图 + 目标姿态检测**
  得到的 source-target 相似度，本实验 tier-2 只取人脸裁剪，数值不可直接比较。
