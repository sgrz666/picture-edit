# SMPL-X 跨体型基准（HBW P0+P1）

2026-11-09。验证“A 的姿态 + B 的体型能否恢复 B 的真实身材”，以及单图 β 估计的精度与稳定性，为 Native V5 的 body-shape 条件监督做决策。

## 里面有什么

- `scripts/`：数据准备、GT β 拟合、姿态选择、SMPLest-X 审计、SHAPY 推理/收集、P1/P0 评测、图与报告。
- `report/REPORT.md`：完整中文报告（数字、配对 CI、对照、局限、预注册判定）。
- `report/figures/`：方法对比、配对图、稳定性、P0 分组、G0 正对照、围度、3 张 P0 对照图。
- `report/figures/gallery/`：6 张直观对比图（10 人体型三列 lineup、真实照片 + A→B swap、误差热图、020 的 6 姿态网格、018→020 三视角）。
- `report/figures/ab_sequence/`：3 张五格序列（A 图、B 图、A mesh、B mesh、A+Bβ）；B 的 β 当 GT、只估 A，第 5 格与第 4 格恒等，真正的误差是 A 估 β vs A GT / vs B。
- `results/`：`p1_summary.json`、`p1_per_subject.csv`、`p0_summary.json`、`p0_per_subject.csv`、`p0_measurements.csv`、SHAPY `coverage.json`。
- 重产物在 `/data/shangguanrz/shape/`：705 张预测、720 个 mask、8100 行 P0 指标、36 张 P0 面板。

## 关键结论

- **单图**：SHAPY P2P-20K 23.78 mm vs SMPLest-X 26.54 mm；受试者级配对差 **−2.76 mm，95% CI [−7.18, +1.04]，7/10，p=0.344** → 不显著，不替换单图 SMPLest-X。
- **稳定性**：SHAPY ICC(1,1) **0.539** vs SMPLest-X **0.246**；SHAPY 多图均值把 P2P 降到 **21.34 mm**（−2.43 mm，CI [−3.57, −1.36]，10/10），SMPLest-X 多图均值 25.30 mm。
- **P0**：v2v 23.62（G0，不换）→ 19.16（G1）→ **17.21 mm（G2）**；IoU 0.828 → 0.852 → 0.870；G0 与真实体型差 Spearman 0.606（正对照通过）。
- **10 维 β 的表达上限**：GT 拟合表面 v2v 残差 3.2–6.4 mm，但围度下限 waist 31.95 mm、hips 23.85 mm；300 维表面 1.23–1.83 mm。围度级监督不能只靠 10 维 β。
- **β 距离会误导**：G2 的 β L2 更大（3.87 vs 3.07），P2P 却更小。

## 与其他实验的关系

上游是 HBW（真实体型注册网格）与 SHAPY/SMPLest-X 的既有预测；下游是 Native V5 的 body-shape 条件监督。本轮**没有改** native_v5 / workflow 的生产代码。MTP / FAUST / 3DBodyTex / 4D-DRESS 需要账号/许可，P2/P3 未做。

## 复现

```bash
RUN=/data/shangguanrz/shape
cd $RUN/code/scripts
python prepare_hbw.py && python fit_gt_betas.py && python select_poses.py && python run_p1.py
# SHAPY demo 见 report/REPORT.md 第 7 节（需要 build_shapy_layout.py 与本地兼容补丁）
python collect_shapy.py && python run_p1.py
python run_p0.py && python report.py
```

环境：`/data/shangguanrz/shape/env/shapy`（deepgen 的 system-site-packages + 仅补 yacs/fvcore/nflows/kornia/pytorch-lightning）；SHAPY 的 torch2.8/py3.12/numpy 兼容补丁只改实验副本，不碰 deepgen/qwen21。

## 已知边界

10 人、705/781（90.27%）、GT 是注册网格不是原始扫描、neutral 模板、SHAPY 用 GT 关键点/bbox/性别（对其有利）、P1 无真实图像 mask、033 腰围单点异常、n=10 功效有限。
