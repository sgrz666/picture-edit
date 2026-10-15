# DeepGen V6.3 独立测试集 Benchmark 评测报告

## 一、 评测环境与数据设置

- **评测模型**：DeepGen V6.3 Native Adapter（全量 20,000 步训练最优检查点 `Step 12,000`）
- **评测协议**：严格在**独立测试集**（`baseline_50_eval_pairs.jsonl`，涵盖全新未见人物，如 `019_2`、`001_11`、`001_19`、`024_7` 等）进行只读前向推理（`eval` 模式，无梯度计算，不更新任何模型参数）。
- **去噪步数**：30 步（Flow Matching）
- **分类引导尺度 (Guidance Scale)**：4.5
- **几何控制强度 (Geometry Strength)**：1.0 (Adapter ON) vs 0.0 (Adapter OFF)
- **基准对照**：与前代 130M 参数量基线适配器（Baseline）在完全相同 50 对测试样本上进行 1 对 1 定量比对。

---

## 二、 核心 Benchmark 指标对比表（可直接写入报告）

| 评测方法 / 模型 | PSNR (dB) ↑ | SSIM (skimage) ↑ | SSIM (标准) ↑ | MAE ↓ | MSE ↓ | 相对基准提升与评价 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Adapter OFF（无控制底模）** | 17.35 | 0.8033 | 0.8733 | 16.06 | 1468.74 | 缺乏姿态控制，肢体位置频繁失控 |
| **Baseline 130M 前代适配器** | 22.30 | 0.8548 | 0.9583 | 8.10 | 444.47 | 初步建立姿态引导，复杂动态姿态存在偏差 |
| **DeepGen V6.3 Native (Step 12,000)** | **24.28** | **0.8783** | **0.9759** | **6.91** | **271.37** | **全面超越：PSNR +1.98 dB，MAE 误差降低 14.7%** |

### 核心结论：
1. **控制有效性验证 (vs Adapter OFF)**：
   - 开启 V6.3 适配器后，PSNR 提升了 **+6.93 dB**，SSIM (skimage) 提升了 **+0.0750**，MAE 相对误差大幅降低了 **57.0%**，方差 MSE 降低了 **81.5%**！
2. **算法优越性验证 (vs Baseline 130M)**：
   - 相比前代 130M 适配器，V6.3 在全新未见人物的测试集上，PSNR 实现了 **+1.98 dB** 的显著提升，MAE 误差从 8.10 降到 **6.91**（降幅 **14.7%**）。这充分证明了 Scheme A 单人高精度控制方案、DWPose+法线+深度+部位图 4 模态融合架构的优异泛化性。

---

## 三、 本地目录文件说明

当前目录 `D:\codeplus\pictureedit\可视化\testdata` 下包含：

1. `eval_metrics.json`：50 对测试样本的全量原始指标与汇总统计（含 Adapter ON 与 Adapter OFF 每样本 PSNR、SSIM、MAE 及平均指标）。
2. `images/`：**已升级为双排专业对比图**（每张图 1704×936 高清宽屏）：
   - `eval_{idx}_{app}_{src}_{tgt}.png`：50 张双排对比板：
     * **第 1 排（输入控制信号）**：`[DWPose (姿态骨架) | Normal (几何法线) | Depth (空间深度) | Part Map (身体部位解析)]`
     * **第 2 排（视觉效果与消融比对）**：`[原图 (Source) | 目标真值 (Ground Truth) | 无控制基线 (Adapter OFF) | 所提方法 (Adapter ON V6.3)]`
     * **定量指标直观标定**：无控制基线与所提方法底部均精确标注该样本的 `PSNR` 与 `SSIM`，并计算指标差值收益。
   - `pred_{idx}_{app}_{tgt}.png`：50 张 V6.3 模型的单独生成结果预测图；
   - `src_{idx}_{app}_{src}.png`：50 张输入源参考图；
   - `gt_{idx}_{app}_{tgt}.png`：50 张目标真值 Ground Truth 图。
3. `conditions/`：200 张独立条件模态图（50 样本 × 4 模态：`dwpose`、`normal`、`depth`、`part`）。
4. `images_1row_backup/`：原单排 4 联对比图的备份目录，安全保留原始版本。
5. 合成脚本：`scripts/build_two_row_testdata_strips.py`，可一键复现或根据论文版面调整格子尺寸。

