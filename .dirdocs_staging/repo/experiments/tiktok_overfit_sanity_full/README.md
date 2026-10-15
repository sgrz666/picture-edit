# experiments/tiktok_overfit_sanity_full/ —— V6.5 面部适配器两阶段过拟合门禁的**权威裁决实验**（结论 FAIL）

> 工作区路径：`/home/shangguanrz/project/pic-edit/experiments/tiktok_overfit_sanity_full`
> 本级：文件 8 个 / 子目录 4 个　|　子树规模：约 93 个文件、逻辑体积约 7.9 GB（checkpoints 6 个 .pt 占 7.9 GB）
> 数据来源：本文件所有数字取自 `summary.json`、`train.log`、`gen_eval_history.jsonl`、`train_metrics.jsonl`，读取时间 2026-10-12。目录未冷归档，实体在本机（无符号链接）。

## 一、这一层在干什么

这是整个"面部/细节适配器过拟合门禁族（V6.4–V6.5）"的**判决性实验**：用一套两阶段（Phase A/Phase B）、六条准则（C1~C6）的过拟合协议，裁决 `DeepGen-1.0 + V6.5 Face Adapter` 在单条 TikTok 样本上到底有没有"正向贡献"。脚本入口为仓库根 `train_tiktok_overfit_sanity.py`。裁决结果 **`VERDICT: FAIL`**，直接终结了"在 MMDiT 单模型中间层强行单层注入面部特征"的路线，是促使技术路线转向 `experiments/qwen21_iper_body/`（Qwen-Image-2.1 单流基座）与"后置两阶段面部精修"（见 `experiments/16-test/` 的 BFS 方案）的关键证据。上游依赖 `tiktok_v65_face_overfit_00001` 的同一条件缓存；下游结论被 `../README.md`（experiments 总览）第 4 族引用。

## 二、层级地图

| 子目录 | 装什么 | 内容实证 | 与本级关系 |
| :--- | :--- | :--- | :--- |
| `checkpoints/` | 两阶段训练权重 | `checkpoint_best.pt`(1.96 GB)、`checkpoint_step_000500/001000/001500.pt`（各 1.964749759 B）；共 6 文件、7.9 GB | 门禁运行落盘的适配器权重，可复推 Phase B 曲线 |
| `condition_cache/` | 源帧离线条件张量 | `00001_0014.pt`(3.62 MB)：`reference_latent`(1,16,64,64) bf16 + `sequence`(1,425,4096) bf16 + `pooled`(1,2048) | 与 `tiktok_v65_face_overfit_00001`、`tiktok_v65_test_2steps` 等同指纹（`ebc193e3…`/源 `de3b6215…`），共享底层依赖 |
| `gen_evals/` | Phase B 真实生成对比图 | 每 100 步三件套 `step_*_gen_off.png` / `_gen_on.png` / `_panel.png`，共 50 文件、25 MB | C4/C5/C6 判据的可视化来源 |
| `previews/` | Phase A 固定步 t=150 的 face on/off 重构图 | `step_000000..001500_face_on/off.png`，共 34 文件、8.6 MB | C1/C2 确定性记忆的可视化来源 |

## 三、关键文件与实证内容

### `summary.json`（门禁裁决全文，唯一权威）
- 运行配置：`architecture_version=v6.5`、`dataset=TikTok`、`sequence=00001`、`source_stem=0014`→`target_stem=0074`、`seed=42`、`eval_seed=1234`；**Phase A 前 200 步固定 `timestep=150`，Phase B 步 201~1500 用 `uniform[0,999]` 全轨迹随机时间步**（`timestep_sampling_phase_b`）；`full_adapter_training=True`，优化器管 778 个参数张量。
- `non_face_sha256_before≠after`（`de3b6215…`→`9719f3c9…`，`non_face_changed=true`）：证明面部注入**确实改动了非面部权重**，是背景/服装被扰动的权重级旁证。
- `grad_flow_report`：`face_preparer` 149 个参数**零梯度**（`with_nonzero_grad:0`），`condition_injector` 150/179、`control_interface` 146/266 有梯度——说明梯度大量漏进了全局条件/接口分支，而非只作用于面部。
- 两阶段 flow 损失（`phase_a_flow_first=0.65047 → tail20=0.41958`；`phase_b_flow_first=0.15210 → tail20=0.23826`）：**Phase B 的 flow 损失不降反升**（0.152→0.238），这是 C3 失败的根因。

### 六条准则实测值（`checks[]` 原样，与 `train.log` 尾部的 "OVERFIT SANITY TEST REPORT" 完全一致）

| 准则 | 判据 | 实测值 | 结果 |
| :--- | :--- | :--- | :--- |
| C1 接口不变量（初始 ON≡OFF，零初始化头） | `max\|Δpred\| ≤ 1e-5` | 0.0 | **PASS** |
| C2 Phase A 确定性记忆（flow 充分下降） | `tail(0.4196) ≤ 0.7×init(0.6505)=0.455` | 0.4196 | **PASS** |
| C3 Phase B 全轨迹 flow 持续下降 | `tail(0.2383) ≤ 0.6×init(0.1521)=0.0913` | 0.2383 | **FAIL** |
| C4 真实生成显著优于初始 | `ΔPSNR ≥ 5.0 dB` | +7.947 dB | **PASS** |
| C5 接近完美拟合（stretch，非门禁项） | `PSNR ≥ 20.0 dB` | 20.039 dB | **PASS** |
| C6 适配器正向贡献（final ON ≥ OFF） | `ON − OFF ≥ 0 dB` | **−0.044 dB** | **FAIL** |

> `verdict=FAIL`（门禁项取 C1,C2,C3,C4,C6；C5 只是拉伸目标）。`elapsed_seconds≈712`。

### `gen_eval_history.jsonl` / `train.log` 尾部（C6 的直接数字）
末步 `[gen-eval @1500] full PSNR off=20.08 on=20.04 | face PSNR off=23.08 on=22.56`。注意：**开启面部适配器后连"面部 ROI PSNR"都从 23.08 掉到 22.56**，全图也 −0.044 dB。结合 `diag_comparison_history`（step0 ON≡OFF，随后 `face_on_similarity` 升到 0.99 但生成端反而变差），可断定：单层面部注入在 Phase A 的固定步重构里能"记住"脸，但放到 28 步全轨迹去噪中会扰乱全局流，净效果为负。

## 四、注意事项 / 调用路径

- **这是本族唯一的"权威门禁裁决"**；同目录的 `tiktok_overfit_sanity_smoke/`（40 步）与 `tiktok_v65_test_2steps/`、`iper_v65_face_overfit_smoke/`（2 步）均为冒烟，其 FAIL 不代表科学结论，只有本目录的 1500 步结果算数。
- 复现入口：仓库根 `train_tiktok_overfit_sanity.py`，输出目录写死为本路径（见 `train.log` 末行 "artifacts saved in experiments/tiktok_overfit_sanity_full"）。
- `condition_cache/00001_0014.pt` 与族内多个实验字节级同指纹，**属共享底层依赖，勿删**；删除会使 `tiktok_v65_*`、`tiktok_objective_eval` 无法免编码复跑。
- 交叉引用：族总览见 `../README.md` 第 4 族；被取代关系——V6.4（`iper_v64_detail_overfit_024_6`，含手部）→ V6.5 纯面部（本目录裁决失败）→ 转向 `../qwen21_iper_body/`。
