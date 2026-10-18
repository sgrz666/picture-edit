# AGENTS.md — workflow/ 单人动作迁移流水线协作须知

通用仓库约定（先读后写、图片/视频永不删移、scp 投递纪律、大文件不进 Git 等）见
仓库根 [`../AGENTS.md`](../AGENTS.md)。本文只写**本目录特有的纪律**。

---

## 1. 每个 run 完成后必须出「标准六图面板」（硬性收尾）

任何 run——新样本、复跑、消融、对照——在生成阶段结束后，都要用
`scripts/six-picture-create/` 出一张**标准六图对比**，作为该样本的人工审图入口。
**没有六图面板的 run 视为未完成交付**，不要只丢一句 "done" 或只报指标数字。

```bash
cd workflow/scripts/six-picture-create

# 单个 run（-o 输出到批次子目录，保持同类 run 聚在一起）
python3 script/six_picture_create.py ../../runs/<run名> -o output/<批次>/<run名>_six_panel.png

# 整个批次目录（--all 固定输出到 output/<run名>_six_panel.png；注意 --all 与 -o 不能同时用）
python3 script/six_picture_create.py --all ../../runs/<批次目录>

# 只校验六件产物是否齐全，不出图
python3 script/six_picture_create.py ../../runs/<run名> --check
```

六格内容（按流水线顺序，2×3 版式）：

| # | 文件（相对 run 目录） | 角色 |
|---|---|---|
| 1 | `inputs/A_reference.png` | 源照片 · 身份/服装/场景 |
| 2 | `inputs/B_rgb.png` | 目标照片 · 仅提供动作 |
| 3 | `geometry/B_poseguide_overlay.png` | 估计器在 B 上的拟合校验 |
| 4 | `geometry/B_3d_dwpose.png` | 对齐后引导 · **唯一进入生成器** |
| 5 | `geometry/A_poseguide_alignment.png` | 对齐引导叠到 A · 诊断用 |
| 6 | `generation/final_image.png` | 最终生成结果 |

细则：

- **消融 run**（只重跑 geometry/generation，`inputs` 在 `fixed` 列表里）第 2、3 格会缺：
  用 `--fallback ../../runs/<基线run>` 从基线补齐，图内会注明「取自基线 run」；
  消融倍率会显示为「基线 scale = X × 消融倍率 Y」。
- 图题自动读 `geometry/stage.json` 的 `metadata.camera_checks`（scale / 画布尺寸 / 有效 B 关节数）；
  页脚按 scale 分级提示对齐风险：**<0.5** 骨架偏小、生成器易把骨架渲染成"小人"、迁移易失败；
  **0.5–0.65** 偏小；**≥0.65** 正常。读到异常分级时先怀疑几何对齐，再怀疑提示词。
- 输出惯例：`scripts/six-picture-create/output/<批次>/<run名>_six_panel.png`；批次目录可在
  `runs/<批次>/` 下建 `panels` 软链指向它（示例：`runs/tiktok/panels`）。
- 阶段没跑完的 run 也能出图（末格显示「缺失」），便于中途检查。

## 2. 跑批与共享 GPU

- 本流水线（`run.py`/`batch.sh`）与 `workflow-multiple`（双人线）**不共享锁**，两条线各自
  串行调度；并发前先实测显存：单人生成峰值约 **34 GiB**，用
  `python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f/1024**3)"` 核对余量。
- 新样本批量：参考 `batch.sh`（auto → `--subject largest` 回退、逐对 `evaluate.py`）；
  写自己的批量循环时保持**串行**与**幂等**（已完成的 run 跳过）。
- 重跑或改设置一律用新 `--name`（或 `--force`），**不覆盖既有产物**。

## 3. 产物、缓存与评测

- run 落 `runs/<name>/`；`stage.json` 记录输入/产物 SHA256，绝对路径不进指纹——**run 目录
  可搬迁**（搬入批量子目录后按 stage.json 做 sha 对账即可；示例：`runs/tiktok/` 8 个 run
  搬迁后 204/204 通过）。
- 手工提示词一律走 `--prompt-file`（显式输入，不被 PE 覆盖）；对照实验的检测标准：副本产物
  与对照 SHA256 完全相同 = 实验变量没生效。
- 评测闭环：`evaluate.py`（几何解释器，身体是否落到请求骨架）+ `evaluate2.py`（qwen21 解释器，
  DWPose 第二读者 / 身份 / 场景）+ `scorecard.py`（跨 run 汇总）。六图面板与这两份评测一起，
  构成一个样本的完整交付。
