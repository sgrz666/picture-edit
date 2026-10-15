# qwen-2.1-test/ —— Qwen-Image-2.1 最早的双人/交互编辑探索原型

实验目的
- 在 Qwen-Image-2.1 基座上做项目最早的双人交互编辑探索：构建候选人物库、交互测试集与 dual adapter 冒烟，验证「姿态重定向 + 物理接触」的可行性。

里面有什么
- `data/`：数据集与资产。
  - `candidates/`：heavy / slim 候选人物图与 `contact_sheets/`（heavy/slim 对照索引），`_rejected/` 为被剔除样本。
  - `testset/`：A_heavy / B_slim 候选图；`pairs/` 下 `hard_01..05`、`medium_01/02` 共 7 组源-目标对（每组含 A_source.png / B_target.png 与 README）。
  - `interaction/testset/`：12 组交互对 `inter_01..12`（太极/拥抱/背抱/舞蹈/姿态），每组带 source/target 与 A/B 掩码。
  - `experiment/code`、`experiment/results` 见下。
  - 注：`data/testset/README.md` 为现存文件，本目录不另写子 README。
- `experiment/`：实验代码与产出。
  - `code/`：hi4d_conditions.py、overfit_inter01.py、qwen21_body_adapter_dual.py、test_conditions.py 等。
  - `results/`：gen_v1（hard/medium 生成图与 results.json）、interaction_v1（12 组交互生成图）、overfit_inter01（adapter_overfit.pt、metrics.json、train_log.json、images 0~500 步 off/on 对照）、overfit_smoke、smoke。

关键结论
- `overfit_inter01/metrics.json` 验证：交互样本 500 步过拟合，ON PSNR 由 12.354 dB（step0）攀升至 29.846 dB（step500），SSIM 由 0.4009 升至 0.9245（即 +17.50 dB）。gen_v1 + interaction_v1 共产出约 309 张评测生成图。这是 Qwen 迁移线最早的双人/交互能力原型。

与其他实验的关系
- 属于 Qwen-Image-2.1 迁移线的早期原型（[../README.md](../README.md) 第 6.3 节），为 `qwen21_hi4d_dual`、`qwen21_iper_body` 提供数据与原型基础。见 [../../README.md](../../README.md)。
