# experiments/qwen21_iper_body/native_smoke/ —— 关控制（原生基座）推理冒烟

> 路径：`…/qwen21_iper_body/native_smoke`　|　`images/` 等约 4 文件 / 610 KB　|　2026-10-12

## 一、在干什么
`controlled_smoke/` 的对照组：小样本走"不开适配器、无几何注入"的原生 Qwen21 基座直出路径，确认 off 臂推理与图像保存可用。**非实验结论。**

## 二、内容实证
`images/` 数张原生直出图（小样），对应四臂 `qwen21_off` 路径预演。

## 三、关系
与 `controlled_smoke/` 成对；正式 on/off 定量对照见 `four_arm_eval_50_s40/`。
