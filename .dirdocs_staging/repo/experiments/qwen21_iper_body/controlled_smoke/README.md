# experiments/qwen21_iper_body/controlled_smoke/ —— 带几何控制的推理冒烟

> 路径：`…/qwen21_iper_body/controlled_smoke`　|　`images/` 等约 5 文件 / 996 KB　|　2026-10-12

## 一、在干什么
在评测流水线正式跑 50 样本前，用极少量样本验证"注入 SMPL-X 目标几何控制 + 适配器 ON"这条推理路径能出图、指标采集不报错。**非实验结论。**

## 二、内容实证
`images/` 数张受控生成图（小样）。对应四臂里的 `qwen21_on` 路径的小规模预演。

## 三、关系
与 `native_smoke/`（关闭控制、原生基座直出）配对，构成"开/关控制"的最小冒烟对照；正式对照见 `four_arm_eval_50_s40`。
