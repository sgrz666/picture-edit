# experiments/qwen21_iper_body/four_arm_eval_50/ —— 四臂交叉评测（初版，50 样本 / 40 步）

> 路径：`…/qwen21_iper_body/four_arm_eval_50`　|　3 文件 + `strips/`（50 拼图）/ 53 文件 / 42.7 MB　|　数据源：`summary.json`，2026-10-12

## 一、在干什么
与 `four_arm_eval_50_s40/` 同为四臂（deepgen_off/on、qwen21_off/on，seed 42、steps 40、50 held-out 对）交叉评测，是本评测协议的**首个版本**；`_s30`/`_s40` 为其步数变体。四臂均值与 s40 高度一致（qwen21_on PSNR 25.86、PCK@0.02 98.00%、Face ID 0.3510；deepgen_on PSNR 24.54），差异主要在计时口径（本目录 `seconds` 含 KV-cache 命中，deepgen_off 3.87s 低于 s40 的 5.08s）。

## 二、内容实证
- `summary.json.arms`：deepgen_off PSNR 17.35 / deepgen_on 24.54 / qwen21_off 17.53 / qwen21_on 25.86；PCK@0.02 分别 0.174 / 0.955 / 0.203 / 0.980。
- `qwen21_control_gain` Face ID −0.468、PSNR +8.33；`deepgen_control_gain` PSNR +7.19。与 s40 结论一致。
- `per_sample.json`（4,401 条）、`strips/`（50 拼图）。

## 三、关系与注意
`s40` 为对外引用的**最终权威版**；本目录保留作协议初版与计时对照。引用项目里程碑数字优先用 `four_arm_eval_50_s40`。
