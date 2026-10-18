# scripts：双人线私有脚本

双人线私有脚本目录。它与 `../`（流水线主脚本）、`../../workflow/scripts/`（单人线脚本）、`../../scripts/`（仓级脚本）是**三套互不相同的集合**——双人线可执行流水线（build_testset / run_dual / evaluate_dual / compare_dual / fit_dual_report / matrix_audit / torso_ratio 等）全在 `workflow-multiple/` 顶层，本 `scripts/` 只放一个辅助可视化工具及其产物。

## 里面有什么

- `panels/`：结果拼图工具与产物：
  - `code/panel_dual.py` —— 把每个 run 的八张图（①A 源帧 ②B 目标 ③B+骨架叠加 ④A+骨架叠加 ⑤⑥单人骨架 ⑦合并骨架 ⑧生成图）拼成带中文标注的面板；按 `generation.json` 的 `reference_images` 角色给格子打"生成器输入"标（skeleton_only 下 ② 打"仅 PE 可见"）。用法 `--run <name>` / `--all` / `--run a,b`。
  - `*.png` —— 已生成面板图（按 run 命名）。
  - `index.json` —— 面板清单：每 run 的 panel 路径、`missing`（缺哪些输入图）、`eval` 布尔。
  - `eight-pictures/`、`output/` —— 脚本创建的子目录，本次运行里为空（疑似中间/输出暂存，待补充）。

## 与上下层的关系

- 输入来自 `../runs/<name>/`（八格图来源）；产物供人工核对。
- 流水线主脚本在 `../`；单人线对照 `../../workflow/scripts/`，仓级对照 `../../scripts/`；权威 ../README.md 未单列本目录。

## 注意与坑

- 顶层**不存在** `workflow-multiple/panels/`；面板实际落在 `scripts/panels/`。`index.json` 里 panel 路径写的是失效旧值 `workflow-multiple/panels/...`，与当前落盘不符——以磁盘 `scripts/panels/*.png` 为准。
- `scripts/` 不是"双人线脚本总目录"，别在此找 run_dual.py 等；它们在 `../`。
- 八格版式（4+4）复刻人工面板，⑦合并骨架不进生成器、仅核对用；skeleton_only 给生成器 ①③⑤⑥，with_b 给 ①②③⑤⑥，PE 都看全四张。
