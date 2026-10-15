# pictures/ —— 流水线默认 A/B 输入图

`workflow` 的"默认配对"输入区。`run.py` 在未显式指定 `--a`/`--b` 或 `--pair` 时，会从 `pictures/A` 与 `pictures/B` 各取唯一一张图作为源（A，身份/服饰/背景保持不变）与目标（B，仅贡献姿态）。它也是 `demo` 运行的实际输入来源。

## 里面有什么

- `A/`：源人物图目录，含 `A1.jpg`（约 180 KB）。A 的人物身份、服饰、背景在生成图中须被保持。
- `B/`：目标姿态图目录，含 `B1.jpg`（约 107 KB）。B 只提供身体动作、头部朝向、可观察的手部配置与表情，不提供身份与外观。

## 与上下层的关系

- 上游：人工放置样图。
- 下游：`../run.py` 的 `discover_image` 读取（强制"每目录恰好一张图"）；`demo` 运行把 A1/B1 复制为 `f2ce…`/`d889…` 进 `runs/demo/inputs/`。
- 互补：批量实验（hard_xx、testset_*）改用 `--pair` 指向 `experiments/qwen-2.1-test/data/testset/pairs/...`，**不使用本目录**。
- 相关：`../runs/README.md`、`../../inputs/README.md`（顶层演示样图）。

## 注意与坑

- `discover_image` 强制"A/B 每目录恰好一张"，多图会直接报错；换默认图直接替换 `A1.jpg` / `B1.jpg` 即可。
- 本目录与顶层 `inputs/`（person1.jpg / dual_person.jpg）是两回事，后者仅作演示，不进入流水线。
