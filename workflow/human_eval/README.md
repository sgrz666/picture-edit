# human_eval —— 主观成对盲评

合同 `task/人物图像姿态编辑任务书_v3.docx` 的三条验收（交互姿态主观合理比例 ≥70%/90%、
编辑主观质量与手部细节优胜率比 Qwen-Edit 基线 +5%、身份可辨识/动作准确/交互自然/手部无结构错误）
都是**主观读数**，仓库里原先没有任何对应基础设施。这个目录补上出题、打分、算数三段。
纯 CPU、不读缓存指纹、不占 GPU，随时能跑。

## 里面有什么

- `protocol.md` —— 打分轴、档位、强制成对、样本量反推、与合同条目的对应关系。**先读这个**。
- `prepare.py` —— 从 run 目录出题：只取两臂都存在的配对，随机决定左右，把真名藏进 `truth.json`。
  图片默认 **symlink** 进 `bundle/img/`（根分区常年 95%，不要复制几十张 PNG），Windows 或不支持软链时才复制。
  `--selfcheck` 不复制任何东西，只验证配平与文件是否还在。
- `annotate.html` —— 单文件离线标注页。`bundle/questions.js` 由 `prepare.py` 生成，
  浏览器直接双击打开即可（用 `questions.js` 而不是 `fetch`，因为 `file://` 下 fetch 被同源策略挡掉）。
  每题对**左右两张各打四轴**，再强制选一张更好；答案存 localStorage，导出 jsonl。
- `tally.py` —— jsonl → win_rate（按参照臂取号，正数=候选更好）+ **按题** bootstrap CI95、
  主观合理比例、逐轴不合格率、Fleiss κ（成对选择）、两两 Pearson（轴分）。
  哈希不符的判定会被剔除并计数，不会静默混进均值。

## 怎么用

```bash
cd /home/shangguanrz/project/pic-edit/workflow/human_eval
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
# 臂名要写完整后缀（含定标档），否则同一臂有多个 run 时无法成对：
$PY prepare.py --pair-dir ../../workflow-multiple/runs \
    --a-arm skeleton_only.b_canvas --b-arm with_b.b_canvas --out bundle
$PY prepare.py --selfcheck bundle --a-arm skeleton_only.b_canvas --b-arm with_b.b_canvas
# 把 bundle/ 整个发给标注者（别带上 truth.json），标注完收回 jsonl 放进 responses/
$PY tally.py responses/ --truth bundle/truth.json --index bundle/index.json --markdown
```

单人线同样可用：`--pair-dir ../runs --a-arm _torso --b-arm _manual` 这类任意两个变体后缀都能成对。
`prepare.py` 只出**两臂都有图**的题，其余列在 `UNPAIRED`/`AMBIGUOUS` 里，不会静默少题。

## 注意与坑

- **参照臂必须是"更该被超越的那个"**。`--a-arm` 是参照（基线），`--b-arm` 是候选；
  win_rate 的符号按候选计。合同要的是"比 Qwen-Edit 基线高 5%"，而**基线臂（不给任何姿态条件）
  现在还不存在**——实现它要改 `run.py`，`run.py` 在缓存指纹的 `INPUTS_CODE` 里，
  会作废全部既有 run 的输入身份，所以排在批量收尾窗口与 `--condition` 消融同期做（见
  `docs/workflow_大方向优化调研_20261014.md` P2/P4）。在那之前这套只能用于两臂/两定标的相对胜负。
- 随机左右由 `--seed`（默认 20261014）决定并写进 `truth.json`；换 seed 重跑就是独立的一轮，
  不要把两轮混在一起算 κ。
- `truth.json` 不能出现在标注者拿到的目录里。
- 每题不足 2 名标注者时 κ 与两两 Pearson 输出 n/a，这是有意的：单人不构成一致性证据。
- bootstrap 重采样的是**题**不是判定，所以 3 人评同一题不会把样本量当成 3 倍。
