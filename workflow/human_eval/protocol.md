# 主观评测协议（合同三条验收的执行口径）

合同 `task/人物图像姿态编辑任务书_v3.docx` 有三条验收挂在主观读数上，机器指标替代不了：

| 合同原话 | 本工具给出的数 |
| --- | --- |
| 典型双人交互姿态生成**主观合理比例** ≥70%（阶段一）→ ≥90%（阶段二） | `plausibility_rate`，逐动作族分列 |
| 交互姿态编辑**主观质量优胜率**比 Qwen-Edit 基线高于 5% | `win_rate` 与 bootstrap 95% CI 的下界 |
| 手部细节合理**主观优胜率**比 Qwen-Edit 基线高于 5% | 同上，只看 `hand` 轴 |
| 图像姿态编辑主观达到身份可辨识、动作基本准确、交互自然、手部无明显结构性错误 | 四轴各自的 5 档均分与"不合格率"（≤2 档占比） |

## 对照是谁：Qwen-Edit 基线 = 不给任何姿态条件

合同写的是"相比 Qwen-Edit 基线"。本仓库现在**每一次生成都把骨架图喂给了生成器**，
纯文本编辑的对照一次都没跑过，所以 +5% 目前没有可减的东西。基线臂定义为：
同一张 A、同一句提示词、同一 seed/steps/resolution，**只去掉姿态条件图**。

实现它要改 `run.py`（在缓存指纹的 `INPUTS_CODE` 里），会作废全部既有 run 的输入身份，
因此排在批量收尾的同一个窗口里做，与 `docs/workflow_大方向优化调研_20261014.md` P0/P2 同期。
在那之前，本工具仍可用于两臂/两定标的成对比较（双人线的 `skeleton_only` vs `with_b`、
`b_canvas` vs `source_fit` 就是现成题目）。

## 题目怎么出

```bash
cd /home/shangguanrz/project/pic-edit/workflow/human_eval
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$PY prepare.py --pair-dir ../../workflow-multiple/runs --a-arm skeleton_only.b_canvas \
    --b-arm with_b.b_canvas --image generation/final_image.png --out bundle
$PY prepare.py --selfcheck bundle          # 不复制图，只验证配平与隐藏标签
```

`prepare.py` 只做三件事：按共同配对键（对子名）取两臂都存在的 run、随机决定左右、
把真名藏进 `truth.json`（标注者看不到）。随机种子固定并写进 `bundle/index.json`，
所以同一批题目可以复现，换 seed 才是独立一轮。

## 打分规则

标注者在 `annotate.html` 里对每题回答两件事，**都必须作答才能提交**：

1. 绝对 4 轴 5 档，**左右两张各打一遍**：`identity`（还是不是同一个人）/ `action`（动作是否落到请求姿态）/
   `interaction`（接触是否自然、有无穿模）/ `hand`（手部结构是否可信）。1=明显错误，5=无可挑剔。
   必须两侧都打，`plausibility_rate` 才是"这张图合不合理"的比例，而不是两图之间的相对偏好。
2. 强制成对选择：`left` / `right` / `tie`，问的是"哪一张更符合任务要求"。
   不允许"差不多"作为逃避——tie 只在四轴都看不出差别时选。

键盘：`1`–`5` 给当前轴（自动跳到下一个未答轴），`←`/`→` 选左/右，`0` 记 tie，`Enter` 下一题，
`U` 撤销上一题。导出走浏览器下载，落到 `responses/<标注者>.jsonl`，一行一题。

## 统计

```bash
$PY tally.py responses/ --truth truth.json --markdown
```

- `win_rate = (W − L) / N`，W/L 按**参照臂**计（参照臂在左时在右取值），方向永远以参照为负。
- 95% CI 用**按题**有放回 bootstrap 5000 次、种子固定；CI 下界 > +0.05 才算满足"+5% 优胜率"。
- `plausibility_rate` = 请求侧四轴均 ≥3 的题占比；同时报逐轴 ≤2 的不合格率。
- 标注者一致性：轴分用 ICC(2,1) 的近似（同一题两两 Pearson 的均值），成对选择用 Fleiss κ。
  κ < 0.4 时 tally 会打印 `AGREEMENT_LOW`，该批结论不进报告。
- 报告必须同时列：题数、标注者数、tie 率、被剔除的题（图片哈希与 run 不符者）。

## 样本量不是按"有几张图"正推，要按要检出的差反推

检出 ±5% 的优胜率差、80% 功效，需要每臂数百判定量级。单人测试集只有 7 对，
双人线 17 对，都远不够。扩题的来源（都在盘上）：iPER `baseline_50_eval_pairs.jsonl` 的 50 对真帧、
TikTok `test_pairs.jsonl`、Hi4D 的 17 对。建议首轮 25–30 题 × 3 标注者。

## 与机器指标的分工

`tally.py` 的读数必须和 `../scorecard.py` 的 `pckDW*` / `torsoG` / `idcos` / `bgMAE` 同时出现。
两条硬性纪律：

1. **机器指标不达标的题进"必审队列"**，不能被平均数稀释：`joints_excluded_ratio` 高（人出画了）、
   `torsoG` 远离 1.0（人被改尺寸）、`persons_detected` 与请求不符（多画/少画人）的图单独送评。
2. **MLLM 预筛不充当验收**。本地 `Qwen-Image-2.1/text_encoder` 是带 `vision_config` 的 Qwen3VL，
   可以离线复刻 VIEScore 的 SE/OQ/D 来缩小送人工的子集，但 self-preference 与手部/计数盲区
   决定了它只能做预筛（详见 `docs/workflow_无GT定量评测调研_20261013.md` §轴5）。
