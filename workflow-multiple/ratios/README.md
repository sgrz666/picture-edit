# ratios：定标常数测量（躯干 ÷ 人体框高）

双人线「定标常数：躯干 ÷ 人体框高」测量目录，对应 ../README.md 同节。量的是共享包 `TORSO_TO_PERSON_BOX_HEIGHT = 0.31` 在双人/单人数据上是否仍是常数。

## 里面有什么

- `torso_ratio.json` —— 唯一文件。结构：
  - `summary`：`measured`/`samples_raw`/`unique_inputs`/`median`/`mean`/`p25`/`p75`/`min`/`max`/`by_frame`/`constant`/`ratio_to_constant`。
  - `pairs`：17 对双人测量的 list。
  - `single_runs`：16 条单人测量的 list。

## 与上下层的关系

- 测量脚本：`../torso_ratio.py`（A 用 pair.json 的 mask 框 + 本机 CPU DWPose 关键点，B 用 pair 自带关键点 + `geometry.npz` 语义掩码框；按 图片哈希+框 去重，原始次数与独立人体数都写出）。
- 对比对象：共享常量在 [`../../src/pose_edit/geometry.py`](../../src/pose_edit/README.md)（0.31 现仅作兜底，见其 README「注意与坑」）。
- 出图尺寸复核：../logs_batch/size_compliance.json 与 ../tests/test_size_compliance.py。

## 注意与坑

- 关键读数：双人 68 次 → 38 独立人体，中位 **0.285**（IQR 0.273–0.300，0.31 落在 p75 之外）；单人 16 次 → 8 独立图，中位 0.353、全距 **0.259–0.524**（比值本身不是常数）。
- 结论 #21：双人线 `source_fit` 不动逐人缩放、用不到 0.31；该常数归属单人线整改项，且应改成"实测 A 躯干"而非重标常数。本轮批量未改。
- `ratio_to_constant` 列即 `0.31/比值`，是出图被画成几倍的尺度误差；单人多落到 0.59–1.20×，说明框高这个代理量本身不含躯干信息。
- `summary.by_frame` 是逐帧明细；`pairs`/`single_runs` 是去重前的原始测量，去重前次数在 `summary.samples_raw`。
