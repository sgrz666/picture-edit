# 指标报告：matrix_20261015T134829

- 运行目录：`/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/runs/matrix_20261015T134829`
- 参数：40 步、seed [42, 43]、CFG 1.0、bfloat16、KV cache False、单卡串行 + 官方 group offload
- 应产出 48 张，实际评估 48 张，complete=True；缺 0 个样本×配置组合
- 生成失败 0 次；DWPose 完全未检到人 0 张（保留在结果里，不当成生成成功）

## 三关状态

| 关口 | 是否通过 | 依据 | 产物目录 |
|---|---|---|---|
| 1 官方复现 | True | 权重 180 键全部加载，零初始化残留 [] 个 | /data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/runs/official_formal_20261015T133449 |
| 2 接入契约 | True | 无参考图与原版逐位相同；scale 0 与不加控制逐位相同；控制张量长度 = 参考+目标、通道 129；换参考图顺序输出会变 | /data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/runs/contract_gate_adapter_20261015T133049 |
| 3 效果比较（48/48） | True | 改善 3 / 退化 0 / 身份退化 1，共 6 个样本 | /data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/runs/matrix_20261015T134829 |

## 逐样本逐配置指标

严格口径（strict）把漏检或未对齐的人记 0 分；`persons_detected` 与 `人数对齐` 用来说明低分到底是姿态错还是检测器没分开人。身份分是 DINOv2 余弦相似度，双人另有互换计数。

| 样本 | 类型 | 配置 | n | PCK@0.2 strict | PCK@0.5 strict | MPJPE(归一) | 脸 MPJPE | 手 MPJPE | 检测覆盖 | 检出人数 | 人数对齐 | 身份分 | 疑似互换 | 耗时(s) | 峰值(GiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hard_02 | single | native | 2 | 0.750 | 0.786 | 0.066 | 0.031 | 0.170 | 0.935 | 1.00 | — | 0.916 | — | 157.9 | 3.75 |
| hard_02 | single | union_s0 | 2 | 0.750 | 0.786 | 0.066 | 0.031 | 0.170 | 0.935 | 1.00 | — | 0.916 | — | 241.0 | 3.76 |
| hard_02 | single | union_s05 | 2 | 0.857 | 0.929 | 0.071 | 0.026 | 0.247 | 0.980 | 1.00 | — | 0.908 | — | 236.1 | 3.76 |
| hard_02 | single | union_s10 | 2 | 0.964 | 0.964 | 0.054 | 0.012 | 0.028 | 0.992 | 1.00 | — | 0.585 | — | 236.1 | 3.76 |
| hard_05_torso | single | native | 2 | 0.389 | 0.667 | 0.378 | 0.085 | 0.320 | 1.000 | 1.00 | — | 0.920 | — | 156.8 | 3.75 |
| hard_05_torso | single | union_s0 | 2 | 0.389 | 0.667 | 0.378 | 0.085 | 0.320 | 1.000 | 1.00 | — | 0.920 | — | 236.1 | 3.76 |
| hard_05_torso | single | union_s05 | 2 | 0.389 | 0.806 | 0.326 | 0.038 | 0.319 | 0.985 | 1.00 | — | 0.894 | — | 236.2 | 3.76 |
| hard_05_torso | single | union_s10 | 2 | 0.389 | 0.944 | 0.286 | 0.028 | 0.313 | 0.989 | 1.00 | — | 0.577 | — | 236.4 | 3.76 |
| medium_02 | single | native | 2 | 0.844 | 0.875 | 0.101 | 0.071 | 0.096 | 0.818 | 1.00 | — | 0.907 | — | 158.6 | 3.75 |
| medium_02 | single | union_s0 | 2 | 0.844 | 0.875 | 0.101 | 0.071 | 0.096 | 0.818 | 1.00 | — | 0.907 | — | 239.6 | 3.76 |
| medium_02 | single | union_s05 | 2 | 0.906 | 0.906 | 0.091 | 0.050 | 0.130 | 0.746 | 1.00 | — | 0.925 | — | 239.7 | 3.76 |
| medium_02 | single | union_s10 | 2 | 0.938 | 0.969 | 0.070 | 0.021 | 0.034 | 0.909 | 1.00 | — | 0.865 | — | 239.3 | 3.76 |
| hug_pair00_084 | dual | native | 2 | 0.068 | 0.131 | 0.788 | — | — | — | 1.50 | 0.500 | 0.700 | — | 195.9 | 3.89 |
| hug_pair00_084 | dual | union_s0 | 2 | 0.068 | 0.131 | 0.788 | — | — | — | 1.50 | 0.500 | 0.700 | — | 300.1 | 4.70 |
| hug_pair00_084 | dual | union_s05 | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.711 | — | 299.1 | 4.70 |
| hug_pair00_084 | dual | union_s10 | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.685 | — | 299.0 | 4.70 |
| dance_pair02_120 | dual | native | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.630 | — | 196.6 | 3.90 |
| dance_pair02_120 | dual | union_s0 | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.630 | — | 299.2 | 4.71 |
| dance_pair02_120 | dual | union_s05 | 2 | 0.269 | 0.327 | 0.307 | — | — | — | 1.50 | 0.500 | 0.721 | 0/1 | 298.5 | 4.71 |
| dance_pair02_120 | dual | union_s10 | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.715 | — | 299.3 | 4.71 |
| cross_backhug_pair02_037 | dual | native | 2 | 0.456 | 0.548 | 0.405 | — | — | — | 3.50 | 0.000 | 0.839 | — | 195.2 | 3.82 |
| cross_backhug_pair02_037 | dual | union_s0 | 2 | 0.456 | 0.548 | 0.405 | — | — | — | 3.50 | 0.000 | 0.839 | — | 296.1 | 4.68 |
| cross_backhug_pair02_037 | dual | union_s05 | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.664 | — | 296.2 | 4.68 |
| cross_backhug_pair02_037 | dual | union_s10 | 2 | 0.000 | 0.000 | — | — | — | — | 1.00 | 0.000 | 0.708 | — | 294.6 | 4.68 |

## 每个样本相对 native 的变化

- hard_02（single）：native 严格 PCK@0.2 0.750 → 最佳 Union 档 union_s10 0.964（改善，Δ=+0.214）；身份分 0.916 → 0.585（掉 0.332，超过阈值 0.05）
- hard_05_torso（single）：native 严格 PCK@0.2 0.389 → 最佳 Union 档 union_s0 0.389（持平，Δ=+0.000）；身份分 0.920 → 0.920
- medium_02（single）：native 严格 PCK@0.2 0.844 → 最佳 Union 档 union_s10 0.938（改善，Δ=+0.094）；身份分 0.907 → 0.865
- hug_pair00_084（dual）：native 严格 PCK@0.2 0.068 → 最佳 Union 档 union_s0 0.068（持平，Δ=+0.000）；身份分 0.700 → 0.700
- dance_pair02_120（dual）：native 严格 PCK@0.2 0.000 → 最佳 Union 档 union_s05 0.269（改善，Δ=+0.269）；身份分 0.630 → 0.721
- cross_backhug_pair02_037（dual）：native 严格 PCK@0.2 0.456 → 最佳 Union 档 union_s0 0.456（持平，Δ=+0.000）；身份分 0.839 → 0.839

## 双人身份与对应关系细节

- hug_pair00_084 native seed42：检出 2 人，髋部配对 歧义（成本 243.21039406612604 px，与次优差 6.876512578679552 px），身份分 0.668，疑似互换 None（阈值 0.02），pair kind=same_identity
- hug_pair00_084 native seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.732，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 native seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.643，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 native seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.618，疑似互换 None（阈值 ?），pair kind=same_identity
- cross_backhug_pair02_037 native seed42：检出 3 人，髋部配对 歧义（成本 90.48105681350215 px，与次优差 0.38591735966748786 px），身份分 0.866，疑似互换 None（阈值 ?），pair kind=cross_pair
- cross_backhug_pair02_037 native seed43：检出 4 人，髋部配对 歧义（成本 101.98063737214059 px，与次优差 2.243563898136955 px），身份分 0.811，疑似互换 None（阈值 ?），pair kind=cross_pair
- hug_pair00_084 union_s0 seed42：检出 2 人，髋部配对 歧义（成本 243.21039406612604 px，与次优差 6.876512578679552 px），身份分 0.668，疑似互换 None（阈值 0.02），pair kind=same_identity
- hug_pair00_084 union_s0 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.732，疑似互换 None（阈值 ?），pair kind=same_identity
- hug_pair00_084 union_s05 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.693，疑似互换 None（阈值 ?），pair kind=same_identity
- hug_pair00_084 union_s05 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.729，疑似互换 None（阈值 ?），pair kind=same_identity
- hug_pair00_084 union_s10 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.626，疑似互换 None（阈值 ?），pair kind=same_identity
- hug_pair00_084 union_s10 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.744，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 union_s0 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.643，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 union_s0 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.618，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 union_s05 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.713，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 union_s05 seed43：检出 2 人，髋部配对 明确（成本 71.30322630803721 px，与次优差 214.90378966490107 px），身份分 0.730，疑似互换 False（阈值 0.02），pair kind=same_identity
- dance_pair02_120 union_s10 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.728，疑似互换 None（阈值 ?），pair kind=same_identity
- dance_pair02_120 union_s10 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.703，疑似互换 None（阈值 ?），pair kind=same_identity
- cross_backhug_pair02_037 union_s0 seed42：检出 3 人，髋部配对 歧义（成本 90.48105681350215 px，与次优差 0.38591735966748786 px），身份分 0.866，疑似互换 None（阈值 ?），pair kind=cross_pair
- cross_backhug_pair02_037 union_s0 seed43：检出 4 人，髋部配对 歧义（成本 101.98063737214059 px，与次优差 2.243563898136955 px），身份分 0.811，疑似互换 None（阈值 ?），pair kind=cross_pair
- cross_backhug_pair02_037 union_s05 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.676，疑似互换 None（阈值 ?），pair kind=cross_pair
- cross_backhug_pair02_037 union_s05 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.652，疑似互换 None（阈值 ?），pair kind=cross_pair
- cross_backhug_pair02_037 union_s10 seed42：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.717，疑似互换 None（阈值 ?），pair kind=cross_pair
- cross_backhug_pair02_037 union_s10 seed43：检出 1 人，髋部配对 明确（成本 ? px，与次优差 ? px），身份分 0.698，疑似互换 None（阈值 ?），pair kind=cross_pair

## 判定

**需要进一步输入适配**。

规则依据：只有 3/6 个样本改善（0 个退化、身份退化 1 例），控制通道有效但覆盖面不足，退化点需要先定位在输入侧还是权重侧。

规则写在这里以便复核：把「最佳 Union 档相对 native 的严格 PCK@0.2」提高超过 0.02 记一个改善样本；身份分下降超过 0.05 记一个身份退化样本；改善样本数 ≥ 75% 且无身份退化 → 直接使用；有改善但未全覆盖 → 需要输入适配；全不改善 → 需要另立训练实验。

小样本结论只决定下一步方向，不自动替换现有工作流；遮挡场景请结合 `panels/` 人工面板看，不要把检测器漏检直接算成生成失败。
