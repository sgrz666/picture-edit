# research/bfs_transfer_audit_20261004/ —— BFS 控制移用性审计证据包

本目录是 2026-10-04 对 **BFS（Body-Face-Skeleton）控制能否从基座移用到本项目**的审计证据包，全部为 JSON 结论与一张对比面板。它回答三个问题：768 分辨率下 BFS 头部 LoRA 相比 512 body 是否更保身份、加/不加 LoRA 的差异、以及 BFS 权重在本机 Qwen 2.1 上能否加载（CPU 兼容性）。对应 `docs/BFS_面部保持移用评估_20261004.md` 的选型结论与 `docs/experiments/` 的落地 runbook。

## 里面有什么

| 文件 | 内容 |
|---|---|
| `results.json` | 主结果：`config`/`summary`/`per_pair`/`body_swap`。`summary` 含 `body`（n=6, identity_mean≈0.417, pck 1.0）、`stage2_nolora`/`bfs_head`/`bfs_head_rev`（identity_mean≈0.327~0.334, pck≈0.990）、`source_target_ceiling`（identity_mean≈0.627） |
| `corrected_768_pose.json` | 修正后的 768 姿态探针：bfs_head_768 与 nolora_768 的 PCK@0.02、per_pair 明细与误差说明 |
| `existing_probe_summary.json` | 既有探针汇总：body_512 / bfs_512 / bfs_768 / nolora_768 的源 A 身份均值（bfs_768≈0.492 > body_512≈0.417）、bfs_768 优于 body_512 的对数与局限 |
| `probe768.json` | 31 帧 768 探针：每帧 `identity_768_vs_A`、`body_512_vs_A` |
| `probe768_nolora.json` | 31 帧无 LoRA 768 探针：`identity`、`pck` |
| `cpu_compatibility_verification.json` | CPU 权重映射核验：`passed`、原始/转换张量键、target_modules、rank、与已装 Qwen 2.1 的形状匹配 |
| `panel_004.png` | 对比面板图（414 KB） |

## 与上下层的关系

- 上游：`docs/BFS_面部保持移用评估_20261004.md`、`docs/BFS_扩大样本对照_20261004.md`。
- 下游落地：`docs/experiments/2026-10-08-bfs-body-face-joint-*.md`（「Adapter 控姿 + BFS 768 ROI 精修面部」两阶段级联）。
- 实现层：`src/pose_control/qwen21_full/`（冻结 BFS 底座对照）。

## 注意与坑

- 样本量小（6 对身份 / 31 帧探针），`existing_probe_summary.json` 已列 5 项 limitations，结论为「倾向可用、不足以判定全面优于 Full Face」，勿当大规模统计结论引用。
- 身份均值口径是**源 A 脸余弦相似度**，天花板 `source_target_ceiling≈0.627` 是 A/B 自身参考上限，跨文件比较须同口径。
- 这些是**审计快照**，权重与代码可能已更新；复现须核对 `cpu_compatibility_verification.json` 中的 `weight_path` 与 diffusers/peft/torch 版本。
