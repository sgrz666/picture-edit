# docs/superpowers/plans/ —— 八份带日期的实施计划（含执行状态）
> 工作区路径：/home/shangguanrz/project/pic-edit/docs/superpowers/plans　|　本级：0 个子目录 / 8 份实体计划（合计约 50 KB；单份 0.7–17.8 KB，`ls -la` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md

## 一、这一层在干什么
每个大改动一份"任务清单式"计划：文件地图、按步 TDD（先写失败测试）、验证命令、发布/回滚方式。多数已执行完毕，正文保留勾选状态，可当工程考古的时间线读。

## 二、层级地图（按时间线）
| 计划 | 工作线 | 内容与状态实证 |
|---|---|---|
| `2026-09-12-champ-deepgen-single-sample.md`（17.8 KB） | CHAMP→DeepGen 单样本闭环 | 建 `src/integration/champ_deepgen.py`、`patches/deepgen_pose_control.patch`、去除 monkeypatch 的完整 TDD 计划；配套 spec 同名 |
| `2026-09-12-deepgen-no-control-ablation.md`（16.2 KB） | 纯 DeepGen 消融对照 | `call_deepgen_without_control`/`image_error_metrics` 契约测试与 runner；配套 spec 同名 |
| `2026-09-15-adapter-reasoning-v1.md`（0.7 KB） | DeepGen V6.2 | 实现记录：`SMPLXAdapterReasoner(ConditionBundle)→InternalControlState`，结构/前向-only、无训练；正文即摘要（头尾重复段落） |
| `2026-09-15-smplx-condition-injection-v1.md`（1.5 KB） | V6 条件注入 | 全部 `[x]`：ConditionBundle/bridge/CHAMP 后端（MIT、固定 commit）；回滚基线 tag `rollback/adapter-v6-condition-injection-v1-pre`@`9804541b` |
| `2026-09-16-deepgen-dynamic-control-interface-v1.md`（2.8 KB） | V6.3 动态控制接口 | 7 任务清单 + "Locked behavior"（geo 1.0/int 0.8、六组注入块段、零等价 1e-6）；即 `docs/adapter_v6_architecture.md` 的来历 |
| `2026-10-03-project-organization.md`（4.3 KB） | 目录治理第一轮 | 全部 `[x]`：62 实体归类、7 项大资产冷归档（157.506 GiB）、验收记录交付；执行结果见 `docs/目录清理执行记录_20261003.md` |
| `2026-10-04-bfs-expanded-comparison.md`（3.3 KB） | BFS 30 对扩测 | 全部 `[x]`；产物 `/home/shangguanrz/nihaochen/qwen21_full/bfs_compare_30_20261004/`，结论 `docs/BFS_扩大样本对照_20261004.md` |
| `2026-10-08-bfs-body-face-joint-pilot.md`（2.4 KB） | 联合试点（512 旧协议） | 顶部标注"被用户叫停"，任务 1 完成（`data/lock.json`），其余被原生画幅 handoff 取代 |

## 三、注意事项
- 计划内的文件路径是当时的（部分入口后来迁到 `scripts/` 子目录）；照抄命令前先核对现路径。
- 这些文件与 `../specs/` 同名日期文档配对阅读。
