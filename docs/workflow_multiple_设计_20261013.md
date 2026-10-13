# workflow-multiple 设计记录与阶段状态（2026-10-13）

范围：`workflow-multiple/`（双人姿态编辑与交互迁移基线）。命令、输出表与实测数字只写在 [`workflow-multiple/README.md`](../workflow-multiple/README.md) 一处，本文记录**为什么这样设计**与**还剩什么**。

## 任务与已定决策

输入 A（双人照片，出身份/服装/场景）与 B（双人交互照片，出交互姿态），无 adapter、无 LoRA，让 Qwen 基座把 A 中两人的姿态与相互关系改成 B 的交互。

| 决策 | 选择 | 理由 |
|---|---|---|
| 人数 | A 双人 → 输出双人，人数与身份守恒 | Hi4D 天然如此，且人数可被机器检查 |
| B 的骨架来源 | 一期用 Hi4D 预计算的逐人 DWPose-134；二期再接触入 multi-HMR | 把"编辑行不行"与"拟合误差"分开测量，否则失败无法归因 |
| 骨架落位 | 默认 `b_canvas`（整对等比 letterbox 到 A 画布）；另有 `source_fit`（一个统一缩放+平移贴到 A 的两人尺度） | 单人线的 hard_05 证明"按人物框内切"在 A/B 框长宽比相反时把请求姿态压到 41% 尺度；两档都绝不逐人分别缩放 |
| 生成器看到哪几张 | 做成 `--generator-references with_b / skeleton_only` 两臂 | 用户原案给四张（含 B），单人线基线刻意不给 B；只有两臂对照才能量化外观泄漏 |
| 人物对应 | 生成侧不主张对应；评测侧按髖心试两种分配取较者，近 tie 标 `ambiguous` | `source_index_for_target` 字段无任何代码消费，不可当依据；拥抱时两人髖心可只差 1px |
| 共用代码 | 提到仓库级 `src/pose_edit/`，两条线都 import | 避免第二份逐字节 fork（仓库级审查 P1-1 的教训） |

## 阶段状态

| 工作包 | 状态 | 证据 |
|---|---|---|
| WP0 资产核实 | 完成 | `workflow-multiple/assets_check.md` |
| WP1 共享包抽取 | 完成 | sg `d17d26f`；单人线 54 项测试通过且 hard_02 端到端重跑指标逐位一致（PCK 0.944 / MPJPE 0.057 / 手 0.179 / 脸 0.021） |
| WP2 测试集 | 完成 | 12 对 same_identity + 5 对 cross_pair；去掉时间戳后两次构建的 `pair.json` 完全相同，图像与 npz 哈希全等 |
| WP3 双人 geometry | 完成 | 骨架与 Hi4D 人体掩码数值对齐：鼻子落在头部带内、90.8%/100% 关键点在各自人体框内；5 项不变量测试 |
| WP4 双人 PE | 完成 | 四图角色契约 + 4→3 编译；`citation_rule` 单一约定（PE 按自己收到的编号写，重编号只由编译器做） |
| WP5 生成两臂 | 完成 | 四图 34.04 GiB / 142 s，三图 86 s；`generation.json` 记录每张参考图角色与 SHA256 |
| WP6 评测与索引 | 完成 | `evaluate_dual.py` + `report_dual.py`；两臂对照见 README 实况表 |
| WP7 multi-HMR 拟合臂 | 未开始 | 环境与权重在位（`add/envs/multihmr`、`multiHMR_896_L.pt`）；须先出"拟合误差报告"再接入 |
| WP8 文档与入库 | 本文与 README；代码入库见下一个提交 | — |

## 过程中修掉的两个约定/设计陷阱

1. **DWPose ONNX 是 133 点，本仓库渲染与 Hi4D 文件是 134 点**。COCO-WholeBody 顺序（17 body + 6 foot + 68 face + 42 hand）与 OpenPose 兼容顺序（18 body，插 neck）不同名同数会静默错位。加了 `src/pose_edit/geometry.py:cocowholebody133_to_dwpose134`（neck 取两肩中点，与 MimicMotion 包装一致）并用槽位语义测试钉住。
2. **PE 的引用编号约定自相矛盾**。最初指令让 PE"按编辑器编号重写"，而编译器又假设它按 PE 编号写再做映射，于是正确的回答被判成违约、两次重试后失败。改为 `citation_rule(arm)` 单一约定：PE 永远按自己收到的 1..4 编号写，只有 `compile_prompt` 负责重编号。

另外两处是本轮自己造成并清理的：`tar -C workflow … | ssh 'cd 仓库根 && tar -xf'` 把 7 个 workflow 文件解到了仓库根（其中游离 `run.py` 会遮蔽 workflow 的模块，已删）；两个测试套件都用 `_repo_path.py` 时 pytest 只导第一个，导致合并跑炸 19 个（改为各自唯一文件名 + 仓库根 `conftest.py`）。

## 尚未解决

- **身份/服装保持没有指标**。那套双人评测原先用的 VGG16 权重不在这台机器上；DINOv2 权重在位，是候选实现。当前只能靠肉眼与"人数"这一硬指标。
- **检出被藏起来的 limb 不算错**：`evaluate_dual.py` 排除检测器没找到的点，所以"藏手臂"的图会虚高，需要配合覆盖率（`body_joints`）一起读。
- **`run.json` 的 `status` 反映最后一次调用而非最远完成阶段**：对既有 run 只跑到 `--until geometry` 会把它标成 `partial`，即使生成产物完好。应改成按阶段有效性判定。
- **`source_fit` 尚未与 `b_canvas` 在同一对上对照**；`--framing` 的常数（躯干/框高 0.31）是从单人线 5 对反推的，双人场景两人重叠时是否仍成立未验证。
- 16 对没有运行；cross_pair 那 5 对没有可信的人物对应，只能靠 best-of-two。
- 仓库级遗留：sg 与 Windows 的 main 历史分叉（本轮只提交到 sg 并经 `git bundle` 取回 Windows 分支，未 push）；根 `tests/` 收集期 13 个既有错误与本轮无关。
