# workflow-multiple：双人姿态编辑与交互迁移

部署位置：`sg:/home/shangguanrz/project/pic-edit/workflow-multiple`。设计依据与整改记录见 [`docs/workflow_multiple_设计_20261013.md`](../docs/workflow_multiple_设计_20261013.md)；单人线是 `../workflow`，两条线共用 `../src/pose_edit`（缓存与指纹、DWPose-134 关节表、画布预处理、估计器封装、提示词契约的通用部分）。

输入 A（双人照片，出身份/服装/场景）与 B（双人交互照片，出交互姿态），让 Qwen 基座把 A 中两个人的姿态与相互关系改成 B 呈现的交互。**无 adapter、无 LoRA**，B 是否交给生成器是一个被测量的变量（见下）。

## 一期用什么数据

B 的骨架来自 Hi4D 预计算的**逐人 DWPose-134**（`target_dwpose_A/B`），不做拟合；A 的人物区域来自 Hi4D 的人体掩码。这样先把"编辑本身行不行"和"拟合误差"分开测量，multi-HMR 单图双人拟合臂是二期（`add/envs/multihmr` 与 `add/tools/multi-hmr/models/multiHMR/multiHMR_896_L.pt` 已在位）。

## 构建测试集

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit/workflow-multiple
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
M=../datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_test.jsonl
$PY build_testset.py --manifest $M --out pairs --list     # 只看选中的对
$PY build_testset.py --manifest $M --out pairs            # 写 pairs/ 与 pairs_cross/
```

选择规则确定可复现：按（目标动作族 × CROSS/SAME_ACTION × 接触顶点分档）轮转，同一动作族最多 2 对，逐人至少 12 个身体关键点分数≥0.5。产出 `pairs/<name>/{A_source.png, B_target.png, B_person_{1,2}_keypoints.npz, pair.json, GT_NOTE.md}`，`pair.json` 记录 manifest 行号、逐人关键点哈希、A 侧人体框、骨架 slot 顺序（按 B 中两人髖心 x 从左到右）与配对判定。`--count`/`--cross-count`/`--max-per-action` 可调。

## 跑一次

```bash
bash run.sh --check
bash run.sh --pair pairs/hug_pair00_084_c4_r000                     # 默认 with_b 臂
bash run.sh --pair .../pairs/hug_pair00_084_c4_r000 --name hug_skeleton_only \
     --generator-references skeleton_only                           # 不给生成器看 B
bash run.sh --pair .../pairs/hug_pair00_084_c4_r000 --name hug_fit --framing source_fit
bash run.sh --name hug_skeleton_only --pair .../pairs/hug_pair00_084_c4_r000 \
     --generator-references skeleton_only --verify                  # 复验，不推理
GEOPY=/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python
$GEOPY evaluate_dual.py --run hug_skeleton_only
$PY report_dual.py
```

`--generator-references` 两臂：`with_b` 给生成器 A、B、骨架1、骨架2 共四张（PE 与生成器同一组，无需重编号）；`skeleton_only` 只给 A 与两张骨架，PE 的四图回答由 `dual_prompts.compile_prompt` 重编号成三图并拒绝任何对 B 的引用。PE 两种情况下都看全四张图。

`--framing` 两档：`b_canvas` 把两人作为一个整体等比 letterbox 到 A 画布（保 B 的布局）；`source_fit` 再用**一个**统一缩放+平移把整对贴到 A 的两人尺度。两档都绝不逐人分别缩放——单人线的 hard_05 就是被"按人物框内切"在宽矮目标框上压到 41% 尺度而失败的。

`--prompt-file` 与单人线同义：手工提示词是显式哈希输入，PE 产物 `prompt/pe_compiled_prompt.txt` 只读、不会被覆盖。

## 输出

| 文件，相对 `runs/<name>/` | 用途 |
| --- | --- |
| `generation/final_image.png`、`final_canvas.png` | 生成结果（裁掉 A 的留边 / 未裁） |
| `generation/generation.json` | 每张参考图的角色、尺寸与 SHA256，臂名，无 adapter/LoRA 记录，提示词来源与哈希，显存与耗时 |
| `geometry/skeleton_{1,2}.png`、`skeleton_both.png` | 每人一张骨架图与合并图（合并图不跨人连线） |
| `geometry/dual_keypoints.npz` | 画布坐标下的请求骨架与有效性（评测直接读它） |
| `geometry/dual_geometry.json` | 定标变换、逐人有效点数与分数范围、输入哈希 |
| `geometry/{A,B}_skeleton_overlay.png` | 骨架叠 A / 叠 B 的肉眼核对图 |
| `prompt/pe_compiled_prompt.txt` | PE 输出经角色编译后的文本（只读） |
| `prompt/generation_prompt.txt`、`generation_prompt.json` | 实际交给生成器的文本及其来源（PE 或手工） |
| `prompt/pe_result.json`、`pe_attempts.json`、`pe_raw.txt` | 引用规则、采样、重试留档 |
| `evaluation/eval_dual.json`、`eval_overlay.png` | 指标与"白圈=请求、彩点=检出"叠加图 |
| `run.json`、`outputs.json` | 状态、逐阶段证据、实测校验结论、产物索引 |

缓存判定与单人线一致：`identity`（设置、代码哈希、模型身份、输入内容哈希）进指纹，绝对路径只进 `runtime`，产物按相对 run 目录记录，所以改 run 目录名或搬仓库不会让内容失效；失效时 `stage_reasons` 逐条指名是哪个设置/代码文件/产物变了。收尾会重跑本轮涉及阶段的校验，把结果写进 `run.json` 的 `verification.status`。

## 评测口径

`evaluate_dual.py` 用 DWPose ONNX（`yolox_l` + `dw-ll_ucoco_384`）在生成图上检出人体，先查**人数**，再按髖心做两种分配的尝试取较者（近tie 时标 `ambiguous`），逐人给 PCK@0.1/0.2/0.5、MPJPE（按请求躯干归一化）、躯干比例，另给两人髖距误差。原始 ONNX 输出是 COCO-WholeBody 133 点，评测经 `cocowholebody133_to_dwpose134` 换算到本仓库渲染用的 134 槽（neck 由两肩中点插入，与 MimicMotion 包装一致）。

它只测姿态与人际几何。身份/服装保持由 `src/pose_edit/appearance.py` 补上：把 A 的两人 mask 框与生成图上按姿态分配到的两人检出框各裁一张，过 DINOv2 ViT-S/14 取 CLS 特征算余弦相似，逐人给分数，并把"按外观最优配对"与"按姿态配对"做对比——两者不一致且人均差距超过 0.02 才报 `identity_swap_suspected`。它是外观代理而不是生物身份：重画的脸、被挡的脸都会降分而不证明人被换掉；框内背景变了也动同一个数。检出的点若被生成器藏起来，该点被排除而不算错，所以"藏 limbs"的结果会虚高。

**人数这一硬指标本身被查实是有条件的**（2026-10-14，用 `count_probe.py` 量的）：YOLOX 在**真实 Hi4D 目标照片** `pairs/dance_pair02_059_c4_r424/B_target.png` 上，nms_thr=0.45 只给 1 个 >0.3 的人框；把 NMS 放宽到 0.95 只多出几个几乎重合的框（[411,300,713,998] vs [410,310,681,1011]，同一具身体），并不是第二个人的框。也就是说**相拥的两个人会被检测器并成一框**，与生成器无关。因此 `person_count_ok=0` 不能直接读成"图里人不对"。

补上的口径是 `region_count`：对**每一个请求的人**各裁其 landmark 外扩 20% 的区域，在区域内单独跑一次检测，看那里有没有身体（`summary.person_count_regions_ok`）。实测对照：

| run | 整图检出 | 整图人数对 | 各区域内检出 | 区域全有 |
| --- | --- | --- | --- | --- |
| `dance_059.skeleton_only.b_canvas` | 1 | 否 | [1, 1] | 是 |
| `dance_059.with_b.b_canvas` | 1 | 否 | [1, 1] | 是 |
| `hug_skeleton_only` | 2 | 是 | [2, 2] | 是 |
| `hug_pair00_084`（with_b） | 3 | 否 | [3, 2] | 是 |

读法：dance 那两行**不是**"两臂都失败"，而是检测器在纠缠姿态上并框——肉眼看结果图确实是两个人在相拥（我下载核对过图）；hug 的 with_b 那一行才是真的多画了一个人（第一个区域内就有 3 个框）。所以：**整图人数只在两人分开时可信，纠缠姿态要用 `person_count_regions_ok`**；而姿态类指标（PCK/MPJPE/身份）在整图并框时会直接量不出来（dance 两行为 `None`），这类对子要么等区域化重算，要么在结论里单列，不能混进均值。这条修正来自②刚开始跑就出现反例——如果只按原来那一对 hug 下结论，"skeleton_only 人数正确、with_b 必失败"会被当成普适结论。

`target_import` 是另一件事：**直接**测"把 B 的人搬进来"，此前只能靠人数间接推断。它对生成图上每一个检出人体（含没有分配的第 3 个）分别取"与 A 两人的最大相似"和"与 B 两人的最大相似"，后者比前者超过 0.02 就计入 `bodies_liking_target_more`，汇总成 `summary.target_import_suspected`。它只在 A、B 是不同人的 `cross_pair` 上有意义：`same_identity` 对里 A 与 B 是同一对演员、同一身衣服，"像 B"与"像 A"是同一句话。这不是理论顾虑而是实测——`hug_skeleton_only`（我们认定成功的臂）在 same_identity 对上有 2/2 个人体"更像 B"（例：更像 A 0.800、更像 B 0.862），若不按 kind 门控就会把成功的臂标成搬运者；门控后该列输出 `None`，并在 json 里带 `not_informative_reason`。开发这段时被"跑一遍才发现"抓出两个静默错误：掩码键大小写敏感（`semantic_A/semantic_B`，按小写比对就永远取不到框、指标恒为 unavailable），以及框缺失时 `crop_person(None)` 直接崩——现在两种拼法都试、缺框走带原因的 unavailable。

`run.json` 的 `status` 由 `furthest_valid_stage()` 按阶段有效性算出（走到哪一段的指纹与产物哈希还成立），不再等于"最后一次调用要到哪"；同时写 `furthest_valid_stage` 字段。旧 run 的 `status` 保持当时的写法，要按新口径判定就重跑一次 `--verify`。

## 实况

17 对可用（12 对 same_identity + 5 对 cross_pair）。两臂对照已在 `hug_pair00_084_c4_r000` 上完成，768 分辨率、30 步、seed 42、`b_canvas` 定标，各一次运行，收尾自检均 `VERIFICATION validated`：

| 臂 | 生成图检出人数 | 人数正确 | body PCK@0.2 | MPJPE（归一化） | 躯干比例 | 髖距相对误差 | 身份相似 | 疑似换人 | PE 耗时 | 生成耗时/峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_b`（四图全给） | 3 | 否 | 0.000 | 0.942 | 1.066 | 1.355 | 0.681 | 是 | 259 s | 142 s / 34.04 GiB |
| `skeleton_only`（不给 B） | 2 | 是 | 0.933 | 0.061 | 0.921 | 0.022 | 0.739 | 否 | 289 s | 86 s / — |

结论：把 B 原图交给生成器时，模型**没有把 A 的两人摆成拥抱，而是把 B 的那两个人搬进了画面**（人数 3、PCK 0.000、两人距离比请求远 1.35 倍）；只给 A 与两张骨架图时，人数正确、拥抱真的迁移过去且 A 的服装鞋子保留，指标与单人线的良好对处于同一量级。身份指标印证了同一件事并给出方向：`with_b` 那一张里生成的人**各自**与 A 的某个人很像（对角相似 0.948 / 0.921），但按外观配出来的对应关系与按姿态配出来的相反（人均差 0.254，故报"疑似换人"），也就是把人搬到了对方的位置上；`skeleton_only` 的对角与姿态一致（0.800 / 0.678，人均差 0）。两行数字都是 2026-10-13 用当前 `evaluate_dual.py` 重测（`--force`）得到的，之前记录的 0.947 / 1.068 是换算修复前的旧评测。

批量对照在跑：`batch_dual.sh` 对 12 对 same_identity 跑 `{with_b, skeleton_only} × b_canvas` 加 `skeleton_only × source_fit`，对 5 对 cross_pair 跑两臂 `b_canvas`，共 46 个 run，逐个 `evaluate_dual.py`，进度与失败记在 `logs_batch/index.tsv`，收尾自动出 `report_dual.py`（已加身份两列、按"臂 × 定标"分组的均值块，且只在人数正确的 run 上求姿态均值）。写这一段时完成 3/46（2 成功、1 失败——失败是本机驱动的 NVML 断言，不是流水线），`retry_dual.sh` 会在批量结束后只重跑失败的 run（几何与提示词阶段仍命中缓存，一次重试只花生成那 2–8 分钟）。

这张卡与别人的长任务共用：会话开始时空余 60.7 GiB，2026-10-13 23:50 实测只剩 9.8–21 GiB（对方的 8arm 任务已跑 2 天 6 小时），而一次 768/30 生成实测峰值 34 GiB，池子不够时会被驱动的 NVML 断言直接打死。为此加了 `gpu_gate.sh`：每 5 分钟测一次 `torch.cuda.mem_get_info()`，空余 < 40 GiB 就 TERM 掉 batch 循环（正在跑的那个 run 不动，让它自己跑完或失败），回到 40 GiB 以上就重新拉起 `batch_dual.sh` —— 拉起时已记录 ok 的 run 会被跳过，恢复只花缓存检查的时间。看 `logs_batch/gate.log` 判断当前处于 RESUME / HOLD / PAUSE 哪一档。

批量要回答的是"同一对上，另一臂/另一档定标更好吗"，而 `report_dual.py` 的按臂均值回答不了这个——两臂若落在不同的对上，均值比的是对而不是臂。`compare_dual.py` 做**成对**比较：`--by arm` 比 `skeleton_only` 与 `with_b`、`--by framing` 比 `b_canvas` 与 `source_fit`，只取同一对两边都 complete 的 run，逐对给 left−right 的差与判定（PCK/身份/人数越大越好，MPJPE/髖距误差越小越好，躯干比例只作信息因为它是"离 1.0 的距离"，0.02 以内算平），并明确列出"只有一边跑成过"的对。

一条从实测里长出来的口径（2026-10-14 00:22，7 个 run 完成时的读数，往下列作对照不作覆盖）：
- 23:39 前后：3 个 run 完成（2 成 1 败）。
- 00:22：7 个 run 完成。`with_b` 三对全部人数不对（backhug_037 检 1 人、backhug_039 检 1 人、hug_084 检 3 人），因此**臂对照的姿态与身份两列结构性地是空的**：not measured 不是"数据还不够"，而是这一臂根本产不出可计分的两人图——成对表把它显示为 `count_ok` 判给 skeleton_only，其余列为 `.`。把 with_b 的结论写成"PCK 更低"会是假的，它没有 PCK。
- 同一时刻定标对照已经能投票（同臂 skeleton_only、两档定标都成的对）：backhug_037 上 MPJPE 与髖距误差偏向 `b_canvas`（0.230 vs 0.321；16.7 vs 25.4），PCK 打平 0.500，身份相似偏向 `source_fit`（0.788 vs 0.658）；backhug_039 上 `b_canvas` 连人数都没保住而 `source_fit` 保住了。两对都是 backhug，正是 WP7 里向导本身不可信的姿态，**不能当两档定标的结论**，要等非缠绕的 12 对跑完。
- 也验证了闸门与缓存的配合：00:16 被闸门打断的那个 run 生成阶段其实已经跑完（产物时间戳 00:15–00:17），00:21 批量重启后同一 run 三个阶段全部命中缓存、4 秒收尾并补上评测——被中断不浪费贵的那一步。

## 拟合臂（WP7）实测：multi-HMR 能不能当向导

`fit_dual_report.py` + `fit_dual_worker.py`（跑在 `add/envs/multihmr`，权重 `models/multiHMR/multiHMR_896_L.pt`）对 17 对的 B 原图各拟合一次，与两套参照比较：本线一直在用的 DWPose-134 逐人关键点，以及 Hi4D SMPL 关节经该帧自带内参投影到像素的结果。外参方向没写在任何资产说明里，所以按"哪个方向落在这张图的 DWPose 点上"选出 `world_to_camera`；SMPL-24 的关节顺序同样不靠假设，用骨盆=两髖中点、颈=两锁骨中点这两个刚性关系验（实测 0.087–0.088 m 与 0.090–0.091 m，与 SMPL 骨骼一致）。误差统计里剔掉 neck 槽：DWPose 的 neck 是两肩中点，SMPL-X 的 neck 是脊椎顶端，差几厘米是口径差不是错。

17 对全部拟合成功，汇总（`fits/fit_dual_report.json`）：

| 指标 | 拟合 vs DWPose | DWPose vs SMPL 投影（参照自身噪声底） |
| --- | --- | --- |
| body PCK@0.1 / @0.2 | 0.592 / 0.902 | 0.686 / 0.909 |
| MPJPE（像素 / 躯干归一） | 23.0 / 0.122 | 19.8 / 0.105 |
| 3D 绝对 MPJPE | 0.159 m | — |
| 3D 对齐骨盆后 MPJPE | 0.078 m | — |
| 两人髖距误差 | 0.012 m（相对误差中位数级 3–13%） | — |
| 人数正确 / 按髖心对应正确 | 17/17 / 16/17 | — |

门禁判定：`fit pck@0.2 ≥ 底-0.05`、`fit mpjpe_norm ≤ 底×1.25`、`髖距误差(躯干归一) ≤ 0.10`、`对应正确率 ≥ 0.90`、`人数正确率 ≥ 0.90`、`躯干比例 ∈ [0.85,1.20]` —— 前两项、后三项通过，只有髖距 0.120 未过。**未过这一项的责任不在估计器**：三张 backhug（两个 same_identity 对 + 用同一张 B 的一个 cross 对）里，DWPose 向导说两人髖心相距 4.2–4.4 px，而 SMPL 几何投影说 72.6–73.5 px，multi-HMR 说 72.3–72.4 px——拟合跟着真几何走，偏差 1 px 内，是现有向导错了约 68 px。这三对同时也是 `floor PCK@0.2 = 0.500` 与唯一一例对应错（backhug_039）的出处：纠缠姿态下逐人 mask 框里含另一个人，DWPose 的逐人关键点被污染了。

因此这一期的口径定为：**宽松姿态（dance/hug/pose/taichi/yoga）用 DWPose 向导与用拟合骨架向导等价（差异在参照自身噪声以内）；拥抱缠绕类（backhug）以 multi-HMR 的 3D 拟合为准，且评测必须同时报 `projected_gt_px` 那一列，否则把向导错了算成生成器错了。**

拟合骨架接入生成链**已经做完并验证**（`--keypoints-source fit`）：`dual_fitting.py` 是槽位表的唯一住处（`fit_dual_report.py` 与 `dual_geometry.py` 都从它取，报告和向导不可能各说各话），fit 分支读 `fits/<pair>.npz` 的 `j2d_native`，按**髖心 x 从左到右**排成 slot 1/2——与测试集编号 slot 的同一规则，向导不主张身份，provenance 里写 `side_basis: hip centre x order, identity not asserted`；人数不是 2、或任一人缺髖，直接报错而不是猜。走的是与 GT 完全相同的 letterbox/定标/渲染管线，所以 fit 与 dwpose 两种向导的差**只在关键点来源**，一次实验只换一个变量。

两条安全性验证（都是实测，不是声明）：
- **默认路径逐位不变**：`hug_pair00_084_c4_r000` 用重构后的 dwpose 分支重画，`skeleton_1/2/both.png` 与已完成 run 里的三个文件 SHA256 完全相同（`1f414ba7a116d1d4 / f90d0aa7765a6db0 / 55b7154844e16a3a`）。因为 PE 阶段的代码清单不含 `dual_geometry.py`，且骨架图逐位相同，批量里已缓存的 PE/生成阶段**继续有效**，只有 geometry 指纹变了一次（重算约 2 秒）。
- **fit 路径真跑得通**：同一对用 `--keypoints-source fit` 出图，两人各 13 个在框内的点、产物齐全。

fit 向导的生成实验本身还没跑——它在等显存窗口，且按计划排在批量之后，免得两组结论搅在一起。踩到的坑记一笔：新参数最初叫 `source`，与 `produce()` 里存 A 图的局部变量 `source` 同名，于是 PIL Image 被当来源名传下去、直接撞进"未知来源"分支；改名为 `keypoints_source` 后一致。

踩过的坑，记下来避免重复：`ImageOps.contain` **返回新图而不原地缩放**，第一版 `pad_geometry` 读原对象的 size，得到 scale=1.0 与负 pad，于是所有拟合点少除了一次 0.7 的缩放并被平移 (22,192) px——表现是 2D PCK@0.2 只有 0.14、3D 绝对误差 1.9 m，而两人髖距误差却只有 1 cm（常量平移与统一缩放不改变相对距离），差点被当成"模型精度差"。修正后同一套代码得到上表。自检教训写进了代码注释：**模型自己 j2d 与 j3d 的重投影一致性（0.17 px）查不出这类坐标错，因为它两侧用的是同一个错内参；只有与外部参照对比才能发现。**

## 定标常数：躯干 ÷ 人体框高

共享包里的 `TORSO_TO_PERSON_BOX_HEIGHT = 0.31` 是单人线 `source_torso` 定标用的（期望躯干 = 源人物框高 × 0.31），当年由 5 对单人数据反推。双人线自己的 `source_fit` 不做逐人缩放、用不到它，所以这个常数在双人数据上一直是空白。`torso_ratio.py` 直接量它：A 侧用 pair.json 里的 mask 框 + 在本机 CPU 上跑的 DWPose 关键点，B 侧用 pair 自带关键点 + `geometry.npz` 的语义掩码框；同一人体被多个 run 目录或共用源帧重复量到时只算一次（按 图片哈希+框 去重），原始次数与独立人体数都写出来，明细在 `ratios/torso_ratio.json`：

- **双人线**：68 次测量 → **38 个独立人体**（A 侧 16、B 侧 22），中位数 **0.285**，IQR 0.273–0.300，极值 0.256–0.333；
- **单人线**（同一把尺，但框用流水线自己记录的检测框 `geometry/B_camera.json` 的 `detections.A`）：16 次 → **8 个独立图**，中位 0.353，IQR 0.288–0.443，极值 **0.259–0.524**；
- 两线合并 46 个独立人体：中位 0.288。

读数比第一版更硬也更麻烦。双人侧：0.31 落在 IQR 之外（p75 = 0.300），且掩码框比检测框贴身体、**掩码口径的比值是检测口径的上界**，所以按这类身形 0.31 偏高约 5–9%，用它缩放会把迁移姿态放大同样比例。单人侧更关键：**在它出身的那条线上这个比值根本不是常数**——8 张源帧从 0.259 跨到 0.524，一倍有余。也就是说「A 框高 × 常数」预测躯干在单人线上本来就不成立，双人线只是把它照搬过来。

因此正确的修法不是把 0.31 调成 0.29：A 的躯干可以直接测（仓库里已有 DWPose ONNX 路径，本脚本就是这么量的），应当改成用实测的 A 躯干、只在 A 关键点不可得时退回框高常数。这一改动会动 `src/pose_edit/geometry.py` 的代码指纹、连带让批量中的 PE 阶段全部失效，所以**留到本轮 46 个 run 跑完再做**——对照中途不换尺子。本轮没有改常数。

同一份数据换算成流水线自己的单位才看清严重程度（`sizing_error`：`source_torso` 把迁移后的躯干定为 `0.31 × A框高`，而实测躯干是 `比值 × A框高`，所以人会被画成 `0.31 / 比值` 倍）：**单人线 8 张源帧的倍数中位 0.90、IQR 0.70–1.08、全距 0.59–1.20，只有 2/8 落在 ±10% 内**；双人 38 个独立人体中位 1.09、IQR 1.03–1.14、全距 0.93–1.21，55% 在 ±10% 内。结论分两层：双人式的紧致掩码框上，0.31 只是把人稳定画大约 9%；而在它出身的单人线上，同一个常数能从 0.59× 跳到 1.20×——**这不是"常数取值偏了一点"，是框高这个代理量本身不含躯干的信息**。所以 #21 的形状是"换成实测 A 躯干"，不是"重标定一个更好的常数"。
