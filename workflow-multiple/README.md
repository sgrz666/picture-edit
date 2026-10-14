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

**控制组把 `target_import` 打回"未验证"**（2026-10-14 10:15，55 个 run 全量聚合，逐身体 `target_minus_source` = 与 B 照片的相似度 − 与 A 照片的相似度）：

| 组 | 身体数 | 中位 | 均值 | 更像 B 的比例 |
|---|---|---|---|---|
| `same_identity` × `skeleton_only`（A/B 是同两个人同衣服，方向本应无意义） | 52 | **+0.084** | +0.077 | 81% |
| `same_identity` × `with_b` | 39 | **+0.109** | +0.066 | 72% |
| `cross_pair` × `skeleton_only`（flag 只在这一组被允许） | 23 | **−0.023** | −0.049 | 39% |

控制组不仅不为零，反而比实验组更大且方向相反。也就是说这个量主要由**裁剪/尺度与参照集差异**驱动，不是"把 B 的外观漏进画面"的证据；此前本节写的"没喂 B 原图仍有 3/17、4/16 被 flag，说明有人比像 B 更像 A"是把噪声当发现，**撤回**。要用它必须先给同一 run 内的基线（逐身体的差值减去 same_identity 组同幅度分布），在那之前报告只给数字、不给布尔判定（`identity_swap_suspected` 不受影响，见下）。

**`identity_swap_suspected` 是有信号的**：9/55 被 flag，其中 `skeleton_only` 7、`with_b` 2，`b_canvas` 与 `source_fit` 两档都有，两种检测基准各有（whole_image 5、regions 4），超阈值幅度 0.022–0.120（门槛 0.02）。可核查例：`dance_pair02_059_c4_r424.skeleton_only.b_canvas` 相似度矩阵对角 0.651/0.664、非对角 0.734/0.737 → 外观配对 [1,0] 与姿态配对 [0,1] 相反，`swap_margin_per_person` 0.078。矩阵与两个配对都存在 `eval_dual.json` 的 `appearance` 块里，判定可逐条复核——这才是①要的那块"看不见的失败面"。

**为什么说它"看不见"——被 flag 的 9 条里，姿态指标本身很好看**：

| run | PCK@0.2 | MPJPE(归一) | 整图人数 | 基准 |
|---|---|---|---|---|
| `dance_059.skeleton_only.b_canvas.fit` | **1.000** | 0.064 | 1（错） | regions |
| `dance_059.skeleton_only.source_fit` | 0.950 | 0.068 | 0（错） | regions |
| `dance_059.with_b.b_canvas` | 0.917 | 0.051 | 0（错） | regions |
| `yoga_263.skeleton_only.b_canvas` | 0.941 | 0.057 | 1（对） | whole_image |
| `dance_059.skeleton_only.b_canvas` | 0.913 | 0.068 | 0（错） | regions |
| `dance_059.with_b.source_fit` | 0.875 | 0.059 | 0（错） | regions |
| `hug_084.skeleton_only.b_canvas.fit` | 0.662 | 0.158 | 1（对） | whole_image |
| `backhug_037.skeleton_only.b_canvas` / `.source_fit` | 0.500 | 0.230 / 0.321 | 1（对） | whole_image |

`dance_059` 这一对的五个 run 全被 flag（两臂 × 两定标再加 fit 孪生），它们的 PCK 在 0.875–1.000、MPJPE 0.051–0.068——按姿态看是全场最好的那批图。也就是说**这两个人站到了对方的位置上，关节却摆对了**：只看姿态与人数都会放行。`backhug_037` 是另一类（PCK 0.500、人数正确），flag 与姿态问题同时出现。这一条不依赖 `target_import`，所以①的成果仍然成立，只是"漏外观"那一半被撤。

**import 偏差的来源已定位到口径**（不是猜）：A 侧裁剪用 Hi4D 语义**掩码**框（`evaluate_dual.py:167` `source_boxes_canvas` 读 `person['region']['bbox_xyxy']`），生成侧用 DWPose/YOLOX **检测**框，`crop_person` 两边都只各扩 2%——掩码贴身体、检测框含更多背景，再加上 `A_reference.png` 与 `B_target.png` 的取景/分辨率本就不同，余弦差的**符号**可由口径预测。所以 #25 的第一步是把两侧裁剪几何对齐（同长边正方形化）再重测控制组，第二步才谈阈值。

**0.02 这个 swap 阈值是量出来的，但空隙很窄**（`swap_calibration.py`，56 个带方阵的 run）：存的 `swap_margin_per_person` 在被 flag 的 9 条里为 **0.022–0.120**，未 flag 的 47 条最高 **0.018** ——— 两侧之间有间隙，阈值落在间隙里，不是随手取的。两点必须同时写出来，否则这组数会被读过头：

- 间隙只有 **0.004**（0.018 vs 0.022）。`hug_pair00_084_c4_r000.skeleton_only.b_canvas.fit`（0.022）这条判定是脆的，阈值挪 ±0.005 就会翻。
- 幅度 ≥0.055 的 **7 条**才是稳的（0.055/0.070/0.078/0.079/0.082/0.092/0.120），另有 0.022 与 0.031 两条属边界；引用"疑似换人"要按幅度分层，不能把 9 条当一个同质集合。
- 判据必须用**按外观最优配对之后**的 margin：朴素的"非对角均值 − 对角均值"与它对不同号（9 条被 flag 的 run 里该原始值最小 **−0.092**），因为相似度矩阵不对称。矩阵与非对称性都在 `eval_dual.json` 里，逐条可复核。

**人数这一硬指标本身被查实是有条件的**（2026-10-14，用 `count_probe.py` 量的）：YOLOX 在**真实 Hi4D 目标照片** `pairs/dance_pair02_059_c4_r424/B_target.png` 上，nms_thr=0.45 只给 1 个 >0.3 的人框；把 NMS 放宽到 0.95 只多出几个几乎重合的框（[411,300,713,998] vs [410,310,681,1011]，同一具身体），并不是第二个人的框。也就是说**相拥的两个人会被检测器并成一框**，与生成器无关。因此 `person_count_ok=0` 不能直接读成"图里人不对"。

补上的口径是 `region_count`：对**每一个请求的人**各裁其 landmark 外扩 20% 的区域，在区域内单独跑一次检测，看那里有没有身体（`summary.person_count_regions_ok`）。实测对照：

| run | 整图检出 | 整图人数对 | 各区域内检出 | 区域全有 |
| --- | --- | --- | --- | --- |
| `dance_059.skeleton_only.b_canvas` | 1 | 否 | [1, 1] | 是 |
| `dance_059.with_b.b_canvas` | 1 | 否 | [1, 1] | 是 |
| `hug_skeleton_only` | 2 | 是 | [2, 2] | 是 |
| `hug_pair00_084`（with_b） | 3 | 否 | [3, 2] | 是 |

读法：dance 那两行**不是**"两臂都失败"，而是检测器在纠缠姿态上并框——肉眼看结果图确实是两个人在相拥（我下载核对过图）；hug 的 with_b 那一行才是真的多画了一个人（第一个区域内就有 3 个框）。所以：**整图人数只在两人分开时可信，纠缠姿态要用 `person_count_regions_ok`**；而姿态类指标（PCK/MPJPE/身份）在整图并框时会直接量不出来（dance 两行为 `None`），这类对子要么等区域化重算，要么在结论里单列，不能混进均值。这条修正来自②刚开始跑就出现反例——如果只按原来那一对 hug 下结论，"skeleton_only 人数正确、with_b 必失败"会被当成普适结论。

顺着这条把姿态指标也救回来了：整图人数不对时，评测改用**逐人区域检测**（`pose_basis: regions`，每人区域内单独检一次、取该区域内可信点最多的那个身体，slot 归属按位置而非髖心推断）。dance_059 两臂于是有了姿态数字，成对表立刻能投票：

| 对子（skeleton_only vs with_b） | 人数 | PCK@0.2 | MPJPE(归一) | 躯干比例 | 身份相似 |
| --- | --- | --- | --- | --- | --- |
| dance_059（basis=regions） | 0.000\|0.000 = | 0.913\|0.917 = | 0.068\|0.051 = | 0.958\|0.979 i | 0.658\|0.621 **L** |

也就是说在这一对上，**with_b 的姿态完成度与 skeleton_only 相当（PCK 差 0.004，判平）**，差别只出现在身份相似（skeleton_only 更像 A 的两个人）。原来"with_b 姿态全废（PCK 0.000）"的印象来自 hug_084 那一例——那里它确实多画了人（区域 [3,2]）、PCK 为 0。两件事要分开说：**with_b 的风险是引入多余身体与换人，不是必然把姿态做坏**。`report_dual.py` 的臂均值改为"凡姿态可计分者入均值"（不再要求整图人数正确），并新增 `bas` 列标明每个 run 用的是哪种检测基准，避免把两种基准的数字混着读。

`target_import` 是另一件事：**直接**测"把 B 的人搬进来"，此前只能靠人数间接推断。它对生成图上每一个检出人体（含没有分配的第 3 个）分别取"与 A 两人的最大相似"和"与 B 两人的最大相似"，后者比前者超过 0.02 就计入 `bodies_liking_target_more`，汇总成 `summary.target_import_suspected`。它只在 A、B 是不同人的 `cross_pair` 上有意义：`same_identity` 对里 A 与 B 是同一对演员、同一身衣服，"像 B"与"像 A"是同一句话。这不是理论顾虑而是实测——`hug_skeleton_only`（我们认定成功的臂）在 same_identity 对上有 2/2 个人体"更像 B"（例：更像 A 0.800、更像 B 0.862），若不按 kind 门控就会把成功的臂标成搬运者；门控后该列输出 `None`，并在 json 里带 `not_informative_reason`。开发这段时被"跑一遍才发现"抓出两个静默错误：掩码键大小写敏感（`semantic_A/semantic_B`，按小写比对就永远取不到框、指标恒为 unavailable），以及框缺失时 `crop_person(None)` 直接崩——现在两种拼法都试、缺框走带原因的 unavailable。

`run.json` 的 `status` 由 `furthest_valid_stage()` 按阶段有效性算出（走到哪一段的指纹与产物哈希还成立），不再等于"最后一次调用要到哪"；同时写 `furthest_valid_stage` 字段。旧 run 的 `status` 保持当时的写法，要按新口径判定就重跑一次 `--verify`。

## 实况

17 对可用（12 对 same_identity + 5 对 cross_pair）。**两臂与定标的结论以本节末尾的“②的最终成对读数”为准。**下面这张表是最初只在 `hug_pair00_084_c4_r000` 一个对上做的一次对照（768、30 步、seed 42、`b_canvas`，各一次运行，收尾自检 `VERIFICATION validated`），保留作个案与方法证据，**不能当臂的结论读**：

| 臂 | 生成图检出人数 | 人数正确 | body PCK@0.2 | MPJPE（归一化） | 躯干比例 | 髖距相对误差 | 身份相似 | 疑似换人 | PE 耗时 | 生成耗时/峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_b`（四图全给） | 3 | 否 | 0.000 | 0.942 | 1.066 | 1.355 | 0.681 | 是 | 259 s | 142 s / 34.04 GiB |
| `skeleton_only`（不给 B） | 2 | 是 | 0.933 | 0.061 | 0.921 | 0.022 | 0.739 | 否 | 289 s | 86 s / — |

个案读数（仅此一对）：把 B 原图交给生成器时，模型**没有把 A 的两人摆成拥抱，而是把 B 的那两个人搬进了画面**（人数 3、PCK 0.000、两人距离比请求远 1.35 倍）；只给 A 与两张骨架图时，人数正确、拥抱真的迁移过去且 A 的服装鞋子保留，指标与单人线的良好对处于同一量级。身份指标印证了同一件事并给出方向：`with_b` 那一张里生成的人**各自**与 A 的某个人很像（对角相似 0.948 / 0.921），但按外观配出来的对应关系与按姿态配出来的相反（人均差 0.254，故报"疑似换人"），也就是把人搬到了对方的位置上；`skeleton_only` 的对角与姿态一致（0.800 / 0.678，人均差 0）。两行数字都是 2026-10-13 用当前 `evaluate_dual.py` 重测（`--force`）得到的，之前记录的 0.947 / 1.068 是换算修复前的旧评测。**但这是 n=1**：扩到 12 对后 with_b 在可投票的姿态指标上反而略优（均值 PCK 0.807 vs 0.786），它稳定输掉的是人数（7/12 vs 12/17）——见②节。单对上的“搬进画面”是真的，写成“with_b 这一臂失败”是不真的。

**批量进度（滚动更新，实时数以 `matrix_audit.py` 为准，别看下面这段的文字数）**：第一轮 04:09 跑完 45 格、04:12 retry 收尾、04:18 finalize 统一重算并出 `report_dual.py` 与三份 `comparisons/compare_*.json`。审计发现设计矩阵其实是 17 对 × 两臂 × 两定标 = **68 格**，旧 `batch_dual.sh` 把 `with_b` 限制在同身份的 b_canvas（n=1 理由），于是欠 23 格。矩阵已改成全交叉，第二轮 09:39 起在补空格（`done_run()` 跳过已 ok 的，重试只花生成那 2–8 分钟，几何与 PE 命中缓存）；09:51 实测 covered 50 / planned 85（含 17 个孪生格）。链条：batch → `watch_and_retry.sh` → `finalize_dual.sh` → `run_fit_after_batch.sh`（14 个拟合孪生）→ `FIT_ALL_DONE`，四个标记齐了才算完。

这张卡与别人的长任务共用：会话开始时空余 60.7 GiB，2026-10-13 23:50 实测只剩 9.8–21 GiB（对方的 8arm 任务已跑 2 天 6 小时），而一次 768/30 生成实测峰值 34 GiB，池子不够时会被驱动的 NVML 断言直接打死。为此加了 `gpu_gate.sh`：每 5 分钟测一次 `torch.cuda.mem_get_info()`，空余 < 40 GiB 就 TERM 掉 batch 循环（正在跑的那个 run 不动，让它自己跑完或失败），回到 40 GiB 以上就重新拉起 `batch_dual.sh` —— 拉起时已记录 ok 的 run 会被跳过，恢复只花缓存检查的时间。看 `logs_batch/gate.log` 判断当前处于 RESUME / HOLD / PAUSE 哪一档。

批量要回答的是"同一对上，另一臂/另一档定标更好吗"，而 `report_dual.py` 的按臂均值回答不了这个——两臂若落在不同的对上，均值比的是对而不是臂。`compare_dual.py` 做**成对**比较：`--by arm` 比 `skeleton_only` 与 `with_b`、`--by framing` 比 `b_canvas` 与 `source_fit`，只取同一对两边都 complete 的 run，逐对给 left−right 的差与判定（PCK/身份/人数越大越好，MPJPE/髖距误差越小越好，躯干比例只作信息因为它是"离 1.0 的距离"，0.02 以内算平），并明确列出"只有一边跑成过"的对。`matrix_audit.py` 回答"这一轮到底覆盖了几格、每个 compare 维度真有几对可投票"（覆盖判据 = `run.json` status=complete **且** 有 `evaluation/eval_dual.json`，光看目录会把跑完但无人能打分的 run 算成证据）；`size_compliance.py` 回答"人是按请求尺寸画出来的吗"（逐人 实测躯干 ÷ 请求躯干）。两个都只读已有产物、不需要 GPU，批量在跑的时候也能执行；退出码 1 表示矩阵还有空格，别把"跑完了"当成"覆盖了"。

### ②的成对读数（三段：04:18 finalize 重算的 45 格表 + 10:08 第二轮进行中的两个分档表）
本节的三个块各有自己的样本量：45 格表是 finalize 的统一重算值（可引用），两个分档表是第二轮跑到一半时的行数（18/22 行，`with_b.source_fit` 与跨身份对还没补完，只能算早期信号）。`FIT_ALL_DONE` 之后用 finalize 的重算值统一替换，并且**臂的结论必须按定标分档写、定标的结论必须按臂分档写**——这一轮已经证明合并会抹掉真实的交互。


覆盖：下表是 **04:18 那一次的 45 格**（臂 12 行 / 定标 16 行 / 向导 3 行）。**行数会随第二轮补格上升，实时格数与配对行数一律以 `matrix_audit.py`（落盘 `logs_batch/matrix_audit.txt`）为准，不要把本表的行数当现状读**：09:51 实测已经是 planned 85 / covered 50、`--by arm` 14 行、`--by framing` 18 行、`--by source` 3 行，还欠 35 格（21 个 dwpose 向导格 + 14 个拟合孪生）。覆盖判据刻意取 run.json=complete **且** evaluation/eval_dual.json 存在：光看目录会把"跑完但没人能打分"的 run 算成证据。同一身份的 12 对跑满三格（`skeleton_only.b_canvas`、`skeleton_only.source_fit`、`with_b.b_canvas`），跨身份的 5 对只跑 `skeleton_only`（不给 B 的原图才有意义），另有 3 对配了 fit 向导孪生 run。合计 48 个 run 目录、45 条带评测的批量 run。评测口径与批量期间的两次修补见后一节。

**分组均值**（identity/pck/mpjpe 只在能打分的人数正确 run 上取均值；count 与 swap 覆盖组内全部 run）：

| 组 | run | 人数正确 | 区域都有人 | swap flag | import flag | identity | PCK@0.2 | mpjpe_norm |
|---|---|---|---|---|---|---|---|---|
| `skeleton_only.b_canvas` | 17 | 12/17 | 17/17 | 3 | 3（另 13 无定义） | 0.729 (n=16) | 0.786 | 0.157 |
| `skeleton_only.source_fit` | 16 | 8/16 | 16/16 | 2 | 4（另 12 无定义） | 0.728 (n=16) | 0.765 | 0.195 |
| `with_b.b_canvas` | 12 | 7/12 | 12/12 | 1 | 0（全部无定义） | 0.745 (n=11) | 0.807 | 0.148 |
| `skeleton_only.b_canvas.fit` | 3 | 1/3 | 3/3 | 2 | 0（全部无定义） | 0.716 (n=3) | 0.887 | 0.103 |

**臂（`skeleton_only` vs `with_b`，同臂同定标，12 对）**：count 3 胜 / 8 平 / 1 负，PCK 4/5/2，MPJPE 3/4/4，髖距 2/2/5，identity 5/3/3。
读数：**上一轮"with_b 把 B 的人搬进来所以失败"在这 12 对里不成立**。with_b 在能投票的姿态指标上略优（均值 PCK 0.807 vs 0.786、MPJPE 0.148 vs 0.157，髖距 5:2 领先），代价集中在**人数**（7/12 vs 12/17）；身份反而更高（0.745 vs 0.729）、swap 只 flag 1 对。`target_import` 在 with_b 上恒为无定义，这是设计如此：给了 B 的原图，"更像 B"就是预期结果，flag 没有信息量——所以这个指标只在 `skeleton_only` 侧读（3/17 与 4/16 被 flag，即生成图里有人比像 A 更像 B，而这一臂并没有把 B 的照片喂给出图模型）。

**臂的结论随定标变号，这是分档后才看见的（`compare_dual.py` 现在按 held 上下文分开报，2026-10-14 10:08，`--by arm` 18 行）**：

| 定标档 | 行数 | count(skel/平/with_b) | PCK | MPJPE | identity |
|---|---|---|---|---|---|
| `b_canvas` | 12 | 3 / 8 / 1 | 4 / 5 / 2 | 3 / 4 / 4 | 5 / 3 / 3 |
| `source_fit` | 6（进行中，n 尚小） | 3 / 3 / 0 | **2 / 1 / 3** | **1 / 2 / 3** | 3 / 2 / 1 |

读法：**人数**在两个档都是 `skeleton_only` 稳赢（3-8-1 与 3-3-0），但**姿态**在两档方向不同——`b_canvas` 档 PCK 4/5/2 偏 `skeleton_only`，`source_fit` 档 PCK 2/1/3、MPJPE 1/2/3 反过来偏 `with_b`（中位差 −0.042 / +0.049）。合并两档的池化块会把这个交互抹平（池化 PCK 是 6/6/6），所以**引用时必须指名档位，池化块不作结论**（工具已把这一点写进输出：`--- POOLED over N held contexts: not a single-framing result ---`）。`source_fit` 档只有 6 行、`with_b` 的 12 格仍在补，`FIT_ALL_DONE` 后以 finalize 的重算值替换这两行。

反方向（`--by framing` 按臂分档，同一时刻）也一并记下来，因为它把"定标影响什么"说得更准：

| 臂档 | 行数 | count(b_canvas/平/source_fit) | PCK | MPJPE | identity |
|---|---|---|---|---|---|
| `skeleton_only` | 16 | 6 / 8 / 2 | 6 / 3 / 6 | 5 / 7 / 3 | 3 / 7 / 5 |
| `with_b` | 7（进行中） | **3 / 4 / 0** | 1 / 4 / 1 | 1 / 4 / 1 | 3 / 2 / 1 |

读法：**定标的作用是人数**——两臂都是 `b_canvas` 在人数上领先（`with_b` 档 3 胜 0 负），而姿态在两臂都以平局为主（`with_b` 档 PCK 6 行里 4 行判平）。也就是"给不给 B 原图"与"用哪档定标"影响的量不是同一个：前者在 `source_fit` 档能改姿态胜负，后者主要改人数。两档样本都还在补，finalize 后替换。

**定标（`b_canvas` vs `source_fit`，同臂同向导，16 对）**：count 6 胜 / 8 平 / 2 负，PCK 6/3/6，MPJPE 5/7/3，髖距 7/3/2，identity 3/7/5。
读数：中期那份"b_canvas 在每个可投票姿态指标上全胜"是 5 对时的表象。**扩到 16 对后 PCK 打平（6:6）**，`b_canvas` 的优势只剩两处：人数正确率（6 胜 2 负）与髖距（7 胜 2 负）；MPJPE 略偏 `b_canvas`；identity 反而 `source_fit` 略好（5 胜 3 负）。两条定标在姿态上没有稳定的胜负关系，`b_canvas` 的领先集中在人数与髖距两处。**这一档与 0.31 常数无关**（先前本节把它归因给常数是错的，已改）：双人线的 `source_fit` 用的是实测的两人 extent 比 `a_pair[extent_px]/b_pair[extent_px]`（`dual_geometry.py:171-185`），`TORSO_TO_PERSON_BOX_HEIGHT` 只存在于单人线的 `align_to_source_bbox(..., framing="source_torso")`。所以这里量到的是"用整体 extent 定标与用 B 自己的画布定标差不多"，而不是"某个常数对不对"。

**向导来源（`dwpose` GT 骨架 vs `fit` multi-HMR 拟合骨架，3 对孪生）**：count 1/2/0，PCK 1/0/2，MPJPE 1/1/1，identity 2/0/1，髖距 1/0/0。逐对：backhug_pair02_037 拟合赢 3 项、hug_pair00_084 GT 赢 4 项、dance_pair02_059 1/2/1 基本平。
读数：**n=3，不构成结论**，方向与 WP7 的门一致——缠绕的 backhug 上拟合向导更好（那里 DWPose 向导自己就错了），宽松姿势上 GT 更好。fit 孪生有 2/3 人数不对，但能打分时 PCK 均值 0.887 最高，所以这一格现在只能写成"值得扩到全部 12 对再判"，不能写成"拟合向导更好"。

### 逐轮计数（追加，不覆盖）：
- 2026-10-14 09:43 **②的覆盖被审计出缺口并已开跑第二轮**：`batch_dual.sh` 旧矩阵把 `with_b` 限制在 12 个同身份对的 `b_canvas`（注释理由是"失败就在那一格复现过"，即 n=1 的取舍），于是全交叉的 17×2×2=68 格里只有 45 格有 ok 记录，**缺 23 格**（12 个 `with_b.source_fit` + 5 个跨身份对的 `with_b` + `cross_backhug…skeleton_only.source_fit`）。矩阵已改成 pair×arm×framing 全交叉（`896e222`），09:39 起跑第二轮，`done_run()` 只补空格；watch_and_retry / finalize / gpu_gate 全挂上（起时显存 62 GiB 空闲，闸门记 OK free=82GiB）。上面"②的最终成对读数"里的臂对比因此只在 `b_canvas` 一档成立，第二轮跑完需重读 `compare_dual --by arm`（应出现 source_fit 档）与 `--by framing`（应出现 with_b 档）。向导那一档同步扩样本：`run_fit_after_batch.sh` 原本硬编码 3 个对，已改成从 `fits/*.npz` 派生（`5d006c4`），待跑 14 个孪生（9 同身份 + 5 跨身份，拟合缓存都在），排在第二轮 finalize 之后，跑完 `--by source` 从 n=3 变 n=17。

- 2026-10-14 09:35 **②收尾**：批量 04:09 BATCH_DONE、retry 3 轮 04:12 RETRY_DONE、finalize 04:18 统一重算 45 条 run（`EVAL_FAILED=0`，用的已经是修好的评测器）、fit 孪生 04:35 FIT_AFTER_DONE。成对读数从 5 对扩到 12（臂）/16（定标）/3（向导），上面中期表已被最终表替换。三处中期结论被数据改写：with_b 在姿态上不再"更差"（略优，输在人数）、b_canvas 的 PCK 全胜变成打平、`target_import` 在 with_b 上恒无定义是设计而非缺测。
- 2026-10-14 01:45 评测阶段两处崩溃已修（`evaluate_dual.py` 只出现在评测指纹里，几何/PE/生成缓存与批量都不受影响）：
  ① 没有任何可评人员时 `np.mean([])` 把 NaN 写进 `eval_dual.json`，`json.dumps` 抛 `Out of range float values are not JSON compliant: nan`，整条 run 记成 `evaluate_failed`。现在 `finite()` 递归把 NaN/inf 写成 `null`——"测不出"在数据里就是缺值，与"测不出 ≠ 更差"的报告口径一致。
  ② `pose_basis=regions` 时 assignment 按请求人数给（长度 2），而 `draw_overlay` 拿的是整图那次检测（可能只有 1 人）→ `IndexError`；它发生在 `eval_dual.json` 写完**之后**，所以数字其实已经算对，却被连带作废。现在覆盖图按真正用于打分的人员清单画。
  实测补回一格：`dance_pair02_120_c4_r459.skeleton_only.b_canvas` → persons_detected=1、count_ok=0、regions_ok=1、PCK@0.2=0.500、mpjpe_norm=0.336、identity=0.706、swap=0。tests/ 七个模块全绿。

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

### 门的参照口径修正（2026-10-14 10:25，用今天的代码重跑报告时发现）

重跑 `fit_dual_report.py` 得到 **VERDICT FAIL**：判定项 `hip_gap_error_norm`（拟合 vs **DWPose** 髖距）均值 0.120 > 0.10。逐行查来源，失败全部来自三条 DWPose 自己就错的记录——`backhug_pair02_037`、它的 cross 孪生、`backhug_pair02_039`：DWPose 报两髖心相距 4.2–4.4 px，而拟合与**投影 Hi4D 几何**都报 ~72 px（72.4 vs 73.5、72.3 vs 72.6）。把同一项改用它本该参照的投影几何后，17 对逐对误差 0.001–0.058，**均值 0.021 ≤ 0.10**；DWPose 参照那个值仍在报告里，但降级为 `DIAGNOSTIC` 并注明"分歧可能是向导错"。

这不是放宽阈值：**阈值没动（仍是 0.10），动的是拿什么当参照**，且两个方向的数都同时印出（`GATE PASS hip_gap_error_norm_gt max 0.1 measured 0.021` + `DIAGNOSTIC hip_gap_error_norm=0.120`）。修正后六项全过：correspondence 0.941、person_count 1.000、hip_gap_gt 0.021、torso_scale_ratio 1.096、fit vs DWPose PCK@0.2 0.902（下限 0.859）、mpjpe_norm 0.122（上限 0.132）→ `VERDICT PASS: the fitted skeleton is accurate enough to guide generation`。于是④那句"报告过关才把拟合骨架接进生成链"变成**机械可判定**的，不再靠人工归因兜底。契约测试 `tests/test_fit_gate.py`（GATE 不含 DWPose 项、DIAGNOSTIC 含它、缺投影参照时不编造数值）。复现：`fit_dual_report.py --out fits_recheck2`（写新目录，不覆盖 10-13 那份 blessed 报告）。

## 定标常数：躯干 ÷ 人体框高

共享包里的 `TORSO_TO_PERSON_BOX_HEIGHT = 0.31` 是单人线 `source_torso` 定标用的（期望躯干 = 源人物框高 × 0.31），当年由 5 对单人数据反推。双人线自己的 `source_fit` 不做逐人缩放、用不到它，所以这个常数在双人数据上一直是空白。`torso_ratio.py` 直接量它：A 侧用 pair.json 里的 mask 框 + 在本机 CPU 上跑的 DWPose 关键点，B 侧用 pair 自带关键点 + `geometry.npz` 的语义掩码框；同一人体被多个 run 目录或共用源帧重复量到时只算一次（按 图片哈希+框 去重），原始次数与独立人体数都写出来，明细在 `ratios/torso_ratio.json`：

- **双人线**：68 次测量 → **38 个独立人体**（A 侧 16、B 侧 22），中位数 **0.285**，IQR 0.273–0.300，极值 0.256–0.333；
- **单人线**（同一把尺，但框用流水线自己记录的检测框 `geometry/B_camera.json` 的 `detections.A`）：16 次 → **8 个独立图**，中位 0.353，IQR 0.288–0.443，极值 **0.259–0.524**；
- 两线合并 46 个独立人体：中位 0.288。

读数比第一版更硬也更麻烦。双人侧：0.31 落在 IQR 之外（p75 = 0.300），且掩码框比检测框贴身体、**掩码口径的比值是检测口径的上界**，所以按这类身形 0.31 偏高约 5–9%，用它缩放会把迁移姿态放大同样比例。单人侧更关键：**在它出身的那条线上这个比值根本不是常数**——8 张源帧从 0.259 跨到 0.524，一倍有余。也就是说「A 框高 × 常数」预测躯干在单人线上本来就不成立，双人线只是把它照搬过来。

因此正确的修法不是把 0.31 调成 0.29：A 的躯干可以直接测（仓库里已有 DWPose ONNX 路径，本脚本就是这么量的），应当改成用实测的 A 躯干、只在 A 关键点不可得时退回框高常数。这一改动会动 `src/pose_edit/geometry.py` 的代码指纹、连带让批量中的 PE 阶段全部失效，所以**留到本轮 46 个 run 跑完再做**——对照中途不换尺子。本轮没有改常数。

同一份数据换算成流水线自己的单位才看清严重程度（`sizing_error`：`source_torso` 把迁移后的躯干定为 `0.31 × A框高`，而实测躯干是 `比值 × A框高`，所以人会被画成 `0.31 / 比值` 倍）：**单人线 8 张源帧的倍数中位 0.90、IQR 0.70–1.08、全距 0.59–1.20，只有 2/8 落在 ±10% 内**；双人 38 个独立人体中位 1.09、IQR 1.03–1.14、全距 0.93–1.21，55% 在 ±10% 内。结论分两层：双人式的紧致掩码框上，0.31 只是把人稳定画大约 9%；而在它出身的单人线上，同一个常数能从 0.59× 跳到 1.20×——**这不是"常数取值偏了一点"，是框高这个代理量本身不含躯干的信息**。所以 #21 的形状是"换成实测 A 躯干"，不是"重标定一个更好的常数"。

**归属已经查清（2026-10-14 09:53，读代码而非猜）**：`grep -rn source_torso` 只有 `workflow/run.py:242`、`workflow/smplx_worker.py:85` 与单人线测试引用 `align_to_source_bbox`，`workflow-multiple/` 里一处都没有——**这个常数不在双人线的路径上**，#21 因此是单人线的整改项，它仍然动 `src/pose_edit/geometry.py`（SHARED_CODE），仍然要等双人批量跑完。

**新量到的一条支持这个归属**（直接来自已有 `eval_dual.json` 的 `requested_torso_px` 与 `detected_torso_px`，未新增任何推理）：生成图里逐人的 实测躯干 ÷ 请求躯干 中位数 = `skeleton_only.b_canvas` 0.987 (n=25)、`skeleton_only.source_fit` 0.988 (n=25)、`with_b.b_canvas` 0.993 (n=19)、`with_b.source_fit` 0.970 (n=5)、fit 孪生 0.937 (n=3)。**出图模型基本按请求尺寸画人（±1.3%）**，所以"人被画错大小"这类失败要往请求几何（常数/extent 比）上找，不该记在生成器头上。
复现命令：`python3 size_compliance.py`（只读已有 `eval_dual.json` 的两个躯干字段，零新推理，所以批量在跑也能执行；落盘 `logs_batch/size_compliance.json`，契约测试 `tests/test_size_compliance.py`）。同一份输出给出带宽：±5% 内占比 `skeleton_only.b_canvas` 0.80、`skeleton_only.source_fit` 0.80、`with_b.b_canvas` 0.89、`with_b.source_fit` 1.00、fit 孪生 0.33；±10% 内分别 0.92 / 0.96 / 0.95 / 1.00 / 0.67。注意 fit 孪生只有 3 人，带宽不稳，别当成"拟合向导会让尺寸跑偏"。
