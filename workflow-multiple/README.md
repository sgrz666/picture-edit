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

它只测姿态与人际几何。**身份/服装保持尚未测量**（离线机器上没有那套评测用过的 VGG16 权重；DINOv2 权重在位，是候选实现）。检出的点若被生成器藏起来，该点被排除而不算错，所以"藏 limbs"的结果会虚高。

## 实况

17 对可用（12 对 same_identity + 5 对 cross_pair）。两臂对照已在 `hug_pair00_084_c4_r000` 上完成，768 分辨率、30 步、seed 42、`b_canvas` 定标，各一次运行，收尾自检均 `VERIFICATION validated`：

| 臂 | 生成图检出人数 | 人数正确 | body PCK@0.2 | MPJPE（归一化） | 躯干比例 | 髖距相对误差 | PE 耗时 | 生成耗时/峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_b`（四图全给） | 3 | 否 | 0.000 | 0.947 | 1.068 | 1.355 | 259 s | 142 s / 34.04 GiB |
| `hug_skeleton_only`（不给 B） | 2 | 是 | 0.933 | 0.061 | 0.921 | 0.022 | 289 s | 86 s / — |

结论：把 B 原图交给生成器时，模型**没有把 A 的两人摆成拥抱，而是把 B 的那两个人搬进了画面**（人数 3、PCK 0.000、两人距离比请求远 1.35 倍）；只给 A 与两张骨架图时，人数正确、拥抱真的迁移过去且 A 的服装鞋子保留，指标与单人线的良好对处于同一量级。`with_b` 那一行的 `status` 显示 `partial`，是因为该 run 最后一次调用只跑到 geometry（`--until geometry --force`）——阶段产物与评测都还在，状态反映的是最后一次调用而不是最远完成阶段，这一条记在待改清单里。

尚未做的：其余 16 对没有运行；`source_fit` 与 `b_canvas` 的对照尚未跑（只有 hug 一对的 b_canvas 数据）；身份/服装指标未实现；multi-HMR 拟合臂未接入；`report_dual.py` 尚无"每对多臂"的横向对比视图。
