# A 实测尺度与骨盆锚点：真实生成对照

2026-10-13，已在 sg 完成四个坏例、一个较好例对照，以及 hard_02 的单句提示词变体，共 **6 张实际 Qwen 生成图**。

结果目录：`/home/shangguanrz/project/pic-edit/workflow/runs/scale_anchor_20261013/`。本目录是独立实验脚本与报告；生产工作流和原来的七组结果没有改写。

## 本次改变了什么

第一轮五组只改变骨架条件，原 A、画布预处理、生成提示词全文、模型、seed 42、30 步、resolution 768、true CFG 1、KV cache 关闭均保持与各自基线相同；没有 adapter/LoRA，没有重新调用 PE。

骨架用 A 的现有 CPU DWPose 测量定标。两边统一取“肩中点→髋中点”，避免将 DWPose 的合成肩中点与 SMPL-X 的颈部关节当成同一点：

```text
scale = A肩中点到髋中点长度 / 原骨架肩中点到髋中点长度
新骨架 = A髋中点 + scale × (原骨架 − 原骨架髋中点)
```

画外骨段只在绘制时裁剪，关节坐标不挪回画面。因此近景 A 不再被缩小来容纳 B 的完整腿。这仍是**二维统一缩放与锚定**：B 的各骨段比例仍保留，没有实现 A 形体参数和逐骨段三维重定向，也没有恢复真实标定相机。

肩/髋四个 A 锚点置信度均需至少 0.3。四项测试覆盖肩中点定义、相对姿态/NaN 保持、画外关节保留、低置信度拒绝以及骨段跨边界裁剪，已红→绿验证并在 sg 通过。

## 实際生成结果

下表为同一个 DWPose 模型测量的肩中点到髋中点二维长度，除以 A 的对应长度。它会受姿态、朝向和检测误差影响，不是人体真实尺寸或身份保持分数。

| 结果目录名 | 原结果/A | 新结果/A | 人工观察 |
| --- | ---: | ---: | --- |
| `hard_01` | 0.582 | 0.964 | 恢复近景体格，交叉双臂；原来硬挤进画面的整条腿消失，背景人物仍在 |
| `hard_02` | 0.498 | 0.952 | 缩小问题改善，但原 PE 提示词冲突使图里出现三只手；保留这张失败证据 |
| `hard_04` | 0.636 | 0.971 | 上身与头部尺度接近 A，双臂举起，恢复截断构图 |
| `hard_05` | 1.002¹ | 0.913 | 胸腹前的小人物消失，转身效果仍主要是侧身，不能认定严格复现 B 的背身姿态 |
| `medium_02` | 0.887 | 0.854 | 没有全面改善；面部投影高度更接近 A，但躯干长度偏差略增，成图下方裁剪也改变 |
| `hard_02_pose_consistent` | 0.498 | 0.912 | 使用与 hard_02 新版相同的骨架，仅修一条提示词；两只手交叉并握住一支麦克风，多余手臂消失 |

¹ hard_05 原测量对应仍保留的大人物，因此无法反映额外的小人物，不能只按此比值判断质量。

头部投影高度的独立参考：hard_01 的脸部轮廓高度/A 从 0.592 到 0.918，hard_04 从 0.565 到 0.991；hard_02 单句修正版从 0.504 到 0.921。头部朝向和关键点误差同样影响这些数值。

这轮结果支持“原骨架的尺度/定位错误导致近景人物缩小”的判断。它没有证明二维统一定标对所有动作都更好：medium_02 的对照、hard_05 的背身动作与剩余身份/手指细节，均需要继续检查。

## hard_02 为什么额外跑了一版

原提示词同时要求“双臂交叉、双手在对侧上臂附近”和“原来举起的手与麦克风保持嘴边位置”，与新骨架存在冲突。第一轮严格冻结提示词，所以保留了多余手的失败结果。

`hard_02_pose_consistent` 使用完全相同的 A、骨架、预处理和采样设置，只把该麦克风句子改成：一支麦克风由一只交叉的手握住，随手移到胸前，明确总共两只手和两条手臂。保留物品的外观，不继续固定与目标动作冲突的原手位置。替换前后全文与哈希记录在该 run 的 `prompt/generation_prompt.json`。

## 看图与复查

- `comparison_1.jpg`：hard_01/hard_02 的纯几何对照，包含三只手的失败结果。
- `comparison_2.jpg`：hard_04/hard_05 的纯几何对照。
- `comparison_3.jpg`：medium_02 对照。
- `recommended_comparison_1.jpg`：hard_01 与 hard_02 单句修正版；后者同时包含几何和提示词修正，不能当作纯几何对照。
- `microphone_prompt_comparison.jpg`：A、原提示词的近景生成、单句修正版，直接比较麦克风/手臂冲突。
- `measurements_summary.json`：六组 A/原生成/新生成的关节测量、置信度与检出人数。

在 sg 执行：

```bash
cd /home/shangguanrz/project/pic-edit/workflow/diagnostics/2026-10-13-scale-generation
QWENPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$QWENPY run_experiment.py --verify
$QWENPY run_prompt_variant.py --verify
$QWENPY -B -m unittest test_geometry_ablation -v
```

六张图的生成证据与输入/输出哈希均已验证通过。`run_experiment.py` 和 `run_prompt_variant.py` 不加 `--verify` 时会生成尚未完成的对应实验，已完成且指纹一致时复用缓存。重新测量和构建比较图：

```bash
$QWENPY analyze_results.py --cases hard_01 hard_02 hard_04 hard_05 medium_02 hard_02_pose_consistent
```

每个结果在 `generation/final_image.png`；完整画布为 `generation/final_canvas.png`。`geometry/operation.json` 保存实际缩放系数、源锚点、置信度与画内点数；`generation/request.json`、`stage.json`、`generation.json` 保存模型、采样、实际参考图、提示词及生成产物证据；`experiment.json` 区分这个实验与普通端到端 PE 流程。

首次 hard_02 模型加载遇到 CUDA/NVML 错误，尚未采样，日志保留为 `generation/failed_load_attempt_1.log`。逐组件诊断确认 BF16 模型可加载后，原配置重试成功；没有更改驱动、模型精度、显存分配器或采样参数。[PyTorch 的错误报告路径](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/c10/cuda/CUDACachingAllocator.cpp)说明 NVML 初始化也可能在显存分配失败的报告过程中触发，不能仅凭该断言判断模型代码本身损坏。
