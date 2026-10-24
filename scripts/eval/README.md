# scripts/eval/ —— 正式评测与阶段对照

本目录承载系统的量化评测与多分支对照实验，是训练线（DeepGen V6、Qwen21 Body/Face/Full、joint pilot）产出的「打分与可视化」层。它读取 `experiments/<run>/` 下已生成的 RGB 图像或缓存，在固定 pair、固定随机种子、固定参考契约下计算指标（PSNR、SSIM、PCK、BFS 对照等），并产出可供 `scripts/reports/` 写成 Word/图表的原始结果。多数脚本明确标注「inference-only / 不训练 / 不改权重」，把生成与打分解耦。

## 里面有什么

### iPER 四臂 / V6 / Qwen21 主干评测

| 文件 | 一句话职责 |
|---|---|
| `eval_iper_four_arm.py` | 用同一套 CPU 指标与 pair 评测已保存的 DeepGen/Qwen21 图像（四臂对照） |
| `eval_iper_v6.py` | 在 iPER val/test 上加载 V6 adapter checkpoint，跑 ControlledDeepGenPipeline，算 MAE/MSE/PSNR/SSIM 并出对照条图 |
| `eval_iper_qwen21_body.py` | 在固定 iPER 对照 pair 上生成 Qwen21 adapter OFF/ON 图像 |
| `eval_iper_qwen21_face.py` | 用冻结的 Qwen21 Body checkpoint + 单独训练的 Face 分支做采样 |
| `eval_iper_qwen21_full.py` | 对保留的完整 controller 与 Qwen21 做 30 步完整采样 |
| `multi_stage_eval_v6.py` | 多阶段 checkpoint（Step 500…3000）在同测试 pair 上的推理与演化对照网格，对照 130.28M 基线 adapter |

### Qwen21 BFS（四臂平衡对照）系列

| 文件 | 一句话职责 |
|---|---|
| `eval_qwen21_bfs_rgb.py` | inference-only 的四案 BFS 审计，锁定 A-RGB + B-RGB 参考 |
| `eval_qwen21_bfs_skeleton.py` | locked native A+skeleton 协议的 BFS 对照（先 `--prepare`，再每变体一个进程） |
| `bfs_compare.py` | 在 sg 上做平衡、匹配 prompt 的 BFS 实验（不训练、不生产编辑） |
| `score_qwen21_bfs_rgb.py` | 用不变的交叉人物指标给 12 个 RGB 参考输出打分 |
| `score_qwen21_bfs_skeleton.py` | 给三个固定 BFS skeleton 臂打分，保留旧协议 |
| `build_bfs_comparison_report.py` | 构建已保存图像的 BFS 对照面板（target B 仅用于评测） |
| `summarize_bfs_comparison.py` | BFS 对照汇总 |

### Qwen21 面部修复 / 变体 / 诊断

| 文件 | 一句话职责 |
|---|---|
| `eval_qwen21_face_repair.py` | 用冻结输入渲染并给一个有限 Face 修复 checkpoint 打分 |
| `eval_qwen21_face_cross_person.py` | 定性 A/B 跨人检查：B 只供几何，绝不供 identity 像素 |
| `eval_qwen21_face_references.py` | 配对 Qwen21 Body 20k 探针：source 图带/不带自身人脸裁切 |
| `eval_qwen21_face_variants.py` | 在同 pair 与有效人脸子集上测已保存 Qwen21 Face 变体 |
| `diagnose_qwen21_full_face.py` | 有限 Face 稳定性诊断（不训练、不做最终测试采样） |

### Qwen21 native-value 与组合对照

| 文件 | 一句话职责 |
|---|---|
| `eval_qwen21_native_value.py` | 固定输入的 native/Body/Face 对照，从不训练或改权重；prompt 显式给出，B 的 RGB 从不作 adapter 参考 |
| `score_qwen21_native_value.py` | 固定交叉人物 Qwen native/control 对照的 CPU 评测（B 供姿态，不供 RGB 重建目标） |
| `qwen21_native_value_prompts.py` | 推理-only Qwen value 对照的可审计详细 prompt |
| `qwen21_condition_combinations.py` | 补充因子审计用的显式有序 native 参考（`COMBINATION_REFERENCES`/`NEW_METHODS`） |
| `prepare_qwen21_condition_combinations.py` | 归档旧报告并锁定补充实验，不改旧资产 |
| `evaluate_qwen21_weight_inheritance.py` | 固定继承对照的 CPU 指标，不合成图像 |

### Hi4D 双人接触 / 绑定评测

| 文件 | 一句话职责 |
|---|---|
| `eval_hi4d_native_binding.py` | locked、inference-only 的 Hi4D 人对姿态对应实验 |
| `eval_hi4d_qwen21_dual.py` | 生成并给 native/geometry/full Qwen21 Hi4D 控制打分 |
| `score_hi4d_native_binding.py` | 显式复核 A identity 后评测锁定 Hi4D 实验；检测不按 target 位置分配身份 |
| `eval_hi4d_separate_depth.py` | 锁定、inference-only 的 Hi4D 直连对应实验：分开姿态 ± 联合深度；`--prepare` 锁定输入与提示词，`--validate-only` 跑坐标 / 深度测试；旧实验与权重只读 |
| `eval_hi4d_separate_depth_bilingual.py` | 锁定、inference-only 的 Hi4D 细骨架 / 深度**中英文双语**实验，复用 `eval_hi4d_separate_depth` 与 `eval_hi4d_native_binding` 的模型与 REVISION |
| `score_hi4d_separate_depth.py` | 对六个直连 Hi4D case 做 CPU 检测 + 人工身份分配打分 |
| `score_hi4d_separate_depth_bilingual.py` | 对四个锁定 Hi4D 双语臂做身份复核评测 |
| `eval_hi4d_depth_combinations.py` | 5 种锁定 native 输入组合的英文 inference-only 评测：skeleton / skeleton_joint_depth / joint_depth / separate_depth / joint_separate_depth（默认根 `qwen21_full/hi4d_depth_combinations_20261013`） |
| `score_hi4d_depth_combinations.py` | 对 5 种锁定英文 Hi4D 输入组合做身份复核评测，定义深度 vs 骨架等对照（`CONTRASTS`：individual_depth_vs_skeleton、joint_depth_added_to_individual 等） |

### Qwen21 joint pilot 评测

| 文件 | 一句话职责 |
|---|---|
| `eval_qwen21_joint_pilot.py` | 用单一冻结 Qwen/BFS 基座 + joint Body+Face controller 采样；一次只加载一个 adapter checkpoint 与至多一个 LoRA，B-RGB 仅离线导出 |

## 与上下层的关系

- **上游输入**：`experiments/<run>/`（训练/采样产物）、`scripts/data/` 产出的 pair 与条件缓存、`models/`（DeepGen、Qwen21 各基座）。
- **产出物**：每个评测目录下的结果 JSON/CSV、对照图条、BFS 面板；这些产物被 `scripts/reports/` 的 `write_*_report.py` / `build_*_report.py` 消费成 Word 与 PNG。
- **契约守护**：`scripts/audit/` 的 `verify_qwen21_condition_combinations.py`、`verify_qwen21_full_runtime.py` 等验收这些评测的协议与可复现性。
- 相关索引：父级 [`../README.md`](../README.md)；指标口径由 [`../../tests/README.md`](../../tests/README.md) 的 `test_iper_v6_eval_parity.py`、`test_qwen21_bfs_*.py` 等测试固定。

## 注意与坑

- 评测与生成解耦：大多数 `eval_*` 读取**已保存图像**，不重新训练；但 `eval_iper_qwen21_*`、`eval_qwen21_joint_pilot.py` 等会真正跑推理生成图像，需确认 GPU 空闲。
- 多套「参考契约」（A-RGB、B-RGB、skeleton、native）由 `qwen21_condition_combinations.py` 集中定义，改对照矩阵前先核对 `COMBINATION_REFERENCES`，避免误用 skeleton 输入契约。
- B 的 RGB 在整套评测中**只作评测目标，从不当作 adapter 的参考输入**——这是反复在 docstring 里强调的不变量，回归测试（`test_qwen21_dual_inference_boundary.py`）守护它。
- **竖图评测（2026-10-23 起可用）**：`eval_iper_qwen21_full.py` 的出图尺寸不再写死方形，改从样本声明的画布取
  （`src/data/canvas_contract.sampling_size()`）。跑竖图：
  `PYTHONPATH=. python scripts/eval/eval_iper_qwen21_full.py --pairs datasets/TikTokDataset/TikTok_v3/manifests/pairs/val_pairs.jsonl --dataset-type tiktok_v3 --align-mode aspect_preserving --max-pixels 65536 --modes body --steps 4 --max-samples 1 --cache <新目录> --output <新目录>`。
  复现历史方形结果必须显式 `--align-mode resize --legacy-policy-ok --square-sampling`（三个都要，缺一个就报错）。
  实测对照（pair 0、同 seed、body、4 步）：等比画布出图 192×352、PSNR 12.85 / SSIM 0.588；方形老路 512×512、PSNR 13.24 / SSIM 0.638。
  **别把这两行读成"竖图更差"**：`metrics.metric_resolution=512` 会把 352×192 上采样后再算分，
  而且两档都用方形时代训的 checkpoint（`canvas_policy=None`），等比画布对它是分布外输入。
- 还有 **8 个出图点仍写死方形**，尚未接入 `sampling_size()`：
  `eval_iper_qwen21_body.py:109`、`eval_iper_qwen21_face.py:118`、`eval_qwen21_bfs_skeleton.py:165`、
  `eval_qwen21_face_cross_person.py:89`、`eval_qwen21_face_references.py:107`、`bfs_compare.py:135`、
  `eval_qwen21_native_value.py:430/683`、`../run/run_qwen21_weight_inheritance.py:217`。
  这些脚本的竖图结论在接上之前不要引用。
- `eval_iper_qwen21_full.py` 与多数旧入口**没有 repo-root `sys.path` 兜底**，按路径直接执行会
  `ModuleNotFoundError: No module named 'src'`，必须带 `PYTHONPATH=.`（与 `scripts/train/` 同类问题）。
