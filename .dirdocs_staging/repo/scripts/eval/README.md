# scripts/eval/ —— 正式评测与对照实验入口（采样、打分、诊断）
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/eval　|　本级：0 个子目录 / 28 个实体文件（约 1.1 MB，`du -sh --si` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md、`__pycache__`

## 一、这一层在干什么
把 `scripts/train/` 产出的 checkpoint 变成可比对的定量证据：对固定配对集采样、计算身份 cosine / PCK / PSNR / SSIM / MAE、跑多方法同底座对照。绝大多数入口遵守同一纪律：B 的真实 RGB 只做离线评测不进生成条件、评测与训练分离、`--score-only` 允许对已存图片重评不重采样。

## 二、层级地图（按实验族分类）
| 实验族 | 文件 | 实证（默认输出/关键设置） |
|---|---|---|
| iPER × Qwen21 分支评测 | `eval_iper_qwen21_body.py`（Body ON/OFF 对比）、`eval_iper_qwen21_face.py`（冻结 Body + 独立 Face 支路）、`eval_iper_qwen21_full.py`（完整版控制器 30 步全采样）、`eval_iper_four_arm.py`（DeepGen 与 Qwen21 已存图片同指标对比） | 配对与资产默认 `datasets/iPER/iper_sampled_6src64tgt`、`iper_assets_512_nvdiffrast` |
| DeepGen V6 线 | `eval_iper_v6.py`（MAE/MSE/PSNR/SSIM + 四栏对比图；默认 pairs `dev_val_pairs.jsonl`、输出 `experiments/iper_v6_eval_results`、条件缓存复用 `experiments/iper_adapter_finetune_v1/condition_cache`）、`multi_stage_eval_v6.py`（Step 500–3000 中间 checkpoint 同种子横评） | — |
| Hi4D 双人 | `eval_hi4d_qwen21_dual.py`（native/geometry/full 三组；默认对 606 条测试对采样到 `experiments/qwen21_hi4d_dual/test_eval`；开发用验证集须显式 `--allow-val`，见 `docs/qwen21_hi4d_dual.md`）、`eval_hi4d_native_binding.py` + `score_hi4d_native_binding.py`（锁定推理-only 的 person↔pose 对应实验；打分前要求人工逐图审身份） | — |
| BFS（外部 head/body-swap LoRA）对照 | `bfs_compare.py`（30 对/8 人物/10 衣着组两阶段对照）、`summarize_bfs_comparison.py`（按 actor 聚类的配对统计）、`build_bfs_comparison_report.py`（面板）、`eval_qwen21_bfs_skeleton.py` + `score_qwen21_bfs_skeleton.py`（三臂骨架协议）、`eval_qwen21_bfs_rgb.py` + `score_qwen21_bfs_rgb.py`（A RGB + B RGB 四例） | 复现命令在 `docs/BFS_扩大样本对照_20261004.md` 末尾；结论：BFS V1.1 身份对 A cosine 0.4422 vs 基干 0.5210，头部朝向更贴中间图 C |
| Qwen 价值审计（native value） | `eval_qwen21_native_value.py` + `score_qwen21_native_value.py` + `qwen21_native_value_prompts.py`（固定输入 native/Body/Face 对比）、`prepare_qwen21_condition_combinations.py` + `qwen21_condition_combinations.py` + 归档旧报告的因子组合审计 | 与 `audit/check_qwen21_native_value_prediction.py`、`audit/verify_qwen21_condition_combinations.py` 配对使用 |
| Face 专项 | `eval_qwen21_face_variants.py`、`eval_qwen21_face_references.py`（Body20k 上源图带/不带自家脸裁片的配对探测）、`eval_qwen21_face_cross_person.py`（B 只供几何的跨人定性 A/B）、`eval_qwen21_face_repair.py`（冻结输入下渲染打分一个修复 checkpoint）、`diagnose_qwen21_full_face.py`（有界稳定性诊断） | — |
| 联合试点 | `eval_qwen21_joint_pilot.py`（单 checkpoint + ≤1 LoRA；`--score-only` 重评 24 TikTok + 12 iPER 锁定集，见 `docs/experiments/`） | — |
| 权重继承 | `evaluate_qwen21_weight_inheritance.py`（CPU 指标，无合成） | 配合 `weights/inherit_qwen21_full_weights.py`、`run/run_qwen21_weight_inheritance.py` |

## 三、关键文件与实证内容
- `bfs_compare.py`：`--per-identity --stage2-resolution --steps --seed` 全锁定，`--self-test` 可 CPU 自检；历史 102 次生成（30 Body + 24×3 修复）结论见上表引用文档。
- `eval_qwen21_joint_pilot.py`（36 KB）：图像路径进记录、导出 B RGB 仅供离线比较；配合 `reports/summarize_qwen21_joint_pilot.py` 出交接文档里的指标表。
- `eval_iper_v6.py`：docstring 即指标定义（MAE/MSE/PSNR/SSIM + comparison strip）。

## 四、注意事项 / 调用路径
- Qwen 线一律用 `/home/shangguanrz/nihaochen/envs/qwen21/bin/python`；V6/DeepGen 线用 `/home/shangguanrz/miniconda3/envs/deepgen/bin/python`（见 `docs/adapter_v6_architecture.md` 的验证命令）。
- `experiments/iper_adapter_finetune_v1/condition_cache` 是多个评测共享的 VLM 条件缓存，不可删。
- 根目录软链 `scripts/eval_qwen21_face_repair.py → eval/eval_qwen21_face_repair.py` 是历史兼容入口，实体在本目录。
- 多个脚本要求"先 `--prepare` 一次，再一个 variant 一个进程"（如 bfs_skeleton），并行复用缓存会互相踩。
