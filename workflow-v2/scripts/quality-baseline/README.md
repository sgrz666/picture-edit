# A+B+C 正式质量基线

固定旧 Workflow 的 A/B 与站立、握手、拥抱 C，比较 Qwen-only、Native Full、Body + Interaction、Full 关闭 Source Detail。此工具不训练模型，也不修改重定向算法。

## 使用

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit/workflow-v2
bash scripts/quality-baseline/script/run.sh --campaign quality_baseline_20261028_v1 --phase all
bash scripts/quality-baseline/script/run.sh --campaign quality_baseline_20261028_v1 --verify
```

`--phase smoke|formal|swap|diagnose|report` 可分阶段执行；同一命令重启时先只读验证已完成任务，再继续缺失任务。`formal` 使用 768/30 和 seeds 42、43、44。`smoke` 使用 512/12、seed 42。若分阶段执行，先完成 smoke，再 formal、swap、diagnose、report。

总计 68 张：Smoke 12、正式主矩阵 36、来源/角色交换 4、诊断 16。诊断比较握手/拥抱的 Body-only、关闭 Hand、关闭 Face，以及拥抱 Qwen-only 仅改 Prompt 或仅改单色骨架。诊断不混入正式主矩阵统计。

产物位于 `/data/shangguanrz/pic-edit-workflow-v2/validation/<campaign>/`：

- `freeze.json`：实际代码、模型、输入、checkpoint 与固定矩阵。首次完整计算 SHA，任务之间检查源码内容和资产文件状态，最终 `--verify` 再完整检查 SHA。
- `jobs/`、`logs/`、`progress.json`：独立任务状态、命令、追加式运行日志与进度。失败立即停止；重试保留原失败日志。
- `runs/`：各动作与分辨率的共享几何/Prompt，以及同级、独立的生成与评测阶段。
- 各 run 的 `appearance_cache/`：按来源 RGB、mask、Canvas、几何和模型 SHA 绑定的脸手外观特征，复用前验证 payload SHA；不替代实际 Qwen Source64 编码。
- `paired_metrics.json`、`metrics.csv`、`summary.json`、`audit.json`：成对数值与证据审计。
- `report/index.html`、`report/报告.md`：整图浏览和汇总；各 `evaluation/` 中保存脸手放大图。

## 单次消融

原 `run.sh` 增加 `--variant full|body_interaction|no_source_detail|body_only|no_hand|no_face` 与 `--generation-tag`。例如在同一几何 run 下，使用不同 tag 保存种子与消融：

```bash
bash run.sh --source-a /path/A.jpg --source-b /path/B.jpg --interaction /path/C.jpg \
  --name abc_quality --mode native_v5 --checkpoint /path/pinned_v5.pt \
  --variant no_source_detail --generation-tag s42_no_source_detail --observe-native
```

`--observe-native` 保存 `native_residuals.json`，只有标量，不保留完整 Hidden。Body/Interaction/Face/Hand 的 RMS 来自实际注入残差；局部 identity/texture/geometry/detail 的 RMS 位于共享输出投影之前，二者不可直接比较绝对大小。记录同时包含控制参数、条件有效性和两份来源绑定。

不指定 variant/tag 时保持默认 Full 和原目录路径。非 Full variant 未指定 tag 时自动使用 variant 名称。所有不同生成阶段均为同级目录，不向已校验的 Full 目录新增子目录。`--diagnostic photo_prompt|monochrome_guides` 必须给独立 tag，仅用于明确标记的诊断。

Source Detail 与 Depth 的逐人细节编码是两个不同控制项：`no_source_detail` 仅关闭前者。关闭 Face/Hand 会同时停用它们承载的 Source Detail，因此完整 Local-off 等价于 Body + Interaction，不重复计算为独立路线。

## 判读边界

固定 checkpoint SHA256 为 `6a340c0a9bde9fbea943bfb26cbfce425a893ac2201d1b23651e0508e4bd20e2`。训练来源数量从 checkpoint 读取；本轮推理使用两个独立 Source。

站立、拥抱优先使用 Hi4D 官方 SMPL/接触与火柴人骨架；官方数据没有细手指姿态。握手照片只有上半身，估计出的腿部不是标注真值。RGB 接触指标仅为接近可检测关节的二维代理，不代表三维接触。

背景评测排除画布留边及原人物/目标人物区域。颜色比例与锐度仅提示复查，不能单独判定骨架泄漏或手部质量。检测不到、遮挡、区域重复检测均保留不确定性。没有配对输出 D，不计算伪造的输出真值指标。

冻结代码或资产变化时换 campaign 名称；不在既有批次上混入修复结果。CUDA 分配检查独立于 NVML；不因 NVML 告警自动降级或安装驱动。

批次通过 `--asset-snapshot` 与其固定 SHA 向 runner 传递已经完整计算的模型指纹；每个阶段仍记录同样的模型内容 SHA，并检查文件清单、大小与修改时间。输入图片和阶段产物继续实际计算 SHA。最终批次 `--verify` 在逐任务校验前后重新完整读取模型文件计算 SHA；普通 `run.sh --verify` 不使用该加速路径。

## 阶段 1 正式质量基线 V2（2026-10-30）

本轮完成 68/68：Smoke12、768/30 三种子主矩阵36、交换4、诊断16；全资产／阶段只读复验通过。独立重建16个目标，最大mesh误差7.63e-6m、自投影0.000182px，满足1e-4m/.25px契约；源码回归107通过、2跳过（另8个subtests通过）。没有新训练。

**质量尚未达标。** Full共同可测人物ArcFace在9组中均低于Qwen-only，中位差-0.1294；覆盖下界PCK总体中位+2.78个百分点，剔除匹配歧义组后约0。Hand-off／Face-off诊断分别改善手部重复轮廓／脸部软化，Source Detail有明确负面影响但并非全部原因。另有原人物残留、骨架泄漏、拥抱Prompt动作冲突及接触几何残差；不能把工程通过当成质量通过。

完整结果：`/data/shangguanrz/pic-edit-workflow-v2/validation/quality_baseline_20261029_v2/`。先看`report/质量分析与验收.md`、`report/index.html`；数值与歧义分层在`analysis/paired_quality.json`，实际校验记录在`final_checks/`。

共享主干在长运行中被其他工作继续修改，V2最终使用该批次归档的精确源码运行快照，保留主干现状。固定批次复验：

```bash
cd /data/shangguanrz/pic-edit-workflow-v2/validation/quality_baseline_20261029_v2/runtime_snapshot/workflow-v2
bash scripts/quality-baseline/script/run.sh --campaign quality_baseline_20261029_v2 --verify
```

新实验仍从主干统一入口运行并使用全新名称；当前主干拒绝旧代码指纹属于预期行为，不要改SHA或覆盖旧图来绕过校验。V1仅保留历史材料，不混入V2生成统计。V2站立geometry/Prompt只读复用的依赖记录在`parent_campaign.json`；调度将缓存别名正规化到相同的物理缓存目录，并继续逐项检查SHA，未更改冻结推理代码。
