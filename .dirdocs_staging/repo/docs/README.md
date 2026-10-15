# docs/ —— 当前规范、实验结论文档与目录整理记录
> 工作区路径：/home/shangguanrz/project/pic-edit/docs　|　本级：2 个子目录（`experiments/`、`superpowers/`）/ 根级实体文档 12 份（`ls -la` 2026-10-12 实测；本级体积约 275 KB，`du -sh --si`）　|　另含生成物 README.md、目录说明.md

## 一、这一层在干什么
项目的"纸质决策层"：架构说明、训练/复现 runbook、带日期的实验结论、目录整理执行记录与架构审查。区别于 `reports/`（交付 Word）与 `experiments/*/report.md`（单 run 记录）：这里写的是跨 run 的规范与结论，工程师接手项目先读这里。

## 二、层级地图（按工作线归类根级 12 份文档）
| 工作线 | 文档 | 记录内容与关键结论（读正文摘出） |
|---|---|---|
| 架构 | `adapter_v6_architecture.md` | DeepGen V6.3 动态控制接口规格：静态 prepare + 每去噪步双分支残差、6 组 zero-head、注入块 0-3/4-7/8-11/12-15/16-19/20-22（block 23 不注入）、零初始化时 Adapter 与无 Adapter 输出差 <1e-6；冒烟命令三条 |
| 训练计划 | `iper_v6_training_plan.md` | V6.3 iPER 方案 A 四阶段计划：227.98M 参数=169.89M 可训练+58.09M 冻结；4 组分层 lr；人体掩码加权 flow loss（w_human=3.0）；训练峰值显存 16.50 GiB；checkpoint 上限 9.2 GB；全量 20,000 步配置命令 |
| Qwen21 线 | `qwen21_hi4d_dual.md`（Hi4D 双人入口/两阶段 lr/822 对全量验证）、`tiktok_qwen21_full_training.md`（TikTok→Qwen Full 冒烟与 pilot 全流程；严格池 8,574/1,120/1,156 对；Face lr 教训：1e-4/5e-5/1e-5 两个样本更新即把固定噪声 loss 从 0.437 推到 2.792） | — |
| CHAMP 线 | `champ_deepgen_single_sample_runbook.md` | 单样本数据契约（8 通道 Normal+Depth+Pose+Semantic）、`patches/deepgen_pose_control.patch` 应用/回滚命令、无控制消融产物清单与有效性判据（std≥5.0、近白近黑≤0.95） |
| BFS 对照 | `BFS_面部保持移用评估_20261004.md`（六对初评 + LoRA 权重兼容核验：V1.1 转换后 384 张量/192 模块、gate-first 拆分误差 0）、`BFS_扩大样本对照_20261004.md`（30 对/8 人物/102 次生成；BFS V1.1 对 A 身份 cosine 0.4422 低于基干 0.5210、PCK 高但均低于保留 Body C；区域外 MAE 4.35 vs 3.13——结论：BFS 与基干各有取舍，不能宣称全面更优） | 两文互为续篇，后者为准 |
| 目录治理 | `PROJECT_STRUCTURE_20261003.md`（工作线-目录职责-大资产位置总览）、`目录清理执行记录_20261003.md`（62+31 入口归类、7 项大资产冷归档 4,764 文件/157.5 GiB、根分区 95%→80%、246+188 测试验收）、`目录清理执行记录_20261009.md`（二轮：撤 54 个兼容链接留 8 个、清 34 个缓存、验收 passed）、`架构审查_20261012.md`（全仓只读审查：P0 git 三处分叉、pytest 收集 7 错、NVML 驱动不匹配；P1 v6 与 qwen21_full 47 文件字节级 fork、入口迁移半路；P2 无打包 229 处 sys.path hack） | 架构审查直接催生了 10-12 的 scripts/ 实体化迁移 |

## 三、子目录
- `experiments/`：两份 BFS 联合试点交接文档，见其 README。
- `superpowers/`：AI 协作工作流的计划（plans/）与设计（specs/）档案，见其 README。

## 四、注意事项
- `README.md` 与 `目录说明.md` 当前是 `gen_folder_docs.py` 生成物，非手写。
- 带日期文档是时点快照：10-03 结构文档中"根目录实体+分类链接"布局已被 10-12 迁移取代，读治理类文档时以日期最新者（10-09 记录、10-12 审查）为准。
- 复现命令均假定在仓库根执行、解释器用文档内绝对路径。
