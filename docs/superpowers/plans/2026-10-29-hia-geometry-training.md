# HIA 几何补全与训练接口实施计划

**目标**：实测 HIA/TikTok 可用规模，补 HIA 人体几何，并让最新 Native V5 读取真实样本通过数据检查。用户已授权执行，未要求启动长训练。

**方案**：保留发布版 RGB、2D pose、视频归属与质量门。独立生成原生画布人体 SMPL-X、camera、3D DWPose、投影骨架、metric depth、法线、mesh mask 与部位标签。深度/法线只描述人体，不宣称背景或真值；单目估计与显示归一化分开保存。生成失败或几何未完成的帧不能伪造条件入训。

- [x] 从现盘 JSONL 重算 HIA 主集/strict 与 TikTok usable/strict pairs，区分原始帧、筛选帧、配对和实际几何覆盖。报告在实验 `report/data_comparison.json`。
- [ ] 复用既有 TikTok/SMPLest-X 流程，新增实验私有批处理；模型保持常驻，先真实样本测试，再以有界批量完成补全。输出绑定 RGB SHA、模型/代码 SHA、相机约定与原生尺寸；失败单独记账、支持恢复；不删移媒体、不覆盖旧版本。
- [x] 扩展 Native V5 数据准备接口，输出 Source/Target 条件、视频 group 与显式区域有效性。出画/遮挡手不能因新几何重建而恢复监督；保留原始 2D 和估计 3D 的来源区别。三 split 样本已通过真实 Native 数据读取和 consumer/loss 检查。
- [ ] 运行契约测试、远端 py_compile、真实几何 smoke、数据装配/预检；全量完成前不宣称训练就绪。
- [x] 写中文 README、数量对比和可复现运行/验证命令。训练预算未确定，只做训练接口和预检，不启动长训练。

2026-10-29 实施状态：几何与 Native V5 接口均已实现；三 split 各一对真实估计/装配和 Native V5 数据预检通过，仍须等待全量。代码测试本地 25 passed、1 skipped（本地未同步远端 Native loss 模块），远端真实 loss 消费验证另做。不把子集 smoke 写成全量验收。全量后台作业为 `HIA_geometry_v1_20261029/jobs/20261029T151254Z_df4ba96b`，依次生成几何、SHA 检查、全量 typed cache、DataPlan 预检；全量步骤尚未完成，因此以上相关复选框保持未完成。

2026-10-31 增补：全量几何阶段结束于 22,263 完成 + 13 拒检。经 `experiments/hia_geometry_repair_20261031/` 逐帧诊断，6 帧为上游 SMPLest-X→ultralytics 的 YOLO 输入通道序误杀（BGR 修正后 IoU 0.78–0.99），2 帧无检出走 fallback，4 帧检测碎裂/主体漏检判定 primary，1 帧（017323_000023）因 LED 大屏翻拍污染排除。修复在独立签名副本（`df6f8f5c`）完成 12/12 帧并零改动冻结根，账目闭合至 22,275 完成 + 1 排除 + 0 待处理；全量 typed cache（train 33,650+/41,902 pairs）与 DataPlan 预检仍未完成，相关复选框保持未完成，训练就绪结论继续推迟。

职责：geometry worker 独占新 `hia_geometry.py` 与相应测试；training worker 独占新 `hia_training.py` 与相应测试；主任务负责报告、作业运行、集成验证和 README。共享改动先沟通，不撤销其他人的编辑。
