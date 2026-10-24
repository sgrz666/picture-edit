# Native Depth V2 执行计划

用户已批准完整 P0–P3 方案；本人实施，不派 subagent。本轮仅静态核对，不运行测试、模型、训练、渲染、数据导出或迁移。

1. 快照 sg 当前 Body21/Depth V1 与调用点，比较本地工作树和远端源码。
2. 新增 DepthConditionV2、DepthFusionContext，固定 23 几何通道及 29 Gate 上下文；FP32，mask 独立，Root 来自 FK pelvis，尺度来自同 beta 标准身高。
3. 源 Depth 网格构建差分、边界和法线一致性；声明的 asset/canvas 相机变换只应用一次。PNG 关闭 Camera-Z 通道；不伪造米制或置信度。
4. 增加 1/16 Residual CNN、512→256 Mixer、逐位置/通道 Gate 和唯一零初始化输出投影；CHAMP V1、Body21、Native Body 与 Qwen 序列保持原接口。
5. 接通 reader/collate/runtime、推理/训练参数和七组可训练参数；新增 runtime v3/Depth v2/condition v2，显式迁移只继承旧 Encoder，重置融合输出头。
6. 实现但不运行同源 Camera-Z 导出工具。版本化来源落数据盘，不覆盖已有资产。
7. 静态编译、AST、调用链和来源核对，保存文档、命令和未执行的后续验收清单。
