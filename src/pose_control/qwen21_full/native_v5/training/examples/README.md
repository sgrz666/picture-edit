# 准备阶段配置示例

四个阶段模板都没有预算和资产绑定，只能`prepare`。将Stage配置与真实Sources、初始化、模型manifest、冻结缓存及报告绑定后，才可执行后续preflight／train。模板均未运行。

训练Sources示例字段：`source_id`、`role=train|validation`、`manifest:{path,sha256}`、`groups:{path,sha256}`、`data_profile=full_smplx|hybrid_hi4d|auto`、`scene_kind=single|dual|any`、`weight`。多训练源明确给weight；groups文件覆盖该manifest每个pair_id并声明原始视频／序列group。禁止CHAMP无RGB动作例进入训练。

`body_report.example.json`只是门禁格式，默认false不是验收结果，不能改布尔值冒充真实报告。真实grad报告由独立的grad preflight生成，不提供伪passed示例。
