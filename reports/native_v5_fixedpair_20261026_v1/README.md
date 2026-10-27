# TikTok单人和Hi4D双人V5固定样本实验结果

本目录汇总两个独立的300步固定样本实验。TikTok训练0265→0374，0126仅观察；Hi4D训练000011→000082击掌，000112非接触帧仅观察。两套训练使用相同V5初始化，不串接权重。

两套均已完成300个成功optimizer update，阶段预算为60／60／120／60，最终checkpoint SHA及提交谱系已核验。最终交付在`final_v3/`：图文HTML内嵌原图并支持点击放大，PDF方便顺序阅读，含178个图片展示位、80张去重后的原始图片。

`input/`保存远端只读导出的图像与SHA证据包。`layout_draft_*`、`final_review_*`、`final_v1/`与`final_qa_*`是排版及复查记录；不要将它们当作最终交付。

报告按输入、Qwen-only／V5初始化／300步、每30步变化、脸手固定ROI、多种子和分支关闭对照组织。所有GT都标注为监督／对照；不进入生成外观条件。当前数据仅能支持单样本拟合与同场景观察，不支持跨人物泛化结论。

上游为服务器单人族`tiktok-fixedpair-v5_20261025_v1`及双人族`hi4d-fixedpair-v5_20261025_v1`。报告工具为`scripts/reports/export_qwen21_fixedpair_report.py`和`build_qwen21_fixedpair_visual_report.py`，只消费既有图片和记录，不启动训练。

TikTok仍有脸手软化与未训练帧手臂误差；Hi4D全图变化较小，但放大可见局部手部红色重影，Local off后有所缓解，不能认定当前V5全面优于Qwen-only。运行期间远端另有可选Source Bank接线修改，严格“当前源码等于开训源码”审计未通过，旧快照与漂移账本均保留，详见报告运行审计。
