# 结束核验记录

本研究审阅过程中，服务器的 `src/data/tiktok_dataset.py` 和 `src/data/canvas_contract.py` 发生并行修改。9个源文件的起始hash保存在 `remote_depth_audit_v1.json`，结束hash与2项差异在 `remote_source_hashes_after.json`。

重新读取后，TikTok仍读取relative depth PNG并标记`metric=False`；canvas仍使用16倍VAE网格和32像素倍数。iPER读取方向、官方CHAMP实现、权重shape和本次CPU机制验证结论不受这两项变更影响。本研究没有写入服务器的生产源码。

服务器最初投递的 `REPORT.zh-CN.md` 和 `README.md` 保留原稿。本地报告的最终版补充了本条版本说明。研究结果用于确定后续改造边界，未验证图像训练/生成质量。
