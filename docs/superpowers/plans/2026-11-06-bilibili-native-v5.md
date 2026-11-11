# Bilibili Native V5 Implementation Plan

**Goal:** 完成已确认的 80/10/10 全量划分及现用训练接口对齐，提供实测缺项与使用说明。

**Architecture:** 原数据只读；版本化索引落到 `/data/shangguanrz/`。按 clip 整组划分，重用已实现的 Native V5 几何/sidecar 工具；独立保存 test，真实读取器验收替代冒烟训练。

**Tech Stack:** Python JSONL、SHA256、现有 Native V5 数据工具、SSH/SCP。

- [x] 只读 inventory：49 视频、1,960 帧、12,156 对、7,418 个路径全部有效；5 路几何各齐全；torch CUDA 可用，NVML 版本不匹配。
- [x] 划分契约测试：49 个组分为 39/5/5；固定 seed、输入顺序不影响；配对守恒和媒体不跨 split；原文件不覆写。
- [x] 生成划分：train 9,520 对、val 1,490 对、test 1,146 对，全部保留；group 索引和原字段校验通过。
- [ ] Native 对齐：先检查 `experiments/hia_geometry_training_20261029/scripts/hia_training.py` 与服务器消费者；使用现有几何资产装配，缺项单独报告，不伪造就绪状态。
- [ ] 验证与说明：运行必要的数据契约检查和语法检查，下载证据，更新 README 及训练使用命令；不启动冒烟测试。
