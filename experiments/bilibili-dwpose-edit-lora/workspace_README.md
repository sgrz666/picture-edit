# LoRA 实验工作区

从官方 Diffusers Edit LoRA 的单图训练流程出发，验证人物参考图 A、目标骨架和 Prompt 到配对 B RGB 的训练。首轮为 Bilibili 单人原始 DWPose、统一短指令、32 对训练及 10 对验证，基座/VLM/VAE 冻结，仅训练 DiT LoRA。

- `source/diffusers/` 保存官方 commit `0905b667fe3f37f43b6ef9c6e05ad20636206938` 的未修改脚本、训练 README、Pipeline、许可证及版本记录。
- `experiments/bilibili-dwpose-edit-lora/` 是首轮实验。具体配置、完整命令、检查方法和报告入口见该目录 `README.md`。

数据和模型本体沿用服务器原路径。新索引、预处理副本、缓存、检查点、图片与日志全部保存在本工作区；每轮使用独立名字。

本轮未接入 Native V5；与已在运行的 V5 训练并行。可用接口不等于训练有效，结论以实际对照指标和图像为准。
