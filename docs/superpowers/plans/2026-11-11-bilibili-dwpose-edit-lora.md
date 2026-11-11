# Bilibili 双图 Edit LoRA 已确认实施计划

用户于 2026-11-11 确认实施；全部由主智能体独立完成，不使用 subagent。

- 输入 `[A RGB, B 原始 DWPose 骨架]` 与统一短指令，监督 B RGB。冻结基座/VLM/VAE，只训练 DiT attention Q/K/V/Out 的 rank/alpha=16 LoRA。
- 官方 Diffusers commit `0905b667fe3f37f43b6ef9c6e05ad20636206938` 的脚本、README、Pipeline 和许可证保留原样；适配代码位于独立实验目录。维持服务器 `qwen21` 环境。
- 数据保持 9,520/1,490/1,146 对原视频隔离划分。画布 384×672，原始有效点标记，无随机裁剪/翻转；固定 8 个训练视频各 4 个同 A 不同 B 配对，5 个验证视频各 2 对。
- 冻结编码结果落盘并校验图像、文本、预处理、模型、源码与清单身份。双图 Latents 在前、noisy B 在后，仅预测 B 的 Flow Matching velocity。
- seed=42、BF16、gradient checkpointing、batch=1、AdamW、lr=5e-5、前 10 步 warmup，共 300 次更新，保存 100/200/300 步。
- 用户选择与当前 V5 并行。先验证缓存、真实反向和原生推理，预留至少 8 GiB 显存；失败保留记录并停止本轮，不静默改变量。
- CPU 契约、真实梯度、LoRA 更新、保存重载及第 100 步恢复验收。对照同输入 baseline/LoRA，并做同 A 换骨架与空白骨架诊断。报告 PCK、人脸检测成功数与身份相似度、loss、显存/RAM 峰值及可复跑命令。

完整训练、Native V5 接入不在本轮范围。任务完成要求实验确实运行并报告正面或负面结果，不能把接口跑通解释为生成效果改善。
