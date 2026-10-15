# PE-I2I → Qwen3-VL 任务分支可行性审计

## 目的

核对单人 workflow 是否可以复用 Qwen-Image-2.1 的 Qwen3-VL，通过可关闭的 Prompt LoRA 替代 PE-I2I。读取当前服务器源码、权重文件头及既有运行记录，不运行图像推理或训练。

## 里面有什么

- `audit_readonly.py`：审计快照脚本，只读权重 safetensors 文件头、stage 记录，使用现有 tokenizer 在 CPU 侧统计文本长度；JSON 写到标准输出。
- `audit.json`：2026-10-15 在 sg 执行上述脚本的结果，包含时间、代码 SHA256、模型结构、逐 run 耗时与同一 tokenizer 下的输出长度。
- `quantized_followup_snapshot.json`：00:52 追查 Pocket 的实际 config 与运行中 llama.cpp 的采样参数；仅用于当时状态取证。
- `quantized_gate0.json`、`quantized_gate0_answer.txt`：已有 Q8_0+MTP 首次实验的原始指标与回答副本。
- `quantized_gate1_hard02.json`、`quantized_gate1_hard02_prompt.txt`：00:55 已落盘的 hard_02 两次尝试记录与成功编译的提示词；含重试共 571.7 s，文本通过不等于出图效果通过。
- `quantized_batch_script_snapshot.txt`：正在运行的 gate1 脚本只读副本，用于核对图文顺序、设备和失败统计路径，不应据此覆盖远端脚本。
- `quantized_review_A.png`、`quantized_review_B.png`：hard_02 输入图的只读复制，用于核验动作描述；不是量化模型生成图，不得删除/移动。
- 结论与社区来源见 `../../docs/workflow_PE蒸馏到Qwen3VL任务分支_20261015.md`。

## 复核方法

将脚本用 `scp` 传到服务器独立的 `/data/shangguanrz/` 时间戳目录，以现有环境执行：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python audit_readonly.py \
  --project /home/shangguanrz/project/pic-edit
```

脚本不安装依赖，不加载模型张量到 GPU，不写入 workflow 或模型目录。重新执行时将 JSON 保存到新文件，保留原快照。

## 与既有调研的关系

本次核验：学生 8,767,123,696 参数且有 lm_head；语言层 Attention/MLP rank16 LoRA 预算 43,646,976 参数。23 个三阶段完整 run 的 PE 时间占比中位数为 73.93%；示例的最终 JSON 占同 tokenizer 重新编码后原始输出的 7.15%。这些数字来自文件头和既有日志，不是学生实测效果。

延续 2026-10-14 的 PE 成本及 GGUF 调研，本次关注共享 Qwen3-VL 的训练和部署接口。已有输出的耗时是历史观测，不能当作当前空闲 GPU 上重新测出的基准；学生质量、加速比例、LoRA 关闭后的数值一致性仍须专项实验。

后续追查将成本优先级调整为先完成已有 Q8+MTP 的公平评测。hard_02 首答在三图角色契约上被拒绝，第二次通过，含重试 571.7 s；BF16 与 Q8 都有持物/姿态冲突，尚无量化端到端出图质量结论。详细依据和只读进度检查命令在报告第 11 节。
