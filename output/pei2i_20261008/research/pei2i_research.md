# 研究快照 · Qwen-Image-2.1-PE-I2I

## 权威来源
- 模型卡：https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I
- 姊妹模型（文生图版）：https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I
- 官方代码仓库：Qwen-Image-2.1 仓库 `prompt_rewrite/` 目录（含 transformers / vLLM / vLLM server 三套入口）
- 第三方整理：https://comfyui-wiki.com/zh/news/2026-09-22-qwen-image-2-1-prompt-enhancer

## 关键事实
| 项 | 值 |
|---|---|
| 模型标识 | `Qwen/Qwen-Image-2.1-PE-I2I` |
| 底座架构 | 微调自 **Qwen3.5-VL 9B**（`AutoModelForImageTextToText`） |
| 参数规模 | 9B，BF16 |
| 发布时点 | 2026-09-20，与 Qwen-Image-2.1 基座同期 |
| 参考图上限 | 最多 10 张，指令中以 `<image1>` / `<image2>` … 引用 |
| 官方推荐 | README 明确推荐将其作为准备提示词的默认方式 |
| 许可 | Qwen Research License |

## 官方采样参数（两模型不共用，勿混用）
| 参数 | PE-I2I | PE-T2I |
|---|---|---|
| `max_new_tokens` | 24000 | 16256 |
| `temperature` | 1.0 | 1.0 |
| `top_p` / `top_k` | 0.95 / 20 | 0.95 / 20 |
| `presence_penalty` | **不设** | **1.5**（抑制重复） |
| `enable_thinking` | 开启 | 开启 |

## 输出 JSON 结构
```json
{ "rewritten_prompt": "<精确编辑提示词>",
  "wh_ratio": "",
  "ratio_follow": "<image1>" }
```
`wh_ratio` 与 `ratio_follow` **互斥**，有且仅有一个非空。
解析时需先切分思考过程：`thinking, _, answer = text.partition("</think")`。

## 本地运行事实（本次实测）
- 权重落盘 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I`，18.84 GB，15/15 文件与 HF 元数据一致
- `qwen21` 环境（transformers 5.17.0）直接可加载，无需新建环境；加载耗时 78 s
- 推理耗时：中文 80 s / 英文 154 s
- 环境缺 `flash-linear-attention`、`causal_conv1d` 优化内核，回退到参考实现，速度受影响
