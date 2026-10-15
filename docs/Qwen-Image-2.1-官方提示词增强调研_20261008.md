# Qwen-Image 2.1 官方提示词增强（Prompt Enhancer）调研

**日期**：2026-10-08
**动机**：本项目的实验反复显示**提示词详细程度对身份/服装/背景保持影响极大**——详细版六段式提示词明显压制住了"抄参考图 B"的倾向。因此需要接入官方的提示词重写方案。
**结论**：官方确实提供了**两个专用重写模型**，与 Qwen-Image 2.1 基座**同期发布（2026-09-20）**，且官方 README 明确推荐把它们作为准备提示词的**默认方式**。

> 官方原话：「为获得最佳效果，我们推荐使用官方提示词重写模型，把简短提示词展开为详尽、高质量的描述。」

---

## 1. 两个模型，各司其职

| 模型 | HF ID | 用途 | 输入 | 输出 |
|---|---|---|---|---|
| **PE-T2I** | `Qwen/Qwen-Image-2.1-PE-T2I` | 文生图需求展开 | 任意语言的简短需求 | 长英文提示词 + 推荐宽高比 |
| **PE-I2I** | `Qwen/Qwen-Image-2.1-PE-I2I` | **编辑指令展开** | 一条编辑指令 + **最多 10 张参考图** | 精确编辑提示词 + 宽高比策略 |

**共同底座**：微调自 **Qwen3.5-VL 9B**（`AutoModelForImageTextToText`），BF16。

### 对我们的价值判断

**`PE-I2I` 正是我们 16-test 场景需要的那个。** 它的设计目标就是「多图输入 + 编辑指令 → 精确提示词」，完全对得上我们的「A 图身份 + B 图姿态」任务。

`PE-T2I` 对本项目用处较小（我们不是文生图）。

---

## 2. PE-I2I 的精确用法

### 2.1 提示词格式

用 `apply_chat_template`，messages 含 `system` + `user` 两角色：

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},  # 模型自带的 system_prompt.txt
    {"role": "user", "content": [
        {"type": "image", "image": images[0]},      # <image1>
        {"type": "image", "image": images[1]},      # <image2>
        {"type": "text",  "text": "Place <image1>'s subject into <image2>'s scene"},
    ]},
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt", enable_thinking=True,   # 必须开 thinking
).to(model.device)
```

**关键点**：参考图在指令中用 **`<image1>` / `<image2>` …** 顺序引用（最多 10 张）。

对我们的场景，指令可以写成：
> 「把 `<image1>` 中人物的姿态改为 `<image2>` 中人物的姿态，严格保留 `<image1>` 的身份、服装与背景」

### 2.2 采样参数（官方指定，不要改）

| 参数 | PE-I2I | PE-T2I |
|---|---|---|
| `max_new_tokens` | **24000** | 16256 |
| `temperature` | 1.0 | 1.0 |
| `top_p` | 0.95 | 0.95 |
| `top_k` | 20 | 20 |
| `presence_penalty` | **不设** | **1.5**（防重复） |
| `enable_thinking` | **开启** | 开启 |

**注意两个模型不共用参数**：T2I 需要 `presence_penalty=1.5` 防重复输出，**I2I 则必须不启用**该惩罚。

### 2.3 输出格式（JSON，在 `<think>` 块之后）

```json
{ "rewritten_prompt": "<精确编辑指令>",
  "wh_ratio": "",
  "ratio_follow": "<image1>" }
```

- `rewritten_prompt` —— 展开后的精确提示词，喂给图像编辑模型
- `wh_ratio` —— 新建构图时使用的宽高比（如 `"16:9"`）
- `ratio_follow` —— **就地编辑时继承哪张输入图的宽高比**
- **`wh_ratio` 与 `ratio_follow` 互斥**，有且仅有一个非空

解析时需要先切掉思考过程：
```python
thinking, _, answer = gen.partition("</think>")
result = json.loads(answer.strip())
```

### 2.4 ⭐ 一个直接解决我们设计问题的发现

我们在 16-test 里讨论过「输出用谁的构图」这个口径问题——**`ratio_follow` 就是官方对此的标准答案**。

PE-I2I 会**自己判断**输出应该继承哪张输入图的宽高比。对我们的任务（保 A 的构图），它理应给出 `ratio_follow: "<image1>"`。

这不仅省掉了我们手动指定尺寸，而且是**官方训练过的判断**，比我们拍脑袋更可靠。

---

## 3. 环境与资源需求

| 项 | 要求 |
|---|---|
| `transformers` | **≥ 5.4.0** |
| `torch` | ≥ 2.4.0 |
| 其他 | `accelerate`, `pillow` |
| 精度 | BF16 |
| 显存 | 9B BF16 ≈ **18 GB 权重** + 长上下文（24000 tokens）的 KV cache，建议 **≥40 GB** |
| **下载体积** | 约 **18–20 GB** |

**落盘位置**：按项目约定应放 **`/data/shangguanrz/`**（`/` 已用 83%、只剩 295G；`/data` 剩 21T）。

### ⚠️ 一个需要先验证的兼容性风险

我们的 `qwen21` 环境是 **transformers 5.17.0** ✓ 满足 ≥5.4.0。

**但 PE 模型基于 Qwen3.5-VL，而我们的环境里只有 `qwen3_vl`（Qwen-Image-2.1 的文本编码器）**。需要先确认 5.17.0 是否已支持 `qwen3_5` 架构——如果不支持，就要新建/升级环境，而不能污染现有的 qwen21 环境（`diffusers 0.41.0.dev0` 是为 Qwen-Image-2.1 专门保留的）。

---

## 4. 三种接入方式

| 方式 | 做法 | 取舍 |
|---|---|---|
| **A. 本地 transformers** | 按模型卡示例，`AutoModelForImageTextToText` + `apply_chat_template` | 完全离线、可控；需 18 GB 权重 + 显存 |
| **B. 本地 vLLM** | 官方 `prompt_rewrite/` 提供 vLLM 与 vLLM server 入口 | 吞吐高、可复用；需装 vLLM |
| **C. 云端 API** | 阿里云百炼的 `prompt_extend=true` + `prompt_extend_mode` | 零部署；需 API Key、有网络依赖 |

**API 侧补充**（阿里云百炼）：
- `prompt_extend`（默认 `true`）—— 智能改写正向提示词
- `prompt_extend_mode`：`direct`（DPE，T2I/I2I 都支持）或 `agent`（APE，**仅 T2I**，I2I 传 agent 会 400）
- `enable_thinking`（默认 `true`）—— 需 `prompt_extend=true` 才生效

**建议**：先做 **方案 A**（与我们现有 pipeline 同构，验证成本最低）。

---

## 5. 官方代码与参考

- 官方重写代码：`Qwen-Image-2.1` 仓库的 **`prompt_rewrite/`** 目录（提供 transformers / vLLM / vLLM server 三套入口）
- ComfyUI 适配权重（int8）：`Comfy-Org/Qwen-Image-2.1` → `text_encoders/qwen3.5_9b_qwen_image_2.1_pe_{t2i,i2i}.int8_convrot.safetensors`
- ComfyUI 节点包：`benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer`（含 `QwenImage21_T2IPromptRewrite` 与 `QwenImage21_EditPromptRewrite` 两个节点）
- 参考资料：
  - <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I>
  - <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I>
  - <https://github.com/QwenLM/Qwen-Image-2.1>（`prompt_rewrite/`）

---

## 6. 建议的下一步

1. **先验证兼容性**：确认 `qwen21` 环境（transformers 5.17.0）能否加载 `qwen3_5` 架构。**这一步决定要不要新环境**，应最先做。
2. **下载 PE-I2I**（约 18 GB）到 `/data/shangguanrz/pic-edit-models/`
3. **跑一次对照**：用 PE-I2I 把我们的中文/英文需求重写成 `rewritten_prompt`，再用 Qwen-Image-2.1 基座生成，与本次手写的六段式详细提示词**做同口径对比**
4. 顺带验证 **`ratio_follow` 是否给出 `<image1>`**——这直接回答"输出构图口径"那个悬而未决的问题

**优先级判断**：本次手写详细提示词已经效果很好，PE-I2I 的增量价值主要在于（a）自动化，省去手写长提示词；（b）`ratio_follow` 的构图判断。第 1 步验证成本很低，值得先做。
