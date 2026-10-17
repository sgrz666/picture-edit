# pose_transfer/ —— 单人动作迁移流水线在本仓库的提示词契约

> 2026-10-13 重构：原来散在本目录的底层（缓存指纹、估计器封装、几何/投影、DWPose 关节表、外观相似度）已迁到仓库级 `src/pose_edit/`，由单人与双人两条 workflow 线共用。本目录**现在只剩两个文件**：`__init__.py` 与 `prompts.py`。权威行为描述见 `../README.md` 的「缓存怎么判定」「source_torso 改用实测 A 躯干」两节，以及 `../src/pose_edit/README.md`。

## 里面有什么

- `__init__.py`：模块 docstring，定义"用原生 Qwen + SMPL-X 姿态引导做单人姿态迁移"。仅有说明，无代码。
- `prompts.py`：单人线**专属**的 PE-I2I 提示词编译。文件头写明：与人数无关的一半（采样参数、官方答案解析、生成端引用守卫）住在 `src/pose_edit/prompt_contract`，在本文件 re-export，**保证单人线只有一个 import 站点**。本文件自己只放单人专属内容：
  - `PE_ROLE_SYSTEM_SUFFIX`：本流程特有的「最终输出图像引用契约」，覆盖官方多图引用规则——最终 `rewritten_prompt` 不得出现 `<image2>` 或任何对目标照片的引用，只能引用 `<image1>`（源）与 `<image3>`（骨架），下游把 `<image3>` 映射成生成图 2（`compile_prompt` 内部 `<image3>`→`<image2>`）。
  - `pe_instruction(prop_policy='source_only')`：给 PE 的重写指令全文。`prop_policy` 只接受 `source_only`，否则抛 `ValueError`（道具策略固定：保留 A 已有物品、排除 B 独有物品，如测试 B 的篮球不进生成图）。
  - `compile_prompt(text, prop_policy='source_only')`：把 PE 的三段式 JSON 答案编译成生成器能吃的提示词。先用 `IMAGE_REFERENCE` 正则改写图像引用——`<image1>`→`<image1>`、`<image2>`→直接抛 `ValueError`（生成器看不到 B）、`<image3>`→`<image2>`；再在前面拼上契约前缀、后面拼上后缀（`…Preserve the source person and their original clothing and setting. Render a natural photograph without skeleton lines…`），最后过 `validate_generator_prompt` 守卫。
  - re-export 的共享件（来自 `src/pose_edit.prompt_contract`）：`PE_SAMPLING`、`IMAGE_REFERENCE`、`parse_pe_answer`、`referenced_images`、`validate_generator_prompt`。

## 与上下层的关系

- **`src/pose_edit/prompt_contract.py`（共享层）**：`prompts.py` 里与人数无关的部分都来自这里。`prompt_contract.py` 负责：采样参数 `PE_SAMPLING`（`max_new_tokens=24000, temperature=1.0, top_p=0.95, top_k=20`）、`IMAGE_REFERENCE` 图像引用正则、`parse_pe_answer`（解析官方三段式 JSON、强制 `ratio_follow="<image1>"` 即源画布）、`referenced_images`、`validate_generator_prompt`（生成器只允许指向它实际收到的图像，引用越界即契约破坏）。分工边界：**`prompt_contract.py` 不关心一张图里有几个人；`prompts.py` 在它之上加单人专属的 system 后缀、重写指令与编译拼装**。两者必须同步演进，但职责边界清晰（以代码为准）。注：旧版 `prompts.py` 里 `parse_pe_answer` 等原属本目录，现已挪到共享层；本文件只 re-export，不重复实现。
- **`../run.py`**：`from pose_transfer.prompts import PE_SAMPLING, validate_generator_prompt`；并在指纹代码清单里显式登记 `'pe_worker.py', 'pose_transfer/prompts.py'`。
- **`../pe_worker.py`**：`from pose_transfer.prompts import (PE_ROLE_SYSTEM_SUFFIX, PE_SAMPLING, compile_prompt, …)`——PE 阶段实际在这里取 system 后缀、采样参数与编译结果。
- **`../lowlatency_probe.py`**：`from pose_transfer.prompts import compile_prompt`——低延迟探测的 `--mode pe` arm 用它校验/编译 PE 回答（见 `../README.md`「低延迟探测（llama.cpp / GGUF）」一节）。
- 迁走的三件套（`common.py`/`estimator.py`/`geometry.py`）现在是 `src/pose_edit/` 的 `common.py`/`estimator.py`/`geometry.py`，被所有阶段 `import`；本目录不再持有它们，相关描述见 `../src/pose_edit/README.md`。

## 注意与坑

- **改这里会进阶段指纹**：`pose_transfer/prompts.py` 在 `run.py` 的代码清单里，所以改它（哪怕一个字符）会让全部既有 run 的阶段缓存失效、复验报 `workflow code changed`；产物与 `stage.json` 原样保留，不 `--force` 不重算。这是设计取舍（见 `../README.md`「缓存怎么判定」）。
- `prop_policy` 写死 `source_only`，换其他策略直接 `ValueError`；`compile_prompt` 一旦在答案里见到 `<image2>` 即拒绝，要求把目标动作显式写成文字，因为生成器只收到 A 与骨架。
- 角色与 JSON 校验不保证 PE 的动作理解完全正确，最终动作仍以 `runs/<run>/prompt/generation_prompt.txt` 为准。
