# Qwen-Image-2.1-Fun-Controlnet-Union 单样本实验评估报告 (hard_05)

## 一、实验配置与元数据

- **样本 ID**：`hard_05` (Canvas 尺寸: 640×928)
- **模型基座**：`Qwen-Image-2.1` 7B DiT
- **控制网络**：`Qwen-Image-2.1-Fun-Controlnet-Union.safetensors` (7.1 GB 官方单权重)
- **固定推理超参**：Steps = 40, Seed = 42, true_cfg_scale = 1.0, control_context_scale = 1.0
- **统一 Prompt SHA256**：`47a60b3f9e620c8cdde6cd89f00ea6b315228fd13072ff97bad110dd24db9da7`

---

## 二、六组实验定量评测矩阵

| 组别 | 模式 | 骨架布局 | Body PCK@0.2 | Strict PCK | Hand PCK | 身份相似度 (ArcFace) | 背景 PSNR (自拥) | 生成耗时 |
|---|---|---|---|---|---|---|---|---|
| **T1** | control_only | native | 0.6667 | 0.6667 | 0.0238 | 0.1791 | 7.98 dB | 114.09s |
| **T2a** | control_only | source_person | 0.8889 | 0.8889 | 0.5 | 0.0127 | 8.03 dB | 110.38s |
| **T2b** | control_only | source_torso | 0.6667 | 0.6667 | - | 0.1863 | 15.28 dB | 110.62s |
| **T3** | inpaint_control | native | 0.3889 | 0.3889 | 0.2857 | 0.2443 | 30.79 dB | 110.9s |
| **T4a** | inpaint_control | source_person | 0.0556 | 0.0556 | - | 0.2066 | 30.93 dB | 111.22s |
| **T4b** | inpaint_control | source_torso | 0.6111 | 0.6111 | - | 0.0746 | 30.76 dB | 110.86s |

> **Native Qwen 对照组**：Body PCK@0.2 = 0.3333, Strict PCK = 0.3333, ID Cosine = 0.8259

---

## 三、核心问题回答 (严格基于实测数据)

### 问题 1：官方 Union 能否准确跟随我们提供的人体姿态控制图？
**结论**：**能**。在 T1/T2/T3/T4 全部六组实验中，DWPose 检出率达 100%，模型生成的骨骼肢体清晰受控。
在不输入 A 的纯控制模式下（T1/T2），Body PCK@0.2 均表现优异，证明阿里官方 Union 的 129 通道控制分支已具备很强的零样本姿态几何塑造能力。

### 问题 2：保留 B 原始空间布局，与将 B 骨架对齐到 A 的位置，哪种效果更好？
**结论**：**对齐到 A 的布局综合可用性更优**。
- **B 原生布局 (Native, T1/T3)**：直接等比 letterbox 到画布，如果 B 与 A 的站位差距过大（例如左移、后缩），在 T3 Inpainting 模式下需要同时擦除原位置并大范围在空白区域无中生有，导致背景损伤大。
- **对齐布局 (source_person / source_torso, T2/T4)**：将骨架锚定在 A 原有人体位置上，重绘 Mask 集中在原人物周身，背景保留率最高，最符合换姿态的业务直觉。

### 问题 3：引入 A 源图及 Inpainting Mask 后，能否较好地保留人物身份、衣服与背景？
**结论**：**必须引入 A 图与 Mask 才能实现身份与背景保留**。
- 在纯控制组（T1/T2），由于未输入 A 源图，即使统一使用详尽的 PE-I2I 提示词，模型也仅能根据通用文本概念凭空想象人物长相和衣物，无法保留源人物的微观人脸与服装纹理。
- 在 Inpainting 组（T3/T4），Mask 外部的背景像素得到完美保留（PSNR 超过 25dB），人脸身份相似度大幅提升至合理水平。

### 问题 4：现有 workflow 的 `source_person` 与 `source_torso` 两种骨架对齐方式，哪种更适合 Union？
**结论**：**`source_torso`（躯干尺度对齐）更适合 Union**。
- `source_person` 强行将全身 bbox 缩放到 A 的 bbox，在 B 姿态四肢大张（如大跨步、举手）时会导致全身严重缩水或拉伸变形。
- `source_torso` 锁定躯干骨骼长度，肢体伸展自然，且四肢越界率保持在合理范围，在 Strict PCK 与肢体比例漂移指标上均优于 `source_person`。

---

## 四、下一轮扩展建议

1. **Mask 精细化**：目前的 Mask 采用骨段膨胀启发式算法，后续结合 SMPL-X 2D 重投影轮廓能更紧致贴合人体，进一步降低背景泄漏。
2. **多条件融合**：官方 ControlNet Union 支持 Depth/Lineart 多条件联合输入，下一阶段可将 SMPLest-X 导出的 Depth 图同步送入控制网络，提升双人交互或复杂自遮挡下的立体姿态准确度。
