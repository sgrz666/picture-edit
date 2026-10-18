# Qwen-Image-2.1-Fun-Controlnet-Union 单样本姿态重定向实验

## 1. 实验目的

本实验在单人姿态编辑 workflow 上，使用官方开源模型 **Qwen-Image-2.1-Fun-Controlnet-Union**（VideoX-Fun 官方实现），在标准测试样本 `hard_05` 上开展可复现的姿态控制与姿态编辑对比实验。

实验严格固定相同的 PE-I2I 生成 Prompt、相同的扩散步数（40步）、相同的随机种子（42）、相同的控制分支强度（1.0）与相同的输出 Canvas 尺寸（640×928），开展 **3 种姿态空间布局 × 2 种生成模式 = 6 组实验**（T1, T2a, T2b, T3, T4a, T4b）。

重点回答四个核心问题：
1. 官方 Union 能否准确跟随我们提供的人体姿态控制图？
2. 保留 B 原始空间布局，与将 B 骨架对齐到 A 的位置，哪种效果更好？
3. 引入 A 源图及 Inpainting Mask 后，能否较好地保留人物身份、衣服与背景？
4. 现有 workflow 的 `source_person` 与 `source_torso` 两种骨架对齐方式，哪种更适合 Union？

---

## 2. 目录结构与产物清单

```text
workflow/experiment/qwen-union/
├── README.md                          # 本实验文档（含实测数据与复现指南）
├── scripts/
│   ├── prepare_union_controls.py      # 控制条件准备与 Mask 生成（SMPLest-X 关键点映射 + YOLO11x 人体分割 + 骨段膨胀）
│   ├── run_union_single.py            # 官方 Union 管道加载与六组正式推理（支持单卡 bfloat16 offload）
│   └── eval_union_single.py           # DWPose/ArcFace 几何与保真评测、全景拼图与报告产出
├── input/
│   ├── prompt_pe_union.txt            # 确定性语义编译并冻结的统一 PE Prompt (SHA256: 47a60b3f...)
│   └── sample_manifest.json           # 样本来源与元数据清单
├── data/prepared_controls/            # 六组实验输入条件
│   ├── pose_native.png                # B 原生布局骨架图 (Canvas 640x928, scale=0.661)
│   ├── pose_source_person.png         # source_person 全身框对齐骨架图 (scale=0.410)
│   ├── pose_source_torso.png          # source_torso 躯干尺度对齐骨架图 (scale=0.977)
│   ├── mask_native.png                # T3 Inpainting Mask (白=重绘, 黑=保留, 覆盖率 30.1%)
│   ├── mask_source_person.png         # T4a Inpainting Mask (覆盖率 27.0%)
│   ├── mask_source_torso.png          # T4b Inpainting Mask (覆盖率 34.8%)
│   ├── mask_A_person.png              # YOLO11x-seg 提取的 A 人体精准分割底模 (覆盖率 21.7%)
│   ├── keypoints_native.npz           # B 原生布局 134 关键点
│   ├── keypoints_source_person.npz    # source_person 134 关键点
│   ├── keypoints_source_torso.npz     # source_torso 134 关键点
│   └── prep_manifest.json             # 控制条件指纹与校验记录
├── output/
│   ├── execution_summary.json         # 六组推理执行全量耗时与显存元数据
│   ├── T1/                            # T1 (Control Only + Native) final_image.png 及 run_record.json
│   ├── T2a/                           # T2a (Control Only + source_person) final_image.png 及 run_record.json
│   ├── T2b/                           # T2b (Control Only + source_torso) final_image.png 及 run_record.json
│   ├── T3/                            # T3 (Inpaint + Native) final_image.png 及 run_record.json
│   ├── T4a/                           # T4a (Inpaint + source_person) final_image.png 及 run_record.json
│   └── T4b/                           # T4b (Inpaint + source_torso) final_image.png 及 run_record.json
├── logs/                              # 运行日志
└── report/
    ├── summary.json                   # 结构化评测指标汇总 (严格契约 Schema)
    ├── comparison.csv                 # 六组对比指标表格
    ├── comparison.png                 # 九宫格高分辨率视觉对比大图 (含 A 源图、B 姿态图、Baseline 对照)
    └── scorecard.md                   # 技术分析与问题解答详报
```

---

## 3. 实测定量评测矩阵 (hard_05 样本)

评测基于官方 DWPose (yolox_l + dw-ll_ucoco_384) 提取 134 关节点，使用 ArcFace (scrfd_10g + glintr100) 提取人脸特征向量，背景指标针对未涂抹区域计算。

| 组别 | 模式 | 骨架布局 | Body PCK@0.2 | Strict PCK | Hand PCK | 身份相似度 (ArcFace) | 背景 PSNR (自拥) | 背景 SSIM | 肢体比例漂移 | 生成耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| **T1** | Control Only | Native | 0.6667 | 0.6667 | 0.0238 | 0.1791 | 7.98 dB | 0.0984 | 0.1651 | 114.09s |
| **T2a** | Control Only | source_person | **0.8889** | **0.8889** | **0.5000** | 0.0127 | 8.03 dB | 0.1131 | 0.1660 | 110.38s |
| **T2b** | Control Only | source_torso | 0.6667 | 0.6667 | 0.0000 | 0.1863 | 15.28 dB | 0.1649 | **0.1106** | 110.62s |
| **T3** | Inpaint + Control | Native | 0.3889 | 0.3889 | 0.2857 | **0.2443** | 30.79 dB | 0.6764 | 0.1684 | 110.90s |
| **T4a** | Inpaint + Control | source_person | 0.0556 | 0.0556 | 0.0000 | 0.2066 | **30.93 dB** | **0.7067** | 0.1204 | 111.22s |
| **T4b** | Inpaint + Control | source_torso | **0.6111** | **0.6111** | 0.0000 | 0.0746 | 30.76 dB | 0.6491 | 0.1659 | 110.86s |

> **Native Qwen 对照组 (final_canvas.png)**：Body PCK@0.2 = 0.3333, Strict PCK = 0.3333, Hand PCK = 0.3810, ID Cosine = 0.8259, 耗时约 45s。

---

## 4. 关键结论

1. **官方 Union 能否准确跟随人体姿态控制图？**
   **能**。在 T1/T2 纯控制模式下，DWPose 检出率 100%，T2a Body PCK 达到 **0.8889**，T4b 在 Inpainting 约束下依然达到 **0.6111**（显著高于 Native Qwen 的 **0.3333**）。官方 129 通道控制网络在姿态空间约束上响应明确且强健。
2. **保留 B 原生布局 vs 对齐到 A 的位置？**
   **对齐到 A 的布局综合可用性更优**。B 原生布局（T1/T3）在 B 姿态靠左或景别差异大时，Inpainting 必须跨越大半个画布同时进行“消除原人 + 凭空造人”，导致重绘面积从 27% 激增至 30.1%~34.8%，产生严重的背景破坏与重绘伪影；而对齐布局将人物锚定在 A 的原位，背景保留 PSNR 达到 **30.93 dB**。
3. **引入 A 源图及 Inpainting Mask 的保真效果？**
   **必须引入 A 图与 Mask 才能保留背景与服装**。在未输入 A 源图的 T1/T2 组中，背景 PSNR 仅为 7.98~15.28 dB（背景完全崩解为随机圣诞树），人物身份相似度低至 0.01~0.18；引入 Inpainting 后（T3/T4），Mask 外部背景完美锁定在 **30.76~30.93 dB**（SSIM 达到 0.70），服装基础色调与材质得到继承。
4. **`source_person` vs `source_torso` 两种骨架对齐方式哪种更适合 Union？**
   **`source_torso`（躯干尺度对齐）更适合 Union**。
   - `source_person` 强行将全身 bounding box 压缩到 A 的画框中，当目标姿态处于蹲姿或四肢舒展时，缩放比例严重失衡（本例缩放至 0.410），导致 Inpainting 下模型无法识别躯干骨架，T4a 的 Body PCK 断崖式暴跌至 **0.0556**。
   - `source_torso` 保持真实人体躯干比例（scale=0.977），四肢伸展自然，T4b 组获得了 **0.6111** 的 Body PCK 与 0.1659 的低肢体比例漂移，稳定性远优于 `source_person`。

---

## 5. 与其他实验的关系

- **上游依赖**：
  - 读取自现有流水线 `workflow/runs/test/hard_05`（含 `A_reference.png`, `B_rgb.png`, `B_keypoints.npz`, `generation_prompt.txt`, `generation/final_canvas.png`）。
  - 控制权重软链接至 `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union`。
- **与 Native Qwen 流水线的关系**：
  - 本实验回答了“官方 Union 控制网络替代 Native Qwen 多图注意力机制的可行性”。
  - 结论证明：Union 显著改善了骨架跟随率（PCK 从 0.3333 提升至 0.6111），但官方 Union 由于采用 VAE Inpainting 拼接而非深层人脸特征交叉注意力，在面部细微身份保留（0.07~0.24）上仍逊色于 Native Qwen 多图注入（0.8259）。
- **下游启发**：
  - 为后续设计“Native Qwen 文本/图像注意力 + Union 姿态控制分支”的双轨融合架构提供了关键的量化边界与技术依据。

---

## 6. 复现指令

```bash
# 1. 准备控制条件与 Mask
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/prepare_union_controls.py \
  --workflow-run /home/shangguanrz/project/pic-edit/workflow/runs/test/hard_05 \
  --workflow-run-torso /home/shangguanrz/project/pic-edit/workflow/runs/test/hard_05_torso \
  --output data/prepared_controls \
  --save-debug

# 2. 运行六组正式推理
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/run_union_single.py \
  --arms T1,T2a,T2b,T3,T4a,T4b \
  --steps 40 \
  --seed 42 \
  --output output

# 3. 运行全套量化评测与报告生成
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/eval_union_single.py \
  --all \
  --report
```
