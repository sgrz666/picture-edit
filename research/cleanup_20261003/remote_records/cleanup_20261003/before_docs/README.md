# 人物图像姿态编辑（Picture Edit）

基于三维人体表示的单/双人参考姿态图像编辑研究项目。在保持身份、服饰和场景一致性的前提下，实现身高体型解耦的参考姿态图像编辑，支持握手、搭肩、拥抱等肢体接触交互动作，并精确到手指结构。

## 项目背景

- 任务书要求第三阶段与 Qwen-Edit 类模型适配，生成主干 ≤ 6B，最终验收交互姿态合理比例 ≥ 90% 且主观质量优于 Qwen-Edit 基线 5% 以上。
- 当前使用 **DeepGen-1.0-diffusers**（DiT ≈ 2.47B 存储参数）作为生成主干，符合 <6B 约束；Qwen-Image-Edit-2511（20B）已完成显存可行性分析，定位为质量上界与教师模型。

## 目录结构

```
├── src/                    # 核心源码
│   ├── adapter/            #   PoseConditionAdapter 外挂姿态适配器（零卷积初始化）
│   └── controls/           #   控制信号生成（SMPL-X 骨架/深度/法线/接触图）
├── vram_lab/               # 显存实验包（run/analysis/acceptance/report）
├── tests/                  # 自动化测试
├── research/               # 调研文档（DeepGen 评估、第三阶段显存预算）
├── results/                # 实验结果与可视化 Demo（review_20260908 初版、review_20260908_v2 正式版）
├── QWEN_EDIT_20B_MEMORY_ANALYSIS.md  # Qwen-Image-Edit 20B 显存可行性分析
├── demooutput/             # 演示输出（对比图、控制图、演示报告）
├── inputs/                 # 样例输入照片
├── train_adapter.py        # 适配器训练入口
├── run_demo_pose_edit.py   # Demo 推理入口
├── create_debug_samples.py # 合成调试样本生成（data/ 目录说明）
├── test_control_signals.py # 合成 SMPL-X 人体工具（被 vram_lab.worker 依赖，非一次性测试）
├── build_*_docx.py         # 文档生成脚本
└── *.docx                  # 任务书 / 架构设计 / 进展简报等文档
```

## 技术方案

- **生成主干**：DeepGen DiT（冻结，BF16 前向），约 2.47B 存储参数
- **控制注入**：外挂 CNN 姿态适配器（M 型 130,277,376 参数，FP32），6 个残差头注入 DiT token，参考图 token 补零；零卷积初始化保证基线行为不变
- **控制信号**：SMPL-X 骨架 / 深度 / 法线 / 接触图（`src/controls/generator.py`）
- **训练配置**：冻结基座 + 外挂适配器，条件缓存 + 梯度检查点，micro-batch=1，AdamW FP32，前向 BF16 autocast

## 显存实测（RTX 6000D 84GB）

| 分辨率 | 稳定训练峰值 | 完整流程峰值 | 建议容量 |
|---|---:|---:|---:|
| 512 | 10.57 GiB | 15.96 GiB | 18 GiB |
| 768 | 13.40 GiB | 17.42 GiB | 20 GiB |
| 1024 | 18.01 GiB | 19.35 GiB | 22 GiB |

条件缓存与梯度检查点是显存控制的核心手段（1024 下缓存训练峰值从 37.34 降至 18.01 GiB）。

## 当前结论（截至 review_20260908_v2）

- 120 个运行记录，核心验收 111/111 通过，17 项自动化测试通过，精确复现 PNG 哈希一致。
- 原生编辑：单人招手可完成且配饰保持较好；**双人握手/搭肩仍普遍失败**（新增人物、原人物未起身、身份/衣着错配），问题在人物对应关系而非权重。
- 外挂适配器已通过接口/梯度/显存验收，但训练数据为两张照片自重建 + 人工控制图，**尚不能证明学会姿态迁移**；下一阶段需补齐真实对齐的源图/目标图/结构控制配对数据与人物实例对应信息。

## 运行

服务器端（DeepGen 实验，需 conda 环境 `deepgen`）：

```bash
cd /home/shangguanrz/project/pic-edit
conda activate deepgen
export DEEPGEN_PROJECT=/home/shangguanrz/project/pic-edit
python -m vram_lab.run preflight --root experiments/review_20260908_v2
```

本地查看实验结果 Demo：

```powershell
cd D:\codeplus\pictureedit
python -m http.server 7860 --bind 127.0.0.1 --directory results\review_20260908_v2
# 浏览器访问 http://127.0.0.1:7860/
```

测试与验收：

```bash
python -m unittest discover -s tests -v
python -m vram_lab.acceptance --root results/review_20260908_v2
```

## 说明

- 模型权重（DeepGen 官方权重）与实验运行数据（缓存、检查点、日志）体积过大，不在本仓库内；请从官方渠道获取模型，实验数据保留在服务器。
- `deepgen/` 为上游官方仓库，未包含在本仓库，按官方文档获取。
