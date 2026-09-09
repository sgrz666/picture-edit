# experiments/ —— 实验运行数据目录（不纳入 Git）

本目录存放 `vram_lab` 各阶段实验的**完整运行记录**（含适配器检查点、条件缓存、GPU 采样等大文件），单轮实验约 **26 GB**，不纳入仓库。

> 实验的**文本报告、汇总 JSON 和可视化 Demo 已同步到仓库的 `results/review_20260908_v2/`**（不含大张量），无需本目录即可阅读结论。

## 目录结构

```
experiments/
└── review_20260908_v2/                 # 实验根目录（由 --root 指定）
    ├── runs/                           # 每个运行 ID 一个子目录
    │   └── <RUN_ID>/
    │       ├── config.json             #   完整运行配置（分辨率/编码方式/检查点/种子等）
    │       ├── adapter.pt              #   适配器检查点（约 500 MB，仅训练/推理类运行）
    │       ├── control.npy             #   结构控制张量（人工模拟，约 33 MB）
    │       ├── input_original.png      #   原始输入照片
    │       ├── input_model.png         #   letterbox 后的模型输入（记录 content_box）
    │       ├── skeleton/depth/normal/contact.png  # 模拟控制图
    │       ├── output.png              #   生成结果（如有）
    │       ├── gpu_samples.csv         #   50ms NVML 显存采样
    │       ├── events.jsonl            #   forward/backward/optimizer 阶段事件
    │       ├── updates.json            #   逐次更新的梯度/显存记录
    │       ├── summary.json            #   峰值显存、耗时等汇总
    │       ├── validation.json         #   冻结哈希/零等价/梯度门槛校验
    │       ├── environment.json        #   软件硬件环境
    │       ├── stdout.log / stderr.log
    │       ├── source/                 #   源码快照
    │       └── task_evidence/          #   逐任务证据
    ├── cache/                          # 条件缓存（真实语义条件 + 源图/目标 latent，按内容哈希命名）
    ├── configs/                        # 各运行的配置副本
    ├── logs/                           # 运行日志
    ├── review_sheets/                  # 人工评分拼图
    ├── *_manifest.json                 # 各阶段运行清单（main/stable/q4/...）
    ├── results.json / summary.csv      # 汇总结果
    ├── preflight.json                  # 权重哈希预检
    ├── capacity.json                   # 容量建议
    ├── scores.json                     # 人工/视觉评分
    ├── ACCEPTANCE.json                 # 验收结果
    ├── EXPERIMENT_CONCLUSION.md        # 实验结论（仓库 results/ 有副本）
    ├── MEMORY_REPORT.md                # 显存报告（仓库 results/ 有副本）
    ├── QUALITY_REVIEW.md               # 质量复核（仓库 results/ 有副本）
    ├── RUNBOOK.md                      # 运行手册（仓库 vram_lab/README.md 有副本）
    ├── index.html                      # 可视化 Demo（仓库 results/ 有副本）
    └── evidence.zip                    # 证据包（见 results/review_20260908_v2/EVIDENCE_README.md）
```

## 如何重新生成

在服务器（或已下载模型、装好环境的机器）上：

```bash
conda activate deepgen
export DEEPGEN_PROJECT=$(pwd)
ROOT=experiments/review_20260908_v2

python -m vram_lab.run preflight  --root "$ROOT"   # 权重哈希预检
python -m vram_lab.run smoke     --root "$ROOT"    # 链路冒烟
python -m vram_lab.run cache     --root "$ROOT"    # 条件缓存
python -m vram_lab.run main      --root "$ROOT"    # 显存主矩阵
python -m vram_lab.run stable    --root "$ROOT"    # 100 次更新稳定性
python -m vram_lab.run infer     --root "$ROOT"    # 独立适配器推理
python -m vram_lab.run quality   --root "$ROOT" --groups Q0,Q1,Q2,Q3
python -m vram_lab.analysis capacity  --root "$ROOT"
python -m vram_lab.acceptance --root "$ROOT"
python -m vram_lab.report --root "$ROOT"
```

完整阶段顺序与约定见 `vram_lab/README.md`。注意：实验既有完整结果不可覆盖，相同 ID 配置不同会被拒绝；重跑请使用新的实验目录。
