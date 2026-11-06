# PromptHMR Hi4D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. 用户已要求实际运行，按下列步骤连续执行。

**Goal:** 补齐 PromptHMR 环境与配置，完成 4 张固定 Hi4D 图的双臂推理，并补齐四模型对照。

**Architecture:** 独立 venv 继承现有 mamma；实验私有配置和脚本；只读现有模型、自动框与 GT-free 输入；新目录保存全部运行证据。

**Tech Stack:** Python 3.11、PyTorch 2.7.1+cu128、xformers 0.0.31、PromptHMR、NumPy、Pillow。

### Task 1: 独立运行环境

- [x] 只读核对源码、权重、PyTorch/CUDA、磁盘和显存。
- [x] 创建 `/data/shangguanrz/pic-edit-runtime/prompthmr-hi4d-20261106/env`，安装 `xformers==0.0.31 --no-deps`。
- [x] 实际导入 PromptHMR；默认 Flash3 不支持本机，改为显式 CUTLASS，float16 探针对 Torch 最大差 0.00024414。原环境不写入。

### Task 2: 冻结输入与数值契约

Files: 新建实验 `scripts/contracts.py`、`tests/test_contracts.py`、`data/request.json`。

- [x] 输入契约与路径包含测试先 RED 后实现，拒绝越界/非法框、目录穿越和非有限 JSON。
- [x] 输入只来自 GT-free fit records、自动 selected 和 MultiHMR raw predictions；固定 4 个 id 和双臂参数。
- [x] 服务器 `python -m unittest discover -s tests -v` 18/18 通过；本地 17 通过、1 权限相关跳过。

### Task 3: 模型加载与推理

Files: 新建 `scripts/run_prompthmr.py`。

- [x] 使用 `data/phmr`，完整 checkpoint strict=True 加载 765 键，0 missing/unexpected。旧 Chumpy SMPL 转数值副本，10 字段逐值相同。
- [x] 不传 text/masks/kpts；真实 K、两个自动框，分别跑 interaction=False/True。
- [x] 验证 image SHA256、参数有限、人数、正深度、K 逆仿射；最大投影差 4.946e-5 px，8 次参数网格重建最大残差 0。
- [x] 独占保存；smoke02 2/2、four-images-v1 8/8 成功，源码哈希未变。

### Task 4: 同图展示

Files: 新建 `scripts/render_results.py`、`report/` 运行说明。

- [x] 实际产物生成 12 张图，四模型列齐全，注明 MultiHMR 自动框；旧拼图保留。
- [x] 同相机透视 z-buffer 渲染双臂网格/normal，已目视检查接触场景总图。

### Task 5: 收尾与复现

- [x] 独立 spec 审查和代码质量审查均通过；审查补入模型 YAML/契约哈希、路径包含保护。
- [x] README 已写实测数、环境兼容方式、测试/重跑命令、四样本边界和关联实验。
- [x] 8 份网格下载后独立 SHA256 复核全部一致；只暂存本任务代码/文档/小型 JSON，保留旧产物和主工作区修改。
