# vram_lab/ —— 显存压测与实验证据工具包（独立成包）

`vram_lab/` 是一个**独立成包**的显存压力测试与实验证据工具集，围绕 DeepGen-1.0-diffusers 基座做可控的显存 / 耗时测量与人工复核。它与 `experiments/1.deepgen experiment/`、`experiments/iper/iper_v6_*` 等正式训练实验**无关**，不依赖主仓的训练流水线，自成一套可独立运行的模块（`vram_lab.run` / `vram_lab.worker` / `vram_lab.analysis` / `vram_lab.acceptance` / `vram_lab.report` 等）。实验产物落在 `results/<run>/`，证据通过 `demo.html` 离线查看。

> 注意：本目录不是「姿态迁移效果」的演示。它测的是模型在固定任务、固定分辨率、固定输入下的显存占用与训练 / 推理稳定性；页面里的照片与结构残差图仅用于案例诊断，结构条件始终标记为人工模拟，不能作为姿态迁移质量的证据。

## 包内模块清单

| 模块 | 职责 |
|---|---|
| `common.py` | 共享常量与工具：`MODEL`（`models/DeepGen-1.0-diffusers`）、`PROJECT`、`REVISION`、负向提示词、`TASKS`（S0/S1/S2/D0/D1/D2 六个编辑任务提示词）、`sha256`、`environment`、`preprocess`、`Monitor`、`quality_manifest`、`training_manifest` |
| `run.py` | 串行、可恢复的**实验控制器**。子命令：`preflight / quality / reproduce / cache / smoke / main / stable / infer / q4 / sensitivity`。已完成运行不可变，相同 ID 配置不同会被拒绝 |
| `worker.py` | 单个隔离 GPU 作业；**同一 GPU 上绝不并发**；加载 DeepGen 流水线、按任务测量显存与步时 |
| `analysis.py` | 证据衍生的参数选择与显存报告。子命令：`select / capacity / quality / evidence`。`evidence_index` 仅在逐字节比对训练输入与缓存输入后才补索引；绝不臆造人工评分 |
| `acceptance.py` | 纯 CPU 的**最终验收**，独立于 UI 标签：按 manifest 校验全部 run 的 `config.json` / `summary.json` / `status` / NVML / `gpu_samples.csv` |
| `report.py` | 零依赖、以证据为支撑的 CPU-only 实验报告（产出 `MEMORY_REPORT.md`、`results.json`、`summary.csv`） |
| `contact_sheet.py` | 把未改动的实验图像按 run 标签铺成视觉复核表（`--root / --group / --task`），供人工看原图 |
| `package.py` | 导出可复核证据包（`evidence.zip`），剔除 `.pt / .npy / .zip / .tmp` 等多 GB 张量，保留图片 / 配置 / 日志 / 源码快照 |
| `demo.html` | 零依赖离线静态报告前端（"DeepGen 显存实验台 / Local Experiment Observatory"）：读取 `results.json` / `summary.csv` 与 `runs/` 证据，无 CDN、不占 GPU。六段：输入核对 / 完整结果墙 / 局部与身份 / 控制条件 / 显存与耗时 / 证据与评分 |
| `__init__.py` | 包标识 |

## 服务器运行

> 完成的实验证据实际位于 `results/review_20260908_v2`（旧稿曾写 `experiments/review_20260908_v2`，该路径当前不存在，以磁盘为准）。

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit
conda activate deepgen
export DEEPGEN_PROJECT=/home/shangguanrz/project/pic-edit
ROOT=results/review_20260908_v2
python -m vram_lab.run preflight --root "$ROOT"
python -m vram_lab.run quality --root "$ROOT" --groups Q0,Q1
python -m vram_lab.run reproduce --root "$ROOT"
python -m vram_lab.run cache --root "$ROOT"
python -m vram_lab.run smoke --root "$ROOT"
python -m vram_lab.run main --root "$ROOT"
python -m vram_lab.run stable --root "$ROOT"
python -m vram_lab.run infer --root "$ROOT"
python -m vram_lab.run quality --root "$ROOT" --groups Q2,Q3
# 完成人工/视觉评分，将 scores.json 保存到 ROOT 后才能选参数：
python -m vram_lab.analysis select --root "$ROOT"
python -m vram_lab.run q4 --root "$ROOT"
python -m vram_lab.run sensitivity --root "$ROOT"
python -m vram_lab.analysis capacity --root "$ROOT"
python -m vram_lab.analysis quality --root "$ROOT"
python -m vram_lab.analysis evidence --root "$ROOT"
python -m vram_lab.acceptance --root "$ROOT"
python -m vram_lab.report --root "$ROOT"
python -m http.server 7860 --bind 127.0.0.1 --directory "$ROOT"
```

本地另开一个终端做端口转发，用浏览器查看 `demo.html` 渲染的离线报告：

```powershell
ssh -N -L 7860:127.0.0.1:7860 sg
```

浏览器访问 http://127.0.0.1:7860 。静态 `demo.html` 不占 GPU、无 CDN 依赖；源码与结果同步到本地后，同样可用 `python -m http.server` 打开。

## 核心约定

- 全局 CPU/CUDA RNG 与去噪 generator 都固定为配置中的 seed；每个 GPU 配置独立进程，串行执行。重复任务的图片像素哈希应一致，同一软件 / 硬件下验证。
- 原图和输出分别保存；模型输入记录 letterbox 的 content_box，不能将原始横竖照片拉伸后的形状当作原始人体比例。
- PASS 表示代码与测量完成，不代表编辑质量合格。人物身份看不清时评分为 null，不能当成保持成功。
- 六个结构残差头作用于目标图 tokens；参考图 tokens 补零。CFG 推理将同一结构条件复制至两个分支。结构条件始终明确标记为人工模拟。
- 缓存包含真实完整正向语义条件、源图 latent 和重建目标 latent；输入、提示词、分辨率、编码实现及模型版本共同决定缓存键。生成缓存阶段独立计量。
- 在线训练仅对冻结编码器使用 no_grad；冻结 DiT 保留到适配器的梯度。适配器 FP32、AdamW FP32，前向 BF16 autocast；没有量化、主干 LoRA 或全量训练。
- 短测试 5 次预热 + 20 次测量更新。稳定性测试共 100 次实际更新（5 次预热 + 95 次稳态），交替两照片，三次独立复测。
- 第一轮更新通过阶段事件标记 forward/backward/optimizer；不嵌套重置全局峰值。正式时间对照使用完整更新；梯度门槛检查发生在预热期（smoke 除外）。
- 运行峰值涵盖加载、零等价检查和实际训练；`steady_train` 单独列出稳态 allocated/reserved。CUDA 组件时间不等于端到端耗时，文件保存另记。
- NVML 50ms 采样可能漏掉瞬时尖峰。采样缺失或异常不能产生合格的显存结论。
- 预算按子进程总墙钟时间保守计费：72 小时总额、原生质量实验最多 6 小时。历史修正前消耗通过 `budget_carryover.json` 计入。GPU 有其他任务时等待，不杀其他进程。
- 既有完整结果不可覆盖；相同 ID 配置不同会拒绝。NOT_RUN 在重试前归档。INVALID 必须修复并换新实验目录 / 运行 ID，保留原始失败证据。

## 测试与验收

```bash
# 在项目根目录运行（tests/ 位于仓根，非本包内）
python -m unittest discover -s tests -v
python -m vram_lab.acceptance --root results/review_20260908_v2
```

先看 `preflight.json` 的全部权重哈希，再看 smoke 的 `validation.json`：主干前后摘要一致、六个头与主体梯度有效、实际重计算次数大于一次。最后看 `MEMORY_REPORT.md` 与 `capacity.json`，仅在三次稳定测试、两张照片独立推理和两份缓存全部通过后给出建议容量。

`scores.json` 为按 run ID 索引的字典。每项使用 action/identity/clothing/background/anatomy（0/1/2，identity 可 null）及 added_person/missing_person/wrong_person 布尔值，可附 reviewer、note。评分是有限照片的视觉判断，不是通用 benchmark。

六个 `demo.html` 模块：实际输入、全结果墙、局部放大与人物标记、模拟控制图、显存时间线 / 热力图 / 速度图、证据和评分导出。人物标记只用于界面检查，不加入模型输入。

## 使用本地证据副本

在 `D:\codeplus\pictureedit` 中运行：

```powershell
python -m http.server 7860 --bind 127.0.0.1 --directory results/review_20260908_v2
```

访问 http://127.0.0.1:7860 ，先按任务与分组选图，再从"查看运行"进入配置、局部放大和评分。稳定性运行可在 S0/D0 间切换实际输入与控制图。显存区提供完整流程容量、逐阶段采样和各训练配置对照。

页面修改的评分通过"导出评分 JSON"保存；将文件放回结果目录并命名为 `scores.json`，重新执行 `analysis quality`、`report` 后才会持久更新报告。既有 Q4 输出使用当时选定参数；若人工评分导致参数选择改变，应保留旧实验，在新实验目录运行新的 Q4 对照。

本地 `evidence.zip`（由 `package.py` 生成）包含图片、配置、采样、日志和源码快照，不包含大型 `.pt` / `.npy` 张量。适配器检查点及条件缓存保留在服务器对应运行目录，可直接用于加载与复测。

`analysis evidence` 为已完成的稳定性运行补充逐输入索引：先验证训练保存的原始 PNG 与模型输入 PNG 和缓存运行逐字节相同，再引用缓存的原始照片哈希。该步骤只生成复核索引，不改写原始运行配置或测量结果。
