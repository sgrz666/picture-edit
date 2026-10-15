# /data/shangguanrz 数据盘目录说明——人工手写规范（对齐用户本人笔法）

## 0. 唯一事实来源
用户已在 2026-10-12 15:07 手写 6 份，作为**风格与深度基准**（存于 `data_inventory/_user_docs.txt`，务必先读）：
- `pic-edit-datasets/TikTokDataset/README.md`、`pic-edit-datasets/Hi4D-origin/README.md`
- `pic-edit-models/Qwen-Image-2.1/README.pic-edit.md`
- `pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B/README.pic-edit.md`
- `pic-edit-archive/cleanup_20261003/experiments/iper_v6_comparative_3000/README.md`
- `pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2/README.md`

**这 6 份禁改禁覆盖**。本次任务是"深入写"：把它们没覆盖的**每一个目录层级**补齐。

## 1. 用户笔法的硬特征（必须照做）
标题：`# <目录名>/ —— <一句话定性（它到底是什么东西，不是"存放文件的目录"）>`
紧接一段说明"本目录是软链接还是实体、实体在哪、逻辑体积多少"。小节用**用户原话标题**，按目录性质选：
- 数据/权重/资产类：`## 里面有什么` → `## 与上下层的关系`（生产者脚本、消费者脚本，**点名到文件与行号**，如 `src/pose_control/qwen21_full/execution.py:32`）→ `## 注意与坑`
- 实验/归档类：`## 实验目的` → `## 里面有什么` → `## 关键结论` → `## 与其他实验的关系`
- 若同级已有 `目录说明.md`（旧脚本产物），可在"里面有什么"末尾一行提它，但**不得照抄其句子**。

实证要求：
- 每个数字都要实测得到（`find -maxdepth 1 -type d | wc -l`、`du -sh --si`、`wc -l`、`stat -c '%s'`、`ls -la`、打开 JSON/JSONL 首行、`.pt` 用远端 `python3` 打印 keys/shape/dtype）。
- 编号项目录（`00001…`、`seq_*`、`pairXX`、`sample_*`、单帧）**不逐个写 README**，而是在父级 README 里：写清总数、抽 1–2 项做完整实测（列出真实子结构与文件名规律、分辨率、张量形状、各项大小分布），并指出稀疏/缺失项（例如某些序列只有 1 个子目录）。
- "注意与坑"必须写真问题：宽高比失真、身份未认证、软链落在 `cleanup_*` 有被清理风险、逻辑体积 vs 实占（稀疏文件）、双层同名嵌套路径、历史版本易误引用等。没坑就写"无已知坑"，不要编。

## 2. 落盘与命名
- **禁止在远端直接写文件**（本任务由主会话统一同步）。
- 本地 staging：`D:\codeplus\pictureedit\.dirdocs_staging\data\<相对 /data/shangguanrz 的路径>\README.md`
- 若该目录已有**上游自带** README（如 HF 模型卡的 `README.md`），改用 `README.pic-edit.md`（用户已确立此惯例）。
- 已有用户手写稿的目录跳过；只写它的子目录。
- 只允许新增说明文件；**绝不改动数据集原始件**（`TikTok_dataset/**`、`Hi4D-origin/*.tar.gz`、`License.pdf`、各 `manifests/*.jsonl` 等）。
- 只写分配给你的子树，别越界（多 agent 并行）。

## 3. 远端访问
`ssh sg '<单条命令>'`（免密、无外网、主机 node01）。复杂探查**必须**写到唯一脚本再管道执行：
`ssh sg 'bash -s' < D:\codeplus\pictureedit\.dirdocs_staging\_probe_<分区名>.sh`
（在 ssh 的双引号里嵌套 for/do/done 或 heredoc 会报语法错误；`$` 会被吞。）
远端 `python3` 可用；`jq` 不保证有。大目录统计前先看 `ls | wc -l`。

## 4. 汇报（≤200 字，勿贴正文）
负责的层级、写了多少份、跳过哪些（用户已写/上游自带），以及 3–5 条实测到的关键事实（数字优先）。
