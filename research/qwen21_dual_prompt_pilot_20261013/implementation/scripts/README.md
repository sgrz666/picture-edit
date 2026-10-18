# 运行入口分类

| 目录 | 用途 |
|---|---|
| `train/` | 训练入口，保留 V6、Qwen 简化/Full 和数据集前缀 |
| `eval/` | 正式评测与阶段对照 |
| `data/` | 数据构建、过滤、划分、预计算 |
| `audit/` | 数据/权重/兼容性检查与整理验收 |
| `reports/` | 图表、Word、面板、结果摘要生成 |
| `weights/` | 权重继承和转换 |
| `run/` | 实验编排与收尾 |

2026-10-03 整理：远端 62 个稳定工具迁入上述目录；旧位置保留相对软链接。31 个分类入口指向保留原位的正式或当前开发入口。只有一个源码实体；软链接不代表重新复制源码。今天新增的 TikTok Full 准备脚本保留原位。

`run/home/` 另有四个正式 HOME runner 的链接；原 HOME 脚本继续保留。62 份旧 HOME 诊断/临时工具归入 `archive/home_tools_202609/`，不混入正式入口。

根目录正式 train/eval 源码继续作为实现文件；从分类入口执行时，`Path(__file__).resolve()` 会落到原文件，所以不改变训练的 ROOT、源码哈希记录或模块身份。稳定工具的分类实体已调整项目根定位，可从项目外运行。

```bash
cd /home/shangguanrz/project/pic-edit
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/train/train_iper_qwen21_full.py --help
/home/shangguanrz/miniconda3/envs/deepgen/bin/python scripts/eval/multi_stage_eval_v6.py --help
python3 scripts/audit/verify_project_organization.py
```

数据与旧报告脚本有些在运行时直接写输出，没有通用 `--help`。先检查脚本的 main/参数，不要以批量执行方式探测所有脚本。

本地与远端不是全量镜像：本地还保存图表脚本与独有成果；本轮不从远端覆盖本地已有工作。远端入口映射和原始源码存于 `archive/cleanup_20261003/organization.json`、`before/`。

验收脚本默认只检查结构、链接、小文件哈希和资产是否存在。`--full-assets` 另读完七项资产并核对文件清单、属性和内容哈希。后续正常源码开发列在 `source_changes_since_cleanup`，不会被当成整理损坏；`--strict-source-baseline` 才要求源码与历史整理基线完全一致。
