# 项目清理归类执行计划

> 用户已授权执行（“好的你来清理归类吧，一定要仔细审查”）。整理远端运行树并归类本地成果，不提交、推送或同步覆盖已有改动。

目标：入口按功能集中、成果有统一索引、冷资产移出根分区；旧命令和资产路径继续可用。

## 1. 基线与回退

- [x] 在本地 `research/cleanup_20261003/` 保存操作脚本、Git 状态、源码哈希和验收结果。
- [x] 远端 `archive/cleanup_20261003/` 保存迁移前小文件与操作日志；保存 `.git/index` 备份并验证索引未被改写。
- [x] 逐操作核验源文件哈希；今天新增的 TikTok Full 入口、核心源码和运行目录不迁移。

## 2. 入口分类

- [x] 根目录数据构建/审计/报告辅助脚本迁入 `scripts/data/`、`scripts/audit/`、`scripts/reports/`；旧位置保留相对软链接。
- [x] 原 `scripts/` 的稳定工具按 data/audit/eval/reports/weights/run 分类，调整脚本自身的项目根定位；保护今天正在开发的 TikTok 准备入口。
- [x] 正式 train/eval 与 `test_control_signals.py` 保持源码位置；在分类目录提供指向它们的入口链接，避免改动并发训练源码。
- [x] 验证所有移动脚本可解析；核验 ROOT、兼容链接和关键 CLI；相关数据/Qwen/权重回归测试通过。

## 3. 成果、小文件与重复备份

- [x] 本地 `reportdoc/` 改名为 `reports/`；`expeiment/tiktok/` 迁入 `experiments/tiktok_legacy_evidence_202609/`，同步两份本地图表脚本的路径。
- [x] 远端 `demooutput/` 迁入 `results/legacy_demo_202609/` 并保留旧路径链接；根 `inference.log` 归入 `logs/legacy/`。
- [x] 三组已核验重复备份各存一份归档正文与来源清单，移除入口旁的冗余副本；每次操作重新核验 SHA-256 和引用。
- [x] 清理确认为空的本地目录及自有源码中的 Python/pytest 缓存，不触及环境/第三方/运行目录。

## 4. 大资产冷归档

- [x] 旧 `model-qwen-edit/` → `/data/shangguanrz/pic-edit-archive/cleanup_20261003/models/Qwen-Image-Edit-20B/`。
- [x] `experiments/review_20260908_v2/` 与 `experiments/iper_v6_comparative_3000/` → 同归档根 `experiments/`。
- [x] `nihaochen/qwen21_hi4d_controls/` → `/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls/`。
- [x] `add/imageFiles.zip`、`sequenceFiles.zip`、`camera_tram.zip` → 冷归档根 `raw/3dpw/`。
- [x] 使用保留属性/硬链接的复制；逐文件比较 SHA-256、源树稳定性和属性后，检查进程，再切换旧位置软链接。只有副本校验完全一致才撤除原实体。记录源/目标、哈希、文件数、字节数、切换结果和 df 实测。

## 5. 索引、审查与验收

- [x] 新建 `docs/PROJECT_STRUCTURE_20261003.md`、`scripts/README.md`、`reports/README.md`、`experiments/README.md`、`add/README.md` 与 `configs/asset_registry.json`；README 加入当前状态和新目录导航，保留历史内容。
- [x] 通过 requesting-code-review 技能调用独立 reviewer，审查迁移脚本、操作清单和兼容性；处理重要问题。
- [x] 核验核心及受保护源码哈希、Git 暂存索引、关键资产链接/数据清单、迁移内容哈希、相关回归测试、磁盘释放量。
- [x] 交付 `docs/目录清理执行记录_20261003.md`，列出实际完成项、保留项、新入口、复查与回退方法；不把历史模型质量结果当成本次新验收。

## 执行补充

- [x] HOME 66 份脚本逐份分类：62 份历史工具归档且保留兼容链接，四份正式 runner 保持 HOME 实体并增加分类入口。
- [x] 大资产采用两阶段迁移：入口与 246 项测试通过后再复核并撤除旧实体；完整资产复查 4,764 个文件，76 个临时副本的操作清单精确闭合。
- [x] 其他任务并发修改的四份 Full 核心文件及 canonical prepare 保留；最终 252 个源码绑定与实际通过测试的版本一致。
- [x] 逐对象内容 157.506 GiB、源侧分配 157.519 GiB；根分区 df 增量 259.443 GiB 作为全盘变化单列，不全部归因于本次迁移。

完成结果与使用方法见 `docs/目录清理执行记录_20261003.md`；本轮未提交或推送。
