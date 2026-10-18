# union-adapter —— Qwen-Image-2.1 ControlNet Union 权重实体

## 里面有什么

`Qwen-Image-2.1-Fun-Controlnet-Union/`，VideoX-Fun 官方 ControlNet-Union 控制分支权重：

| 文件 | 说明 |
|---|---|
| `Qwen-Image-2.1-Fun-Controlnet-Union.safetensors` | 7,550,979,904 字节（7.03 GiB），SHA256 `65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd`。180 个 tensor，全 BF16，`__metadata__.format = pt`，键名为 `control_blocks.0..15.*` + `control_img_in` |
| `_checksums.json` | 上游 32 个文件的指纹清单（含 asset/results 的图） |
| `README.md` / `README_zh.md` | 官方模型卡，8 种结构控制条件（Canny/Depth/Grayscale/HED/Lineart/MLSD/Pose/Scribble）+ inpainting 共用一套权重 |
| `configuration.json` | 34 字节，`{"task":"text-to-image-synthesis"}`，ModelScope 下载残留 |

**不带**的：`asset/`（5.5M 示例图）、`results/`（31M 官方生成图）、`.cache/`（172K 下载缓存）——留在 `/data` 的隔离目录里。

## 与上下层的关系

- 2026-10-17 从 `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union/` 迁入。原目录改名成 `Qwen-Image-2.1-Fun-Controlnet-Union.pre-move-20261017/`（7.1G，含示例图）保留在同处；原路径现在是**指向本目录的软链接**。
- 因此仓库里 40+ 处写死 `/data/.../qwen21_union_migration_v1/models/...` 的脚本与 README（`workflow/experiment/qwen-union{,-hybrid}/scripts/*`、`experiments/6.qwen-controlnet union/*`）继续可用，路径没改也能命中这里的实体文件。
- 出图还需要基座：`/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`（本目录只有控制分支，不含基座）。
- 迁移动机：控制分支要拿来改结构/微调，放项目树内便于直接读写，不再依赖 data 盘的实验目录。

## 注意与坑

- **别提交进 Git**：根 `.gitignore` 已加 `/union-adapter/*` + `!union-adapter/README.md`。7G 权重和上游示例图都不入库；`*.safetensors` 本身也在全局忽略里，双保险。
- `/` 盘（sda2）只剩 ~298G、已用 83%。这里一份 + `/data` 隔离目录一份，确认新副本可用后**由人决定**是否清掉隔离目录腾 `/data`（`/data` 有 21T 空闲，不急）。
- 改过权重后 `_checksums.json` 就对不上实体了，它是上游快照，不是本目录的校验凭据；自己产出的 checkpoint 请另记 SHA256。
- `workflow/experiment/qwen-union/scripts/eval_union_single.py` 会把 checkpoint 的 SHA256 写进评测报告——动了权重再跑评测，报告里的指纹会变，与历史 `report/summary.json` 对不上属正常。
