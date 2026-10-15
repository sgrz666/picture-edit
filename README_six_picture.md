# six-picture-create

把一次「单人动作迁移」workflow run 的六件关键产物拼成一张带标注的图，用于快速人工审图。

## 拼的是什么

按流水线顺序取六件产物：

| # | 文件 | 角色 |
|---|---|---|
| 1 | `inputs/A_reference.png` | 源照片 · 身份 / 服装 / 场景 |
| 2 | `inputs/B_rgb.png` | 目标照片 · 仅提供动作 / 表情 |
| 3 | `geometry/B_poseguide_overlay.png` | 估计器在 B 上的拟合校验 |
| 4 | `geometry/B_3d_dwpose.png` | **对齐后姿态引导 · 唯一进入生成器** |
| 5 | `geometry/A_poseguide_alignment.png` | 对齐引导叠加到 A · 仅诊断 |
| 6 | `generation/final_image.png` | 最终生成结果 |

> 4 与 5 是**同一份对齐后的引导**，区别只在底图（黑底 vs A 的照片）。
> 5 不进入生成器，仅供人工判断「骨架落到 A 身上的位置与比例是否合理」。

## 用法

```bash
cd workflow/scripts/six-picture-create

# 单个 run
python3 script/six_picture_create.py ../../runs/demo

# 批处理：把 runs/ 下每个 run 各出一张
python3 script/six_picture_create.py --all ../../runs

# 只校验六件产物是否齐全，不出图
python3 script/six_picture_create.py ../../runs/demo --check

# 指定输出位置 / 自定义标题与注释
python3 script/six_picture_create.py ../../runs/demo -o /tmp/x.png \
    --title "自定义标题" --note "自定义注释"

# 未跑完 generation 的 run 也能出图（最后一格显示「缺失」）
python3 script/six_picture_create.py ../../runs/testset_hard05
```

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-o / --out` | `output/<run名>_six_panel.png` | 输出路径 |
| `--cell` | 620 | 单格边长 px |
| `--gap` | 22 | 格间距 px |
| `--scale-threshold` | 0.65 | 低于此缩放比，页脚提示「骨架偏小」 |
| `--lang` | auto | `zh` / `en` / auto（检测到中文字体就用中文） |
| `--title` / `--note` | 自动 | 覆盖图题 / 页脚注释 |
| `--check` | 关 | 只校验产物，打印每项大小 |
| `--all` | 关 | 把参数当作 runs 目录，逐个处理 |

## 自动读取的元数据

从 `geometry/stage.json` 的 `metadata.camera_checks` 读三项，写进图题副标题：

- `B_to_generation.scale` —— B 姿态等比缩放到 A 画布的缩放比
- `generation_canvas` —— 生成画布尺寸
- `valid_B_joints` —— 有效 B 关节数

页脚注释会**按 scale 自动分级**：

- `scale < 0.5`：骨架只占 A 人物一小部分，生成器可能把骨架当成独立「小人」渲染，姿态迁移容易失败
- `0.5 ≤ scale < 0.65`：骨架偏小，结果里若有多余人形伪影，根源在此
- `scale ≥ 0.65`：骨架尺寸与 A 较匹配，对齐正常

## 环境要求

- Python 3 + Pillow（`python3 -c "import PIL"`）
- 中文标签需要一个 CJK 字体；脚本按以下顺序自动探测，全部缺失时回退英文标签：

```
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc   ← node01 用这个
/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf
/usr/share/fonts/truetype/arphic/uming.ttc
/usr/share/fonts/truetype/arphic/ukai.ttc
C:/Windows/Fonts/msyh.ttc
C:/Windows/Fonts/simhei.ttf
```

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 参数错误；或单 run 模式下六件产物不齐 |
| 2 | 指定了中文但找不到中文字体 |
