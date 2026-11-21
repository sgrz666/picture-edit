# bilibili-single-spatial-train-20261107（服务器技术验证跑的本地报告镜像）

这个目录只放**报告与复现脚本**，不放训练代码和数据。训练本体在服务器
`/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1`，
`run_plan.json` 标 `technical_only=True`，S0–S3 预算 10000/2000/4000/4000 共 20000 updates。
**它不是全量训练线**：全量在 `bilibili-native-v5-training-20261106-v1`，那边装配断在 560/9520 对，训练尚不可启动。

2026-11-20 新增 [Region-Normalized FM v1 设计与 CPU 参考](report/region-objective-v1/README.md)。它先完善新区域目标，再用于后续「无空间加权 vs 新区域加权」对照；不是本页历史技术跑的新增训练结果。历史装配/运行状态不作为新实验的就绪判断。

## 里面有什么

- `report/s0_update1000_gallery/gallery.html` —— update 1000 验证集生成的三联对照页
  （source 身份图 / 生成图 / target GT 姿态图）。点击放大、Esc 关闭；可切"仅生成图"、按 distance 或 time_gap 排序。
  直接双击打开即可，图片走相对路径。
- `report/s0_update1000_gallery/images/` —— 30 张：10 张生成原图（256–320×512，未改动），
  20 张 source/target 原图的 512 长边缩放副本（原图在数据集里，未改动）。
- `report/s0_update1000_gallery/thumbs/` —— 10 张生成图缩略图，供会话内嵌展示用。
- `report/s0_update1000_gallery/manifest.json` —— 每对的 pair_id、生成图 sha256、source/target 原始路径、
  distance、time_gap_sec、motion score、scene_kind、S0 分支开关。
- `scripts/build_gallery.py` —— 服务器侧 staging 脚本：读生成 commit.json 与 v4.jsonl，配对 source/target，
  生成缩放副本与 manifest。只新建副本，不写 run root 和数据集。staging 目录已存在时拒绝覆盖。
- `scripts/build_html.py` —— 本地侧：由 manifest 生成 gallery.html 与会话内嵌 fragment。
  其中 VIZ 路径是会话专属，换会话需改；gallery.html 的输出路径跟着 GAL 走。

复现：`scp scripts/build_gallery.py sg:/data/shangguanrz/tmp/` 后用 qwen21 环境跑，tar 取回 staging，
再本地跑 `build_html.py`。换 update 号要改 `build_gallery.py` 里的 `GEN` 路径。

## 关键结论

- 10 对全部 `same_clip`、`cross_shot=false`；seed 42、20 steps、variant native；
  `target_rgb_used_as_condition=false`（GT 没泄漏进条件）。
- S0 只启用 **body** 分支，face/hand/detail/interaction 全关；base 侧 identity 开、depth 为 full_geometry。
  所以这批图不能按最终效果评判。
- 目视：身份、服装、场景复刻较好；**姿态只是近似**（#01 更接近 source，#05 介于 source 与 target 之间）。
  与 held-out loss 在 600→1700 步基本平盘一致（velocity 全程 0.6080–0.6090，波动 <0.2%）。
- 生成图里的粉色"柚子"、bilibili 标、时间戳是**数据帧烧录的水印残留**，模型在复刻它们，不是幻觉。
- 运行事故：11-08 01:02:48 `S0.stage_train` 真 OOM（结果文件 `error_type=OutOfMemoryError`），
  先被短消息匹配误判成 terminal；12:16:54 journal 记 `resource_failure_reclassified` 后从 checkpoint 500 恢复，
  停摆约 11.5 小时。恢复后连续 1200+ 步无失败，update 1000 生成探针通过（显存净增约 0）。

## 与其他实验的关系

上游是服务器 run root 与 `/data/shangguanrz/datasets/bilibili`（val 切分 5 视频 / 200 帧 / 1490 对）。
平行线是 `experiments/bilibili-native-v5-training/`（全量装配，尚未就绪）；本目录的结论只针对 technical_only 跑，
不要外推到全量线。几何渲染任务 `bili_vertical_dance_20261103` 已于 11-07 停在 3630/3998 帧，与本目录无数据依赖。
