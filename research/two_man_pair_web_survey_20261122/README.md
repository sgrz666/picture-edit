# 双人配对样本互联网广泛搜集 — 第一轮调研（2026-11-22）

## 1. 实验目的

xhs 账号下载通道被平台风控反复限制（硬限 3h / 软限已持续 12h+），单人/双人配对原料断供。
目标：转向互联网公开来源，广泛搜集与模板同型的原料——**同一对双人 + 同场景同装 + 不同姿态**的
全身照/视频——供既有配对管线（`two_man_pair_mine.py` / `two_man_video_pair_mine.py` /
`video_pair_mine.py` / `export_clean_pairs.py`）挖掘训练配对。本轮先回答"哪里有、有多少、抓不抓得动"，
下一轮按用户选定方向批量开采。

## 2. 配对模板（判定标准）

以 `two-man-data/pairs/pairs/pair_001_27_68e500da`（a_8 / b_4 双图对）为模板：

- 同一对双人（男女情侣，衣着完全一致：男 navy 长袖+军绿裤+卡其帽，女条纹上衣+米色百褶裙）；
- 同一场景（"BEES" 门店前，雨棚/海报栏/桌椅/栏杆逐像素一致）；
- 姿态明显不同（a 双人敬礼站姿；b 女抬腿+举手、男插兜）；
- 全身、正面、单帧可辨双人骨架。

## 3. 来源摸底与验证结果（本轮实测）

| 来源 | 原料形态 | 本轮实测 | 配对潜力评估 |
|---|---|---|---|
| **B 站双人舞视频** | 单视频 30s-3min，同人同景多姿态帧 | ✅ Scrapling 搜索页全抓通：8 关键词 × 42 卡，**278 个去重 BV**（含标题/UP主/播放量），`bilibili_duo_candidates.json` 已落 `two-man-data/web_collect/` | **最高**：仓库已有 bilibili 下载+抽帧+`two_man_video_pair_mine.py` 全套管线；一个双人舞视频可挖 5-30 对；候选里情侣/姐妹/师生翻跳都有，按标题粗筛后可直接进管线 |
| **婚纱旅拍客片画廊** | 一个套系 = 同 couple 同装多景多 pose 20-50 张 | ✅ Scrapling 普通 HTTP 可抓：纪梵希全球旅拍 `/work/list` 结构清晰（115 个分类/套系链接，图片 CDN 直链 `txbizimg.picbling.cn`），脚本 `scripts/probe_gallery.py` 验证通过 | **高**：天生就是"同 couple 同装多 pose"，且公开可抓；缺点是婚纱装占比大、景别混有半身/特写，需按全身+站姿过滤 |
| **Pinterest 双人 pose 板** | 单图为主，同 shoot 图分散在不同 pin | 🔍 搜索定位到 230-610 图的大板（`bayofamez/couple-poses-ideas`、`pastorangelo22/pre-wedding`），未抓取 | 中：量大但同 shoot 聚类需额外匹配（同人同景同装判别），工程量高于前两者 |
| **xhs 公开页** | 多图笔记（现管线同款原料） | ⏸ 账号通道受限中，暂停采集（用户指示） | 高但暂不可用；等风控解除后与 web 来源并行 |

结论：**B 站双人舞视频是当前最优开采面**（管线现成 + 原料密度高），婚纱客片画廊次之
（图源干净、无账号依赖），Pinterest 作为补集。

## 4. 里面有什么

- `scripts/harvest_bili.py` — B 站多关键词搜索采集器（Scrapling StealthyFetcher，8 关键词，输出 BV/标题/UP主/播放量 JSON）
- `scripts/probe_bili.py` — B 站搜索页解析冒烟测试
- `scripts/probe_gallery.py` — 婚纱画廊站结构探针（纪梵希旅拍，普通 HTTP 验证）
- `scripts/bilibili_duo_candidates.json` 的服务器权威副本：`sg:/data/shangguanrz/xiaohongshu/two-man-data/web_collect/bilibili_duo_candidates.json`（278 条：BV/标题/UP主/播放量/命中关键词）
- 本地运行环境：Scrapling 已装于独立 venv `C:/Users/sg/AppData/Local/Temp/scrapling_env`（Python 3.13 + `scrapling[fetchers]`，BSD-3-Clause）

## 5. 关键结论（带数字）

- B 站搜索页 Scrapling 全抓通：8 关键词平均 42 卡/页，278 去重 BV，无验证码拦截（2026-11-22 实测）；
- 婚纱画廊站普通 HTTP 即抓（无需隐身浏览器），套系→图片直链结构清晰；
- Scrapling 在本地 venv 完成安装与两次实抓验证，未见反爬拦截；
- 双人舞视频对产率参考：`two-man-data/vedios/` 29 个双人视频 → 37 对（xhs 样本）；
  B 站候选若按 30% 成对率、278 个视频估算，理论可挖 **数百至上千对**。

## 6. 下一步（待用户定向）

1. **B 站方向**：把 278 个 BV 过一遍标题粗筛（双人+真人+全身向），导入既有 bilibili 管线下载抽帧，
   跑 `two_man_video_pair_mine.py` 出对——预计一轮就能出百对级配对；
2. **画廊方向**：写批量脚本分页抓纪梵希等旅拍站全站套系图（同 couple 同装多 pose），
   按"全身+站姿"过滤后直接可当照片对候选（同一套系内两两组合）；
3. 两条线可以并行，产物统一落 `/data/shangguanrz/xiaohongshu/two-man-data/web_collect/`，
   挖出的配对并入 `two-man-data` 既有 manifest 体系。

## 7. 三来源配对样例（2026-11-22 补充实测）

`samples/` 下三份样例，对应三来源各挖一份：

- `bilibili_duo_pair_sample.jpg` — BV11XWR6ZERs（BESTie 双人翻跳正面机位）：yt-dlp 480p 下载 → ffmpeg 抽帧 96 张 → 人工挑 t=14s / t=60s 两帧，同双人同练习室、姿态差异大。**管线全通**。
- `wedding_gallery_pair_sample.jpg` — Junebug 真实婚礼特辑（Hycroft Manor，62 张同 couple 画廊）：普通 HTTP 抓文章 → images.junebugweddings.com CDN 直链 → 人工挑 w_28 / w_29（同一面爬山虎石墙、站姿不同）。**管线全通**；注意纪梵希旅拍站是 XHR 懒加载不可用，公众号文章（mmbiz 图床）通道已验证可用但需先解决文章检索（搜狗微信的 /link 跳转有反爬）。
- `pinterest_harvest_sample.jpg` — 板集 8 pin 直采（pinimg 736x 直链）。** harvesting 通，配对未通**：board 页 DOM 变体多变、pin 链接时有时无，同 shoot 配对需下一轮做 pin 详情页/carousel 挖掘（__PWS_DATA__ 里 id 字段格式待适配）。

## 8. 与其他工作的关系

- 模板与验收标准来自 `two-man-data/pairs/pairs/pair_001_27_68e500da`（xhs 呆呆比耶数据系）；
- 消费方是 `two-man-data` 既有的两套配对管线（照片对 / 视频对），web 原料只是**新上游**；
- xhs 账号管线（author 下载）暂停中（风控），恢复后与本线并行，互不依赖。
