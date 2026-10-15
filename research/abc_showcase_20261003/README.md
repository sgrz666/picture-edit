# A/B/C 候选数据集官方公开样例（2026-10-03）

下载位置：/data/shangguanrz/dataset_samples/abc_showcase_20261003/
目的：在正式申请/下载完整数据前，用官方公开样例目检真实感、人物构成与清晰度。

## freeman/  （需申请：签协议 + 表单 → HF/OpenXLab 权限）
- FreeMan_paper_Scenes1.png / Appendix_scene_example.png：论文 10 类日常场景实拍
  （Sport Court/Corridor/Lobby/Library/Dance Room/Park/Cafe/Platform/Courtyard/Square），
  全部亚洲真人、真实光影。来源 arxiv.org/html/2309.05073v4。
- 注意：样例为论文拼图；部分画面有棋盘格标定板、人物偏小；完整包 1920x1080 需申请。
- A/C 用法：同 session 同相机不同帧；SMPL 为粗拟合，需自补 SMPL-X/DWPose。

## dna_rendering/  （完整数据需申请：协议 + 机构邮箱；Part1 约 192GB）
- DNA_sample_20220419_yuxinrui_*.mp4 / 20220816_zhangchu_*.mp4 / BENGAMIM_*.mp4：
  官网首页按演员公开的样例（抠像人物 + SMPL 网格叠加），中文名演员为主，另 1 名外籍。
- DNA_teaser.mp4 / DNA_final_compressed.mp4：官方宣传片。
- 来源 dna-rendering.github.io。完整包 12 视角 4096x3000，Part 1/2 含 SMPL-X+mask+相机。

## mvhumannet/  （HF 门控：GAP-LAB-CUHK-SZ/MVHumanNet 需获批；表单可先要 100 套样本）
- MVHumanNet++_Teaser.jpg：官方宣传图。
- MVHumannet_sample_0380_img.jpg：repo 公开的原始相机帧（绿幕穹顶棚拍、亚洲真人、衬衫日常装）。
- 000075_smplx.obj / 000075.json / projected_smplx.png / sample_100_description.json：
  SMPL-X 网格、相机参数与文本描述样例。来源 github.com/GAP-LAB-CUHK-SZ/MVHumanNet（已克隆）。
- 注意：公开包图片缩小为原尺寸一半并遮挡部分背景，清晰度须实测。

## humanvid/  （真实部分公开：Pexels 视频 URL 清单，官方 repo 提供）
- HumanVid_pexels_12908928 / 3048882 / 4324564 .mp4：Pexels 真实 1080P 视频
  （已在本机 add/datasets/HumanVid/raw_cache，此前管线下载，共 728MB / 46 段）+ 每段抽帧。
- 日常环境真实光影，但无 SMPL-X 真值；注意官方 HF 新增部分是合成数据，勿混。

## pku_dymvhumans/  （HF 完全公开：zxyun/PKU-DyMVHumans，part1/part2 共约 60 个 tar）
- Dance_Dunhuang_Single_f14_sample/：敦煌舞单人序列样例（HTTP Range 取 tar 头部 12MB 解出）：
  overview_fme_200.png（60 相机总览）+ cam_10 若干完整帧 PNG。
- 真实中国舞者，但为暗光舞台环境、非日常场景；人物在远景中偏小；无 SMPL-X（需自行拟合）。
- 完整 tar 单个 7-19GB；CDN 到本机较慢（约 40KB/s，走 hf-mirror）。

## 结论速览（针对日常真实 + 亚洲为主 + A/B/C 可构建）
- 日常场景最贴：FreeMan（需申请）> HumanVid 真实部分（可立即批量取）
- 高清 SMPL-X 现成：DNA-Rendering Part 1/2（需申请，棚拍）
- 身份/服装规模：MVHumanNet++（需获批，公开图有缩水）
- 完全公开可直接下：HumanVid(Pexels)、PKU-DyMVHumans
- 无任何一家提供现成严格跨身份 A/B/C；A/C 同人不同帧 + 跨身份 B 检索的路线不变。
