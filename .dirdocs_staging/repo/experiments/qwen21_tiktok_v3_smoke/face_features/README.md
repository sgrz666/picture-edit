# experiments/qwen21_tiktok_v3_smoke/face_features/ —— 面部特征离线预缓存（冒烟规模）

> 叶子目录，10 个 `.pt` + `precompute_stats.json`　|　2026-10-12

## 干什么
`face/` 训练所需的面部识别/感知特征张量的预计算产物（人脸嵌入、ROI 特征等），按编号存为独立 `.pt`。

## 实证
`precompute_stats.json`：apps=2、sources=8（source_valid=8）、targets=16（target_visible=16）——8 源脸 / 16 目标脸全部通过可见性校验。10 个 `.pt` 文件名如 `00001.pt … 00127.pt`。

## 注意
仅覆盖冒烟样本；不可作为全量 TikTok 面部特征库。删除会让 `face/` 重跑触发预计算。
