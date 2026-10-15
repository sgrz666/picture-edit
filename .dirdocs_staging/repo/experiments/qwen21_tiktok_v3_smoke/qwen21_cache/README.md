# experiments/qwen21_tiktok_v3_smoke/qwen21_cache/ —— 身体潜变量离线缓存（冒烟规模）

> 叶子目录，9 个 `.pt`　|　2026-10-12

## 干什么
`body/` 训练消费的 Qwen21 VAE 潜变量缓存，命名 `source_<16字节hash>.pt` / `target_<hash>.pt`，按图哈希去重存储，避免每步重复编码。

## 实证
9 个 `source_*`/`target_*` `.pt`（仅覆盖冒烟用到的少量源/目标帧）。

## 注意
本目录内为冒烟小子集；生产级身体缓存全量位于 `nihaochen/qwen21_iper_cache`（iPER 线，他人目录，不可进入/删除）。此处缓存可按需重算，非权威资产。
