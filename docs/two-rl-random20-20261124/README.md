# two-rl 随机 20 对目检图（2026-11-24）

## 目的

从 `sg:/data/shangguanrz/datasets/xiaohognshu(two-rl)` 随机抽 20 对，按「左 A（source）｜右 B（target）」拼成两张对照图，用于快速目检双人配对的人数、身份与背景一致性。

## 抽样口径

- 来源：`pairs/pairs.jsonl`，过滤 `disk_status=complete` 且 `source_rgb_path` / `target_rgb_path` 均存在 → **814 对**可用（全库 2,395 对，其余为视频帧路径缺失）。
- 随机：Python `random.seed(20261124)`，`random.sample(rows, 20)`；**不做任何人工筛选**。
- 每对固定 A = `source_rgb_path`，B = `target_rgb_path`；图片按高度 420 px 等比缩放，不裁剪。

## 里面有什么

- `two_rl_random20_sheet1.png` —— 第 1–10 对（1544×2584）
- `two_rl_random20_sheet2.png` —— 第 11–20 对（1544×2584）
- `sampled_pairs.json` —— 20 对的 pair_id / note / source_tag / 文件名，供复核

## 抽到的 20 对

| # | pair_id | source_tag | note 前缀 | A / B 文件 |
|---|---|---|---|---|
| 01 | `pair_178_68e500da` | author1 | `68e500da` | `68e500da0000000004022164_2_05.jpg` / `68e500da0000000004022164_2_07.jpg` |
| 02 | `pair_255_6aeec43b` | author2 | `6aeec43b` | `6aeec43b000000000e026d54_16.jpg` / `69429554000000000d039059_3.jpg` |
| 03 | `pair_004_68e500da` | author1 | `68e500da` | `68e500da0000000004022164_6.jpg` / `68e500da0000000004022164_5.jpg` |
| 04 | `pair_046_68e500da` | author1 | `68e500da` | `68e500da0000000004022164_9.jpg` / `68e500da0000000004022164_5.jpg` |
| 05 | `pair_252_69786fad` | author2 | `69786fad` | `69786fad000000002202d7b2_6.jpg` / `6a101b1b0000000037036550_8.jpg` |
| 06 | `pair_273_6a958599` | author2 | `6a958599` | `6a95859900000000070043b5_7.jpg` / `6a95859900000000070043b5_6.jpg` |
| 07 | `pair_142_6976046d` | author1 | `6976046d` | `6976046d000000001a0226e8_3_02.jpg` / `6976046d000000001a0226e8_3_05.jpg` |
| 08 | `pair_181_6a69cd0b` | author2 | `6a69cd0b` | `6a69cd0b000000000301f351_16.jpg` / `6ac7854900000000140391f6_3.jpg` |
| 09 | `pair_034_6a69cd0b` | author2 | `6a69cd0b` | `6a69cd0b000000000301f351_17.jpg` / `6a69cd0b000000000301f351_4.jpg` |
| 10 | `pair_059_6a101b1b` | author2 | `6a101b1b` | `6a101b1b0000000037036550_10.jpg` / `6a113377000000003601c8ea_9.jpg` |
| 11 | `pair_167_68e500da` | author1 | `68e500da` | `68e500da0000000004022164_2_03.jpg` / `6976046d000000001a0226e8_8_07.jpg` |
| 12 | `pair_353_6a113377` | author2 | `6a113377` | `6a113377000000003601c8ea_6.jpg` / `6a113377000000003601c8ea_7.jpg` |
| 13 | `pair_006_69786fad` | author2 | `69786fad` | `69786fad000000002202d7b2_8.jpg` / `69786fad000000002202d7b2_12.jpg` |
| 14 | `pair_439_6a9291cd` | author1 | `6a9291cd` | `6a9291cd000000002501c3db_3.jpg` / `6a9291cd000000002501c3db_8.jpg` |
| 15 | `pair_371_6a113377` | author2 | `6a113377` | `6a113377000000003601c8ea_8.jpg` / `6a113377000000003601c8ea_9.jpg` |
| 16 | `pair_143_6a9fdd8d` | author2 | `6a9fdd8d` | `6a9fdd8d000000000d027646_3.jpg` / `6a9fdd8d000000000d027646_10.jpg` |
| 17 | `pair_253_68f77f85` | author1 | `68f77f85` | `68f77f850000000004016d63_8.jpg` / `6976046d000000001a0226e8_3_01.jpg` |
| 18 | `pair_078_68e500da` | author1 | `68e500da` | `68e500da0000000004022164_2_04.jpg` / `68e500da0000000004022164_2_06.jpg` |
| 19 | `pair_308_6951ff85` | author2 | `6951ff85` | `6951ff850000000022022704_10.jpg` / `69786fad000000002202d7b2_6.jpg` |
| 20 | `pair_002_68f77f85` | author1 | `68f77f85` | `68f77f850000000004016d63_3.jpg` / `68f77f850000000004016d63_8.jpg` |

## 复现

```bash
# 1) 远端抽样（814 对里取 20，打包到 /tmp/two_rl_sheet.tar.gz）
scp scripts/sample_and_pack.py sg:/tmp/ && ssh sg "~/nihaochen/envs/qwen21/bin/python /tmp/prep_sheet.py"
scp sg:/tmp/two_rl_sheet.tar.gz . && tar xzf two_rl_sheet.tar.gz
# 2) 本地合成（PIL，两列五行 × 两张，输出 out/）
python scripts/compose_sheets.py
```
