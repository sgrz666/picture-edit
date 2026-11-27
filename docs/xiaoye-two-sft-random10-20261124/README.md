# 小野一舟 two SFT 随机 10 对目检图（2026-11-24）

## 目的

从 `sg:/data/shangguanrz/xiaohongshu/author/3.小野一舟/two/pairs-final/sft` 随机抽 10 对，按「左 A（source）｜右 B（target）」拼成一张对照图，检查双人配对的姿态变化与背景一致性。

## 抽样口径

- 来源目录共 **1,344 对**（`pair_*` 照片 42 对 + `vpair_*` 视频 1,302 对），每个目录恰好一组 `a_*.jpg` / `b_*.jpg`。
- 随机：Python `random.seed(20261124)`，`random.sample(valid, 10)`；**不做人工筛选**。
- 图片按高度 420 px 等比缩放，不裁剪。

## 里面有什么

- `xiaoye_two_sft_random10.png` —— 10 对，2 列 × 5 行（1544×2584）
- `sampled_pairs.json` —— 抽中的 pair_id 与文件名
- `scripts/sample_and_pack.py` / `scripts/compose_sheet.py` —— 抽样与拼图脚本

## 抽到的 10 对

| # | pair_id | A 文件 | B 文件 |
|---|---|---|---|
| 01 | `vpair_1194_697b3f5f` | `a_697b3f5f000000001a0269b8_f00039.jpg` | `b_697b3f5f000000001a0269b8_f00279.jpg` |
| 02 | `pair_008_69_69f15a40` | `a_69f15a40000000001e00c549_3.jpg` | `b_69f15a40000000001e00c549_6.jpg` |
| 03 | `vpair_050_697765c4` | `a_697765c40000000028009c59_f00160.jpg` | `b_697765c40000000028009c59_f00282.jpg` |
| 04 | `vpair_1129_697debec` | `a_697debec000000001a0364a0_f00160.jpg` | `b_697debec000000001a0364a0_f00281.jpg` |
| 05 | `vpair_897_68e86b79` | `a_68e86b790000000005033ae4_f00640.jpg` | `b_68e86b790000000005033ae4_f00821.jpg` |
| 06 | `vpair_608_69f095c2` | `a_69f095c20000000023015a43_f00160.jpg` | `b_69f095c20000000023015a43_f00463.jpg` |
| 07 | `vpair_658_6ab50986` | `a_6ab50986000000001402e741_f00101.jpg` | `b_6ab50986000000001402e741_f00345.jpg` |
| 08 | `vpair_1175_689b17da` | `a_689b17da000000001c00b2eb_f00346.jpg` | `b_689b17da000000001c00b2eb_f00437.jpg` |
| 09 | `vpair_553_6a4f7f3c` | `a_6a4f7f3c00000000080322df_f00289.jpg` | `b_6a4f7f3c00000000080322df_f00401.jpg` |
| 10 | `vpair_525_694fe482` | `a_694fe482000000001e0225e7_f00099.jpg` | `b_694fe482000000001e0225e7_f00219.jpg` |

## 复现

```bash
scp scripts/sample_and_pack.py sg:/tmp/prep_xiaoye.py
ssh sg "~/nihaochen/envs/qwen21/bin/python /tmp/prep_xiaoye.py"
scp sg:/tmp/xiaoye_sheet.tar.gz . && tar xzf xiaoye_sheet.tar.gz
python scripts/compose_sheet.py
```
