# experiments/qwen21_tiktok_v3_smoke/ —— Qwen21 身体+面部双适配器在 TikTok v3 上的 2 步冒烟

> 路径：`/home/shangguanrz/project/pic-edit/experiments/qwen21_tiktok_v3_smoke`　|　本级：4 个子目录 / 直属 0 文件　|　子树：39 文件 / 2.3 GB，0 张 PNG
> 数据来源：`body/run_manifest.json`、`face/manifest.json`、`face_features/precompute_stats.json`，读取 2026-10-12

## 一、这一层在干什么
把 `qwen21_iper_body` 训好的 Qwen21 单流身体适配器迁移到 TikTok v3 数据，并新增面部适配器（解耦的身体/面部两段注入）。本目录**只是 2 步冒烟**：验证 body→face 的加载链、面部特征预计算和离线潜变量缓存管线能跑通，**不含任何质量结论**。归属 Qwen21 身体控制族（与 `qwen21_iper_body` 同一基座、同一 `body_adapter`）。

## 二、层级地图
| 子目录 | 装什么 | 内容实证 | 关系 |
| :--- | :--- | :--- | :--- |
| `body/` | TikTok 身体适配器 2 步冒烟 | `checkpoint_best/last.pt`（771.9 MB 级）、`train_metrics.jsonl`（末行 step=2, loss 0.1431, val_loss 0.5235, peak 30.76 GiB）；`run_manifest.json` 记 `v6_checkpoint=iper_v6_native_20k_v1/checkpoint_last.pt`、TikTok_v3 `pairs_strict_v1` 清单 | 供 `face/` 初始化 |
| `face/` | 面部适配器头 2 步冒烟 | `checkpoint_best_loss/last/step_000002.pt`（各 215.9 MB，仅头部参数）；`train_metrics.jsonl` 末行 step=2：flow_loss 0.1732、identity/perceptual/outside_loss 均 0.0、val_velocity_mse 0.3150, peak 20.998 GiB；`manifest.json` 记 `body_checkpoint=body/checkpoint_last.pt` | 依赖 `body/` |
| `face_features/` | 面部特征离线预缓存 | 10 个 `.pt` + `precompute_stats.json`：apps=2、sources=8(source_valid=8)、targets=16(target_visible=16) | `face/` 训练输入 |
| `qwen21_cache/` | 身体 VAE 潜变量缓存 | 9 个 `source_<hash>.pt` / `target_<hash>.pt` | `body/` 训练输入 |

## 三、关键文件与实证
- `body/run_manifest.json`：`model=Qwen-Image-2.1`，TikTok 清单在 `datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/`，`align_mode=square_crop`。
- `face/manifest.json`：`train_sha256 / val_sha256` 锁定数据、`auxiliary_decode_resolution=512`、`body_checkpoint` 指向 body 输出。
- `face_features/precompute_stats.json`：8 个源人脸、16 个目标人脸全部通过可见性校验，说明面部特征提取链路健康。

## 四、注意事项 / 调用路径
- 本目录是**冒烟**，检查点不可用于任何评测/生成；正式身体权重仍是 `experiments/qwen21_iper_body/full_20k/checkpoint_best.pt`。
- `qwen21_cache/` 与 `face_features/` 是训练消费的离线缓存；若 TikTok v3 正式训练启动会复用其生成逻辑，但本目录内的少量张量仅覆盖冒烟样本，不代表全量缓存。
