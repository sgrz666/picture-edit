# WP0 资产核实（node01 / sg 实测，2026-10-13）

只读检查，未改动任何既有文件。命令都在 `/home/shangguanrz/project/pic-edit` 下执行。

## 解释器

| 用途 | 路径 | 实测 |
|---|---|---|
| 几何 / 检测 / 评测 | `add/envs/smplestx_bw/bin/python` | `import torch,smplx,ultralytics,numpy,PIL` 通过（单人线 `--check` 已在用）；含 `onnxruntime`，可跑 DWPose ONNX |
| PE 与生成 | `/home/shangguanrz/nihaochen/envs/qwen21/bin/python` | torch 2.8.0+cu128，`from diffusers import QwenImage21Pipeline` 可用 |
| 双人 SMPL-X 拟合（WP7） | `add/envs/multihmr/bin/python` | 目录存在，multi-HMR 权重在位 |

## 模型与工具

- DWPose ONNX：`add/tools/dwpose_mimicmotion_official/models/yolox_l.onnx`、`dw-ll_ucoco_384.onnx` 均存在。`src/evaluation/dwpose_onnx/{onnxdet,onnxpose}.py` 只 import `onnxruntime`，模型路径由调用方传入（`eval_hi4d_qwen21_dual.py` 那样自己建 session）。
- multi-HMR：`add/tools/multi-hmr/models/multiHMR/multiHMR_896_L.pt`，入口 `demo_smplx_only.py`（`load_model` / `forward_model`），依赖 `add/tools/dinov2_e1277af`（通过 `MULTIHMR_DINOV2_REPO` 指定）与 `XFORMERS_DISABLED=1`，用法见 `research/hi4d_smplx_pilot_20261011/initialize_multihmr.py`。
- SMPLest-X（单人线用）：`add/tools/SMPLest-X`，其 `person_detection_confidence` 依赖本地 hook（`main/inference.py` 读 `SMPLESTX_DET_CONF`），该改动已冻结在 `workflow/patches/`。双人一期不用它。

## Hi4D 数据

`datasets/Hi4D_pilot_v1`（18G，实体在根分区）；`final_assets_v4_2/manifests/main_pairs_{train,val,test}.jsonl`，test 606 行，抽查首行：

| 字段 | 实测 |
|---|---|
| `source_rgb` / `target_rgb` | 存在，940×1280 |
| `source_person_mask_A/B`、`target_person_mask_A/B` | 存在，940×1280 |
| `target_dwpose_A/B` | `xy_native(134,2)`、`xy_normalized(134,2)`、`scores(134,)`、`width/height`、`generator` |
| `target_smpl` | `betas(2,10)`、`global_orient(2,3)`、`body_pose(2,69)`、`transl(2,3)`、`verts(2,6890,3)`、`joints_3d(2,24,3)`、`contact(2,6890)` → **SMPL，无手无脸** |
| `target_camera` | 8 相机 `ids/intrinsics/extrinsics/dist_coeffs` |
| `target_geometry_npz` | `semantic_A/B`、`depth_A/B`、`contact_A/B`、`normal_scene`、`K_runtime`、`front_owner`、`interperson_overlap`、`A_occluded_by_B`、`B_occluded_by_A`、`occlusion_mask` |

test split 全为 `N2C_SAME_IDENTITY_CAMERA`（同演员同相机），`pair_relation` CROSS_ACTION 276 / SAME_ACTION 330；`source_index_for_target` 取值 0/1/2（234/186/186），只在 `scripts/dataset_notes.py` 有一句"目标视角下的源索引"注释，**无任何代码消费**，不作为人物对应依据。

## 磁盘与 GPU

- 根分区 `/dev/sda2` 1.8T，已用 96%，剩 78G；Hi4D pilot 与 workflow 都在根分区上。
- `/data`（`/dev/sdb1`）22T，剩 21T，模型与归档在此。
- 单卡 83 GiB，实测空闲 60.7 GiB；单人线一次生成峰值 36.7 GiB。`pgrep -f workflow/run.py` 会把发起检查的 shell 自己也匹配上，判断进程时要看 `nvidia-smi` 之外再用 `ps` 精确匹配（本轮 `GPU_BUSY` 即该假阳性）。

## 一期写入预算

每 run 产物约 10–20 MB（两张骨架图 + 若干 npz/json + 生成图），16 对 × 3 臂 ≈ 1 GB，写在 `workflow-multiple/runs/`（根分区）可接受；若扩到百对级别，改 `--runs-root` 指到 `/data`。默认 `resolution` 取 768 而非单人线的 1024：双人线一次要给 4 张参考图，先保守，WP5 实测显存曲线后再决定是否提高。
