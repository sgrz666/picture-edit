# Bilibili 数据集 SDPose 骨架与数值关键点（2026-11-17）

## 结论

`/data/shangguanrz/datasets/bilibili` 全量 49 个视频、1,960 帧已用 **SDPose-Wholebody + YOLO11x** 跑完，
产出骨架渲染图 + 可计算的 134 槽数值关键点，**1,960/1,960 成功、0 失败、0 帧无人体检出**，耗时 18.0 分钟（1.81 帧/秒）。

## 产物位置

```
/data/shangguanrz/datasets/bilibili/geometry/skeleton/
├── sdpose_2d/<bvid>/<fid>_skeleton.png        1,960 张  黑底 134 槽骨架，与源图同分辨率（83 MB）
├── sdpose_2d_kp/<bvid>/<fid>_keypoints.json   1,960 份  逐帧坐标/置信度/有效掩码（39 MB）
├── sdpose_2d_kp/summary.json                            全量统计 + 逐视频明细
├── sdpose_2d_kp/run_manifest.json                       本次推理参数快照
└── sdpose_2d_overlay/<bvid>/<fid>_overlay.jpg 49 张     原图叠骨架目检图（每 40 帧抽 1，9.6 MB）
```

命名与既有 `sapiens2_2d/<bvid>/<fid>_skeleton.png` 完全一致，两个后端可**按路径直接互换**。

## 推理口径

参数全部继承 `src/evaluation/sdpose.py`（`docs/SDPOSE_POSE_EVAL_SPEC.md` 定义的项目统一后端），未在生成脚本里另设：
YOLO11x `conf=0.25 / iou=0.65 / classes=[0]`、`GetBBoxCenterScale(padding=1.25)`、
`TopdownAffine(768x1024, use_udp)`、`timesteps=[999]`、`flip_test=True`、float32；
有效性判据 `finite ∧ 画布内 ∧ score>=0.30`；槽位序 `geometry`（`14=R_eye, 15=R_ear, 16=L_eye, 17=L_ear`）。
单人舞蹈帧内多人时取**面积最大**的人体框，并同时记录 `raw_persons_detected` / `fg_persons_detected`。

## 质量指标

- 单帧有效点：均值 **129.12/134**，min 78，max 134；**1,932 帧 ≥100 点（98.6%）**
- 无人体检出帧：**0**
- 逐视频有效点最低 5 个：`BV1Aynt6zEQd` 111.83、`BV1Mfsd6qEtC` 118.03、`BV1f54A6CEiQ` 118.92、`BV1tqhz6uE2u` 120.47、`BV1dQx364Ex2` 122.17 —— 低值均因画面在膝下裁切、脚踝落在画布外被阈值正确剔除，不是漏检。
- 与既有 DWPose（`frames/poses`）全 1,960 帧逐点比对，torso 对角线归一化后：
  body-18 中位误差 **7.04 px**，`PCK@0.05 = 98.16%`、`PCK@0.10 = 99.95%`；
  肩/肘/膝中位 8–10 px，腕 11–12 px，髋 15–17 px（两模型对髋点定义差异最大）。

## 两个坑（已写入规范文档）

1. **眼耳槽位序不同**：`frames/poses`（DWPose）是 **openpose 序**（`15=L_eye, 16=R_ear`），
   `sdpose_2d_kp` 是 **geometry 序**（`15=R_ear, 16=L_eye`）。逐槽直比时这两个槽误差 61–132 px，
   交换后降到 1.3–8.7 px；其余 16 槽不受影响。跨后端比对必须交换。
2. **`CUDA_HOME` 必须显式设置**：本机无 `/usr/local/cuda`，`mmengine._strategy` import `deepspeed`
   会在版本探测处抛 `MissingCUDAException`，崩在 `pipelines.SDPose_D_Pipeline` 的 import 链上。
   固定 `CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21`（自带 nvcc 12.8.93，不编译扩展）。

## 复现 / 续跑

```bash
# 全量（已存在产物自动跳过，中断可续）
bash scripts/data/launch_sdpose_bilibili_skeletons.sh

# 与 DWPose 全量一致性复核
python scripts/audit/check_sdpose_vs_dwpose_agreement.py
```

本次运行日志：`/data/shangguanrz/pic-edit-experiments/sdpose_bilibili/logs/run_20261117_233347.log`

## 本目录图片

| 文件 | 说明 |
|---|---|
| `01_three_way_source_sapiens2_sdpose.jpg` | 三帧 × 三列（原图 / Sapiens2 / SDPose），含有效点最低的两个视频 |
| `02_overlay_arms_up_BV14NHK6SE9d_000095.jpg` | 双臂抬起帧，手臂/手指/脸部轮廓贴合良好 |
| `03_overlay_ankles_offcanvas_BV1Aynt6zEQd_000007.jpg` | 画面在膝下裁切，脚踝被 `score<0.30` 正确剔除 |
| `04_overlay_worst_agreement_BV1T3Hy6FErW_001172.jpg` | 与 DWPose 分歧最大帧（187 px），目检 SDPose 落点合理 |
