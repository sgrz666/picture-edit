# PromptHMR Hi4D 四图试跑

本实验补齐 PromptHMR 的独立环境、权重路径和自动框输入，在已有四模型拼图的相同 4 张原图上，比较 box-only 与 interaction 两个设置。text、mask 均为空，使用真实相机 K。

代码在 `scripts/`，固定配置在 `data/config.json`，输入契约测试在 `tests/`，最终轻量证据副本在 `report/summary.json` 和 `report/model.json`。服务器产物写 `/data/shangguanrz/pic-edit-experiments/mete-test/hi4d-prompthmr-comparison-20261106/four-images-v1`，包含冻结的 `request.json`、权重审计 `model.json`、两臂各 4 份 `geometry.npz/result.json`、`summary.json` 和 `figures-v1/` 的 12 张诊断图。旧模型、原缓存和旧拼图均只读。

2026-11-06 实测：两臂各 4/4 成功，总计 8/8 双人输出；完整 checkpoint 765 个键 strict=True 加载，missing/unexpected 均为空。导出的 body/shape/translation/mean-hands 参数重新构建网格，8 次最大残差均为 0；原图 K 与模型 K 逆仿射投影最大差 `0.00004946 px`。服务器 18 项契约测试全部通过，本地 17 项通过、1 项因 Windows 符号链接权限跳过，服务器已覆盖该分支。峰值 PyTorch allocated 显存 `3,257,196,032 bytes`（约 3.03 GiB）；首次冷推理 3.574 s，其余前向 0.099–0.135 s，不含加载、输入准备、导出或绘图。共享 GPU 上的耗时仅供运行记录，不能作为独立性能基准。

8 份实际网格下载到本聊天的可视化目录后再次计算 SHA256，全部与逐帧记录一致；`report/verification.json` 保存该核查。interaction 相对 box-only 的平均顶点位移在 separated/light/hug/severe 四帧分别为 8.00/5.23/10.11/25.93 mm；这只度量预测变化，不是 GT 质量改善。

已按同一相机把输出叠回 RGB。四张图中开启 interaction 的可见变化较小；复杂拥抱仍存在局部肢体与轮廓误差。本次没有做 GT 质量量化，也没有验证消穿透能力，不能将这 4 张试跑称为完整 Hi4D 评测。normal 来自 SMPL-X 网格，use_mean_hands=True 使用官方均值手姿，不代表恢复了精细手部或衣物表面。

上游为 `dual_mesh_refine_20261103_v1` 的 GT-free 记录及 MultiHMR 自动选人结果；下游为本聊天的 PromptHMR / Multi-HMR / Sapiens2 / Multi-HMR2 同图对照。PromptHMR 的框来源必须标注为 MultiHMR 自动框。

## 环境补齐

独立 venv 为 `/data/shangguanrz/pic-edit-runtime/prompthmr-hi4d-20261106/env`，使用 `mamma` 的 system-site-packages（PyTorch 2.7.1+cu128），仅在新 venv 安装 xformers 0.0.31。完整包清单保存于该 runtime 根目录的 `environment.json`。

RTX 6000D 是 SM120。xformers 自动选择的 Flash3 在实际 CUDA 探针报 invalid argument，因此在进程内明确选择官方 CUTLASS 实现；对 PyTorch SDPA 的 float16 探针最大绝对差 `0.00024414`，未修改模型文件。SMPL 原始 pickle 依赖 Chumpy，`prepare_assets.py` 用既有 `smpl_clean` 环境生成纯数组副本，10 个数值字段 array_equal，原 SHA256 保持 `94082a25a6cc161e71dc0052bf0b3dfb52c8ec5c387e81491a7104276685af82`；副本写 runtime 的 `assets/smpl/SMPL_NEUTRAL.pkl`，配置只在当前进程指向副本。

权重目录修正为现有 `data/phmr`。构造阶段的 CLIP 额外预训练下载被关闭，可能先打印随机初始化警告；随后从完整 release checkpoint 严格加载全部 765 个键，最终图像/文本编码器均来自 checkpoint，不使用随机权重推理。checkpoint SHA256 为 `a3ef04ef8a12c3068682b03c62c95f8959cd8554424e105c63bf97f6c8e97e99`。配置、模型 config.yaml、数值 SMPL、runner 和 contracts 的 SHA256 都已保存。

## 查看、测试和重跑

`figures-v1/four-models-contact.png` 是拥抱/重遮挡两行的四模型骨架图，`four-models-all.png` 是全部四行；PromptHMR 列采用 box-only。每列颜色区分本列人物，跨模型颜色不表示身份一致。Sapiens2 在手权重只有二维输出，四模型图统一比较 12 个身体点投影。

`figures-v1/prompthmr-mesh-normal-contact.png` 对照原图、box-only 网格、interaction 网格和两份 normal；`prompthmr-mesh-normal-all.png` 包含四张图。normal 用 `(nx,-ny,-nz)` 编码，黑色为背景，渲染使用统一的真实相机和透视 z-buffer。

登录 `ssh sg` 后执行测试：

```bash
cd /home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/hi4d-prompthmr-comparison-20261106
PY=/data/shangguanrz/pic-edit-runtime/prompthmr-hi4d-20261106/env/bin/python
$PY -B -m unittest discover -s tests -v
```

重跑必须使用新名称，已有运行默认拒绝覆盖：

```bash
PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/run_prompthmr.py --config data/config.json --name four-images-v2
OUT=/data/shangguanrz/pic-edit-experiments/mete-test/hi4d-prompthmr-comparison-20261106/four-images-v2
$PY -B scripts/render_results.py --run-dir "$OUT" --output-dir "$OUT/figures-v1"
```

每份 npz 含 2×10475 顶点、faces、22 关节 body/global rotmat、betas、transl、独立左右均值手姿矩阵、原图和模型 K、逆仿射信息、原图 12 点/25 点投影。模型返回的含 epsilon/clamp 的 2D 输出另外保留，不混作独立相机一致性校验。`smoke01` 的历史加载失败和 `smoke02` 的单图双臂成功记录也保留在服务器，最终使用 `four-images-v1`。
