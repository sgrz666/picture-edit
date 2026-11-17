# Native V7 双人单配对过拟合

## 实验目的

验证真实 Qwen Image 2.1 + Native V7 能否学习一个固定双人姿态配对，同时保留两人的身份。训练只使用 `pair12/highfive12/cam4` 的 Source 000011 → Target 000082；同人物 Target 000112 仅观察，不参与优化，不作为泛化测试集。

首轮上限 300 个 optimizer updates，分为 Body 30、Head 30、PoseLink Full 120、Joint 120；必须逐阶段通过数据、真实梯度和冻结检查。固定 seed 42、原生 1280×960 Canvas、4Ref、B1，先尝试保持原几何与图像尺寸，不静默缩放。当前 GPU 有其他服务，保留其运行；实验进程最多使用显卡总显存的 50%，如果放不下，记录失败与峰值，再显式另建低分辨率对照。

## 里面有什么

- `scripts/audit_inputs.py`：核对 typed 样本、生产初始化权重和冻结 evaluator。
- `scripts/prepare_assets.py`：匹配真实检测的人脸与稳定人物槽位，复用 `training_sidecar()`、冻结 evaluator 工厂和原始 Source Mask 生成新侧车。
- `scripts/configure_run.py`：生成 SHA 绑定的 1 训练行、1 观察行与 30/30/120/120 阶段配置。Head 阶段同时预热 Face/Hand/Interaction/PoseLink，避免 Joint 门禁碰到零头。
- `scripts/gpu_action.py`：共享 Trainer 的实验包装，保留 50% 显存上限、首次 optimizer update 显存测量和严格执行策略身份；可选官方 VAE tiling 或精确 Hidden CPU 暂存。
- `scripts/run_training.py`／`launch.py`：逐阶段真实门禁、保存、技术 Body 检查、生成与 PID/状态；不自建训练循环。
- 服务器实验根目录：`/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1/`。新增合格监督与 repair 在 `data-v2/`；四图缓存在 `cache/native-4ref/`；生产迁移在 `initialization/`；所有实际日志、失败和资源峰值在 `logs/`、`report/`，不会覆盖旧 pilot。

## 检查标准与当前状态

1. 两个人的 GT 五点必须来自真实检测，并与稳定人物槽位及 Source 身份核对；不可见人脸不填伪标签。
2. repair 只使用源人物 Mask 和已核对的 Source→Target Canvas 变换，不能从 Target RGB 反推。
3. 从未对本实验目标配对训练的生产初始化迁移到 V7，不用旧 Hi4D 300 步结果或 CPU 合成权重冒充新实验。
4. 看固定噪声 FM、逐人物 ID/对比项、SDPose 姿态与可视化生成；记录训练与观察样本的差异。Loss 下降不单独证明成功，观察样本改善也不证明跨身份泛化。

目前优化更新次数为 **0**，不能宣称已经过拟合。已完成真实数据资格、生产迁移和四图缓存，但原生尺寸带完整 ID 监督的真实梯度预检遇到显存限制：

|预检方式|实测结果|
|---|---|
|原生 VAE 解码|在 41.52GiB allocator 上限触发 OOM；保留 ID Loss，没有优化更新|
|官方 VAE tiling|反向图仍占内存，仍触发 OOM；没有降低 Loss 或伪造门禁|
|大 Hidden 暂存 CPU，原生 VAE|暂存 34 个 Tensor、7,028,080,640 字节，仍在 VAE 卷积处触发 OOM|

训练帧两人的 Source–Target ArcFace 余弦为 **0.8364／0.8285**，观察帧为 **0.8656／0.8081**，交叉人物余弦为负；四个人脸实例都通过真实五点检测、唯一人物匹配及 ONNX/TorchScript ≥0.999 一致性检查。两条样本的辅助资格均为 `[true,true]`，repair 只用源 Mask。CPU数据准备回归与真实 CUDA Hidden 暂存输出/梯度逐值一致性检查通过。

显卡还有独立 ComfyUI 服务占约 35GiB，实验没有停止或清理它。进一步训练需选择低分辨率的完整 Canvas 重建、释放显存或 CPU 辅助解码；会为新策略建立独立记录，不把先前失败当作训练完成。

Hidden 暂存使用 [PyTorch saved tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)，只移动大的三维 4096 通道激活，排除权重和 Mask。对应策略、脚本 SHA、VAE 实现 SHA 都进入真实 Backend identity；没有修改图像、模型参数或损失计算精度。

复核数据（`--report` 必须是新的文件）：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
RUN=/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native-v7-singlepair-overfit-20261117-v1
$PY scripts/train/train_qwen21_native_full_v7.py preflight --level data \
  --config "$RUN/configs/native-v1/S0.json" --report "$RUN/report/data-check-new.json"
tail -n 5 "$RUN/logs/pilot300-offload-v1-events.jsonl"
cat "$RUN/report/pilot300-offload-v1/failure.json"
```

私有脚本回归：`python -m pytest tests/test_native_v7_singlepair_preparation.py tests/test_native_v7_singlepair_driver.py -q`。真实 CUDA 暂存测试在 CPU-only 机器会显式跳过；服务器可用 `V7_PILOT_SCRIPT_ROOT` 指定实验脚本所在路径后运行该测试，检查输出/梯度逐值一致。

## 与其他实验的关系

上游为 Native V7 架构（`1838e58`）和 Hi4D r4 grounded binding pilot 的两条 typed 4Ref 数据。本实验补齐训练资格与修复条件，沿用共享 Trainer；后续少样本、多人物组合训练以本实验的真实可学性与效果为依据。
