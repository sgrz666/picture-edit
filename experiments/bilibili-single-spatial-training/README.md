# 单人 Native V5 空间加权正式训练

目的：在 Bilibili 原视频隔离的 train/val/test 上执行 S0/S1/S2/S3 共 20,000 次成功 optimizer 更新。使用 Sapiens2 v1 几何和全量 `visible_region_v2` 空间监督；保留所有样本及原质量标记，不做冒烟训练。

运行根为 `/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1`；本地隔离分支为 `codex/bilibili-single-spatial-20k`。本目录的 `scripts/` 提供后台主管、数据准备和阶段 worker；`source/legacy/` 保存既有数据装配工具原样副本。服务器 `source/remote-baseline-20261107T0113/` 固定了实施前代码，正式版本是独立的 `source/runtime_repo_v7/`，不修改远端主工作区。早期 `source/runtime_repo/` 与根 `scripts/` 保留原样，只作历史快照，启动使用下面的 v6 路径。

已确认划分为 train 9,520 对、val 1,490 对、test 1,146 对；已有 1,960 帧物理资产和 Source 参考特征，561 对 Native 配对产物可校验后复用。全量 v2 空间索引已生成并通过独立审计，Body Mask 保持原样。

阶段预算固定为 10,000/2,000/4,000/4,000，seed=42、micro-batch=1、accumulation=4、单 GPU、workers=0，long-side=512；Loss 为 velocity=1、Body=.5、Joint=.5、Hand=1、max=2.5，旧 Face/Hand 局部 velocity 和 ID/perceptual=0。训练对象是 Native Adapter，权重图保持冻结；Interaction 关闭。

2026-11-07 性能更新实施时，现行主管已在运行 `prepare_data`，optimizer 更新为0。v6 在完整数据准备被验收后自动接续，复用原产物；实际生效版本以 `status.active_repo_root` 和 journal 的 `code_revision` 为准。完整 Native 清单、冻结缓存和真实梯度验收由主管顺序完成，CPU 回归不代替生产模型验收。

## 启动、查看与恢复

先 `ssh sg`，以下命令在服务器执行。启动只需一次；原始 `run` 拒绝已有 journal，恢复必须用 `resume`。

```bash
RUN=/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
REPO="$RUN/source/runtime_repo_v7"
SCRIPTS="$REPO/experiments/bilibili-single-spatial-training/scripts"
bash "$SCRIPTS/launch.sh" run --run-root "$RUN" --repo-root "$REPO" --python "$PY" --gpu 0
```

主管通过 `nohup + setsid` 脱离 SSH，关闭聊天不影响它。`fcntl` 锁保证单主管；独立 worker 锁保护仍存活的子进程。启动会核对已固定的代码 SHA，不允许中途改代码后继续同一运行。

```bash
# 状态包含当前任务、owner_alive、已提交更新、每阶段断点和有实测速率后的 ETA
"$PY" "$SCRIPTS/supervisor.py" status --run-root "$RUN"
ls -lt "$RUN/logs"
# 对上一步输出的具体日志文件执行 tail -f；主管和每次 worker 尝试各有独立日志

# 停止：只终止本运行的进程组，保留最近已提交边界及全部未提交证据
"$PY" "$SCRIPTS/supervisor.py" stop --run-root "$RUN"
# 等到 owner_alive=false 后恢复；服务器重启后同样使用此命令
bash "$SCRIPTS/launch.sh" resume --run-root "$RUN" --repo-root "$REPO" --python "$PY" --gpu 0

# 完整 SHA 验收，可能耗时；它不创建模型、不做训练
"$PY" "$SCRIPTS/supervisor.py" verify --run-root "$RUN"
```

`status=running` 同时检查 `owner_alive`；主机掉电后 journal 的旧状态可能仍是 running。`failed` 或 `completed` 是终态，拒绝自动 resume；SHA、资格、梯度或数值失败须先检查原因，不能删 journal 或修改失败记录绕过。瞬时/资源故障最多自动重试 2 次；不够显存时等待，不清理其他人的进程。停止可回退最多 99 个未提交更新，计数以 committed checkpoint 为准。

## 流程与产物

执行顺序是 `prepare_data → prepare_cache → data_preflight → S0.grad/train → S1.grad/train → S2.grad/train → S3.grad/train → eval_test → export_final`。每阶段真实梯度验收不创建 optimizer；成功后才允许该阶段训练。S1 暖输出 Head，S2 开对应局部分支，S3 联合训练；阶段间保留 AdamW moments、全局 RNG 和 FM-noise Generator，Sampler 按新阶段路线初始化。

每 100 次成功更新运行固定验证 loss 并保存；每 1,000 次及阶段末生成固定 10 对验证图，5 个验证视频各 2 对。每阶段单独建立 loss baseline，连续 3 次超过 2 倍或非有限值即停止。最终 loss 覆盖全部 1,490 对 val 和 1,146 对 test；test 只在完整 20k 后读取为评估源。test 生成图与像素指标只覆盖固定 10 对，像素指标不声明人工画质达标。

|位置|用途|
|---|---|
|`prepared.json`、`data_audits/`|全部 12,156 对 Native 真读取、空间图及输入绑定报告|
|`frozen_qwen/`、`frozen_qwen_test/`|带逐 pair commit 的冻结缓存；test 缓存独立|
|`journal/`、`requests/`、`results/`、`logs/`|主管 hash 链、输入、验收和每次尝试日志|
|`stages/S*/config.json`、`execution_config.json`|匹配阶段的冻结配置、梯度报告引用|
|`stages/S*/checkpoints/*.commit.json`|不可变模型/optimizer/RNG 完整恢复断点，每 100 更新|
|`stages/S*/output/images/`|固定验证图，文件绑定模型、数据、配置与种子|
|`evaluation/final_validation_loss.json`、`evaluation/test/`|全量最终 val/test loss，固定 test 图和像素指标|
|`export/adapter.commit.json`、`export/summary.json`、`export/README.md`|不含 optimizer 的推理权重、SHA、每阶段实际更新/尝试耗时/验证结果|

只有四阶段实际提交更新恰为 10,000/2,000/4,000/4,000、最终验证与独立 test、权重和摘要导出均成功，主管才写 `completed`。最终完整断点仍可恢复 optimizer；导出的 Adapter 使用 `NativeFullRuntimeV5.from_checkpoint(torch.load(path, map_location='cpu', weights_only=True))` 加载。

## 测试与交接

服务器 CPU 回归（不运行生产 optimizer）：

```bash
cd "$REPO"
"$PY" -m pytest tests/test_native_spatial_masks.py tests/test_native_spatial_weights.py \
  tests/test_native_spatial_audit.py tests/test_qwen21_native_v5_training.py \
  tests/test_native_unattended_core.py tests/test_single_spatial_data_prepare.py \
  tests/test_single_spatial_supervisor.py tests/test_qwen21_fixedpair_training_v5.py -q
```

真实旧产物复用检查命令是 `"$PY" "$SCRIPTS/data_prepare.py" verify-reuse --run-root "$RUN" --repo-root "$REPO"`，只用 CPU 和真实 Native reader；全量 12,156 对读取由 `prepare_data` 门禁执行。运行时数据准备使用原 CUDA 几何 rasterizer，不重新拟合 SMPL-X。空间图和旧产物先校验再复用，已匹配旧缓存的任何 SHA/Canvas/读取失败直接停止，不重建掩盖错误。保留局部浮点遮挡修正及原精细手部有效标记。NPZ 仅采用标准无损 DEFLATE level 3，数组/精度不变。

交付验证命令为 `"$PY" "$SCRIPTS/verify_delivery.py" --run-root "$RUN" --repo-root "$REPO" --deployment-manifest "$RUN/source/deploy_20261107_final_v7_manifest.json" --report-path "$RUN/source/delivery_verification_v7.json"`。首次执行验证部署 SHA、语法和上述 CPU 回归，并生成本版本独立的 `source/runtime_code_manifest_*.json`、`source/delivery_verification_v7.json`；再次执行校验既有交付闭包，不创建训练作业。监管 agent 可直接查看记录中的测试摘要和未完成门禁。

交给监管 agent 时先读 `HANDOFF.md` 和最新状态，关注 `prepare_data`、`prepare_cache`、阶段切换、首批 optimizer 更新与终态。不要额外启动冒烟训练、重复数据作业或修改已固定的运行代码。GPU 查询在该机 NVML 版本不匹配时使用 CUDA runtime；仅对本运行设置 `cudaMallocAsync`，不修改驱动或环境包。

上游是 `/data/shangguanrz/datasets/bilibili`、既有 Bilibili Native 准备目录及 `native-spatial-supervision-20261106-v1`；下游复用仓级 `native_v5/training` 正式接口。

## v6 等价提速与授权代码更新

只改变三处工程实现：采样阶段利用已验收的 scene kind，完整样本在前向时只读一次；梯度finite/nonzero/norm逐项照旧计算，在同一device汇集标量后一次回传；参数绑定/冻结检查每轮只枚举一次模型参数。B1累积4、全部20k成功更新、sigma/noise序列、FP32 Adapter和AdamW moments、loss/LR、验证/出图/保存频率均不变。SHA与场景失配仍停机。

`audit_throughput.py` 将v5/v6源码差异限定为上述3个核心文件，比较3对真实Native样本所有字段，并用真实Adapter参数形状的合成CUDA梯度缓冲比较四阶段审计结果、缓冲区SHA和耗时。它不加载Qwen、不创建optimizer、不执行训练更新；此计时仅代表审计环节，不能当作整程训练倍速。结果在运行根 `source/throughput_audit_v6.json`。生产梯度验收仍由新版本正式入口重新执行。

用户已明确授权解除旧代码SHA对本次修改的限制。实现为追加新代码指纹/plan引用，原 `run_plan.json`、日志和旧版本全部保留；数据、模型、缓存、checkpoint SHA继续严格。`update_training_code.py` 等待prepare_data验收通过后停止旧主管，确认无生产grad/train尝试或checkpoint，再追加 `code_revision` 并 `resume` 同一run，已验收任务不重做。只允许训练前切换；若已开训则拒绝，避免丢失optimizer/RNG或重新计数。

一次性后台接续命令（相同控制锁拒绝重复接续）：

```bash
bash "$SCRIPTS/launch_code_update.sh" --run-root "$RUN" --repo-root "$REPO" \
  --authorization-json "$RUN/source/performance_update_authorization.json" \
  --audit-report "$RUN/source/throughput_audit_v6.json"
ls -lt "$RUN"/logs/code_update_* "$RUN"/control/code_update_*
```

如已在后台等待接续，不要再启动第二个主管。主管状态多出 `active_plan`、`active_repo_root`；旧脚本不能解析版本更新，后续恢复使用v6入口。`code_update_completed_*.json` 表示代码接续成功，不能冒充20k训练完成。


v7采用按需冻结输入。所有12,156对Native数据已经完整真读验收；不再等待11,010对VLM/VAE缓存全部生成。固定描述文件绑定全部输入，每pair缓存独立SHA/commit，checkpoint绑定已使用缓存，缺失时按原采样顺序等待编码，不跳样本。S0标准Body资格复用已通过Joint资格并逐项对照；S1/S2/S3保留各自真实资格/梯度门禁。编码器仅编码时上GPU，随后退CPU；40GiB资源门槛保证显存余量，训练精度/loss/LR/B1累积4及20k预算均不变。实际开训以正式metrics中optimizer_updated=true为准。
