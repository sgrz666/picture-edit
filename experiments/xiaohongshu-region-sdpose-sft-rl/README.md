# 小红书：空间区域目标、SDPose 与 GRPO

## 实验目的

在 `/data/shangguanrz/datasets/xiaohongshu(single)` 比较无加权／空间加权及继续 SFT／GRPO。SDPose 同时用于训练骨架和生成评价，保留所有生成图。用户已授权扩大为 **512 train／40 val／64 test**；沿用 note 和重复 RGB SHA 隔离，选取身体／手部可评测的固定子集，不声称覆盖原始 1,307 对全部分布。

扩大版锁定 141 train notes／16 val notes／20 test notes，共 747 张不同原图。已核对可逐 SHA 复用旧 172 帧几何，其余 575 帧在依赖作业结束后补算。训练样本从 78 增至 512（6.56 倍），SFT 每臂从 300 增至 **1,000 updates**（Body 200、Head 150、Full 400、Joint 250），accumulation=4；继续 SFT／GRPO 各 **50 updates**，仍为单 seed。

## 内容与使用

`scripts/prepare.py --expanded` 锁定笔记级划分、重复 RGB SHA 分组和轮流取样；`geometry.py --reuse-index` 校验后复用已有真实几何并补算新增帧；`assets.py` 构建 Native 数据、新 SDPose 条件及逐手区域侧车。`waiting.py` 以 boot id／start ticks 绑定指定 PID，显存探测只调用 NVML，不创建 CUDA context，瞬时查询失败会重试。

当前扩大版根为 `/data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-expanded-v1`。`launch_request.json` 固定 PID **1916769**（start ticks `366098433`）、起始权重、样本、预算和资源限制；结束后检查至少 64 GiB 空闲显存再自动接续，Torch 单进程上限为总显存 75%。启动凭据为 `logs/automatic-v1.pid.json`，阶段事件为 `logs/events.jsonl`；文件锁拒绝重复程序。旧 78／10 根 `...-20261121-v1` 原样保留，其旧 CUDA 显存探针曾因 OOM 退出，不再作为有效等待器。

测试源有独立 `test` 角色，不进入 optimizer pools、训练路由、validation loss probe、RL reward、checkpoint 选择或调参。所有优化完成后，initial／uniform_S3／regional_S3／continue_sft／grpo 各生成固定 **64 对 test 图，共 320 张**（seed42、20 steps），然后用 SDPose＋身份／背景评价。val 的阶段图与 test 最终图分别存放；汇总逐图检查 split、pair ID、真实 checkpoint SHA 和图片 SHA，输出各自的 8 列面板（Source、SDPose、GT、五份权重）。

## 先前 pilot 的准备证据

磁盘已核对：原始数据 1,307 pairs／198 notes／942 frames，SDPose 942 帧已完成。seed42 的完整划分为 train 1,091 对／158 notes、val 88 对／20 notes、test 128 对／20 notes；同一 note 和重复 RGB SHA 不跨 split。Pilot 锁定 78 train／10 val、176 张不同原图，已完整导出真实 SMPLest-X 参数。

首轮旧 YOLO 检测器漏帧被完整性门禁拦下，失败日志与结果保留。第二轮只在 run-local estimator 副本中，把 crop 框输入替换为已固定的 SDPose／YOLO11x 人体框；不改 SMPLest-X 模型或裁剪／投影算法，不改原图字节。176／176 帧尺寸与参数顺序均通过核对。

按用户最新指示，共同初始化改为 Bilibili 2 万步：`/data/shangguanrz/pic-edit-experiments/one/bilibili-single-spatial-train-20261108-ungated-v1/stages/S3/checkpoints/update_00020000_f51df6cd2ace4a97b06ab037bd03b939.pt`，SHA256 `d1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3`。实际 S3 已完成 4,000／4,000，累计 20,000，commit／父链／journal 已核验；不再使用 SourceBG192。通过 V5→V6→V6.1→V6.2→V7 显式迁移保留已训练参数，新增架构参数按 seed42 初始化，逐次序列化严格读回；不继承旧优化器。目录迁移前的旧路径只作来源记录，原 checkpoint／commit 不改写。

这轮测的是从共同 2 万步起点出发的小红书增量微调收益；旧起点本身已有历史训练与旧空间策略，不解释成从零训练的因果对照。两个新臂只有区域目标不同，必须用各臂新产物评价。

迁移后共同初始化为 `<run>/initialization/v7.pt`，SHA256 `57022f5cdcae2931f30ec93090c6e53626089eb69f8893761d8e9c901448060e`。CPU 实权重验收确认 1,054 个继承张量／225,955,980 个元素与源权重逐值相等；176 个 mesh 均为正相机深度，真实一帧 importer／完整 mesh FK 检查通过，表达基绑定一致。小红书新架构训练更新数当前为 0。

真实 SDPose 原始 heatmap score 最大 1.14435，ROI 的 q 显式裁到 `[0,1]`，不改原 SDPose JSON。先前本地与服务器各 87 项针对性回归通过；扩大版新增 PID／NVML、test 隔离、样本均衡、几何复用和真实测试权重绑定回归，当前本地合计 98 项通过。validation 和 test 汇总分别用合成图及真实格式 committed checkpoint 做过 1,000-step 轨迹核验冒烟；这些测试不代表真实训练收益。小红书 GPU 训练尚未开始。

先前源码 `source/runtime-v2` 的包 SHA256 为 `1fbbaf39346954747524b850d8253ea7fb78aa401b96308103e9af895e93ff15`，其 PID 1967731 已退出，旧日志保留。当前扩大版仅使用新目录和新启动凭据，模型字节仍固定为上述 Bilibili20k／V7 SHA。

查看进度：

```powershell
ssh sg "tail -n 12 /data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-expanded-v1/logs/events.jsonl"
```

本地复测数据隔离、置信范围与单人 Canvas 门禁：

```powershell
python -m pytest tests/test_region_pilot_automation.py tests/test_native_test_split.py tests/test_xiaohongshu_region_prepare.py tests/test_native_v62_conditions.py tests/test_region_objective.py tests/test_region_supervision.py tests/test_native_region_backend.py -q
```

`scripts/audit_initialization.py --root <run> --repo <冻结源码目录> --output <新的报告路径>` 可在 Qwen21 环境用 CPU 复核继承张量、176 帧原图／几何和真实 importer／FK；第一次审计结果在 `report/initialization-acceptance-v1.json`，本地镜像在 `report/live-run/`。不把该 CPU 验收称为真实 GPU 梯度验收。

扩大版的 CPU 状态命令为 `Qwen21-python <run>/scripts/status.py --root <run>`，可检查依赖 PID、启动器、预算、阶段与终态，不加载 CUDA。最终验证图在 `output/generation/<权重>/`，测试原图在 `output/generation/test/<权重>/`，RL 候选图在 `output/grpo/rollouts/`，逐图指标在 `report/evaluation/`；`report/comparison-ready.json` 与 `report/test-comparison-ready.json` 分别指向验证／测试表和面板。重新出图和复测报告必须使用新名字，旧产物保留。

## 与已有实验的关系

沿用 `bilibili-single-spatial-train-20261107/region-sdpose-sft-rl` 的目标设计与评估，旧 Bilibili run 已在初始生成的单人 Depth Canvas 门禁报错退出，没有新增 SFT 权重；其缓存、源码和日志保留。小红书是新 run，不复用旧样本或旧条件缓存。Native 几何由 SMPLest-X 另行估计，SDPose 只承担 2D 骨架／区域／评价职责。
