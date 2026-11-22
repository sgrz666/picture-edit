# 小红书：空间区域目标、SDPose 与 GRPO

## 实验目的

在 `/data/shangguanrz/datasets/xiaohongshu(single)` 比较无加权／空间加权及继续 SFT／GRPO。SDPose 同时用于训练骨架和生成评价，保留所有生成图。用户已授权扩大为 **512 train／40 val／64 test**；沿用 note 和重复 RGB SHA 隔离，选取身体／手部可评测的固定子集，不声称覆盖原始 1,307 对全部分布。

扩大版锁定 141 train notes／16 val notes／20 test notes，共 747 张不同原图。已核对可逐 SHA 复用旧 172 帧几何，其余 575 帧在依赖作业结束后补算。训练样本从 78 增至 512（6.56 倍），SFT 每臂从 300 增至 **1,000 updates**（Body 200、Head 150、Full 400、Joint 250），accumulation=4；继续 SFT／GRPO 各 **50 updates**，仍为单 seed。

## 内容与使用

训练已实际开始。最近核对无加权 S0 **133／200 updates**，梯度有限且非零；747 帧 Native 装配、616 对缓存绑定、真实 data／grad 门禁和首轮 40 张初始化验证图均已完成。当前 SGD **PID 3346829** 保留，吞吐主管 **PID 3413941** 已接管原主管等候工作，未重启或重置当前优化器。原主管退场不代表 SGD 退出；查看 `logs/throughput-v1.log`／`logs/throughput-v1.pid.json`。

为减少阶段间等待，完整验证／测试出图与指标统一后置到全部优化完成之后；各阶段 checkpoint 和 generation config 照常锁定，真实梯度门禁保留。训练核心仍使用 `source/runtime-v3`，只更换独立编排脚本，未改精度、学习率、采样、初始化或预算。GPU 原观测 utilization 91%，此优化减少的是阻塞式出图墙钟，不声称单步算子已经提速。

图片已增加到 **1600 张**：五份最终权重的 test 各 64 对×3 seeds（42／123／2026），共 **960 张**；最终 val 各 40 对×2 seeds（42／123），中间六阶段 val 各 40 对×seed42，共 **640 张**。现有 40 张 seed42 初始化图复用不覆盖。每份权重只加载一次 Qwen 连续完成所有 split／seed，SDPose／身份／背景集中批量评价，再按模型／split／seed分组；seed42 主对照路径保留，额外种子在 `output/generation/multi-seed/` 与 `report/evaluation/multi-seed/`。

图片计划凭据为 `report/deferred-image-plan.json`；只有全部优化完成才写 `optimization-complete.json` 并启动完整图片批次。多种子图片／指标完成后记录 `report/deferred-render-complete.json`，再生成原验证／测试主对照报告。原始 source／SDPose／GT／初始化生成图的 6 对预览在 `report/baseline-panel-v1.png`，它不是小红书新训练权重效果。

吞吐／出图代码包 SHA `6f4550181e1848e85044e5cae7760a75387fbc9220058ed11dd3404c673640ff`，本地 36 项相关回归、服务器 4 项真实脚本回归通过，覆盖活动训练接管、多 seed 数量、同模型只加载一次、既有图片复用。当前仍在训练，不把计划图片数当作已生成数。

`scripts/prepare.py --expanded` 锁定笔记级划分、重复 RGB SHA 分组和轮流取样；`geometry.py --reuse-index` 校验后复用已有真实几何并补算新增帧；`assets.py` 构建 Native 数据、新 SDPose 条件及逐手区域侧车。`waiting.py` 以 boot id／start ticks 绑定指定 PID，显存探测只调用 NVML，不创建 CUDA context，瞬时查询失败会重试。

当前扩大版根为 `/data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-expanded-v1`。`launch_request.json` 固定 PID **1916769**（start ticks `366098433`）、起始权重、样本、预算和资源限制；结束后检查至少 64 GiB 空闲显存再自动接续，Torch 单进程上限为总显存 75%。启动凭据为 `logs/automatic-v1.pid.json`，阶段事件为 `logs/events.jsonl`；文件锁拒绝重复程序。旧 78／10 根 `...-20261121-v1` 原样保留，其旧 CUDA 显存探针曾因 OOM 退出，不再作为有效等待器。

扩大版首个启动器 PID 2226038 已退出：依赖 PID 1916769 结束后，575 个新增几何完成，随后 SAM2 因 Qwen21 缺少 `hydra` 而在 Native 资产开始前失败；没有启动 SFT 或生成 test 图。部署时缺少真实 SAM 环境检查，不能把之前的排队状态当作训练进展。938 个源码 SHA／编译和 98 项回归已通过，但不替代此 GPU 环境验收。

已用 `prepare_sam_dependencies.py` 从现有 DeepGen 环境原样复制 5 个纯 Python 包到本实验 `source/sam-dependencies-v1`，276 个文件按 SHA 固定，拒绝二进制模块与 Torch／diffusers／transformers；不修改 Qwen21 环境。依赖 manifest SHA 为 `d0a1424cad02401a07ac76e2d955525d26adc895955277bc859f96b26fa61981`。真实 SAM GPU 单帧通过：score .94922、人体提示点覆盖 1.0、峰值 allocated 1.591 GiB；框架仍为 Torch 2.8.0+cu128／diffusers 0.41.0.dev0。

Hydra 依赖恢复后，旧主管 PID 2939783／资产子进程 2939801 已退出：第 130 帧为植物遮挡人像，SAM body 提示覆盖 .61538，低于 .65。原始图、三种提示诊断和所有失败日志保留。现在保留 .65 标注资格门槛；未通过分割的样本不丢弃，其分割相关 person／joint／hand／background 空间支持和 q 均置零，背景评价不计入；global FM 保留，缺失权重不重新分配。Source 的非空 SAM 代理只作弱条件，另行记录，不冒充合格 GT。

用户要求不再串行等待后，已拆为独立 Qwen 编码与 Native 装配。此前并行主管 PID 3078223、装配 PID 3079034 的凭据 `logs/parallel-v2.pid.json` 只作历史记录。747 帧所用的 616 个最终 Qwen 条件独立锁定，Native manifest 引用同一条件 SHA；编码无需 SAM／DINO／Native mesh。**616／616 对缓存编码完成**，峰值 allocated 17.49 GiB，索引 SHA `852db52f99065a53b39b870bad5648a47ea73e059be75b3a9313f15095b04620`；Native 现已 **747／747 帧完成**。

当前准备源码为 `source/runtime-v3`：从已验收 `runtime-v1` 原样保留训练核心，仅叠加准备脚本和回归；准备 delta SHA `e14d782698b252c4d69804467f171c19202973dae5021ab244c9012136d5a64f`，945 个文件校验／编译和服务器 **104 项回归通过**。Qwen 编码进程未因主管交接而重启。装配使用设备 20% 上限，编码使用 35%；正式训练仍按原请求使用 75% 上限／64 GiB 门槛。样本、预算、画布、主线权重、Qwen 和 evaluator 不变。

`cache_ready.json` 已经通过真实 DataPlan 对完整条件集合、Source／GT RGB SHA 和 Canvas ID 的逐项核对，训练已按上方当前状态启动。缓存记录在 `report/independent-cache-complete.json`，准备日志 `logs/parallel-v2.log`／`logs/native_assets_parallel_*.log` 保留。

测试源有独立 `test` 角色，不进入 optimizer pools、训练路由、validation loss probe、RL reward、checkpoint 选择或调参。所有优化完成后评价 initial／uniform_S3／regional_S3／continue_sft／grpo，均为 20-step 出图。val 与 test 分开存放；汇总逐图检查 split、pair ID、真实 checkpoint SHA 和图片 SHA。seed42 主对照的 8 列面板保持原口径（Source、SDPose、GT、五份权重），其他 seed 单独归档。

新运行使用共享 Native RL 更新内核，GRPO 的 50 updates 是 50 次 `optimizer.step()`。背景只奖励有像素对应证据的 Source 场景。当前配对清单没有提供这项证据，`background_reward` 为 `null`，`valid_mse` 保留作 Target GT 诊断。训练前固定每组 Body、Hands、ID 和背景的评分资格，生成漏检计零分，评测程序异常时该组停止更新并记录错误。

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

扩大版 CPU 状态命令为 `Qwen21-python <run>/scripts/status-throughput-v1.py --root <run> --launcher throughput-v1`，训练日志查看 `logs/throughput-v1.log` 和 `logs/train_*.log`，实际步数以 `output/<臂>/<阶段>/segments/*/metrics.jsonl` 的 committed update 为准，不加载 CUDA。吞吐和多 seed 复测为 `python -m pytest tests/test_deferred_rendering.py -q`（4 passed）。最终验证图在 `output/generation/<权重>/`，测试原图在 `output/generation/test/<权重>/`，额外种子在 `output/generation/multi-seed/`，RL 候选图在 `output/grpo/rollouts/`；主对照索引为 `report/comparison-ready.json`／`report/test-comparison-ready.json`。重新出图和复测报告必须用新名字，旧产物保留。

## 与已有实验的关系

沿用 `bilibili-single-spatial-train-20261107/region-sdpose-sft-rl` 的目标设计与评估，旧 Bilibili run 已在初始生成的单人 Depth Canvas 门禁报错退出，没有新增 SFT 权重；其缓存、源码和日志保留。小红书是新 run，不复用旧样本或旧条件缓存。Native 几何由 SMPLest-X 另行估计，SDPose 只承担 2D 骨架／区域／评价职责。
