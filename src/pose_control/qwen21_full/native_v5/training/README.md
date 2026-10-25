# V5 Training Preparation v1

本训练层以sg部署`68daace`为基线，保留`native_full_v5`模型版本及全部参数名称／形状。Runtime、Integration、Local和编码器不改；只扩展既有Loss的显式权重和统计接口。旧Full工程机制被重新组织为V5管理代码，不导入DeepGen训练循环。

**本轮只实现代码和静态核对。没有执行测试或收集、数据预检、缓存构建、权重加载、模型前反向、optimizer或训练，没有安装依赖。** 下文命令均供后续单独启动，不表示已经通过运行验收。

## 安全入口与使用顺序

正式入口仍是`/home/shangguanrz/project/pic-edit/scripts/train/train_qwen21_native_full_v5.py`。没有动作参数时默认为`prepare`，只输出规范配置及缺失项，不加载数据、权重、模型或optimizer，也不会因传了`--profile`而训练。

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python

# 配置检查；示例预算和资产未绑定，只能prepare
$PY scripts/train/train_qwen21_native_full_v5.py prepare \
  --config src/pose_control/qwen21_full/native_v5/training/examples/s1_heads.json

# 以下步骤本轮没有执行：需先填写真实路径、SHA、预算、划分和缓存
$PY scripts/train/train_qwen21_native_full_v5.py preflight --level data \
  --config /data/shangguanrz/.../training.json --report /data/shangguanrz/.../data_preflight.json
$PY scripts/train/train_qwen21_native_full_v5.py preflight --level grad \
  --config /data/shangguanrz/.../training.json --report /data/shangguanrz/.../gradient_preflight.json

# 正式训练必须显式命令且配置引用匹配的真实梯度报告
$PY scripts/train/train_qwen21_native_full_v5.py train --config /data/shangguanrz/.../training.json
$PY scripts/train/train_qwen21_native_full_v5.py resume --checkpoint /data/shangguanrz/.../committed.pt
$PY scripts/train/train_qwen21_native_full_v5.py advance-stage \
  --checkpoint /data/shangguanrz/.../committed.pt --config /data/shangguanrz/.../next_stage.json
```

旧`--checkpoint/--manifest/--profile/--steps/--loss-profile/--lr`可转换成单阶段配置，`--preflight-only`映射到data preflight；新增必需的验证划分、group索引和资产绑定不能自动猜测。分组freeze／unfreeze／LR覆盖保留，违反阶段或inactive约束时报错。S2／S3从显式阶段推进启动，不把重置optimizer包装成续训。

`prepare`报告始终`training_ready=false`；真实门禁由后续preflight和执行入口检查。配置中的`gradient_report`是报告文件引用，不参与自己的计算配置签名，避免循环依赖；其SHA仍单独验证和保存在checkpoint里。

## 配置、阶段和参数角色

`TrainingConfig`分别保存Stage、Routes、Sources、Optimizer、Loss和资产绑定；`NativeV5Trainer`构造函数没有执行副作用。新增训练状态schema为`native_v5_training_v1`，不是新的Adapter架构。

|阶段|默认训练参数|控制与推进|
|---|---|---|
|S0|Native Body、Body21、双人Geometry、CHAMP CNN／Attention、Depth Detail／融合及Body Strength|关闭局部／独立Interaction／Source Detail；继承Geometry默认冻结|
|S1|Face／Hand／Interaction最终输出Head|Encoder、Controller及Strength冻结；Detail路由使用共享Face／Hand Head，不创建detail_heads|
|S2|所选Face／Hand／Detail／Interaction的Head、Controller、必要Encoder、对应Strength|其他分支冻结；显式选择分支|
|S3|全部有效分支，继承Geometry默认冻结|显式all-native／unfreeze_geometry才开放共享Geometry；默认LR倍率0.1|

阶段没有自动推进。直接从S1启动需Body验收JSON绑定本次初始化SHA，包含`body_accepted=true`和`manual_acknowledgment=true`；S0→S1推进也要求对应父checkpoint的Body记录。其他推进需父阶段有实际更新、目标阶段匹配的梯度报告，且所选新Head已非零。相同phase的显式修订可以修改预算／路由等并保留moments，不能伪装成严格Resume。

参数目录覆盖所有Adapter参数，按规范名称排序并拒绝重复对象／共享Head重复登记。Native Body的最终Head在训练层细分；Qwen、旧Interaction融合及当前模式不用的Head永久不进optimizer。精确逐参数白名单在安装后复查，不改网络结构。

|LR预设|Head／允许训练的Strength|Controller|Encoder|
|---|---:|---:|---:|
|protect_inherited，默认|1e-4|5e-5|明确新初始化5e-5；继承／未知来源1e-5|
|full_legacy|1e-4|1e-5|5e-5|
|legacy_flat|1e-4|1e-4|1e-4|

继承属性取checkpoint迁移账本，不凭模块名宣称预训练。未知来源列入unknown并保护。默认AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01、clip_norm=1，均可显式覆盖；这些只是未实测工程初值。

优化器一次登记固定潜在可训练超集，分组键为`runtime_group|role|lineage`。冻结参数`requires_grad=false`且`grad=None`，不发生AdamW decay或moment更新；阶段切换不重建optimizer、不清空已有moments。新激活参数由AdamW在首次梯度时初始化状态。Scheduler默认constant，可选warmup_constant／warmup_cosine，只随成功更新推进。

## 数据、采样、累积和Loss

Sources只接受已验证V5 manifest，训练／验证分别声明。每份group索引是`pair_id -> 原始视频或序列group`的JSON，必须覆盖全部行；group字符串应包含数据集命名空间。Hi4D跨相机同一pair/action使用相同group，不能用camera分开规避泄漏；TikTok用原视频group，不能凭文件名猜。跨训练／验证group重叠直接拒绝。

单个训练Source／Route缺权重时按1，多源／多路由必须明确权重。正权重池为空不会静默重分配；请求的局部／辅助Loss没有任何有效监督也会拒绝。所有原行保留在数据报告，资格矩阵按路由区分字段存在、实际可用、高频cell数和Null-only上下文。

`champ_motions_example`无配对RGB，不参与监督训练。Hybrid的Jaw／Expression／Hand45／BPS／局部Z保持既有无效标志，不做伪标签。Interaction路由只选有效双人；Body Identity启用时需真实Source person可见支持。不同画布／场景／Profile／R数量不拼成一个Qwen批次；本版固定B1、workers0、world_size1。

各路由／源使用独立CPU Generator、打乱循环队列，保存完整排列、游标、epoch、draw状态和attempt。每个累积窗口先确定全部有效候选、sigma和监督数量，再逐微批次前反向，不保留A个图。

- Global Velocity按窗口实际微批次数平均。
- Face／Hand局部项分别按有有效面积的微批次数平均，不因其他行缺模态被稀释。
- ID／Perceptual按有效人脸数平均，默认每4个global optimizer update、sigma≤0.35；可独立关闭和配置权重。
- 只复用已有Owner约束／面积归一化／Qwen64解码与对齐核；增加分子、分母、数量统计，不新增几何Loss。
- schedule关闭的候选记录原因，不占累积槽；attempt默认上限为预算×累积×10，也可显式指定。
- 每个窗口只clip、step和scheduler step一次。NaN、冻结参数收到梯度或未预期断路立即失败；不填Head、不自动解冻。

旧Loss API的默认系数和核心数值公式保留。新Route Loss默认velocity-only；局部模板0.5／0.5，辅助模板0.10／0.05是可配置起点。评估器只按实际请求的ID／Perceptual加载，预测分支保留梯度、目标无梯度。Target RGB只作配对监督，反事实／跨身份原图继续拒绝。

## 冻结latent缓存与验收门禁

训练只读取已有冻结Qwen缓存。`model_manifest`是本地pipeline配置／索引／权重文件的绝对路径＋SHA列表，必须覆盖本地safetensors／bin权重；`cache_identity`为冻结cache index引用，绑定模型manifest SHA、prompt SHA和每个pair文件SHA／Source与Target Canvas指纹。

缓存缺失时显式使用`prepare-cache --config ... --cache-output 新目录`，只运行冻结文本编码／VAE，不做optimizer、估计、渲染或可学习控制缓存；本轮也没有执行它。index的entries键为`source_id + "\u0000" + pair_id`，tensor文件保持实际Qwen64 latent、prompt和mask，读取时逐文件验证SHA和网格。

`preflight --level data`只作数据／资格／划分检查，不能生成梯度通过报告。`--level grad`后续显式加载原生Qwen与初始化checkpoint，检查：

- B1、32层4096↔512生产接口、KV关闭、所有层text／reference前缀摘要一致。
- 精确冻结标志、冻结Native／Qwen／VAE／文本编码器／评估器权重及持久buffer摘要；非Tensor架构信息单独规范化。
- 实际可用模态的Head梯度；上游使用临时非零探针再完整恢复，不保存探针、不创建optimizer。
- 缺BPS／Jaw等叶分支记录未调用／无输入，不要求所有参数都有梯度；需要覆盖的输入组合超过验收预算时不宣称通过。
- 报告绑定初始化SHA、配置、数据、代码／模型／缓存／评估器、环境和参数目录。verification-only小模型报告不等于生产验收。

Qwen保持eval和冻结，但Autograd开启，实际安装的Qwen实现支持在此模式使用非重入checkpoint。Adapter按策略设置训练模式。原始checkpoint没有约定而数据含有效BPS时，先通过上游显式初始化绑定BPS，验收不偷偷改模型header。

`probe_every_updates>0`时可运行固定种子、固定sigma循环的held-out loss probe，保存指标并恢复RNG；不自动出图、不自动选阶段。默认关闭。

## 可靠Resume和不可变产物

新快照保留原模型payload，并新增training_state：配置／输入／参数目录签名、精确活动名单、optimizer／scheduler、Sampler／noise Generator、Python／NumPy／Torch CPU／CUDA RNG、计数器、环境、冻结摘要、门禁和日志已提交前缀。

NumPy RNG转为普通元数据和Tensor保存，使用weights_only安全加载。旧V5仅有optimizer状态、缺少训练schema／RNG／签名的文件只允许初始化，不允许严格Resume。GPU拓扑、后端／版本、数据、Loss、LR、控制或参数目录变化会拒绝严格恢复。

只在optimizer边界保存，micro_position必须0。`--stop-after-update N`可在不改变总预算的情况下停在已提交全局更新边界，用于后续连续／恢复对照；不支持恢复丢失的半窗口。模型、optimizer等重建后再恢复RNG。

checkpoint使用独占新文件＋SHA＋`.commit.json`提交记录，不覆盖旧快照；日志每次Resume／推进使用新segment，SHA链验证已提交前缀。checkpoint之后的日志尾部保留原文件并报告为未提交尝试，不裁剪、删除、移动或自动清理。损坏已提交前缀或缺失commit记录会拒绝恢复。

## 本轮状态与后续测试

使用说明／模板不包含可直接启动的默认训练预算或真实资产绑定。先补齐来源、数据划分、缓存和报告，再分别启动CPU合成测试、真实数据预检及有界梯度验收，最后才决定训练。

`tests/test_qwen21_native_v5_training.py`已编写但未执行／收集，默认测试使用2维合成模块；真实模型用例需另行提供`NATIVE_V5_TRAINING_ACCEPTANCE_CONFIG`。静态结果在实验族`training_prepare_v5_20261025_v1/report/`，不沿用上一轮V5模型测试数字。

**训练接口与训前准备代码接通、静态核对完成；真实数据预检、梯度验收、恢复运行验证和训练均未执行。**
