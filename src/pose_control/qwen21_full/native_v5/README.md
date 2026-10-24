# Native Full V5：完整能力迁移

V5在部署`efc26ba`的V4.1上实现显式Geometry／Interaction分流、Source局部Detail、Face／Hand独立模态、六组强度与sigma调度、局部监督及严格迁移工具。旧Full／V1／V2／V3／V4源码不改，旧checkpoint仍走原入口。

**本次交付只运行源码语法、AST、静态依赖、接口和SHA核对。没有运行测试或收集测试，没有加载／迁移权重、构建缓存、检测、提取、渲染、模型前反向、生成、optimizer或训练；没有安装依赖。** V5不是V4.1的数值等价改名，效果和训练收益尚未验证。

## 架构和尺寸

|路径|维度|用途|
|---|---|---|
|原Geometry／Body21|Global4×256，Joint21×256；目标N×512|旧Full前端和Native增强，保留原数据和权重语义|
|Body控制|4096→512→4096，32层／6组|Geometry＋CHAMP Depth＋Source Body Identity，**没有旧Interaction融合**|
|独立Interaction|对齐N×512，六层512→4096|复用Body的Query投影；直接读取Reasoner／Raster／Part关系，没有中间零输出截断|
|Face|Identity4、Texture8、Texture Delta4、Geometry4，均512|DINO／Perceiver Delta只在Texture路径；Source和Driver独立|
|Hand|Geometry4、Texture8，均512|2D／Hand45／BPS／DINO独立有效性，缺失DINO保留显式Null|
|Source Detail|真实主Source64通道→256→512；6×512×16×16|mask归一化ROI采样，稳定ID及解剖侧映射，不编码第二张Qwen图|
|局部稀疏Query|真实目标cell的4096→512|Face2区域、Hand4区域，各家族共享骨干与六个无bias输出Head|
|调度|8分支×6个可学习log gain|Body、Interaction、Detail、Face ID／Texture／Geometry、Hand Geometry／Texture|
|Qwen|一张Source，32层，4096 Hidden|不增加图像前缀、文本或目标Token；真实主干B=1、KV关闭|

默认注入层为0-based的5、10、15、21、26、31；Body六组为0–5、6–10、11–15、16–21、22–26、27–31。所有动态残差读取同一份未控制的Block输出，FP32合成后只写目标Token。

Source Detail不经过旧16通道变换，不复用DeepGen输出桥。Geometry和Hand高频Detail仍需至少4个有效目标cell；极小ROI只保留有效身份／纹理路径，不把小ROI插值成256个独立Hidden。

`contracts.py`提供Options、Prepared、Source Detail和Training Supervision的`validate/to/index_select/stack`；Options为同批次统一标量。浮点FP32，mask bool，索引int64。`frontend.py`直接调用共享模块，不调用旧融合后的`base.encode`再减Interaction。旧`base.interaction_fusion`权重保留于严格状态中，但在V5处于inactive组且不能解冻。

## 模态与调度规则

Face的原始ArcFace Identity与DINO／Perceiver Delta分开存储。关闭Face Texture同时排除DINO和Perceiver增量；不把Texture Delta加回Identity Tokens。各模态Attention增量按Gain求和，再经过共享无bias输出Head。Gain控制显式残差路径，不表示移除Qwen原生Source或Body中的全部相关信息。

配置默认`strength_preset=constant`：强度1、log gain0、prior1、schedule1。`legacy_full`使用旧策略的工程对照：Face／Detail从progress0.35开始cosine增强，Texture从0.65开始，Hand Geometry从0.15增至1；Face prior为`[.1,.2,.5,.8,1,1]`，Hand为`[.1,.2,.4,.7,1,1]`。进度始终`1-sigma`，不使用采样计数器。

分支Schedule可设置constant／linear／cosine／piecewise。关闭模态、全程强度0或全零schedule时跳过编码；当前sigma为0增益只跳过动态更新，后续需要的静态条件仍准备一次。

显式可选、默认关闭的对照：

- `stage_modulation=film_rank64`：轻量阶段FiLM与rank64残差调制，普通初始化；仅最后Native输出Head置零。
- `canonical_hand=true`：新增左手ROI坐标Canonical 2D投影；不改变已继承Hand45／BPS或空间CNN的解剖约定，不把原姿态权重称为Canonical预训练。
- `hand_fusion=sequential2`：两层Geometry-Q／Appearance-KV对照；默认parallel保持模态增量分开。此对照的Texture更新显式依赖几何Query。
- `identity_pooling=roi48|body8|bodyhead8`：默认48；两个8-token方案使用同一Native64输入，分别为Body2×4和Body2×3＋Head1×2，不导入旧16→1536投影。

`local_mode=dynamic|static`写入checkpoint，不能加载时静默切换。Static不读当前Hidden，每个控制组准备512维局部增量供Body读取。若Native Body输出仍全零且冻结，Static局部heads专训会被入口拒绝；应使用Joint或已训练Body，不填常数绕过梯度。

## 数据和工具（下面命令本轮均未执行）

正式入口在`sg:/home/shangguanrz/project/pic-edit`：

```bash
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$PY scripts/data/prepare_qwen21_native_full_v5.py promote --job promote.json \
  --output /data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native_full_v5_data_v1
```

`promote`要求真实V4 manifest的路径与SHA，并为**每条pair**声明Detail和Supervision；失败样本不能只保留成功行。Body／Local／Source References原文件不重写，V5 manifest引用原manifest＋pair_id，新产物只有侧车、索引和报告。内部使用`NativeFullDatasetV4(load_target=False)`验证原数据，会读取Source RGB，不打开Target RGB、不渲染或提取学习特征。

`examples/promote.example.json`给出显式不可用侧车格式。可用Detail声明需要`role=primary_source_only`、主Source SHA／稳定ID、`support_kind=visible_owned`及六份原图网格mask文件引用，区域顺序固定FaceA、FaceB、HandAL、AR、BL、BR；构建器与原Source person mask及ROI相交，并关闭未知重叠像素。默认使用主References的ROI；若显式空引用没有ROI，可另给同源`boxes_original_normalized [6,4]`及`source_geometry_sha256`，只经过Source Canvas一次，不clamp。

可用训练资格声明通过`record:{path,sha256}`读取JSON：`role=training_only`、`coordinate_space=original_pixels`、`driver_rgb_sha256`、`target_person_ids[2]`、`landmarks5[2,5,2]`、`valid[2]`。使用Driver自己的Canvas变换；空／不可靠对齐不能启用辅助Loss。局部Velocity mask来自最终可见Owner并做内部Feather，不把Target检测结果当作外观条件。

V5 Reader推理时完全不读取Supervision侧车或目标RGB。Source和Driver不同画幅、稳定ID、Body容器SHA／原Driver几何SHA继续使用V4.1规则。Hybrid的Jaw／Expression／Hand45／BPS／局部Camera-Z缺失仍为无效；不要求补齐SMPL-X。

### 多参考与质量工具

`references`复用原Source-only提取器，新增保存`explicit_multi_reference`声明：

```bash
$PY scripts/data/prepare_qwen21_native_full_v5.py references --job source_references.json \
  --output /data/shangguanrz/.../source_references_v5.npz
```

R默认1，最多3；额外参考只进入ArcFace／DINO缓存，不进入Qwen或Detail Bridge。每份参考记录自己的RGB、Canvas、几何SHA、人物ID／ROI和模型SHA；数组顺序必须稳定。第一份必须是实际Qwen主Source，额外参考禁止使用本次监督目标RGB。只有明确请求且有有效ROI的提取模型才初始化；不自动下载或安装。基础job结构沿用V4，示例说明在`examples/README.md`。

`quality-audit`只输出旧Hand门控及按真实Canvas像素计算的复查建议，不更改有效性。`export-repair`只导出同Driver RGB的2D／mask／独立Z有效性；无局部Z时保持无效。`import-repair`需要同源Body、修复结果和带SHA的可信人物可见支持，核对Driver／Camera／Canvas／人物ID，检查腕部及可靠点，再更新2D与保守支持。禁止用修复结果替换Hand45／BPS／网格／Camera-Z。所有输出必须新文件，工具不运行外部Refiner或估计器。

## 权重迁移与审计

```bash
$PY scripts/audit/audit_qwen21_native_full_v5.py --checkpoint /data/shangguanrz/.../v4.pt \
  --state-key state_dict --output /data/shangguanrz/.../v4_header_audit.json

$PY scripts/data/migrate_qwen21_native_full_v5.py --v4 /data/shangguanrz/.../v4.pt \
  --allow-fresh-hand --allow-fresh-detail \
  --output /data/shangguanrz/.../native_full_v5_v1.pt \
  --report /data/shangguanrz/.../native_full_v5_v1_inheritance.json
```

这些工具尚未运行。首版Detail Bridge为新Native64输入，需要显式`allow_fresh_detail`；CHAMP Detail CNN是另一个继承模块。Hand缺训练证据时也需显式允许fresh。Face3000步只是旧Full候选，日志不证明V5有效。

严格继承兼容的Base和静态编码器；原Native V4局部Query／Geometry Attention／Texture Attention／FFN等按白名单继承并标为adapted。V5 Identity Attention、Detail、Interaction及最终Head为新模块；旧V4动态输出Head不放入新分流Head。缺Face basis或BPS完整约定时精确重置对应投影并报告。

V3或更早checkpoint先走现有升级链到V4，不在V5加载时静默转换。不会恢复optimizer或覆盖源权重。可选旧DeepGen Attention／FiLM／低秩warm-start要求明确键映射、源文件SHA，以及checkpoint自身保存的`time_convention=normalized_sigma`、`roi_coordinate_convention=native_target_roi`、`control_width=512`；缺失语义证据就拒绝，不接受一个compatible布尔值。

迁移报告列出inherited、adapted、semantic_resets、discarded、new_initialized、inactive_retained、参数元素数、fresh状态和全部来源。实际迁移后仍需另行严格回读和梯度验收。

## 推理与训练接口

```bash
$PY scripts/run/run_qwen21_native_full_v5.py --checkpoint /data/shangguanrz/.../v5.pt \
  --manifest /data/shangguanrz/.../pairs.jsonl --data-profile hybrid_hi4d \
  --face-texture-gain 0 --detail-strength 1 \
  --output /data/shangguanrz/.../native_full_v5_inference_v1
```

Body Identity由`--no-body-identity`单独关闭；Source Detail由`--no-source-detail`关闭；CHAMP人物Detail由`--no-depth-detail`关闭，三个概念不混用。Face／Hand总体强度及五个模态Gain、独立Interaction、Depth／Relation强度均在request记录。Schedule及结构从checkpoint读取。

训练提供Face／Hand／Detail／Interaction的heads／full、Joint、all-native，保留freeze／unfreeze／group-lr接口，flat AdamW，无自动阶段切换。

|Loss profile|主Velocity|Face局部|Hand局部|ID|Perceptual|
|---|---:|---:|---:|---:|---:|
|velocity_only，默认|1|0|0|0|0|
|local_velocity|1|0.5|0.5|0|0|
|local_aux|1|0.5|0.5|0.10|0.05|

局部Loss独立按真实Owner面积归一化，空区域返回零并记录；不会因面部小而按全图面积稀释。辅助项仅同身份配对、资格通过且sigma≤0.35时启用，默认每4个optimizer step一次；使用固定Target五点网格同时对齐预测／目标，评估器冻结、目标无梯度，预测经Qwen实际64通道unpack／VAE均值方差解码保留梯度。显式Evaluator job格式见示例，不从生成条件读取Target外观。

局部专训保存所有行的eligible报告，仅从符合条件的索引采样；不删除manifest行。Joint保留无局部监督的配对样本。Schedule全关闭的attempt记录为`optimizer_updated=false`，不会填Head或隐式解冻；`--steps`表示实际更新数，`--max-attempts`默认10倍更新数，耗尽后保留部分checkpoint并明确失败，不宣称达到请求更新数。

```bash
# 后续另行启动；本轮连preflight也未执行
$PY scripts/train/train_qwen21_native_full_v5.py --checkpoint /data/shangguanrz/.../v5.pt \
  --manifest /data/shangguanrz/.../pairs.jsonl --profile detail-heads --loss-profile local_velocity \
  --preflight-only --output /data/shangguanrz/.../native_full_v5_preflight_v1
```

## 后续验收与当前状态

已编写`tests/test_qwen21_native_full_v5.py`，本轮不运行或收集。后续先CPU契约，再小模型和7B有界前反向，最后才生成和训练。单／双人、Hybrid／Full、缺失模态、来源冲突、多参考重排、矩形Canvas、未知局部Z、握手／手遮脸和全无效区域都需真实数据验收。

零输出验收相对V5核心，不与V4.1混为一谈；首个反向只要求新Head梯度，再用显式非零探针检查上游，不能用常数填零Head。必须检查prefix、Token数、冻结状态、同一次原Hidden、非重入checkpoint跨forward上下文和所有活动组梯度。完整权重回读由显式`QWEN_V5_ACCEPTANCE_JOB`绑定真实V4／V5 checkpoint、V5 manifest和index后另行运行。

五层结论分别记录结构、权重、运行参与、独立控制和生成效果。**当前仅为“V5架构与工具接通、静态核对完成”；真实权重迁移、数据与运行时验收、效果对照尚未执行。** 不复用前几轮测试数字。
