# Native Dual Geometry V3：架构已接通，尚未运行验收

本轮交付源码、离线工具和入口，不安装依赖、不执行工具、不运行测试、模型前反向、渲染、缓存、迁移或训练。以下命令均供后续使用。不能把单人 V2 的结果算作 V3 验收，也不能据此声称生成质量或身份保持已改善。

基线为 sg `5b9df042ecb18f7c4c5dacbf20cfec9310273310`。此前计划里的 `15b9df04` 混入工具输出序号，实施快照以完整 SHA 为准。旧 Full、单人 Depth V1/V2、Body21、Native Body 与 Qwen block 注入代码保留。新 Runtime 独立使用 `native_dual_geometry_v3`，Depth 为 `native_dual_depth_v3`，条件为 `dual_condition_v3`。

## 数据与同源几何

只接受同一对人物的源帧与目标帧，控制来自目标同一帧的 shape/pose/camera。源图只提供身份；目标 RGB 仅在训练 Reader 的 `load_target=True` 时读取。推理 Reader 不打开或校验目标 RGB 文件内容，但检查其引用与目标几何记录一致，避免把其他 RGB 当作配对监督。

`build_qwen21_native_dual_cache.py` 读取明确的 pairs 与 raw frames manifest，使用对应 gender 的 SMPL 模型、完整69维动作做 FK，按名称提取21关节。SMPL-X 原缓存 Reader 不改；新数据契约记录模型家族与顶点数，不把6890顶点 SMPL 称为10475顶点 SMPL-X。Hi4D 导出器仅处理其原生 SMPL；新神经编码与 typed 输入可区分两种模型家族。

网格、24个原始关节以及重参数化后的相机网格分别要求最大误差≤1e-4。根平移为 `Rc*t + tc + (Rc-I)*J_rest`，两个 Global26 translation 槽同步，根rotation6D采用前两列顺序。全局条件布局不变。Relative11保留宽度，位置取FK pelvis，朝向由 `RA.T@RB` 的前向轴在A的XZ平面投影得到；退化时两项方向通道为零，另记有效性。

默认模型目录为 `add/tools/SMPLest-X/human_models/human_model_files/smpl/`，按 gender 明确选择模型；找不到或重建不一致就拒绝，不换 neutral。数据有非零畸变时要求显式、已校准的 rectified 原图和相机记录，当前导出器不会自动猜测去畸变方式。Hi4D 已检查样本的畸变为零不代表可以跳过其他记录检查。

两人独立渲染，Normal保存CV坐标，Part按三角面LBS部位总权重最大值划分，FCL与Raster共用这个划分。Body旧Normal编码器读取 `(Nx,-Ny,-Nz)`，Depth比较读取原CV值。不复制场景Normal、不做覆盖率驱动的上下翻转。

几何资产默认最长边1024，记录完整 `asset_to_original`；23通道差分和法线在资产网格先算，再映射至目标canvas。目标用等比 `aspect_preserving`，默认像素预算512²、padding为32倍数；参考图单独采用等比512×512 letterbox，可通过 `--reference-side` 更改。参考尺寸与Qwen实际前缀严格匹配，矩形目标不被拉成正方形。

Body25热图采用 `BODY25_BY_NAME` 的SMPL可对应子集及pelvis→midhip；不把head冒充nose，不猜足尖/脚跟。未知通道为零，不宣称这些通道是DWPose估计。Body21的63维姿态与22关节FK仍须完整，不存在缺Pose零填充路径。

## Typed输入和尺寸

`DualDepthConditionV3` 以两个 `DepthConditionV2` 子记录复用23/29通道契约，提供 `person_camera_z/person_depth_valid/person_normal/person_part/root_z/body_height` 堆叠属性；`.to/index_select/stack` 保留bool和int64，几何浮点保持FP32。A/B来源必须同相机、同尺度、同canvas，保留完整投影与可见性，禁止混用。

| 模块／条件 | 输入 → 输出 |
|---|---|
| Body21与模型家族绑定 | A/B各Global26、Body63 → 原4×256 Global＋21×256 Joint |
| Scene CHAMP | Relative＋Valid 2通道 → 256维，每帧一次 |
| 共享Detail CNN | A/B各23通道 → 各256维，4次stride2 |
| 共享Mixer／Gate | concat512→256，原C29 Gate→512 |
| 新Depth投影 | 256→512，权重和bias置零 |
| Relation Raster | 8→16→32→64→128→256→512，输出投影普通初始化 |
| Part关系 | 196×9数值＋部位Embedding→196×256→28×256→空间512维Attention增量 |
| 身份ROI | 原生64通道，身体8×4＋头部4×4→每人48×256→空间512维增量 |
| Interaction Gate | G512→64、I512→64、C12→32；160→128→512 sigmoid |
| 身份Gate | 同上，C14多两个源身份有效性 |
| 两个新融合输出 | 各512→512，仅输出投影置零 |
| 原Native Body | 控制 `[B,N_target,512]`，每层4096↔512，仍32层 |

共享高度为两人标准姿态身高均值，深度中心为两人pelvis Z均值。Scene取有效最近Z后在联合前景归一化为near-one；人物Detail的Relative分别归一化，其物理通道使用各自pelvis/身高。Relation8使用共享归一化Z和ΔZ，不能从两个独立Relative值相减获取前后关系。连续通道asinh，owner容差默认1e-4×共享身高；tie时权重各半。差分阈值沿用身高0.02和Relative0.05，均尚未验证。

Part距离是FCL三角网格BVH查询结果。最近点有效性与距离有效性分离：确认表面相交时距离0可以有效，库没给可靠最近点则 `closest_points_valid=false`，方向清零并输入可用性标志。不把这个标志或距离0当作穿透深度／接触真值。9维关系数值为距离1、位移3、contact/known/avoid/avoid_known4、最近点有效性1。

Hi4D Contact按0无记录、正整数k对应对方顶点k-1解码，保留原方向及多对一关系；不要求互逆。原始correspondence和推断kind分别保存。导出器只写官方对应，不重新做最近邻推断。未标注部位对不自动变成avoid。旧Reasoner最多8组Contact只是继承接口桥接，完整原始对应和196组关系保留在新分支。

`relative_png` manifest必须明确提供 `scene_png`、`scene_png_valid` 的path/SHA及polarity，PNG网格须与几何资产网格相同。只能选scene_only：关闭物理Raster、距离、Contact和人物Detail。Reader仍要求同源Body21/独立表面资产用于人体几何与身份归属，不能用一张PNG替代全部双人几何，也不从PNG推算Z。不存在文件缺失自动降级。

## 身份、融合及参数

Source实例mask来自manifest中的源帧标注／已提供的源mask，头部取源几何head部位与源可见mask交集。两个源实例mask不允许重叠。头部不可用只关闭头部Token，身体参考缺失拒绝；不调用新的人脸/衣服视觉基座。

推理 `bind_native_dual_pipeline` 从Qwen第一次Transformer调用的真实 `hidden_states` 参考切片读取64通道latent，与源mask的canvas严格对齐；只编码一次并限定在一次无梯度推理调用。训练每步直接复用既有Qwen冻结VAE缓存，运行可学习编码，不缓存Joint/Geometry/身份Token。VAE的感受野仍可能包含周边信息，mask并不证明身份已完全解耦。

`source_indices[target_slot]` 指向源人物数组。metadata保存稳定source/target person ID；不按当前左右顺序重排。目标Person Tokens只读取对应人物48个源Token，按目标可见owner权重合成。

`Gd=G0+αΔDepth`，`Ir=I0+β(ΔRaster+ΔPart)`；两个零输出残差把Ir与身份特征融合到Gd。C12顺序固定在 `INTERACTION_CHANNELS`：投影2、可见2、Z有效2、重叠1、front2、tie1、camera-scale1、distance-available1；身份附加源有效性2。零门控未叠加，原Body零head与新增输出零投影仍可能延迟上游梯度，尚未运行梯度验收。

Runtime暴露12组：`geometry_shared/body21/geometry_dual/native_body/champ_cnn/champ_attention/detail_cnn/depth_fusion/relation_raster/relation_tokens/interaction_fusion/identity_binding`。默认仅冻结共享Geometry，旧双人模块显式可训练；`--train-native-all`打开全部，`--freeze-group`可逐组关闭。训练入口也关闭被消融选项禁用的参数组。loss和flat AdamW沿用原实现，没有额外几何loss或调度。

## 后续使用：本轮均未执行

1. 准备独立几何环境。可克隆现有SMPLest-X环境到 `/data/shangguanrz/envs/native_dual_v3`，在克隆环境安装 `python-fcl>=0.7,<0.8`，保留原torch/smplx/nvdiffrast；不要安装到qwen21。导出器记录依赖版本并在真正执行时检查依赖。
2. 使用raw frames manifest建立缓存，示例：

```bash
RUN=/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native_dual_v3_20261024_v1
GEOPY=/data/shangguanrz/envs/native_dual_v3/bin/python
QWENPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python

$GEOPY scripts/data/build_qwen21_native_dual_cache.py \
  --pairs datasets/Hi4D_pilot_v1/pairs_1500_train.jsonl \
  --frames-manifest datasets/Hi4D_pilot_v1/processed_frames_1500.jsonl \
  --output "$RUN/data/cache_v1" --limit 8

$QWENPY scripts/data/migrate_qwen21_native_dual_v3.py \
  --source /data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_geometry_step5_20261024_v1/checkpoints/depth_v2_init.pt \
  --output "$RUN/checkpoints/dual_v3_init.pt"
```

上面的Depth V2源文件需要先按单人V2说明显式构建；本轮没有生成它。V3只从真实Body21＋Depth V2 checkpoint继承，不将Full/Depth V1自动当成V2。迁移严格检查原参数，继承Encoder/Detail/Mixer/Gate/Geometry/Body，报告丢弃旧Depth投影及全部新初始化键，不恢复optimizer。

3. 后续数据预检、推理示例：

```bash
$QWENPY scripts/train/train_qwen21_native_dual.py \
  --checkpoint "$RUN/checkpoints/dual_v3_init.pt" --manifest "$RUN/data/cache_v1/records.jsonl" \
  --output "$RUN/data_preflight_v1" --preflight-only

$QWENPY scripts/run/run_qwen21_native_dual.py \
  --checkpoint "$RUN/checkpoints/dual_v3_init.pt" --manifest "$RUN/data/cache_v1/records.jsonl" \
  --output "$RUN/inference_v1" --pair-index 0
```

预检会读取权重和实际缓存，但不构造控制器或运行Qwen；推理会运行模型。本轮两者都没有执行。所有输出须新目录／新文件。导出器从raw frames manifest查找具体source/target action/camera/frame；源帧不在frames manifest时直接拒绝，需先提供完整原始帧清单。

消融通过显式开关表达：几何基线使用 `--depth-mode off --no-detail --no-interaction --no-raster --no-part-contact --no-identity`；Scene-only增加 `--depth-mode scene_only --no-detail --no-raster --no-part-contact`；Relation必须启用Interaction且mode为full_geometry，否则报错。强度0跳过对应增量。Full几何数据可以用于关掉人物Detail的Scene-only消融，PNG记录不能用于full_geometry。

## 后续验收

先做纯数据与合成几何，再做小模型及真实Qwen前反向，最后才做单例生成与训练决策。检查8帧接触／非接触、交叉、owner切换和至少两台相机；检查FK/Global26/矩形canvas、FCL平面与边缘最近点、Contact索引及无对应、tie/全无效、PNG通道关闭、23/29/8/12/14及256/512接口、源数组重排加mapping不变、头部缺失、目标RGB不入控制、严格checkpoint、旧路径、冻结基座、prefix/序列长度、各活动组梯度。没有执行这些检查之前，状态保持“架构已接通，尚未运行验收”。

方法依据：[Hi4D](https://github.com/yifeiyin04/Hi4D/blob/main/dataset.md)、[SMPL全局变换](https://github.com/vchoutas/smplx#modifying-the-global-pose-of-the-model)、[FCL](https://github.com/BerkeleyAutomation/python-fcl)、[InterControl](https://arxiv.org/abs/2311.15864)、[ContactField](https://proceedings.neurips.cc/paper_files/paper/2024/hash/435422305988b73c6cc00bcb29ba2531-Abstract-Conference.html)。仅复用表示与几何查询思路，没有迁移其生成模型权重。
