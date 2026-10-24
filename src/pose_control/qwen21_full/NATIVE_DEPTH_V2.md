# Native Depth V2：架构已接通，尚未运行验收

本轮按 P0–P3 接通编码与融合，不运行测试、模型前反向、训练、Camera-Z 导出或 checkpoint 迁移。下文命令是后续使用方式，不是已执行记录。旧 Depth V1、Body21 Geometry 和 Native Body 的源码保留，Qwen 仍冻结，不增加 token。

## 结构与尺寸

| 模块 | 输入 → 输出 | 初始化／计算 |
|---|---|---|
| 原 CHAMP Encoder | Relative＋Valid 2 → 256 | 原权重映射不变；attention_scale 默认16，可显式8，最终均为1/16 |
| Geometry Detail Encoder | 23 → 16 → 32 → 64 → 128 → 256 | 4次stride2 Residual Block、GroupNorm、SiLU；没有第二个 Transformer |
| Mixer | concat 512 → 256 | 1×1 Conv＋GN32＋SiLU |
| Gate | G₀ 512→64、F_mix 256→64、C29→32；160→128→512 | sigmoid，逐位置和通道，非零值门控 |
| 输出投影 | 256 → 512 | 仅此残差输出投影权重与bias置零 |
| 融合／主干 | G₀＋αM×Gate×P(F_mix) → 原 Native Body | α默认1，Geometry保持 `[B,N_target,512]`，Qwen保持32层4096↔512 |

Geometry 预处理和新模块均FP32。禁用、强度0、整行无有效Depth时返回原特征；attention仅处理有Depth的行。新Depth只支持单人，启用时遇到有效Person B直接报错；其他旧路径保留。可学习特征每次条件编码计算一次，保留autograd，不进入Qwen逐block循环，不离线缓存。

23通道顺序固定在 `DETAIL_CHANNELS`：

| 索引 | 内容 |
|---|---|
| 0–4 | Relative dx、dy、Laplacian、jump、silhouette |
| 5–9 | Root-relative Z、dx、dy、Laplacian、jump |
| 10–14 | 相机单位射线xyz、Normal cosine、一致性有效性 |
| 15–18 | Depth valid、Geometry valid、confidence、confidence available |
| 19–22 | log1p(Root/身高)、标定身高log、metric available、camera available |

差分以原图归一化坐标步长计算，并使用asinh限制动态范围。jump特征为 `asinh(step/threshold)`；默认阈值为身高的0.02、Relative的0.05，属于尚未验证的工程起点。几何不可用时，5–14、19–22通道显式清零；不能从这些通道的数值是否为0推导有效性。

C29由Part14、Normal3、五种覆盖率（Depth/SMPL/Geometry/Normal/Part）、法线一致性及覆盖率、confidence及可用性、轮廓/跳变、metric标志组成。Part及覆盖率采用平均池化，Normal/一致性/confidence采用有效性归一化平均，边界采用max池化。可信度线索不硬乘成可靠性惩罚，也不宣称是校准过的不确定性。

## 数据契约

`DepthConditionV2` 包含 Relative、独立Depth valid、canvas valid、available、可选Camera-Z及其独立valid、Geometry valid、confidence及可用性、23通道准备特征、`DepthFusionContext` 和来源。两个类型都支持 `.to()`、`.index_select()`、`.stack()`；浮点维持FP32，bool保持离散类型。混合PNG/Camera-Z批次按精确画布分桶，缺Z行的存储零填充只能在独立mask为false时使用。

`DepthFusionContext` 包含目标画布 `K [B,3,3]`、FK pelvis相机Z与同beta标准姿态身高 `[B]`、Normal `[B,3,H,W]`、Part `[B,14,H,W]`、SMPL mask及模态有效性。Root来自Body21的 `joints_cam[0,2]`，不等同于translation Z。

`Z_body=(Z-Z_pelvis)/H_body` 要求同尺度。导出器默认声明 `model_unit/model_scale`，足以计算无量纲几何；不自动称为米制。只有 `metre/calibrated` 且显式给出以下绑定，才将缓存Root/身高转换为米，并启用绝对身高：

```json
{"calibration":{"pose_record_sha256":"与本条pose记录一致的SHA256","metres_per_model_unit":1.0}}
```

PNG行不保存Camera-Z，关闭所有物理通道，保留Relative边界和语义融合。未提供confidence时使用有效区域中性值1、`confidence_available=false`；不创建估计器或伪造置信度。

Reader先校验Depth、mask、可选confidence SHA256，并绑定同一Body21记录、参数、模型、角色和相机指纹。数据源由manifest逐条声明；`auto`允许不同kind并存，不表示“文件缺失就换一个”。声明的资产缺失或来源不一致直接报错。

Camera-Z/Relative差分、深度跳变和反投影法线在源Depth网格计算；Normal、Part、SMPL mask先按各自声明的全帧变换对齐这个网格。`K_asset=inv(A_asset→original)K_original`，`K_canvas=A_original→canvas K_original`，不会重复缩放相机。几何数值图、mask、Part最近邻；边界保守max下采样；CHAMP的Relative保持masked bilinear。相机射线按目标K和像素中心精确生成。

Normal必须声明 `view_xright_yup_ztoward`、`camera_cv` 或 `unknown`。原渲染器的Normal转换为 `(Nx,-Ny,-Nz)` 后比较；这个转换只作用于新上下文，旧Full输入不变。反投影法线与渲染法线朝向统一，在有效、低跳变、未跨已知Part边界的邻域比较。unknown关闭Normal上下文/一致性，不猜符号。

## 调用、参数和版本

- 新建启用Depth的训练入口默认 `--depth-version v2`；`--depth-version v1` 显式保留旧路径。Depth off继续跳过该分支。
- 原推理入口按checkpoint实际版本重建；可用 `--depth-version` 做一致性断言，不能借此静默升级。
- `NativeDepthV2Records.condition(sample)` 返回完整typed输入。可直接传入 `runtime.encode(..., depth=condition)`。
- 可选 `NativeDepthV2Dataset` 给现有dataset附加 `depth_condition_v2`；`qwen_full_collate` 会结构化stack。也可使用 `collate_geometry_pairs`，它保留这种结构化输入并检查精确画布。
- runtime保存 `runtime v3 / Depth v2 / condition v2`；Geometry仍为Body21 v2。Depth V1对应原runtime v1/v2，版本组合严格校验。
- Geometry/Body/CHAMP继承；`upgrade_depth=True` 只从V1导入Encoder，检查但丢弃旧Depth输出投影，重新初始化Detail/Mixer/Gate并将新投影置零。未知key、旧参数缺失、dtype/shape/config不符报错，没有宽泛strict=False。
- V2原样重建直接 `NativeDepthRuntime.from_native_checkpoint(saved)`；原Geometry v1仍需显式legacy或upgrade_pose。新Depth不能挂到legacy Geometry。
- `named_parameter_groups()` 返回七组 `(name, parameter)`：geometry、pose、body、champ_cnn、champ_attention、depth_detail、depth_fusion。`NativeTrainabilityConfig` 独立配置；默认保持原冻结方式，新分支开放。训练入口支持对应freeze参数与 `--train-native-all`，二者不能冲突。checkpoint记录冻结配置；重建时调用者配置决定训练状态，不恢复optimizer。

本轮不改变velocity loss或AdamW算法，不加入新loss、学习率分组/日程或自动阶段解冻。冻结分支无grad时日志记录None。原Body零head和Depth零投影串联仍可能延迟Encoder梯度；新Gate避免增加零值乘法死区，未声称解决了全部训练初期延迟。

## 后续使用命令（本轮未运行）

在 `sg:/home/shangguanrz/project/pic-edit`，先用既有SMPLest-X环境导出目标mesh的原始Z；不运行人体估计。输出须为新目录：

```bash
BODY_RUN=/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1
DEPTH_RUN=/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_geometry_step5_20261024_v1

add/envs/smplestx_bw/bin/python scripts/data/export_qwen21_depth_v2.py \
  --pairs "$BODY_RUN/data/tiktok_pairs.jsonl" --pose-manifest "$BODY_RUN/data/fk_v1/records.jsonl" \
  --output "$DEPTH_RUN/data/camera_z_v1"
```

将 `--kind relative_png` 加入导出命令，可仅为已有PNG写V2 manifest；不生成Z。iPER换用对应pairs。导出工具只创建新NPY/JSON，不覆盖旧图片或缓存。

显式迁移既有Body21/Depth V1 runtime（尚未执行，不生成训练结果）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/data/migrate_qwen21_depth_v2.py \
  --source "$BODY_RUN/checkpoints/native_body21_v2.pt" \
  --output "$DEPTH_RUN/checkpoints/depth_v2_init.pt"
```

Full或Depth-off源需要同时给 `--champ-checkpoint`。Depth V1源则严格继承其Encoder，不能再指定第二个CHAMP来源。投影零起步意味着新V2初始返回G₀，不承诺等于旧V1启用Depth的输出；旧V1依然可以原样复现。

之后才进行资料/模型预检：推理入口提供V2 checkpoint、Body21 pose manifest、V2 depth manifest、`--depth-mode auto --preflight-only`。注意推理preflight会实际构造控制器并前向，因此本轮也没有运行它。训练入口的数据preflight只读数据，不需要CHAMP权重；实际训练则必须明确CHAMP或Native Encoder来源。

后续验收清单：Camera-Z/PNG混合、缺模态、尺度冲突、平面/斜面/边界、Normal基转换、矩形画布与重复缩放拒绝、全无效行、23/29/256/512接口、零投影、legacy、checkpoint回读、全部Native参数梯度及Qwen prefix/序列长度。当前仅语法与静态接线核对，不报告测试通过率或生成效果。

方法依据：[Metric3Dv2](https://arxiv.org/abs/2404.15506)、[GeoWizard](https://arxiv.org/abs/2403.12013)、[UniDepthV2](https://arxiv.org/abs/2502.20110)。仅借鉴相机、Depth–Normal关系、边界与可靠性表示，没有引入这些模型本体。
