# Native Full V4.1：Hi4D Hybrid 使用说明

V4.1为Hi4D增加数据Profile，模型仍是`native_full_v4`。原生SMPL继续提供V3 Body21、Global26、Relative11、Depth、Part、Normal、Contact和距离；原图DWPose只提供可选Face／Hand 2D条件。没有V4→V4.1权重迁移，不重新初始化控制器。完整SMPL-X导入器及旧Full／单人V1／V2／双人V3独立保留。

**本次只交付代码与静态核对。没有执行检测、特征提取、渲染、缓存构建、迁移、测试、模型、生成或训练；没有安装依赖。下面命令均供后续另行执行。**

## 数据链和尺寸

|阶段|输入／输出|来源与约束|
|---|---|---|
|Body|原`NativeDualDataset(load_target=False)` → `native_full_body_record_v4`|必须是真实V3 schema和FK／相机／模型SHA校验通过的缓存；只去掉Source RGB张量，补充pair来源元数据；不改数值或Contact|
|观测|`xy_native [2,134,2]`、`scores [2,134]`、bool有效性|`native_hi4d_hybrid_asset_v1`；133布局必须显式声明；不能读512骨架图|
|Face|`[1,2,72,3]`|68点写入槽4–71，前4点无效；Jaw／Expression无效；可退回V3头部投影与官方可见mask的交集ROI|
|Hand|`[1,2,2,21,3]`|134布局的92:113为解剖左手、113:134为解剖右手；Hand45和BPS均无效|
|局部空间|Face2份＋Hand4份，每份512×16×16|由现有可学习V4 Preparer在线计算，禁止离线缓存；六区域顺序保持V4|
|外观|ArcFace `[1,2,R,512]`；Face DINO `[1,2,R,256,1536]`；Hand DINO `[1,2,2,R,256,1536]`|首版R=1，只允许指定Source RGB；三类有效性和提取请求独立|
|控制|Body `[B,N,512]`、原六层局部动态残差|Qwen32层，4096↔512，层5/10/15/21/26/31，一张Source，无新增Qwen token；真实主干B=1、KV关闭|

`native_hybrid_assets.py`负责来源、布局和人物匹配；`native_hybrid_local.py`负责Canvas、ROI和可见性；`native_hybrid_cache.py`组合现有V3样本和V4记录。推理Reader不运行这些离线构建器。

Manifest必须声明`data_profile=hybrid_hi4d`。缺字段的旧V4记录仍按`full_smplx`读取，不根据文件是否存在猜类型。推理和训练可传`--data-profile hybrid_hi4d`断言；模型、checkpoint头、参数名称和形状不变。

## 来源与人物归属

所有观测按`(pair, action, camera_id, frame_id)`与V3几何联表。不能按行号或文件名A/B配对：已读到的原图观测清单1537行、目标清单1500行且涉及7台相机，这些是清单规模，不是本轮验收样本数。

旧缓存必须在原生成清单中声明RGB、相机文件、原图尺寸、无预裁剪／预缩放及两个人的官方mask。缓存、RGB、mask、camera和生成清单均记录SHA；来源状态明确为“旧清单绑定并重新匹配”，不宣称恢复了原检测执行时未保存的哈希。来源不足或冲突时保留拒绝记录；无可信mask时关闭局部支持，不补造来源。权限拒绝直接向上传播。

人物匹配以官方mask命中率及具名Body21投影锚点的距离共同计分，搜索全局一对一分配；同一检测不能复用，重复检测和低胜出差均关闭对应槽位。逻辑Source slot按稳定人物ID确定，Source数组重排并同步mapping不会改变逻辑ID。左右屏幕位置不参与身份定义。

未验收工程默认值写入`HybridConfig`及每条缓存：点分数0.3、可靠Body锚点至少4、人物匹配分0.5、胜出差0.15；Hand至少6点且腕点可靠，腕点距对应FK锚点不超过`max(16px,0.1×人物bbox高度)`，ROI外扩10%。不把这些阈值称为已校准置信度。

原图像素坐标先归一化，再只经对应Source／Driver的CanvasSpec一次。越界坐标及未裁剪ROI保留，非有限点无效。不硬编码相机、原图尺寸或iPER旧画布。Body的原有Canvas和数值不重写。

## 可见性与缺失模态

Hybrid局部Camera-Z全部无效；粗SMPL手部深度不能充当手指表面。V3 Scene Depth、scale、owner保留。局部支持限制在可信ROI、官方人物可见mask和V3独占owner的交集，Face再与可信头部投影相交；不扩到粗SMPL轮廓之外。V3 tie以及未知局部Z的多个局部区域重叠像素全部关闭，再交由原V4 owner合成。不会从ROI重叠产生Contact。

缺Jaw／Expression／Hand45／BPS不会送零值进相关MLP。仅存储占位为零，valid=false；无有效行的投影模块不执行。Hand有2D仍保留几何路径和显式Null Appearance。ArcFace、Face DINO、Hand DINO各自可缺；有声明的引用文件缺失或SHA不符直接报错，只有`explicit_empty`允许空引用。

`expression_mode=source`沿用原V4刚性点过滤，不伪造Source表情参数；`driver`／`off`可以在配置中显式选择。本路径没有实际表情重定向，仍是原目标帧几何。任何真实反事实几何继续受原Reader／Runtime配对监督拒绝检查。

## 后续使用顺序（本轮未执行）

在`sg:/home/shangguanrz/project/pic-edit`运行。所有路径须替换成已存在并经V3校验的真实产物；历史512缓存不能替代`V3`。以下`OUT`必须是新的版本目录，四个子命令均不覆盖输出。

```bash
V3=/data/shangguanrz/.../真实V3/pairs.jsonl
GEN=/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1/native_interaction_v2/frames_used_with_dwpose_v2.jsonl
OUT=/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hybrid_cache_v1
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python

$PY scripts/data/prepare_qwen21_hi4d_hybrid.py import-observations \
  --v3-manifest "$V3" --generation-manifest "$GEN" --layout dwpose134 --output "$OUT/observations_v1"
```

如果明确需要补检，提交`requests.json`，内容为精确key列表，例如`[["pair00","hug00","4","26"]]`，只允许请求已登记的missing／rejected／ambiguous帧。检测器配置用已有DWPose ONNX路径，参考`examples/native_hybrid_v41/detector.example.json`。原缓存保留，不会启动全数据扫描。

```bash
$PY scripts/data/prepare_qwen21_hi4d_hybrid.py detect-missing \
  --v3-manifest "$V3" --observations "$OUT/observations_v1/observations.jsonl" \
  --requests requests.json --detector-config detector.json --output "$OUT/observations_v2"
```

Source外观有两条显式选择：

1. 暂不提取时，直接在`build`使用`--references-mode explicit_empty`。Face/Hand几何仍按有效性工作。
2. 要外观时，先生成Source-only jobs，配置独立的`arcface`／`face_dino`／`hand_dino`请求以及实际模型SHA。已有`cache_qwen21_native_local_references_v4.py --job ... --output ...`逐个执行。只会为实际可用、已请求的ROI初始化相应模型，无可用ArcFace不会加载ArcFace。模型仅本地读取，不自动下载或安装。每个提取结果用`source_geometry_sha256`和`references:{path,sha256,schema}`组成新references JSONL；job索引里的待执行输出路径不是已提取引用。

```bash
$PY scripts/data/prepare_qwen21_hi4d_hybrid.py reference-jobs \
  --v3-manifest "$V3" --observations "$OUT/observations_v1/observations.jsonl" \
  --appearance-config appearance.json --output "$OUT/reference_jobs_v1"

$PY scripts/data/cache_qwen21_native_local_references_v4.py \
  --job "$OUT/reference_jobs_v1/source_000000.json" --output "$OUT/reference_jobs_v1/references_000000.npz"

$PY scripts/data/prepare_qwen21_hi4d_hybrid.py build \
  --v3-manifest "$V3" --observations "$OUT/observations_v1/observations.jsonl" \
  --references-mode explicit_empty --output "$OUT/full_v4_hybrid_v1"
# 有完整显式Source引用时，替换为 --references-mode manifest --references-manifest references.jsonl
```

`build`调用原`NativeDualDataset(load_target=False)`，会读取Source RGB但不打开target RGB；保存时去掉Source图像张量。输出新Body容器、Local、References、`pairs.jsonl`、所有样本的`report.json`，不提取学习特征或重算Contact。

两个SHA不可混淆：`body_record_sha256`绑定新Body容器（含pair元数据）；`driver_record_sha256`绑定原V3目标几何，且必须与A/B的`pose_provenance.record_sha256`相同。Reader分别检查，不用容器SHA替换几何SHA。

完成数据及小模型验收后，才使用已有V4 checkpoint和原入口，无需V4.1迁移：

```bash
$PY scripts/run/run_qwen21_native_full_v4.py \
  --checkpoint /data/shangguanrz/.../v4.pt --manifest "$OUT/full_v4_hybrid_v1/pairs.jsonl" \
  --data-profile hybrid_hi4d --output /data/shangguanrz/.../新推理目录
```

训练入口同样支持`--data-profile hybrid_hi4d`及原`--profile`／freeze／LR接口。本轮未运行preflight或训练；velocity loss、AdamW和阶段选择逻辑未改。

## 报告与后续验收

`report.json`保留每帧／每人assignment状态、原因和分数；build报告包含每样本Face／Hand ROI来源、可靠点数、腕部检查、6区域像素覆盖率、有效目标cell面积、未知重叠像素数和Source各外观可用性。缺失、拒绝、歧义帧不会被只保留成功检测的筛选丢弃。数据schema或V3来源损坏则直接失败。

`tests/test_qwen21_native_hybrid_v41.py`是**已编写但未执行**的回归源文件。后续先单独启动CPU回归，再提供`QWEN_V41_ACCEPTANCE_JOB`中的真实V3／Hybrid manifests、旧V4 checkpoint与index，核对Body数值往返、推理无targetRGB、严格加载和局部关闭复现V3。完整SMPL-X／小模型／7B prefix、梯度及冻结状态验收仍须另行启动。首次运行测试不能跳过真实资产验收就宣称兼容已验证。

本轮静态记录位于双人实验族`native_full_v41_hybrid_20261024_v1/report/`。**V4.1 Hybrid兼容代码与工具已接通、完成静态核对；真实数据和运行时验收尚未执行。**
