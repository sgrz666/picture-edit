# Native Full V4：Face＋Hand 迁移与使用

**V4 架构已接通，尚未运行统一验收；SMPL-X 资产待补齐。** 本次只执行语法编译、AST、静态导入依赖与源码 SHA 核对。没有运行测试、模型构造、前反向、生成、资产提取、渲染、缓存构建、权重迁移、optimizer 或训练；没有安装依赖。文中的运行命令全部留待后续单独启动。

基线为 `sg:/home/shangguanrz/project/pic-edit` 的 `5fffb58f2f5289389c24855e0feba2e42937be21`，不是本地旧副本。修改前的166份源码及 SHA 在数据盘实验族 `full_local_step7_20261024_v1/source/`；补充保存了包入口和审计 README。远端已有未提交改动保留。旧 Full、单人 V1/V2 和双人 V3 的加载入口仍独立；V4 必须显式选择。

## 模块与尺寸

| 层次 | 文件 | 固定接口 |
|---|---|---|
| 原 Body/Depth/关系/身份 | `native_full_runtime.py` 的 `base` | 一套 V3 参数；单人显式分流，不复制 B |
| 局部契约 | `native_local_condition.py` | `NativeLocalConditionV4`、`NativeLocalReferencesV4`、`PreparedNativeFullControlV4` |
| Face | `native_face.py` | Identity4＋Texture8＋Geometry4，各512；空间512×16×16 |
| Hand | `native_hand.py` | Geometry4＋Appearance8，各512；空间512×16×16；解剖左右手 |
| 稀疏动态/静态对照 | `native_local_control.py` | 当前目标 Hidden4096→512；六个独立512→4096零头；Static为512→512零头 |
| 执行 | `native_full_integration.py`、`native_full_pipeline.py` | 层5/10/15/21/26/31；原生32层4096；不增加 token；真实Qwen仍B=1、KV关闭 |
| 数据/工具 | `native_full_assets.py`、`native_full_data.py`、`native_full_references.py` | NPZ＋JSON树，无pickle几何缓存；读取器不渲染 |
| 权重 | `native_full_migration.py` | V3严格重建＋编码器白名单；默认encoder-only |

Face两个人共享控制器，四只手共享控制器。各层先取得未控制的 Qwen Block 输出，再分别计算 Body、Face、Hand；FP32合成后只写目标 token。Body只新增可选 `return_delta` 接口，旧调用仍沿用原分支。Depth V2只新增默认开启的 `use_detail` 对照开关，旧checkpoint参数键不变。条件前端支持结构化批次；这不代表真实Qwen支持B>1。

六区顺序固定为 `face_a, face_b, hand_a_left, hand_a_right, hand_b_left, hand_b_right`。先在像素层按Camera-Z与Scene-Z判遮挡并分配tie，再平均池化到1/16目标网格。未知Z只用声明的Person/Part可见区域；不从Relative Depth恢复Z。区域重叠总权重不超过1。

动态Query只收集实际被覆盖的目标cell。少于4个cell时跳过空间及几何attention，仍可读取有效身份/纹理；4是待验收默认值。所有输入投影正常初始化，仅最后残差头为零。Static模式不读取当前Hidden，且与Dynamic互斥。分段线性调度只依赖 `1−sigma`，默认恒为1，没有步数计数器。

## 契约和来源

几何浮点、引用特征及可学习控制都保持FP32；mask为bool、索引为int64。三个公共契约具有 `validate/to/index_select/stack`；Prepared对象同时保存每个请求的Body、ROI和局部特征，非重入checkpoint重算捕获该次执行上下文。

| 字段 | 形状 |
|---|---|
| `person_valid / source_indices / source_slot_ids` | `[B,2]`；单人B无效；后两者分别是物理数组位置与稳定逻辑ID |
| Face72、点有效性 | `[B,2,72,3]`、`[B,2,72]`；xy为归一化画布坐标，第三项为检测分数/明确的投影有效指示 |
| Jaw、Expression | `[B,2,3]`、`[B,2,10]`，独立有效性 |
| ArcFace、Face DINO | `[B,2,R,512]`、`[B,2,R,256,1536]`，独立有效性 |
| Hand45、Hand21、BPS | `[B,2,2,45]`、`[B,2,2,21,3]`、`[B,2,2,1024]`，独立有效性 |
| Hand DINO | `[B,2,2,R,256,1536]` |
| ROI | Face`[B,2,4]`、Hand`[B,2,2,4]`，不clamp的归一化xyxy |
| 区域mask/可见性/局部Z/独立Z有效性 | `[B,6,H,W]` |
| Person owner、Scene Z | `[B,2,H,W]`、`[B,1,H,W]` |
| 输出 | `[B,N_target,512]`；前缀、参考图数及Qwen序列长度不变 |

R默认1。R>1只为已声明全部来源的旧多参考缓存兼容，不能隐式读取其他帧；Qwen仍只有一张源参考图。ArcFace/DINO只能从Source提取，Dataset推理模式不打开target RGB。Source/Driver各自使用CanvasSpec，记录原图、裁剪、画布和RGB SHA。坐标越界保留，查询范围仅取实际画布交集。

缺失参数用零作存储占位，但有效性为假，不能被旧MLP当成中性姿态。Face身份/纹理可以在几何缺失时独立使用；ArcFace与DINO均有效时才做纹理对身份的增量。Hand缺45/BPS仍保留21点；缺DINO用显式Null Appearance；Geometry四个token分别标注3D、2D、两者均值和差，后两项只在两路都存在时有效。

`expression_mode=source|driver|off`默认source。参数完整的SMPL-X导入路径：先导入Source，再给Driver导入job增加 `expression_source`，按稳定人物ID和同一Expression basis重建整个人体。必须先证明原始Driver完整网格与已有mesh_cam误差≤1e-4，再重定向表情；新Body21/Depth/局部几何全部来自同一重建。重定向改变参数时标记counterfactual，训练入口拒绝原B RGB监督。若需要直接配对训练，构建job显式选driver/off，且使用原Driver资产。来源缺失的手工typed条件允许source模式刚性回退：关闭表情敏感Driver点和参数，只保留可信头部定位。不能用“source”标签掩盖仍在使用Driver表情。

Face72附件必须明确给出72个三角面ID及重心坐标；不能把原生SMPL-X joints索引当作SMPLest-X137布局。Hand21同样要求明确的21点关节索引和顺序。无附件时这些点标为不可用；可信头部/手部网格仍可提供ROI。Expression basis不是10维时，完整参数保留，但不截断成10维喂旧MLP。

Hand45来自模型返回的完整165维full_pose，即PCA解码并加pose mean后的真实15指关节局部轴角；6D采用矩阵第0列随后第1列。BPS为1024个固定单位球basis点到手顶点的最近距离，先变换到腕部FK局部坐标，再除以同beta标准姿态的腕部到最远手顶点距离；不做左右镜像，不代表接触距离。basis坐标、seed、左右约定、归一化和SHA全部保存。

## 后续资产工具输入

以下工具本轮均未执行。所有输出使用 `/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/` 下新的版本目录，拒绝覆盖。

1. `scripts/data/import_qwen21_native_smplx_v4.py --job import.json --output source_asset.npz`
   - job包含 `parameters` NPZ、`original_hw`、`camera_id`、`scale_group_id`、`provenance`（含`rgb_sha256`）、`people`。
   - NPZ包含 `betas[P,10], global_orient[P,3], body_pose[P,63], left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression, transl, mesh_cam[P,V,3], K[3,3], camera_R[3,3], camera_t[3]`，P为1或2。
   - 每个人明确 `person_id, model_file, model_sha256, model_options`（gender/use_pca/num_pca_comps/flat_hand_mean/num_expression_coeffs）、`expression_basis_sha256, segmentation, part_names`。Expression SHA按该模型`expr_dirs`的little-endian FP32连续字节计算并核对。segmentation NPZ含 `face_part14[F]`、`left_hand_vertices`、`right_hand_vertices`；part_names严格等于仓库PART_NAMES。可选 `face_attachments` NPZ及 `face72_layout=smplx137_65_136_first4_extras_then_dw68`，以及 `hand_joint_indices[2,21]`（按SMPL-X关节名称逐项核对腕、拇指、食指、中指、无名指、小指顺序）。模型必须与原拟合相同，不退回neutral。
   - 可选 `expression_source={path,sha256,schema:native_smplx_asset_v4}`，用于显式Source表情重定向。
   - 默认model_unit/model_scale。只有显式 `calibration={path,sha256}` JSON（含绑定当前参数SHA的`parameter_sha256`和正数`metres_per_model_unit`）才统一缩放网格、FK、平移和标准长度，并标记metre/calibrated；不能只在pair job改units字符串。
2. `scripts/data/cache_qwen21_native_local_references_v4.py --job references.json --output references.npz`
   - job的role必须source_only；显式本地 `arcface_model/arcface_sha256`、`dino_model/dino_files`，后者逐个列模型及processor文件SHA；不下载模型。
   - `source_person_ids`为稳定两个槽位（单人第二槽可为`__inactive__`）；`references`默认一项。每项含 `rgb={path,sha256}, canvas, person_ids, geometry_sha256, people`。
   - people只列真实人物，每人提供Source画布归一化 `face_box, hand_boxes`、`identity_valid, face_texture_valid, hand_texture_valid[2]`；启用ArcFace还需 `arcface_five_points_canvas_pixels[5,2]`。全部来自Source几何/已有检测；不新增估体器。保存ROI和五点相似变换；DINO要求ViT-g/14在224输入产生256×1536 patch。
3. `scripts/data/build_qwen21_native_full_cache_v4.py --job pair.json --output pair_cache_v1`
   - job含 `pair_id, source_asset, driver_asset, references`（均path/sha256/schema）、`source_rgb, target_rgb`（仅path/sha256）、`source_canvas, target_canvas, source_indices[2], source_slot_ids[2]`。
   - `source_masks`为每个真实Source人物的原图可见mask记录；`units/scale_status`明确为model_unit/model_scale或显式已标定metre/calibrated；`surface_distance=fcl|unavailable`不得按包是否存在降级。缺Contact保存unknown，不能伪造avoid。可选 `bps_seed, bps_basis={path,sha256}`（既有1024×3 NPY basis）、`dual_config, expression_mode`。运行时按checkpoint核对basis/frame/归一化，不能随缓存更换。
   - 输出 `body.npz, local.npz, pairs.jsonl`；body含同一SMPL-X的Body21、Depth条件、源人物mask和原图映射，不存RGB或可学习token。读取器只按manifest读取，训练才打开tgt_image。可合并同schema的pairs.jsonl形成数据集；批次按单/双人及两个画布网格分桶。

独立几何环境需已有兼容的SMPL-X、nvdiffrast和（选择fcl时）FCL；外观环境需已有ONNXRuntime、OpenCV和对应DINO加载依赖。本轮不安装它们，也不修改qwen21的diffusers版本。不要为了补局部网格拼接独立SMPL和SMPL-X拟合；V3权重可继承，V4局部资产仍须一致性证明。

## 权重审计、迁移与重建

Face候选为 `/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_last.pt`，已有3000步历史记录，仅作为候选来源，不代表V4效果。Hand来源尚未确认。

先在后续阶段使用 `scripts/audit/audit_qwen21_native_full_sources_v4.py --checkpoint FACE --state-key state_dict --output face_headers.json` 做FakeTensor元数据审计；若旧容器使用adapter_state_dict，显式改state-key，不猜嵌套结构。该工具本轮也未执行。仅对可信旧pickle可显式加 `--trusted-legacy-pickle`。

迁移命令示例（将路径替换为实际新目录）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/data/migrate_qwen21_native_full_v4.py \
  --v3 /data/shangguanrz/.../v3.pt --face /home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_last.pt \
  --face-state-key state_dict --face-prefix face_preparer. \
  --semantics /data/shangguanrz/.../semantics.json --allow-fresh-hand \
  --output /data/shangguanrz/.../native_full_v4_dynamic_v1.pt
```

按审计结果显式调整容器键、prefix和 `--discard-prefix`；不得为掩盖未知键广泛声明前缀。semantics JSON必须有 `policy: encoder-only`，可保存 `face_training_evidence`、`target_face_parameter_basis`、`target_bps_convention`。旧checkpoint metadata中的face_parameter_basis/bps_convention与目标完整匹配时才继承相应投影；缺失即重置，不能用一个兼容bool代替证据。

严格继承V3 Geometry、Depth、Body、Interaction、Identity；Face/Hand静态编码器逐键核对shape/dtype。Hand CNN首层只继承0–21通道，Camera-Z第22通道重置。报告逐项列 `inherited, partial_tensors, discarded, new_initialized, semantic_resets`，以及来源和Hand状态。没有Hand来源时必须 `--allow-fresh-hand`，记录fresh_unverified。可选 `--warm-start-attention`仅继承shape一致的共享self-attention/norm/FFN；默认不继承六阶段adapter、FiLM、1536头或旧Qwen桥。

`--local-mode static`单独迁移成Static对照checkpoint；不在加载时切换架构。V4用 `NativeFullRuntimeV4.from_checkpoint()` 原样严格重建；更早版本先显式升级到V3。任何迁移不恢复optimizer、不覆盖源文件。本轮尚无迁移后权重或实测迁移报告，只有工具与报告格式。

## 推理和训练接口（留待后续）

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/run/run_qwen21_native_full_v4.py \
  --checkpoint /data/shangguanrz/.../native_full_v4_dynamic_v1.pt \
  --manifest /data/shangguanrz/.../pairs.jsonl --output /data/shangguanrz/.../inference_v1
```

当前真实pipeline的Source需正方形letterbox（尺寸由Source Canvas决定），Driver可矩形；两个Canvas仍各自独立。`--no-face --no-hand`跳过局部编码及控制，双人直接走原V3 encode；强度0也关闭对应计算。单人自动关闭Interaction。`--no-detail`关闭共享Detail，`--depth-mode off`关闭Depth；双人Depth-off还需显式关闭raster/part-contact/detail，避免无效消融组合。

训练入口 `scripts/train/train_qwen21_native_full_v4.py` 提供 `--profile face-heads|face-full|hand-heads|hand-full|joint|all-native`，以及 `--freeze-group/--unfreeze-group/--group-lr GROUP=VALUE`。前12组延用V3；新增face_encoder/face_controller/face_heads、hand_encoder/hand_controller/hand_heads。joint默认冻结geometry_shared，其他有效组可训练；all-native显式联合。仍为velocity MSE和AdamW，无新loss、阶段切换或LR scheduler；安装器保留Runtime的requires_grad，不隐式解冻Body。

`--preflight-only`仅检查缓存和配对监督，尚未在本轮执行。训练中Source VAE latent来自原缓存64通道前缀；每step重新计算可学习条件，保留梯度；不缓存learned tokens或调用旧常数Head启动函数。

## 后续统一验收

回归用例在 `tests/test_qwen21_native_full_v4.py`，本轮只编译，未执行。后续先运行该文件及原Body/Depth/V3回归，再补齐真实资产和真实Qwen检查：

- 单/双人、每项缺失模态、稳定ID重排、Source/Driver不同画幅、Head缺失和空ROI。
- PCA/pose mean、完整Expression basis、BPS腕坐标/左右手/标准尺度、完整网格一致性。
- 全无效、越界、极小ROI、握手、手遮脸、Z与tie分配；不越界、不重复增强。
- 局部off、零Head、V3复现、严格checkpoint回读、冻结状态；保留全部prefix。
- 小模型与7B有界前反向、各活动组梯度、非重入checkpoint跨forward上下文、Dynamic/Static消融。

最后才进行单例生成与Face/Hand/双人联合训练。语法/AST通过不是这些验收的替代，也不能据本次交付声称画质、身份稳定性或训练收益。

为后续小型原生Qwen验收预留显式 `NativeLocalConfigV4(verification_only=True, hidden_dim=..., num_layers=..., injection_layers=六个有效层位)`，Body配置须同宽同层数；控制宽度仍512。该标记进入checkpoint，生产推理/训练入口明确拒绝它。默认生产配置仍为32层4096及固定六层位。本轮没有构造或运行小模型。
