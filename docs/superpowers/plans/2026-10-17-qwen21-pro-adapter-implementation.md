# Qwen 2.1 人物任务 Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已批准的Union人物任务设计实现成可准备真实数据、训练、推理、局部精修和复核的共享模块；完整继承16个官方控制块，不改变旧方案B及其失败门槛。

**Architecture:** 新包`src/pose_control/qwen21_pro`复用已验证的`qwen21_union`加载/参考图与flow原语。任务模块只写control target state，在底层Qwen block后、官方after_proj前融合；按人物隔离source bank并显式编码接触端点，冻结原网络但保持反向梯度。整体包含typed条件、源体型重定向接口、控制层接入、真实资产准备、独立训练状态与局部face/paired-hand精修。

**Tech Stack:** PyTorch（本地CPU2.12、服务器2.8.0+cu128）、固定VideoX-Fun、diffusers0.41.0.dev0、NumPy/Pillow；不修改原qwen21环境。共享模型与大产物在`/data/shangguanrz/`。

工作树：`C:/Users/sg/.codex/worktrees/qwen21-union-adaptation/pictureedit`，分支`codex/qwen21-union-dataset-adaptation`，起点`ad07327d`。审批：用户2026-10-17明确“可以改造吧”；按效果优先/85GB继续执行，不再为已批准设计重复确认。

## Task 1: 类型、编码器、人物bank与零输出任务模块

**Files:**
- Create: `src/pose_control/qwen21_pro/{__init__,contracts,encoders,banks,adapter}.py`
- Test: `tests/test_qwen21_pro_{contracts,adapter,banks}.py`

- [x] 定义`ProAdapterConfig(hidden_dim=4096,width=512,heads=8,stage_indices=(0,4,8,12),source_dim=64,identity_dim=512,max_parts=128)`；所有维度正数，width整除heads，stage在0..15且无重复。
- [x] 定义`ContactRelations`：`src_person,src_part,dst_person,dst_part`为[B,E]整数；`relative`为[B,E,6]、`confidence/valid`为[B,E]。端点分别编码再concat；无接触用E=0，不能用相加消除端点配对。
- [x] 定义`TaskCondition`：`target_hw`、稳定`person_ids`、typed `spatial/spatial_valid`；各spatial为[B,P,C,H,W]，modalities为normal3/depth1/mask1/visibility1/contact1/part1；part为整数ID。`person_mask/owner/face_mask`为[B,P,N]，`hand_mask`为[B,P,2,N]，`person_valid`为[B,P]；source_tokens为[B,P,M,config.source_dim]及valid（VAE基线64；真实效果配置VAE64+DINO1536为1600）；face_identity为[B,P,512]及valid；relations、source_fingerprint、provenance。验证有限值、区间、owner和≤1、实际N=H×W、索引和有效人物。元数据递归拒绝target RGB/latent/identity等监督入口。提供`.to()`、`.repeat_batch()`和严格配置/布局签名。
- [x] `ProTaskAdapter.forward_layer(stage,hidden,target_mask,conditions,timestep)`读取当前4096 hidden，按person独立查询bank，用visible owner融合。几何可跨人交换，appearance不跨bank；全套person重标号保持物理语义。无有效ID时不强制查询另一人物，ROI采用soft support避免无query。
- [x] 模块权重保持FP32，输出转换为官方hidden dtype；仅每stage终端head零初始化，其他融合/QKV正常初始化，gate1。实现`.enabled`、`state_dict`配置和trainable审计。
- [x] 先写并观察以下测试失败，再实现：

```python
def test_zero_heads_preserve_joint_and_start_head_gradient():
    from test_qwen21_pro_contracts import make_condition
    cfg = ProAdapterConfig(hidden_dim=32, width=16, heads=4)
    pack = make_condition()
    x = torch.randn(1, 7, 32, requires_grad=True)
    target_mask = torch.tensor([False, False, False, True, True, True, True])
    model = ProTaskAdapter(cfg)
    out = model.forward_layer(0, x, target_mask, pack, torch.tensor([0.5]))
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    out.square().mean().backward()
    assert model.output_heads['0'].weight.grad.abs().sum() > 0
```

补充head更新后encoder梯度、bank A/B交换、source pose参考不能进入bank、invalid/null、重叠owner、微小ROI、person置换、端点配对不混淆、BF16转换保留FP32存储。fixture由真实小尺寸tensor构造，不用mock attention。

Run: `python -m pytest tests/test_qwen21_pro_contracts.py tests/test_qwen21_pro_adapter.py tests/test_qwen21_pro_banks.py -q`。完成后先spec review再quality review；不提交他人文件。

当前实施记录：2026-10-17基线116passed/1skip；Task1先25项red、修复后72项green（含新发现的动态hidden/null bank测试先red后修）。spec review与quality review均通过；可见性/metadata/LN缺口已通过各自先red后green修复。`make_condition`是本任务测试文件中的真实小tensor fixture。

## Task 2: 原生control block与per-call上下文接入

**Files:**
- Create: `src/pose_control/qwen21_pro/{integration,pipeline}.py`
- Test: `tests/test_qwen21_pro_integration.py`

- [x] 复用旧`build_transformer`核对/加载180官方tensor，然后安装控制LoRA（可关闭），最后安装新任务模块；在offload扫描前完成注册。原180参数名保持，不用`wrapper.base.*`改变权重键，不改官方checkout。
- [x] per-instance派生control block，保留官方before合流与历史skip；明确执行以下顺序：

```python
c = native_block_forward(c, **native_kwargs)
c = task_call(stage_index, c, native_kwargs['target_token_mask'])
c_skip = block.after_proj(c)
result = torch.stack([*old_skips, c_skip, c])
```

- [x] 新transformer的forward接收`task_condition`；显式per-call上下文或ContextVar必须绑定具体model并finally恢复。`forward_control`捕获immutable条件到checkpoint闭包，重算不读取已清空的可变当前状态。不缓存trainable编码结果跨optimizer step。
- [x] 校验目标grid、batch/CFG、causal_condition、单卡、关闭KV cache；官方joint hints默认保持，target-only为另开关。fixed权重下task差分不得改prefix；所有新增影响包括LoRA关闭才要求恢复原官方数值。
- [x] CPU小型native blocks证明本层hint已受影响、历史skip不污染、重算后参数梯度、交错调用/异常清理、禁用全增量恢复相同reference Union；服务器用真实固定Qwen block再验证一次。

Run: `python -m pytest tests/test_qwen21_pro_integration.py -q`，并回归旧Union suite。

当前实施记录（spec与quality均通过）：本地Task2为40passed/1skip，全部Pro为112passed/1skip，旧Union为116passed/1skip。2026-10-18服务器固定VideoX-Fun源码的小型32层Qwen实测1passed/38deselected（8.61s，exit0），覆盖zero等价及checkpoint两步梯度。代码快照`/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/code_task2_20261018T002126/`，25个Python文件经compileall，逐文件SHA见`snapshot_manifest.json`；本阶段尚未加载全尺寸模型或训练。

审查修复：补齐链式`ProPipeline.to()`委派，避免`.to()`返回底层pipeline导致丢失task绑定；先red后green验证参数和异常原样传播。修复后父线程复跑全部Pro：112passed/1skip（9.53s），compileall与diff检查通过。

## Task 3: 真实资产准备与source-only缓存

**Files:**
- Create: `src/pose_control/qwen21_pro/{preparation,features,cache}.py`
- Test: `tests/test_qwen21_pro_{preparation,cache}.py`

- [x] 接受锁定B的condition记录与已缓存source VAE latent；按`person_order`重排原始person0/1。源bank只用第一幅source参考latent或源图的冻结特征，不混入后续pose参考。
- [x] Hi4D读取geometry.npz的normal/depth/semantic/front_owner/contact与per-person maps；iPER/TikTok读取已有normal/depth/mask；变换用各自实际resize/offset，normal/深度编码与units/convention登记，scene统一深度归一化。明确未知part/contact relation为invalid，而不是伪造已训练能力。
- [x] 从数值目标关键点/几何产生face/hand soft支持区域；source mask/bbox只从source得到。生成source Face ID可用已安装ONNX作为冻结提取，缺失记录validity与原因。source高分辨率特征可缓存，训练resampler不可缓存。
- [x] 缓存带模型/源码/原资产字节/输入条件/geometry codec/顺序/target_hw指纹，payload SHA校验。新产物落独立root，不修改旧B lock/index；target_rgb只保留在监督路径。测试替换target图为不可读路径仍能完成条件准备。

Run: `python -m pytest tests/test_qwen21_pro_preparation.py tests/test_qwen21_pro_cache.py -q`。服务器实查overfit6的单/双人pack及覆盖率，按新run名称保存QA与资产索引。

实施证据（2026-10-18）：先32项green；spec发现bare cache可错配source，新增`BoundUnionCondition/load_bound_condition`验证expected输入指纹、payload SHA、sample/record、源文件及源段/全payload摘要后43项green；真实six在talk09发现raw contact为0..6335的annotation，改为>0 membership并补共享KP/CanvasTransform代码签名依赖，48项Task3green。最终spec与quality均批准；父线程专属Pro160passed/1skip（50.60s），compileall/diff检查通过。

服务器代码快照`code_task3_20261018T105338`经SCP/归档SHA/38文件compileall核验。真实`runs/conditions_vae64_20261018T105338`与`runs/conditions_effect1600_20261018T105844`均6/6通过（117.89s/107.45s）；首个错误编码失败run`conditions_vae64_20261018T104541`保留。两个Hi4D的contact edges为0/5；单人bank均[1,1,1024,1600]，双人为[1,2,256,1600]；FaceID仅TikTok两人有效(2/8)，Hi4D source缺KP故仅body crop，normal/像素part仍invalid。DINO实际CUDA、ONNX因CUDA13依赖缺失回退CPU，未改环境。有效B producer事件SHA=`ffb3077938662e16277d5e4862af8c3466e316de4f0c3b64ec107981d23ad0ae`。

匹配male模型已在现有环境CPU以literal NumPy data_struct验证：Hi4D两例mesh最大误差2.384e-7m，source/target beta相同。male标签另存`data/smpl_male_v6_parts_20261018.npz`，SHA=`69928b1d17e4187068dcc81635b61ffa7235482549b15c23269a7dd24d8775bd`，原模型SHA=`ed4d55bb3041fefc6f73b70694d6c8edc1020c0d07340be5cc651cae2c6a6ae3`；neutral与male有11/6890个14类标签差异，后续几何优先匹配male，不覆盖neutral证据。

## Task 4: 源体型重定向与接触求解契约

**Files:**
- Create: `src/pose_control/qwen21_pro/geometry.py`
- Test: `tests/test_qwen21_pro_geometry.py`

- [x] `retarget_request(source_shape,driver_pose,camera,body_model)`分离source beta与target pose；单人显式SMPL-X、Hi4D显式SMPL，不混63/69维body pose或10475/6890顶点。禁止调用要求mesh_cam一致的旧reconstruct后私改beta。匹配male SMPL原pkl仅含literal `chumpy.ch.Ch.state.x`，可新增受限兼容读取再交现有SMPL constructor，不装包；模型与loader SHA均登记。
- [x] `solve_contacts(request,relations,body_model,...)`保留driver/requested/solved三层，固定camera和地面锚点；root偏移≤源身高5%、关节偏移≤5°。不收敛/超限返回requested与失败理由；无contact不动姿态。保存求解前后偏差，不声称RGB受3D碰撞监督。
- [x] 真实body backend在已有SMPL环境惰性加载，记录model与params指纹。缺失参数拒绝自动伪造；SMPL-X的144 joints须按真实joint_idx选137后转134，SMPL不伪造缺失的手指/脸点。既有GT精确配对要求driver/requested/solved几何均一致，改变beta或非零求解不得继续配旧GT；相机、法线、深度与poseguide必须从当前请求几何重生成。
- [x] 提供真实三角面renderer的接入（惰性nvdiffrast或已验证NvDiffRasterRenderer），输出有basis、units、pixel变换和slot约定的normal/depth/mask/owner/part/contact；由实际mesh部位可投影source hand/face ROI，不依赖不存在的Hi4D source DWPose，不把SMPL body点冒充134个完整关键点。点云splat不充当dense renderer。
- [x] 测试不同source shape/相同driver pose、两个端点/地面约束、超限回退、camera不改、原数组不被修改；服务器用实际SMPL-X资产作一次前向审计。

Run: `python -m pytest tests/test_qwen21_pro_geometry.py -q`。

实施证据（2026-10-18）：Task4先22项red，真实映射重复/稳定ID/codec与loader边界修复后29passed/3skip；quality发现cached parent package可绑定错误checkout，补三项实际red→green并修正epsilon投影主点，最终32passed/3skip，spec与quality均通过。父线程回归已完成Task1–4：192passed/4skip（31.39s）。

服务器44文件SCP快照`code_task4_20261018T113627`归档SHA=`6dae154ed73e1b4474ff769b21c6cc6c68eb5c9d5e1d96bd0aab3fc18fba0ec6`，compileall通过。真实`runs/geometry_cpu_20261018T113627`与`runs/geometry_render_20261018T113627`均6/6通过；mesh最大误差3.815e-6m，单人投影最大误差2.442e-4px；重定向单人β均变化并拒绝旧GT，Hi4D原β相同可保持精确配对。真实nvdiffrast三角面输出六模态32² typed条件，双人owner标签0/1/2。talk09有8条vertex接触，原始残差1.05–2.27mm，默认1cm容差已满足、0次迭代；不将此称为已优化修复。最早`geometry_cpu_20261018T113313`因审计误把12×16 heatmap当原图像素而失败，原失败保留，未改数据或放宽阈值。

补充真实求解与renderer闭环：`code_task4_20261018T114018`（归档SHA=`a9a293222c79768eec4f2a0ba609f8c7fe7f615f1432d5cd460710a69381771d`）的 `runs/geometry_probes_20261018T114018` 六例通过；已知三角形289像素，metric Z、top-down、CV normal、nearest owner、填充面5项断言通过。真实SMPL的2cm root扰动探针36次迭代收敛，8条残差4.27–6.83mm，root各5.61mm，最大关节4.988°，未改相机。现有SMPL环境无pytest，未装包；改用独立脚本执行同样的GPU几何断言，缺包日志另存并保持失败。

Task4补充复核已完成：8帧旧normal/mask支持完全一致，iPER/TikTok CPU投影IoU分别0.9498–0.9666/0.9861–0.9873，source fallback正确。新renderer的越FoV像素漏进padding问题通过完整inverse affine像素中心裁切修复（5项实际red→green）；spec与quality再次批准。`code_task4_20261018T120424`归档SHA=`7775bf85939d6c67e634f5d1eddb8a1ff8cd9998e4d78d640afcea1cba811ece`经SCP/compileall，`runs/geometry_fov_20261018T120424`六例及已知三角形7项断言通过（9.51s），明确验证padding各模态归零。最终父线程全Pro为255passed/8skip（23.11s）。旧GPU产物保留，其覆盖范围不追溯扩大。

## Task 5: 独立训练状态、区域flow与checkpoint

**Files:**
- Create: `src/pose_control/qwen21_pro/{training,checkpoint,losses,offload}.py`
- Test: `tests/test_qwen21_pro_{training,checkpoint,losses,offload}.py`

- [x] 复用旧schedule/noise stream/transformer_inputs，新增`task_condition`，但optimizer/trainable whitelist/frozen digest/checkpoint独立实现，不套旧“130 LoRA only”审计。
- [x] 区域mask按face→contact→remaining hand→body分区；`L=lambda_global*Lcontent+sum(lambda_r*sum(mask*error)/(C*sum(mask)))`，invalid面积跳过且记录覆盖。DWPose/ONNX不作可微loss。可微feature loss只接收固定geometry crop与保留输入梯度的Torch encoder。
- [x] 只优化`pro_adapter`与明确启用的control LoRA；原始Union/base冻结且checkpoint边界保存frozen hash。新格式完整保存config、task/source/cache/code签名、所有新增参数、optimizer/LR、RNG、sample/noise/timestep与日志游标，保存时独占新目录。
- [x] 2026-10-18实测85GB卡仅空闲20,502,216,704bytes，旧resident训练峰值约29GiB。增加可选`frozen_linear_cpu`训练模式：只让原始冻结大Linear权重常驻CPU，前向/grad_input计算时按原dtype临时传GPU；task与LoRA FP32参数及optimizer常驻GPU。保留原参数对象/名称，不量化、不降分辨率，不用会回填安装时快照的inference group-offload训练。先验证真实小Qwen resident/offload输出、梯度与更新一致及参数设备，再做全尺寸门禁；memory_mode写入checkpoint绑定。推理group-offload必须在装载新权重后注册，避免旧CPU快照覆盖已训练head。
- [x] CPU先做4continuous vs2+2 exact state、跨scope/config/data拒绝、文件损坏拒绝、旧B checkpoint拒绝；再服务器真实CUDA复核。geometry/ID/contact有效分支分别检查head和上游梯度，不用降低resume误差门槛掩盖问题。

Run: `python -m pytest tests/test_qwen21_pro_training.py tests/test_qwen21_pro_checkpoint.py tests/test_qwen21_pro_losses.py -q`。

实施证据（2026-10-18）：loss/offload先13项red→green，training/checkpoint再25项；补齐source64/1600逐条件cache_binding、自定义SampleAttestation、真实refiner keep1+inpaint64、参数/optimizer有序绑定、RNG及完整checkpoint。spec与quality均批准；加载器四类hook及实际VideoX _auto_device_hooks先11项red→green，额外证明placement后新增真实snapshot pre-hook会让直接训练入口复用旧head，补两项red→green并在flow_loss/run前统一拒绝。父线程最终训练相关77passed/3skip（15.45s）；先前全Pro248passed/7skip（29.44s）在末两项hook回归之前完成。

服务器`code_task5_20261018T115036`（归档SHA=`116abf72ec1832ca6b81ff176a3e8e3721b2df59af0a398d3aaebb74b4a99253`）52文件compileall通过；固定32层小型native CUDA有/无LoRA resident/offload两步对照及4continuous=2+2 exactresume，3passed（286.67s）。加载hook guard后的`code_task5_20261018T115624`（SHA=`fd493ddada012f0f7410ca0b43d7b13d8483b3b4c2c1338a4422ef55d4d87055`）再次CUDA exactresume 1passed（214.10s）；随后仅新增入口hook拒绝的CPU回归，未改计算图。真实`runs/conditions_bound1600_20261018T115624`以male标签六例重新提取、条件绑定与缓存读回6/6通过（84.02s）。未加载完整发布模型训练，未产生新训练权重或质量收益结论。

## Task 6: 可运行coarse推理与脸/双手局部精修

**Files:**
- Create: `src/pose_control/qwen21_pro/{inference,refinement}.py`
- Test: `tests/test_qwen21_pro_{inference,refinement}.py`

- [x] denoising支持传入实际H/W、native img_shapes和task grid，不能沿用旧generate_cached固定1024target tokens的假设。正负CFG保持相同source绑定，KV cache关闭。
- [x] 区分精确GT训练的driver条件与无配对source-shape请求；后者从requested/solved的当前mesh重新生成dense和pose/control参考，重新编码原生条件，禁止混用旧driver缓存或旧RGB GT。Hi4D仅SMPL24，不伪造脸/手指关键点；mesh part可提供标注为bounds的ROI。重定向入口必须可从prepare/infer调用，不能只留下未接入函数。
- [x] Hi4D缺source KP时，用source action自己的raw SMPL、camera与source mask建立face/head/hand crop，按稳定person_order重排；禁止借target pose裁source。将新增参数、camera、模型/LBS标签与ROI策略字节加入source指纹，保留低支持invalid。已核实两source图940×1280，raw K与现有K_runtime仅浮点舍入差异，通用逻辑仍绑定source校准。
- [x] `RefinementRequest`固定coarse图、目标geometry ROI、source references、person bindings、white-edit mask、seed和已训练模型签名。hand contact ROI覆盖双方并联合处理；source-only bank重建到ROI坐标。
- [x] 官方ReferenceUnionPipeline以`image=coarse_roi,mask_image=edit_mask,reference_images=source_crops,control_image=geometry_roi,height=width=output_resolution=768,use_kv_cache=False`采样，最后按mask显式合成回coarse，mask外像素保持不变；无有效ROI退回coarse并记录原因。
- [x] builder在CPU的transformer_setup中严格装载Task5权重，再注册inference offload；首次使用同步leaf group（use_stream=False）验证共享task在0/4/8/12多次调用，无装载后旧CPU快照覆盖。完整raw refiner需cached=False的VLM/VAE/DiT各自offload；当前ReferenceUnionPipeline不接收cached image slots，不能去掉VLM后假装同一路径可运行。refiner训练使用真实keep1+masked_coarse64通道，并经Task5的显式sample verifier绑定新缓存。
- [x] 首版refiner训练底图来自锁定stage1 checkpoint离线生成并detach，记录每张底图/ROI/采样协议哈希；GT只入监督。测试坐标roundtrip、face/paired-hands分组、mask外无变化、GT不能充底图、无validity退回以及1024/768尺寸不误用32×32。

Run: `python -m pytest tests/test_qwen21_pro_inference.py tests/test_qwen21_pro_refinement.py -q`。服务器至少执行单人/双人coarse与局部两步smoke；两步图只证明管线可运行，不作质量结论。

实施证据（2026-10-18）：独立spec/quality均批准；fullPro+Union pipeline+实际Reference循环回归327passed/9expectedskip。Encoded首次封口与fullpack/currentgeometry、ROI parent/affine、part端点及bank/native crop逐项签名的错配均通过真实red→green修复。quality补官方逐token CFG L2 norm rescale（旧Union默认False保兼容、Pro/refiner默认True），真实GPU补RGBA→RGB贴回；原失败`inference_task6_20261018T130610`保留。

冻结78文件`code_task6_20261018T132022`（归档SHA=`d3ee7f3b3e8057a88252ac9c85c9ee21da2c8589a2d77bb7ed37a1a9e2df5518`）SCP/compileall通过；完整发布模型新六例2updates门禁通过（432.76s）。`runs/inference_cfg4_task6_20261018T132022/report.json`单人TikTok与真实Hi4D high-five全部通过（443.74s）：每例2步512 coarse+2步768 ROI，CFG4、KVoff、同步leaf offload；local grids均48²，source crops分别1/2，编辑像素13056/2301，mask外逐位不变；两个真实refiner训练工厂/cache/attestation均通过，keep1/inpaint64/GT监督独立。peak allocated/reserved=8,644,746,752/12,859,736,064bytes，initialfree=83,863,470,080bytes。2步仅工程证明，不作质量结论。代码已提交0072032d，Task7a另行开发，未混入该提交。

## Task 7: CLI、实验目录与真实GPU门禁

2026-10-18按实际val盘证补充顺序：先完成Task7a匹配gender的Hi4D backend，再做Task7b CLI/训练门禁。锁定val有iPER/TikTok/Hi4D各24条；Hi4D的pair22/28全部为male+female，不能沿用早期male-only后端。现有`SMPL_FEMALE.pkl`独立CPU重建pair28 source6/target45的female人物，verts/joints最大误差1.19209e-7m，SHA=`6d4a1791b6b94880397e1a3a4539b703a228d2150c57de7b288389a8115f4ef0`，与male同faces/6890顶点、14类LBS标签有11处不同。因此补逐stable-person的gender选择、匹配模型/part标签与permutation、source/driver元数据与模型文件SHA绑定；缺失/不明gender继续拒绝，不能默认male或neutral。先CPU真实forward，再GPU dense和Qwen handoff；不安装包。

- [x] Task7a：`geometry.py/geometry_preparation.py`及对应测试接入matched mixed SMPL；renderer、projected ROI与contact codec接受按人物排序的[P,V]部位标签，同时保持旧[V]兼容。source/driver自己的参数、pair gender元数据、male/female权重与loader文件均进入只读签名；原male路径和旧失败证据保留。

Task7a证据：spec+quality均批准；完整Pro+旧Union455passed/9expectedskip。补齐删除全部mixed标记/绑定后降级male的source/current拒绝，旧shared[V]导出保持兼容。`code_task7a_20261018T140821`（82文件、归档SHA=`459d751b8494bc1425ca5d728523111fb3f1970005404adef75d792ed134b14c`）SCP/compileall通过；`mixed_cpu_20261018T140821`四例8个source/driver状态及人物置换通过（15.72s），mesh/joints最大误差2.38419e-7m；`mixed_geometry_20261018T140821`四例8个source/current worker通过（44.52s），state-order female/male labels逐行pin正确；`mixed_handoff_20261018T140821`Qwen环境4/4通过（7.03s），无smplx导入，geometry-only dummy source未cache/sample/train，不能宣称已提取val外观。两个positive各保留256条raw vertex contact pairs，两个zero E0。

85GB resident工程补充：`code_resident_20261018T134310`由冻结Task6归档派生、仅替换新增memory-mode参数的audit helper；归档SHA=`0e0bb11eaae7c5738493e339967ea70330b2332d89f30b306ebf854dec0b4dbb`。`resident_resume_20261018T134310/gate.json`完整模型4continuous=2+2逐位续跑通过（571.69s），frozen/prefix/各分支梯度通过；peak allocated/reserved=29,799,127,552/30,987,517,952bytes。后续pilot预登记采用resident（与原B resident相同），正式启动前复核实际free；若资源不足改模式须新run/明确协议，不能静默回退。

**Files:**
- Create: `src/pose_control/qwen21_pro/{runner,gates}.py`
- Create: `scripts/train/qwen21_pro.py`, `scripts/run/qwen21_pro.py`
- Create: `experiments/qwen21_pro_adapter_v1/{README.md,scripts/run.py}`
- Modify: `union-adapter/pro-adapter/README.md`
- Test: `tests/test_qwen21_pro_runner.py`

- [ ] CLI提供prepare/cache/audit/smoke/train/infer/refine/status；所有写产物命令要求新name，拒绝覆盖；绝对路径与版本打印到run.json，保存小型source快照，权重/图片不进Git。
- [ ] 新workspace`/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1`，独立代码快照与现有B环境，只增代码不改torch/diffusers。向远端发送脚本一律scp，编译/AST校验；检查Git日志和GPU实际可用内存。
- [ ] GPU smoke逐项保存：180权重不变、zero-pro等价、task prefix差分、头/上游梯度、单/双人、checkpoint恢复、两步coarse/ROI产物与SHA。任一失败先修再继续，失败不改为通过。
- [ ] 跑overfit6新200updates及固定18位置/step0&200图；结果另名、不能续旧B假装新方案。通过后具备按锁定数据扩大训练与val96图的入口；若长期训练仍运行，最终如实报告进程与完成范围，不宣称效果。
- [ ] 对齐旧B的overfit预算：200updates、accumulation=1、train_seed3407、sampler_seed20261015、AdamW lr1e-4/warmup32、循环同六例顺序；正式512的accumulation=3另登记。固定18位置仍每样本重置noise_seed43、timestep index250/500/750；门槛用同定义global content flow MSE下降≥20%，新regional总loss另列，不能混比或更改旧17.7868%失败。
- [ ] 小规模消融实际执行：锁定同seed/noise/timestep的有/无geometry、person bank、contact分支比较，分别报告六例loss与有效模态覆盖；训练拓扑4/8位置要用各自新checkpoint与相同预算，不能拿关层推理替代重训对比。val12×4arm×2seed=96图的40steps协议与可复现入口登记；先执行单/双人pilot并报告实跑数量，完整96图与正式1024图未跑完时保留待完成状态，不冒称质量验证通过。
- [ ] 消融预算先登记为各25updates、accumulation=1：完整4位置（保存主200run的step25）、去dense normal/depth/part但保留owner/ROI的routing、去task bank/FaceID但保留native source refs、去显式contact、8位置完整分支。比较同样18位置与覆盖，报告实际活跃参数量；4/8共享参数初始化核对一致，新增head均零。输入干预另标diagnostic，不代替重训结论；val两seed固定42/43。

Run local: `$paths=(Get-ChildItem tests -File -Filter 'test_qwen21_pro_*.py').FullName; python -m pytest $paths -q`（PowerShell）；服务器`python -m pytest tests/test_qwen21_pro_*.py -q`。

## Task 8: 审查、部署与交付

Task7提前工程证据：`code_fullgate_20261018T121250`由冻结Task5/FoV代码快照派生并加入独立`audit_training.py`（53文件，归档SHA=`5274e710aa889b0a59aa4b86bcf56b717874ec226b7d5803266910f5fffa7710`），SCP/compileall通过。`runs/fullgate_20261018T121250/gate.json`通过完整发布模型真实六例两次平衡更新（346.44s）；180tensor恢复、zero逐值等价、训练后prefix及frozen SHA不变，head/geometry/contact/appearance/identity上游梯度均非零。peak allocated/reserved=8,111,991,296/9,202,302,976bytes，初始free=42,869,784,576bytes；362个frozen Linear共21,781,176,320bytes CPU BF16，task与optimizer GPU FP32。更强的`full_resume_20261018T122119`在独立53文件快照（归档SHA=`ded69a06223a112f9e7b9bb4da3d6f10e71f7b9c0a6b17b348c6bea6428114e0`）中通过完整发布模型4continuous=2+2精确续跑（651.04s），task/AdamW/LR/RNG与noise/timestep/loss轨迹均零容差，参数对象不变，frozen/prefix再次通过；peak allocated/reserved=8,118,249,984/9,307,160,576bytes。coarse/ROI图和200步/消融仍未完成。

- [ ] 每任务先spec再quality review，修复所有会造成错配、梯度截断、缓存污染、旧权重覆盖的问题；最终整体再审查一次。
- [ ] 旧Union回归、全Pro测试、服务器语法与真实GPU证据通过；检查未删移媒体、未改环境、未覆盖旧run、无大文件入Git。
- [ ] 更新README明确训练阶段、实际测试、使用命令与局限；保留源架构报告为设计历史，另记录实施偏差/证据。
- [ ] 提交隔离分支；把实际共享代码与pro-adapter使用入口部署到远端，明确运行import来源。交付时说明完成了哪些能力、怎样测试/使用、哪些训练/质量结论尚未获得，不为普通可逆步骤重复确认。
