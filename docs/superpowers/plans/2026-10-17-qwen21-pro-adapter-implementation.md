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

- [ ] 接受锁定B的condition记录与已缓存source VAE latent；按`person_order`重排原始person0/1。源bank只用第一幅source参考latent或源图的冻结特征，不混入后续pose参考。
- [ ] Hi4D读取geometry.npz的normal/depth/semantic/front_owner/contact与per-person maps；iPER/TikTok读取已有normal/depth/mask；变换用各自实际resize/offset，normal/深度编码与units/convention登记，scene统一深度归一化。明确未知part/contact relation为invalid，而不是伪造已训练能力。
- [ ] 从数值目标关键点/几何产生face/hand soft支持区域；source mask/bbox只从source得到。生成source Face ID可用已安装ONNX作为冻结提取，缺失记录validity与原因。source高分辨率特征可缓存，训练resampler不可缓存。
- [ ] 缓存带模型/源码/原资产字节/输入条件/geometry codec/顺序/target_hw指纹，payload SHA校验。新产物落独立root，不修改旧B lock/index；target_rgb只保留在监督路径。测试替换target图为不可读路径仍能完成条件准备。

Run: `python -m pytest tests/test_qwen21_pro_preparation.py tests/test_qwen21_pro_cache.py -q`。服务器实查overfit6的单/双人pack及覆盖率，按新run名称保存QA与资产索引。

## Task 4: 源体型重定向与接触求解契约

**Files:**
- Create: `src/pose_control/qwen21_pro/geometry.py`
- Test: `tests/test_qwen21_pro_geometry.py`

- [ ] `retarget_request(source_shape,driver_pose,camera,body_model)`分离source beta与target pose；body_model为显式SMPL-X接口，禁止调用要求mesh_cam一致的旧reconstruct后私改beta。
- [ ] `solve_contacts(request,relations,body_model,...)`保留driver/requested/solved三层，固定camera和地面锚点；root偏移≤源身高5%、关节偏移≤5°。不收敛/超限返回requested与失败理由；无contact不动姿态。保存求解前后偏差，不声称RGB受3D碰撞监督。
- [ ] 真实SMPL-X backend在已有SMPL环境惰性加载，记录model与params指纹。缺失参数拒绝自动伪造；已有配对GT资产仅以一致的requested=solved准备训练，非零求解不能继续当原GT精确配对。
- [ ] 测试不同source shape/相同driver pose、两个端点/地面约束、超限回退、camera不改、原数组不被修改；服务器用实际SMPL-X资产作一次前向审计。

Run: `python -m pytest tests/test_qwen21_pro_geometry.py -q`。

## Task 5: 独立训练状态、区域flow与checkpoint

**Files:**
- Create: `src/pose_control/qwen21_pro/{training,checkpoint,losses}.py`
- Test: `tests/test_qwen21_pro_{training,checkpoint,losses}.py`

- [ ] 复用旧schedule/noise stream/transformer_inputs，新增`task_condition`，但optimizer/trainable whitelist/frozen digest/checkpoint独立实现，不套旧“130 LoRA only”审计。
- [ ] 区域mask按face→contact→remaining hand→body分区；`L=lambda_global*Lcontent+sum(lambda_r*sum(mask*error)/(C*sum(mask)))`，invalid面积跳过且记录覆盖。DWPose/ONNX不作可微loss。可微feature loss只接收固定geometry crop与保留输入梯度的Torch encoder。
- [ ] 只优化`pro_adapter`与明确启用的control LoRA；原始Union/base冻结且checkpoint边界保存frozen hash。新格式完整保存config、task/source/cache/code签名、所有新增参数、optimizer/LR、RNG、sample/noise/timestep与日志游标，保存时独占新目录。
- [ ] CPU先做4continuous vs2+2 exact state、跨scope/config/data拒绝、文件损坏拒绝、旧B checkpoint拒绝；再服务器真实CUDA复核。geometry/ID/contact有效分支分别检查head和上游梯度，不用降低resume误差门槛掩盖问题。

Run: `python -m pytest tests/test_qwen21_pro_training.py tests/test_qwen21_pro_checkpoint.py tests/test_qwen21_pro_losses.py -q`。

## Task 6: 可运行coarse推理与脸/双手局部精修

**Files:**
- Create: `src/pose_control/qwen21_pro/{inference,refinement}.py`
- Test: `tests/test_qwen21_pro_{inference,refinement}.py`

- [ ] denoising支持传入实际H/W、native img_shapes和task grid，不能沿用旧generate_cached固定1024target tokens的假设。正负CFG保持相同source绑定，KV cache关闭。
- [ ] `RefinementRequest`固定coarse图、目标geometry ROI、source references、person bindings、white-edit mask、seed和已训练模型签名。hand contact ROI覆盖双方并联合处理；source-only bank重建到ROI坐标。
- [ ] 官方ReferenceUnionPipeline以`image=coarse_roi,mask_image=edit_mask,reference_images=source_crops,control_image=geometry_roi,height=width=output_resolution=768,use_kv_cache=False`采样，最后按mask显式合成回coarse，mask外像素保持不变；无有效ROI退回coarse并记录原因。
- [ ] 首版refiner训练底图来自锁定stage1 checkpoint离线生成并detach，记录每张底图/ROI/采样协议哈希；GT只入监督。测试坐标roundtrip、face/paired-hands分组、mask外无变化、GT不能充底图、无validity退回以及1024/768尺寸不误用32×32。

Run: `python -m pytest tests/test_qwen21_pro_inference.py tests/test_qwen21_pro_refinement.py -q`。服务器至少执行单人/双人coarse与局部两步smoke；两步图只证明管线可运行，不作质量结论。

## Task 7: CLI、实验目录与真实GPU门禁

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

Run local: `$paths=(Get-ChildItem tests -File -Filter 'test_qwen21_pro_*.py').FullName; python -m pytest $paths -q`（PowerShell）；服务器`python -m pytest tests/test_qwen21_pro_*.py -q`。

## Task 8: 审查、部署与交付

- [ ] 每任务先spec再quality review，修复所有会造成错配、梯度截断、缓存污染、旧权重覆盖的问题；最终整体再审查一次。
- [ ] 旧Union回归、全Pro测试、服务器语法与真实GPU证据通过；检查未删移媒体、未改环境、未覆盖旧run、无大文件入Git。
- [ ] 更新README明确训练阶段、实际测试、使用命令与局限；保留源架构报告为设计历史，另记录实施偏差/证据。
- [ ] 提交隔离分支；把实际共享代码与pro-adapter使用入口部署到远端，明确运行import来源。交付时说明完成了哪些能力、怎样测试/使用、哪些训练/质量结论尚未获得，不为普通可逆步骤重复确认。
