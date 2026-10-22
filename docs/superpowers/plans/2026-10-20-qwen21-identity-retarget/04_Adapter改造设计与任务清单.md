# Pro V2 身份路径改造实施计划

> 执行约定：本轮只编写方案；后续按任务顺序由单个智能体执行，可使用 executing-plans 工作流。遵从用户“不派subagent”的要求。每项代码改动先完成契约测试，再接真实模型；本文件不表示以下新接口已实现。

**Goal:** 在保留Union几何能力的前提下，为源人物身份建立可独立启停、可监督、可追溯的路径。

**Architecture:** 复用当前Pro的typed geometry、body appearance bank、人物owner和Union原生block。新增相同容量的FaceBank/IDBranch，支持写入control stream或base target ROI两种实验位置；V1完整保留。

**Tech Stack:** 现有PyTorch/Qwen2.1/固定VideoX-Fun；FP32任务参数、BF16冻结网络；复用现有可微face对齐/评价器能力，不向qwen21环境安装新包。

## 1. 改造范围与文件责任

下表“新增”是计划，不是当前仓库中已有文件。仓级路径以项目根为基准。

| 文件 | 操作 | 责任 |
|---|---|---|
| `src/pose_control/qwen21_pro/contracts_v2.py` | 新增 | V2配置、FaceBank、TaskConditionV2、版本与完整性 |
| `src/pose_control/qwen21_pro/geometry_contracts_v2.py` | 新增，定义见05 | 运行时几何契约，与训练监督许可分离 |
| `src/pose_control/qwen21_pro/identity.py` | 新增 | FaceBank读取、当前hidden查询、零输出ID残差 |
| `src/pose_control/qwen21_pro/features.py` | 修改 | 冻结源脸crop特征提取；保持V1原提取行为 |
| `src/pose_control/qwen21_pro/adapter_v2.py` | 新增 | 复用geometry/body appearance，明确关闭V1的内混FaceID |
| `src/pose_control/qwen21_pro/native_forward_v2.py` | 新增 | 原生联合序列/控制链/base循环中的显式任务参数传递 |
| `src/pose_control/qwen21_pro/integration_v2.py` | 新增 | control/base两种ID写入、mask和checkpoint边界 |
| `src/pose_control/qwen21_pro/identity_supervision.py` | 新增 | 可微ID loss、有效性/对齐/采样覆盖日志 |
| `src/pose_control/qwen21_pro/diagnostics.py` | 新增 | prefix/ROI残差、条件响应、梯度统计 |
| `src/pose_control/qwen21_pro/training.py` | 修改 | 显式V2分派和辅助loss调用，V1入口保持原行为 |
| `src/pose_control/qwen21_pro/gates.py` | 修改 | 新增通用固定评测，不修改旧fixed18结果定义 |
| `src/pose_control/qwen21_pro/checkpoint_v2.py` | 新增 | V2完整断点与warm-start区别，禁止静默加载V1 |
| `src/pose_control/qwen21_pro/pipeline.py`、`runner.py` | 修改 | 新入口/配置分派；scope、route与checkpoint一致性 |
| `tests/test_qwen21_pro_v2_*.py` | 新增 | 下面列出的明确契约与真实集成覆盖 |

不直接修改固定VideoX-Fun checkout。不修改 `union-adapter/fulladapter` 参考副本后期待正式运行自动生效。独立实验脚本未来放 `experiments/qwen21_identity_retarget_v1/scripts/`，服务器归入既有`6.qwen-controlnet union`实验族；共享功能才进入`src/`。

## 2. 先复用已经存在的能力

现有Union共享包支持 `scope="joint"` / `"target"`，由 `scoped_transformer_class.forward_control` 对输出hint施加target mask；没有必要重造该功能。对照入口应把scope显式写到协议和模型构造阶段，而不是forward中随意切换。

现有Pro安装逻辑验证完整16个control block、180个官方参数key、B=1、KV cache关闭；FP32任务参数和恢复测试已有基础。保留这些检查。训练控制流不能包在no_grad中；冻结权重仍要保留对输入/任务残差的Jacobian。

## 3. V2条件与配置契约

拟新增接口形状：

```python
@dataclass(frozen=True)
class ProV2Config:
    body: ProAdapterConfig
    identity_route: Literal['control', 'base']
    injection_scope: Literal['joint', 'target']
    base_layers: tuple[int, ...] = (0, 8, 16, 24)
    identity_width: int = 512
    face_feature_dim: int = 1600
    version: str = 'pro_identity_v2'

@dataclass(frozen=True)
class FaceBankV2:
    person_ids: tuple[int, ...]
    identity: Tensor             # [B,P,512], source only
    identity_valid: Tensor       # bool [B,P]
    detail_tokens: Tensor        # [B,P,Mf,1600], frozen source face patches
    detail_valid: Tensor         # bool [B,P,Mf]
    source_fingerprint: str
    extraction: dict

@dataclass(frozen=True)
class TaskConditionV2:
    body: TaskCondition
    face: FaceBankV2
    geometry: RetargetConditionContractV2  # 05定义；仅生成几何子集，不含监督许可/目标RGB
    version: str = 'pro_task_condition_v2'
```

以上是设计签名，不是可复制运行的完整实现。验证规则必须完整实现：B/P与body一致；person_ids同序；所有tensor设备/数值合法；无效人物不启用特征；来源严格限A；fingerprint覆盖crop、原图、提取模型、空间变换。Mf由实际提取决定，零token必须能安全退化，不强行补随机脸。

初始face细节特征延续现有1600维定义：同源位置的VAE64＋DINO1536，改为合格的源脸crop。需要记录crop到A原图坐标和各网格插值；不能把不同位置的VAE/DINO token直接拼接。源脸crop单独编码不能偷用B/目标RGB。E05所有臂使用相同bank。

V1的ProAdapterConfig与extra_state不增加默认字段，避免旧checkpoint因字典多一个key而失效。V2使用独立配置、缓存schema和checkpoint版本。

## 4. 将身份从原融合路径拆出

V2的geometry/body-appearance路径继续使用当前Pro思路，但显式禁用旧 `banks.identity_projection` 的贡献，并把该旧参数从V2 trainable清单中排除。不能旧FaceID加一次、新IDBranch再加一次而仍称单路径对照。

IDBranch以当前目标hidden为query，按同一人物读取FaceBank；face identity与细节tokens使用独立输入投影，聚合后经分层zero head输出4096维delta。只在target face mask与visible owner的交集写入。头发/大范围人物外观继续由body appearance及原生reference负责；第一版不盲目扩大脸mask覆盖整个头部。

计算顺序：

```text
q = DownProj(LayerNorm(current_target_hidden))
k,v = FaceBank(source identity tokens, source face detail tokens)
face_context = per_person_attention(q, k, v)
delta_id = ZeroHead[layer](Fusion(q, face_context, timestep))
delta_id *= face_mask ∩ visible_owner
```

ID投影和attention上游正常初始化，仅最终head全零；独立gate初值1或固定1，不能再放一个全零乘法gate切断首步学习。有效条件下首次反传预期head有梯度，上游可为0；完成一次head更新后再要求上游能获得非零梯度。

### C：写control

在选定control stage完成原生block及geometry/body task增量后加入ID delta，**再执行官方after_proj**并保存running c。不能只在stack返回后改最后一个c，漏掉本层hint。

### B：写base

在对应base层完成原生block并加geometry hint之后，从当前目标hidden查询相同IDBranch，再写目标face区域。geometry hint和ID delta分别记录，不共用最终output head。

C的control stages0/4/8/12对应B的base layers0/8/16/24。保持IDBranch宽度/输入/输出头数量一致；两组查询的hidden来源与后续传播路径不同，这是实验自变量，不声称函数等价。

## 5. 原生forward与checkpoint必须一起设计

当前Pro仅在control链的kwargs中显式传递task execution，已处理checkpoint重算。直接给base block加读取ContextVar的forward hook，会在退出作用域后的反传重算中读到空/错误条件，因此不作为正式实现。

`native_forward_v2.py`负责显式将**本次不可变执行对象**传入control与base调用及其checkpoint闭包。若上游没有足够的小接口，允许在本仓维护一个固定commit的forward适配版本：保留原联合序列构造、RoPE、时间调制、mask、CFG输出与scheduler含义，只改变有明确定义的block调用参数。文件头记录上游commit和差异清单，不直接patch上游，也不全局monkeypatch。

每次forward重新构建target mask，正负CFG文本长度可以不同；执行对象包含本次condition、timestep、mask、scope和route。多次forward后反传不能串样本。保持B=1/single-card/no-KV的已有边界，第一版不额外扩展batch与分布式。

零任务head的V2必须恢复**相同scope、同source、同控制权重**的Union；不默认恢复native。Union scale0且所有任务支路关闭时才要求与native等价。新base ID若开启，即便Union scale0仍可生效，这应在语义和实验记录中写清。

## 6. ID监督接入

复用Full已有的 `decode_tokens`、`feature` 路径和 `face_loss_alignment.build_arcface_alignment_grid/align_face_rgb` 的思想，先核对其实际网格/输入约定；不直接把Full的1536维假设带入Pro。当前Pro的 `fixed_geometry_feature_loss` 只是可选函数，不等于已经接上训练。

训练目标初版：

```text
L = L_global_flow + L_face_flow + L_hand_flow + L_contact_flow + L_body_flow
    + lambda_id * mean_valid_persons(1 - cosine(F(generated_aligned_face), F(source_aligned_face)))
x0_hat = x_t - sigma * v_pred
```

F为冻结的可微识别网络。其参数无梯度，生成图→crop/align→F→loss→任务参数的梯度必须存在。source embedding可离线提取/缓存；若现有source用ONNX、生成端用PyTorch，需验证同权重、预处理、对齐与特征方向一致，不能直接混用不同识别器输出做cosine。

对齐框/五点来自预先确定的目标几何或离线训练标注；选择依据不能依赖“当前生成图是否容易检出”而跳过差例。GT标签仅在loss路径使用，来源在supervision文件。目标姿态极端、脸过小或监督不可靠时标注invalid及原因，并统计覆盖。

E05采用相同的low-sigma/every4初始调度和lambda，详见03。身份损失不强迫A的源头向/源表情；用目标头向/表达评测防止正面脸复制取巧。同身份目标帧可作为后续视角辅助目标，但跨身份B绝不是ID真值。

不能只证明一次 `backward()` 可运行；必须记录每批有效脸数、实际aux使用次数、各分支grad norm、source ID响应、完整40步采样。单步x0监督是否足够另用短轨迹采样验证，初版不直接承担长轨迹反传的大成本。

## 7. 诊断输出与门控

`diagnostics.py`只读取/汇总，不改变预测。至少记录：每层face/body ROI中的geometry与ID残差RMS，残差与hidden比值；prefix drift；有效owner/脸面积；ID与geometry支路梯度；source替换/禁用条件前后的变化。

上述分支统计按实际注入点记录。C路线的ID经后续control block传播后已经与其他信息混合，不能把最终hint任意拆成“纯geometry＋纯ID”；B路线的独立base ID delta可直接分别统计。03的E06因此限定在具有独立base ID路径的候选上，避免名为调geometry、实际上也减弱ID。

记录RMS不等于优化目标。不同层同RMS可能产生不同结果；先做03的固定gate消融，再考虑可学习gate。geometry与ID各有独立gate，并记录stage与去噪时间的不同含义。冻结正负CFG路径的条件语义，不在一条路径漏掉身份。

## 8. 工程任务与具体验收

### Task A：恢复完整可测试工作区

- [ ] 对账远端shared与冻结root，补齐当前缺失的Union包/Pro测试镜像；只复制有清单的源码，保留较新的共享geometry修正。
- [ ] 记录实际import路径、源码SHA和依赖版本，不升级包。
- [ ] 先运行已有19个Pro测试文件与相关Union测试；把已有失败与新失败分开。

### Task B：实现V2数据与identity模块

**新增测试文件**：`tests/test_qwen21_pro_v2_contracts.py`、`test_qwen21_pro_v2_identity.py`。

- [ ] 测试person顺序不匹配、非有限值、无效人物启用脸特征必须拒绝。
- [ ] 用两个人各自不同的固定bank测试交换bank只交换归属，不改变geometry payload。
- [ ] 测试全masked bank输出有限且为零；有效bank的读出不混人。
- [ ] 测试零head初始等价、更新head后上游梯度可达；两条连续零边界不得通过该测试。
- [ ] 实现对应最小模块后重跑这些测试，再统计C/B参数数目和未使用参数。

### Task C：接入两种位置与原生forward

**新增测试文件**：`tests/test_qwen21_pro_v2_integration.py`。

- [ ] 用tiny模型验证target mask来自actual layout，不改prefix；reference数量/网格不同仍正确。
- [ ] 分别验证control stage与base layer映射，C本层hint确实包含ID增量。
- [ ] 连续两个不同条件forward后再backward，启用checkpoint时梯度与非checkpoint参考一致，不串condition。
- [ ] 验证所有任务关闭恢复同scope Union；再验证control scale0加任务关闭恢复native。
- [ ] 真实模型比较零增量预测/完整采样；确定性后端优先逐位比较，若底层存在数值噪声则预登记数值容差并说明，不能遇到失败才扩大容差。

### Task D：接入身份loss与通用固定评测

**新增测试文件**：`tests/test_qwen21_pro_v2_supervision.py`、`test_qwen21_pro_v2_gates.py`。

- [ ] 用可微toy encoder检查ID loss能反传到head，F参数仍冻结；ONNX或no_grad生成端路径应不能满足此测试。
- [ ] 验证invalid不贡献ID loss，同时记录invalid数；双人按人均衡，不让大脸像素数决定权重。
- [ ] 实现任意N样本×3固定噪声位置的新评测，N=12时为36；旧fixed18和旧gate原样保留。
- [ ] 用12例完成实际有效覆盖/梯度检查，再进入200更新；不把toy测试当真实身份效果。

### Task E：断点、CLI与offload

**新增测试文件**：`tests/test_qwen21_pro_v2_checkpoint.py`、`test_qwen21_pro_v2_runner.py`。

- [ ] V2 checkpoint包含route/scope、source bank/geometry/loss协议、optimizer/LR/RNG/cursor、完整代码/数据SHA；完成manifest最后写。
- [ ] route/scope不同严格resume失败；V1→V2只允许显式warm-start并列出加载/未加载项，不能伪装exact resume。
- [ ] 在真实模型做短连续训练与中断恢复对照；任务参数保持FP32，optimizer绑定的Parameter对象不能被offload替换。
- [ ] 新CLI显式接线scope/route/loss与诊断配置，非法组合拒绝；现有V1命令继续走原入口。

### Task F：运行E05并决定是否保留结构

- [ ] 完成C-F/C-I/B-F/B-I四臂；同时记录硬件时间，不只记录update数。
- [ ] 若B路线无额外收益，不为了“设计更合理”合入默认；若只加loss已经足够，保留较小改动。
- [ ] 只有工程、监督覆盖与完整采样评估都通过，才进入更大训练/接触/高分辨率研究。

## 9. 复用和弃用边界

保留：原生reference、typed geometry与人物bank、正确mask/恢复/offload边界、Full可微face监督经验。暂不动：Qwen基座权重、官方16块数量、129通道语义、已有V1 checkpoint。

不直接复用：Full旧SD3共享残差核心、六组输出重复写32层、没有来源/有效性的脸与手占位特征。若Union结构先验最终未带来联合收益，再按01的备选路线做原生task adapter；不能只因当前Pro200效果不足就一次重写所有模块。
