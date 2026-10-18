# Qwen21 Pro 人物控制

验证完整16个官方 Union 控制块与新人物任务分支能否共同改善姿态、人物身份和双人接触。首轮数据沿用锁定的 overfit6：iPER、TikTok、Hi4D 各2例；正式训练集1536例、验证集72例、测试集128例沿用原划分。

最新交付见 [改造与训练结果报告](report/RESULTS_20261018.zh-CN.md)：完整官方Union已接入，六对样本200步完成，工程门禁通过，全局MSE下降5.67%、未过20%效果门槛。按用户2026-10-18最新要求，消融与后续排队任务已停止；no_dense为0次更新，其余arm未启动，本轮不再追加实验。报告保存样本次数、局部/分数据集loss、原始fixed18和检查点字节复核。

## 里面有什么

共享实现位于 `src/pose_control/qwen21_pro/`。本实验的 `scripts/export_smpl_parts.py` 在现有 SMPL 环境中，从真实6890×24的 LBS 权重导出14类顶点部位，用于 Hi4D 接触端点；它不安装包、不重建或改写原始几何。大产物写入 `/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/`，每轮使用新名称。

`scripts/audit_conditions.py` 校验原缓存生产事件、模型与代码字节，按真实 overfit6 准备条件并读回独立缓存。`vae64` 仅做条件读取基线；`effect1600` 提取冻结 VAE64+DINO1536 bank 及通过质量门槛的源 FaceID。报告记录每人物模态有效性、源 token、face/hand ROI、接触边和缓存 SHA；输入中的目标 RGB 被设为不可读路径。

```bash
P=/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1
/home/shangguanrz/project/pic-edit/add/envs/smpl_clean/bin/python scripts/export_smpl_parts.py \
  --smpl-root /home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files \
  --gender male --output "$P/data/smpl_male_v6_parts_20261018.npz"
```

命令在本实验目录运行；目标文件已存在会报错，复跑请换文件名。输出 `.npz` 的 `parts` 为6890个零基部位ID，旁边 `.json` 保存模型与产物 SHA256、每类顶点数和编码规则。

```bash
B=/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1
"$B/env/bin/python" scripts/audit_conditions.py \
  --legacy-workspace "$B" --project-root /home/shangguanrz/project/pic-edit \
  --run-dir "$P/runs/conditions_effect1600_20261018_v1" --profile effect1600 --device cuda \
  --contact-vertex-parts "$P/data/smpl_male_v6_parts_20261018.npz"
```

脚本只读旧 B。固定生产事件为 `cache/completed_20261016T184014.json`，SHA256 `ffb3077938662e16277d5e4862af8c3466e316de4f0c3b64ec107981d23ad0ae`；当前只有6条condition和6条supervision缓存索引，不能当作全训练集已经编码。上面新运行目录存在时应换名。

`scripts/audit_geometry.py` 在现有 SMPLest-X 环境中核对六例真实身体参数、相机与旧 mesh，再用源 β 重建 requested 几何；原姿态、requested、solved 分别保存。`--render` 检查已知三角形的 metric Z、行序、法线与最近人物归属，再渲染六例；`--solver-probe` 另对真实接触样本登记2cm root扰动，验证反向求解。它不修改源参数或原数据。

```bash
/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python scripts/audit_geometry.py \
  --legacy-workspace "$B" --project-root /home/shangguanrz/project/pic-edit \
  --run-dir "$P/runs/geometry_probes_<新名称>" --render --solver-probe
```

报告和压缩几何数组写入新运行目录。单人源 β 与 driver β 不同，`require_exact_gt_pairing()` 会拒绝沿用旧目标RGB；训练配对必须据此区分，不能悄悄替换几何后继续用旧 GT。

## 当前证据

截至2026-10-18，V4完整Pro、旧Union及私有pilot工具回归共550项通过、9项外部条件跳过（82.43s）；各任务spec/quality及最终整体审查均批准。完整发布模型训练的zero/prefix/梯度/精确续跑门禁、真实单人/双人CFG4 coarse与768精修及其训练缓存均已通过。

`pilot_main200_v4_20261018T161055` 已完成200次更新，工程门禁通过，效果门禁失败：fixed18全局MSE为0.15983020→0.15076623，下降5.6710%，低于20%。global＋区域训练总loss为0.94809722→0.71681113；face/contact/hand/body各分项也下降，但这是同六条训练样本的固定噪声诊断，不能代表身份、姿态或接触的验证集质量。峰值allocated/reserved为29,744,026,112/30,463,229,952bytes；14,646,784个任务参数参与训练，原生Union与基座保持冻结。消融与40步图像对比按最新要求停止，当前不建议扩大训练。

Task6的`conditions_task6_20261018T130005`六例提取与缓存读回6/6通过（64.28s）。4条单人bank为 `[1,1,1024,1600]`，两条双人bank已由早期body-only的256tokens扩为 `[1,2,1024,1600]`，加入source动作自身SMPL/camera投影的head/左右手bounds。Hi4D四个人物有效token分别569/387/473/595。FaceID仍仅两条TikTok有效（2/8）；bounds不等于脸部或手指关键点，也不证明遮挡部位可见。旧Hi4D normal/像素part仍invalid；新的current-geometry路径才使用有明确CV basis和部位codec的重渲染。DINO在Torch CUDA运行，ONNX FaceID回退CPU，环境未修改。

Hi4D 两例标记male，几何后端已接入原始male模型的受限literal NumPy兼容读取；两例target mesh误差均≤2.39e-7m。neutral与male的14类顶点标签有11/6890处差异，正式准备使用male标签，初始neutral审计保留。新增matched mixed backend也已验证四条val，仍不代表全1536例几何已经处理。

几何后端现已接入匹配模型。`geometry_cpu_20261018T113627` 与 `geometry_render_20261018T113627` 均6/6通过，mesh最大误差3.815e-6m、单人投影最大误差2.442e-4px。`geometry_probes_20261018T114018` 的已知三角形5项断言通过；真实SMPL在2cm扰动后36次迭代满足8条接触约束，最终残差4.27–6.83mm，root改动各5.61mm，最大关节改动4.988°，相机不变。该结果只证明已提供约束的求解能力，不是RGB质量或全身无碰撞结论。

renderer现按完整逆仿射裁掉原相机FoV外的像素，避免mesh漏进letterbox白边。`geometry_fov_20261018T120424`六例及7项已知三角形GPU断言通过（9.51s），含白边各模态归零。8帧旧normal/mask支持逐像素一致；其投影IoU为iPER 0.9498–0.9666、TikTok 0.9861–0.9873，source fallback mask可保留。

训练模块位于 `qwen21_pro/{training,checkpoint,losses,offload}.py`。可选 `frozen_linear_cpu` 让原始冻结Linear权重保持CPU原dtype，反向仍计算输入梯度；task/LoRA和AdamW保持GPU FP32。新checkpoint保存完整task、显式LoRA、optimizer/LR、RNG及数据/源码/内存策略绑定；推理用 `load_adapter_weights()`，须在注册offload前调用。CPU训练相关77项通过、3项CUDA跳过；服务器小型Qwen的有/无LoRA两步resident/offload对照、CUDA 4=2+2精确续跑共3项通过（286.67s），新增hook guard后再次通过CUDA精确续跑（214.10s）。以下完整模型门禁单独记录真实峰值。

`conditions_bound1600_20261018T115624` 六例以male标签重新提取并通过缓存读回（84.02s）。typed pack现在绑定对应原生条件的sample、完整payload、prepared source与source VAE SHA，不能把另一条同尺寸pack混进训练；本轮使用新目录，原有缓存保留。

`scripts/audit_training.py` 是完整发布权重的工程门禁。`runs/fullgate_20261018T121250` 已通过六例、两次平衡更新（346.44s）：180个官方tensor恢复、零分支逐值等价、训练后参考prefix与冻结参数SHA保持一致；head/geometry/contact/appearance/identity梯度均非零。362个冻结Linear的21,781,176,320bytes权重保持CPU BF16，任务/AdamW GPU FP32；峰值分配8,111,991,296bytes，含缓存9,202,302,976bytes。此时卡上可用42,869,784,576bytes，不能把它描述为独占85GB压测。checkpoint仅为工程探针，不是效果模型。

`runs/full_resume_20261018T122119` 另通过完整模型4步连续训练=2+2恢复：task参数、AdamW/LR、全RNG、noise/timestep/loss轨迹零容差一致，Parameter对象不变，冻结SHA与prefix再次通过。用时651.04s，峰值分配8,118,249,984bytes、含缓存9,307,160,576bytes。复跑切到 `code_fullgate_20261018T122119`，在下述命令末追加 `--updates 4 --exact-resume` 并换新run名称。

以下命令使用登记的冻结代码与条件缓存；运行目录须换新名称：

```bash
cd "$P/code_fullgate_20261018T121250"
"$B/env/bin/python" experiments/qwen21_pro_adapter_v1/scripts/audit_training.py \
  --legacy-workspace "$B" --project-root /home/shangguanrz/project/pic-edit \
  --feature-run "$P/runs/conditions_bound1600_20261018T115624" \
  --run-dir "$P/runs/fullgate_manual_v1"
```

`scripts/audit_inference.py geometry`在SMPL环境生成source-only bundle与当前目标控制图；`inference`在Qwen环境读回、重新编码native refs/control、生成512 coarse，再跑768 face/paired-hand。额外HH样本固定为train中的`hi4d_train_aeb388ac9f51195bf39345f3`（pair12/highfive12，A右手part7→B左手part4，原始接触距离0.476mm）。overfit6的两条Hi4D没有hand–hand边，不能冒充该正例。

已执行`inference_geometry_20261018T130005`：3份source bundle、2份driver_exact和1份source_shape，共6个worker job通过（14.55s）。source_shape明确unpaired，不能用原GT训练。下面的source index也须传给新版`audit_training.py`，使双人raw feature的几何指纹能被重算验证。

```bash
G="$P/runs/inference_geometry_20261018T130005"
"$B/env/bin/python" experiments/qwen21_pro_adapter_v1/scripts/audit_conditions.py \
  --legacy-workspace "$B" --project-root /home/shangguanrz/project/pic-edit \
  --run-dir "$P/runs/conditions_with_source_geometry_<新名称>" --profile effect1600 --device cuda \
  --contact-vertex-parts "$P/data/smpl_male_v6_parts_20261018.npz" \
  --source-geometry-index "$G/source_geometry_index.json"
```

代码与geometry helper有字节绑定；运行须使用与bundle匹配的冻结代码快照。未封口的`EncodedCondition`只可为新样本提取source features，必须与当前typed pack核对并seal后才能cache/sample/train。局部请求还会核对parent geometry、ROI affine、完整pack和source crop部位/文件签名，不能用face bank配hand references。

`code_task6_20261018T130610`的首次真实coarse/768 ROI采样已执行，但native RGBA贴回RGB时失败，`runs/inference_task6_20261018T130610/failure.json`保留。修复后的`code_task6_20261018T132022`通过完整模型两次更新（432.76s），五分支梯度与frozen/prefix检查通过；`inference_cfg4_task6_20261018T132022`随后单/双人全部通过（443.74s）：各2步512 coarse、2步768 face/paired-hand，CFG4、KVoff，编辑像素分别13056/2301，mask外逐位不变。两个真实精修训练缓存均验证keep1/inpaint64/独立GT监督与48²网格。峰值分配8,644,746,752bytes、含缓存12,859,736,064bytes；初始可用83,863,470,080bytes。这里只证明工程链路，2步图不作质量结论。

`data/val12_conditions_20261018.jsonl`与`val12_supervision_20261018.jsonl`预登记iPER/TikTok/Hi4D各4例；中文`val12_preregistration_20261018.json`保存原manifest SHA、选样规则、source/geometry覆盖与96图预算。四个arm各使用25次更新的独立checkpoint，full4取主200次更新run的step25；同一checkpoint的关分支只算diagnostic。当前0/96张val图，没有用test做训练、生成、评测或选模。legacy入口启动时的全workspace完整性校验会读取train/val/test清单和已登记资产做SHA校验，这与读取test像素做模型实验不同。val Hi4D实际全为male+female；matched mixed runtime已通过spec/quality及真实四例，male/female分别使用自己的权重与14类标签。

`scripts/audit_mixed_geometry.py`分`cpu / geometry / handoff`三阶段。`code_task7a_20261018T140821`中的四例CPU审计通过（15.72s），8个source/driver状态的mesh/joints最大误差2.38419e-7m，人物置换逐位一致；`mixed_geometry_20261018T140821`的8个worker通过（44.52s），`mixed_handoff_20261018T140821`在Qwen环境4/4通过（7.03s）且不导入smplx。handoff只验证真实几何与路由，临时source张量没有缓存、训练或采样；后续val外观编码证据另列如下。

Task7改用与mixed实现匹配的`train_geometry_20261018T142046/source_geometry_index.json`重建六例source bundle（14.35s），`conditions_pilot1600_20261018T142046`提取并校验六例1600维source缓存（58.42s）。`val_geometry_20261018T142046`准备12例当前几何（11.21s）；`val_conditions1600_sync_20261018T143039`实际编码、seal与读回12/12（310.24s），没有读取GT RGB或生成图，峰值allocated/reserved为6,518,726,656/7,132,413,952bytes。该次明确使用同步推理offload；先前常驻编码因实际free不足40GiB预算在加载前拒绝，失败run保留。

冻结`code_task7b_20261018T153556`经SCP、归档SHA与compileall验证；`cli_audit_legacy_20261018T153556`六例通过（68.39s），`cli_audit_val12_20261018T153556`十二例通过（90.00s）。共享入口实际核对原B producer、原始资产、source extractor字节与当前几何重建的typed pack；这些是数据接线审计，未训练或采样。

卡上资源释放后，`resident_resume_20261018T134310`另验证了原生常驻GPU训练：完整发布模型4continuous=2+2零容差通过，frozen/prefix与各分支梯度通过，571.69s。峰值分配29,799,127,552bytes、含缓存30,987,517,952bytes；复跑用`code_resident_20261018T134310`中的`audit_training.py --updates 4 --exact-resume --memory-mode resident`，其余feature/source-index参数与新六例门禁一致。后续200/25步pilot选择resident，原`frozen_linear_cpu`作为明确可选模式保留。

V4运行代码固定为`/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/code_task7b_20261018T161055/`，归档SHA=`bcfb00dd595fadd6e1fab9e845fedef510464f2552f2f3e5475d4fe1240f757e`，46个实际运行依赖的代码签名=`1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e`。`pilot_main200_v4_20261018T161055`已以resident/32GiB新run启动，启动PID1027095，`nohup`后台进程的PPID=1；不能把启动PID当作完成证据，当前状态须核对进程、updates和gate。旧`pilot_main200_20261018T155558`因最终门禁代码修复，在11次更新边界STOP并保存为`incomplete / passed=false`，不改标、不与V4续接。

共享CLI的`cli_smoke_exact_20261018T154445`在V2冻结代码中通过完整发布模型4continuous=2+2：zero/frozen/prefix及五分支梯度均通过，峰值allocated/reserved为29,802,322,432/30,737,956,864bytes。fixed18初始global为0.15983019976152313，旧B为0.15983019976157611；regional总loss单列。该run只有4步，质量字段`passed=false`，工程字段`engineering_passed=true`。后续V3/V4修复监督/设备协议、正式选模与文件名冲突，模型/优化器算术核心未改；V4使用自己的新名称与完整200步。

V3的真实CLI `cli_prepare_valpair_20261018T155558`执行5个body worker（38.31s）：一份双人source bundle、两份driver_exact和两份明确unpaired的source_shape。CUDA可见、device_count=1的服务器还通过真实TinyPro CPU checkpoint→CUDA execution metadata拒绝测试（1passed，2.15s）；该测试验证状态修改前拒绝，不声称执行了跨设备训练kernel。

共享代码已提交`4f9007ba`。远端部署新增40个Python源文件（Pro、Union、dataset reader和两个仓级入口），保留主仓4个已有依赖的SHA；其中`src/pose_edit/geometry.py`含服务器较新的实测torso实现，没有用旧冻结版本覆盖。远端实验入口归在`experiments/6.qwen-controlnet union/qwen21_pro_adapter_v1/`，其`code_task7b_20261018T161055/`保存95文件完整副本，代码签名与`/data/.../code_task7b_20261018T161055/`一致。pilot训练/采样须使用完整冻结根，避免不同版本依赖混用；主仓共享模块供后续新run集成。

`deployments/shared_union_pro_20261018T161055/verification.json`逐项核验40个新增文件、4个保留文件、95个副本文件及同一46文件代码签名。两侧新代码经compileall；从带空格的远端实验族完整副本执行`cli_audit_mirror6_20261018T161055`，六例实际通过（109.69s）。原`pro-adapter/README.md`部署前副本也保存在该部署目录；没有复制或删移媒体/大权重，未改环境。

## Task7共享CLI与复跑方法

共享逻辑为 `src/pose_control/qwen21_pro/{runner,dataflow,gates,ablations}.py`；仓级 `scripts/train/qwen21_pro.py`、`scripts/run/qwen21_pro.py` 和本实验 `scripts/run.py` 使用同一套参数。所有写命令必须给新 `--name`，产物位于 `--output-root/runs/<name>`；已有目录报错。`run.json` 保存绝对输入路径、实际import代码根目录、参数、种子及源码SHA，小型源码副本写入 `code_snapshot/`。签名覆盖Pro/Union包、数据与几何直接依赖和三个入口；记录代码根路径但不将路径混进源码签名，允许相同字节从冻结快照部署到ROOT。下面命令在冻结代码根目录执行，Python使用旧B的现有环境。

```bash
P=/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1
B=/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1
ROOT=/home/shangguanrz/project/pic-edit
PY="$B/env/bin/python"
F="$P/runs/conditions_pilot1600_20261018T142046"
S="$P/runs/train_geometry_20261018T142046/source_geometry_index.json"
PARTS="$P/data/smpl_male_v6_parts_20261018.npz"
COMMON=(--output-root "$P" --legacy-workspace "$B" --project-root "$ROOT"
        --feature-run "$F" --source-geometry-index "$S" --contact-vertex-parts "$PARTS"
        --memory-mode resident --memory-budget-gib 32)
"$PY" scripts/run/qwen21_pro.py status --output-root "$P"
"$PY" scripts/train/qwen21_pro.py smoke --name cli_smoke_20261018_v1 "${COMMON[@]}" --exact-resume
"$PY" scripts/train/qwen21_pro.py overfit --name pilot_main_20261018_v1 "${COMMON[@]}"
```

`smoke`默认2次更新、每次accum3，覆盖六例；`--exact-resume`为4次连续更新与2+2恢复的逐值比较。`overfit`固定200次更新、accum1、warmup32、seed3407、sampler20261015、AdamW 1e-4与clip1；默认effect1600、4融合位置、controlLoRA关闭，完整16个官方控制块和发布权重冻结。`frozen_linear_cpu`可显式选择，训练不安装推理offload hook。默认显存预算在启动时取 `min(.9*total, free−1GiB)` 并固定记录；显式32GiB要求启动空闲至少33GiB。超过预算会失败，不更改精度、分辨率或内存模式。

`fixed_eval_000000.json`、step25、step200分别保存固定18评测：六例各自新建CUDA Generator(seed43)，BF16噪声，1000步schedule中的250/500/750位置。门禁取 `losses.global` 的内容区MSE，区域加权总loss另列；完整200步、global下降至少20%并同时通过完整工程门禁才通过。CPU结果永不记作GPU通过。`microsteps.jsonl`、`updates.jsonl`记录loss/区域覆盖、梯度和资源；`observed_active_parameters`是曾有非零梯度的参数张量所覆盖的元素数，并非逐元素非零梯度计数。检查点为step0、每25步和最终边界。向run目录新建`STOP`文件会在更新边界保存并以`incomplete`结束。恢复用新name加 `--resume <旧checkpoint目录>`，严格校验完整Pro状态、数据/代码/配置/轨迹、实际训练设备/精度/噪声后端、optimizer/LR、cursor与RNG；旧日志和检查点不覆写。

正式选模还要求`gate.final_checkpoint`与实际final200的路径、cursor/accum1、metadata、manifest和state SHA全部一致；同一通过run的step0或step25仍须显式`--engineering-probe`，不能借父目录门禁冒充最终模型。精修的current manifest须与cache绑定路径/资产一致；GT监督读取前会验证driver/requested/solved与实际geometry.pt状态完全一致，不能只改`paired_supervision`布尔值给source_shape配旧GT。

```bash
for arm in no_dense no_appearance no_contact full8; do
  "$PY" scripts/train/qwen21_pro.py ablate --name "pilot_${arm}_20261018_v1" "${COMMON[@]}" --arm "$arm"
done
```

消融各自重训25次更新，使用相同共享张量初始化、样本/噪声/时间顺序、优化器和区域监督。full4取主run的`checkpoint_000025`；full8融合位置为0/2/4/6/8/10/12/14，额外head初始化为0。`no_dense`只屏蔽TASK normal/depth/像素part，`no_appearance`只屏蔽TASK source bank与FaceID，`no_contact`只屏蔽TASK显式relations与dense contact；原生reference保留。屏蔽发生在Adapter前向的局部副本，sealed cache不变；arm写入不可变extra_state，wrong-arm断点会拒绝。

准备与缓存可单独复跑。`prepare --geometry-python <现有SMPL环境python> --geometry-mode all`实际运行独立JSON worker，输出source/current/retarget索引；Hi4D分别使用自己source action和driver action的raw参数、camera与gender metadata。单人source ROI走真实source keypoints，Hi4D走source-only mesh bundle。`current`为driver_exact、`retarget`为source_shape并标记unpaired；后者不能编码原GT监督。`--canvas-hw H W`可显式选择32倍数尺寸，typed grid从实际canvas推导。新缓存用 `cache --kind current --current-geometry-index ... --source-geometry-index ... --conditions ...`，实际执行VLM/VAE与source1600提取并seal；加`--encode-supervision --supervision ...`才单独编码GT。`cache --kind legacy`只读验证旧B producer事件并建立Pro raw/typed缓存。`audit --kind legacy|current`复核缓存绑定，永不代表训练质量通过。

val12已编码的 `runs/val_conditions1600_sync_20261018T143039` 可以直接作为`--cache-run`，无需再次编码。入口只读适配其标准`native`缓存，检查producer实际权重/配置/runtime/code字段、明确的`conditioning_offload`来源、raw source、source bundle与当前geometry重建pack的SHA，不改原report或缓存。该缓存仅有CFG1所需正prompt；CFG4需另建包含负prompt的新缓存。下面先运行预登记单/双人两个样本，实际生成数由新run报告记录。

```bash
V="$P/runs/val_conditions1600_sync_20261018T143039"
VG="$P/runs/val_geometry_20261018T142046"
VAL=experiments/qwen21_pro_adapter_v1/data/val12_conditions_20261018.jsonl
CKPT="$P/runs/pilot_main_20261018_v1/checkpoint_000025"
"$PY" scripts/run/qwen21_pro.py infer --name val_pair_20261018_v1 \
  --output-root "$P" --project-root "$ROOT" --conditions "$VAL" --kind current --cache-run "$V" \
  --checkpoint "$CKPT" --arm full4 --steps 40 --seed 43 --true-cfg-scale 1 \
  --sample-id iper_val_e36352d6b4b72ae362cb1ce3 --sample-id hi4d_val_1cdfa27cfed29647a42be809 \
  --memory-budget-gib 32 --offload sync_group_offload --engineering-probe
```

`infer`实际使用`sample_cached`，保存SamplingResult audit与绑定真实checkpoint/source/current geometry的coarse artifact。验证96图的复跑单位为四个arm（full4/no_dense/no_appearance/no_contact）×seeds42/43×12例，均40步、CFG1；分别传对应25步checkpoint、arm和seed并去掉`--sample-id`过滤即可，每个组合换新name。full8是额外25步结构消融，不包含在预登记96图里；需新建full8配置缓存，不能改sealed full4 pack。

`refine`要求单个`--sample-id`、对应`--coarse <coarse.json>`、`--current-geometry-index`和source index，实际构建768 source crop、ROI VAE/DINO bank、local task、immutable request并运行精修。face显式给`--roi-kind face --roi-person <稳定ID>`；paired-hands给`--roi-kind paired_hands`与两次`--hand-endpoint <稳定ID>:<4或7>`，端点必须来自当前几何真实hand–hand relation。旧低分辨率source mask按完整FoV nearest另存native mask，绑定原mask字节，不改datasets。ROI不可用时报告fallback原因并返回原coarse路径，不算精修完成；有效ROI运行后报告mask外逐位一致。原始reference精修用`cached=False`与KVoff。未通过主200门禁的checkpoint必须显式标记`--engineering-probe`；该标签不构成质量通过。

本入口当前只执行受控六例训练；`smoke --kind current --cache-run ...`可验证另行编码的六例driver_exact训练缓存。正式1536例训练入口、完整val96的汇总评测、正式128×4×2=1024测试门禁仍待后续实现；`test`记录会被拒绝。本地测试仅CPU；完整发布模型的真实GPU smoke与主200步证据见上文，各25步与val单/双人图按实际运行继续登记。

本地回归可运行 `python -m pytest -q tests/test_qwen21_pro_runner.py tests/test_qwen21_pro_gates.py tests/test_qwen21_pro_ablations.py`。它覆盖路径独占/status只读、初始化/arm绑定、sealed条件不变、固定18指标、STOP、真实小Torch的4=2+2恢复、变尺寸coarse与768精修API路由、显存预算和native缓存适配；CPU测试不冒充完整模型GPU验收。

## 与其他实验的关系

私有诊断采样与消融汇总见 [`scripts/README.md`](scripts/README.md)。`render_pilot.py` 保持原B六例条件，生成step0/200各6张40步图；`summarize_pilot.py` 核对独立重训的25步消融、原始fixed18、检查点与门禁，并明确区分CUDA确定性重放与训练时捕获的noise SHA。34项私有工具测试通过，模型代码签名仍为 `1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e`。

本次串行运行清单为服务器 `scripts/pilot_sequence_20261018T170200.py`，dry-run清单保存于 `logs/pilot_sequence_20261018T170200.plan.json`；历史进度见 `runs/pilot_sequence_20261018T170200/events.jsonl`。原登记四个25步arm、12张训练图、4张val图及CUDA重放汇总；用户要求停止后，调度器与no_dense进程均已终止，0个完整消融、0张本轮40步图，后续不会自动启动。停止状态见相应run的 `user_stop.json` 及 `runs/status_report_20261018T170600/verification.json`。首个165900清单的日志参数冲突失败也保留，不将清单或模拟执行当作真实实验。

上游为 `qwen21_union_dataset_adaptation_v1` 的数据锁与冻结缓存，设计见 `research/union_adapter_architecture_20261017/ARCHITECTURE.zh-CN.md`。新方案有独立代码、条件缓存、检查点和门禁；旧方案200步固定loss仅下降17.7868%，仍是失败，不作为新方案的训练结果。官方48张基线已补评，产物在 `qwen21_union_migration_v1/runs/baseline_reaudit_20261017_v2/`。
