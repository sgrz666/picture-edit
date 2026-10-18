# Qwen21 Pro 人物控制

验证完整16个官方 Union 控制块与新人物任务分支能否共同改善姿态、人物身份和双人接触。首轮数据沿用锁定的 overfit6：iPER、TikTok、Hi4D 各2例；正式训练集1536例、验证集72例、测试集128例沿用原划分。

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

截至2026-10-18，冻结Task6的fullPro、Union pipeline与真实Reference循环回归共327项通过、9项外部条件跳过；独立spec与quality均批准。完整发布模型训练的zero/prefix/梯度/精确续跑门禁已通过，数据和几何准备已有真实六例证据。当前正在验证修复版CFG4 coarse/ROI与真实精修训练缓存；200步与消融尚未完成，没有效果提升结论。

最新`conditions_task6_20261018T130005`六例提取与缓存读回6/6通过（64.28s）。4条单人bank为 `[1,1,1024,1600]`，两条双人bank已由早期body-only的256tokens扩为 `[1,2,1024,1600]`，加入source动作自身SMPL/camera投影的head/左右手bounds。Hi4D四个人物有效token分别569/387/473/595。FaceID仍仅两条TikTok有效（2/8）；bounds不等于脸部或手指关键点，也不证明遮挡部位可见。旧Hi4D normal/像素part仍invalid；新的current-geometry路径才使用有明确CV basis和部位codec的重渲染。DINO在Torch CUDA运行，ONNX FaceID回退CPU，环境未修改。

Hi4D 两例标记male，几何后端已接入原始male模型的受限literal NumPy兼容读取；两例target mesh误差均≤2.39e-7m。neutral与male的14类顶点标签有11/6890处差异，正式准备使用male标签，初始neutral审计保留。female/mixed样本尚无匹配后端，不能据这两例声称全1536例几何已覆盖。

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

`code_task6_20261018T130610`的首次真实coarse/768 ROI采样已执行，但native RGBA贴回RGB时失败，`runs/inference_task6_20261018T130610/failure.json`保留。修复后的`code_task6_20261018T132022`通过完整模型两次更新（432.76s），五分支梯度与frozen/prefix检查通过；新的`inference_cfg4_task6_20261018T132022`使用CFG4验证官方L2 rescale和RGBA合成，并额外检查真实768精修训练缓存。运行尚未结束，不能把原失败标为通过。

`data/val12_conditions_20261018.jsonl`与`val12_supervision_20261018.jsonl`预登记iPER/TikTok/Hi4D各4例；中文`val12_preregistration_20261018.json`保存原manifest SHA、选样规则、source/geometry覆盖与96图预算。四个arm各使用25次更新的独立checkpoint，full4取主200次更新run的step25；同一checkpoint的关分支只算diagnostic。当前0/96张val图，未访问test。val Hi4D实际全为male+female，现有female模型已独立CPU验证到1.20e-7m以内；matched mixed runtime正在接线，不能套用male后端。

## 与其他实验的关系

上游为 `qwen21_union_dataset_adaptation_v1` 的数据锁与冻结缓存，设计见 `research/union_adapter_architecture_20261017/ARCHITECTURE.zh-CN.md`。新方案有独立代码、条件缓存、检查点和门禁；旧方案200步固定loss仅下降17.7868%，仍是失败，不作为新方案的训练结果。官方48张基线已补评，产物在 `qwen21_union_migration_v1/runs/baseline_reaudit_20261017_v2/`。
