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

截至2026-10-18，人物任务、控制层接入、数据与几何模块本地192项通过、4项需外部模型/CUDA而跳过；旧Union回归116项通过、1项跳过。服务器小型32层官方Qwen通过零分支等价与checkpoint梯度测试。真实六例VAE64审计和VAE64+DINO1536提取、缓存读回均通过，talk09得到5条部位接触边。训练接口与小型CUDA对照已通过，全尺寸训练和质量评估尚未运行，当前没有效果提升结论。

效果配置的4条单人 bank 均为 `[1,1,1024,1600]`，各含 body/face/两手 crop；双人 bank 为 `[1,2,256,1600]`，因源帧缺关键点只含 body crop。FaceID通过质量门槛的只有两条TikTok（8个人物中的2个），其余明确invalid。Hi4D normal的basis未确认、像素part不存在，仍invalid；depth、mask、visibility、soft contact有效。DINO在Torch CUDA运行，当前ONNX GPU依赖与系统不匹配，FaceID回退CPU，无需改环境。

Hi4D 两例标记 male。已用现有 `smpl_clean` 读取原始male模型的literal NumPy值，再由SMPLest-X环境的现有SMPL constructor做CPU验证：两例target mesh误差均≤2.39e-7m。新几何后端将接入该兼容读取方式。neutral与male的14类顶点标签有11/6890处差异，正式准备优先用male标签；初始neutral审计保留独立产物。

几何后端现已接入匹配模型。`geometry_cpu_20261018T113627` 与 `geometry_render_20261018T113627` 均6/6通过，mesh最大误差3.815e-6m、单人投影最大误差2.442e-4px。`geometry_probes_20261018T114018` 的已知三角形5项断言通过；真实SMPL在2cm扰动后36次迭代满足8条接触约束，最终残差4.27–6.83mm，root改动各5.61mm，最大关节改动4.988°，相机不变。该结果只证明已提供约束的求解能力，不是RGB质量或全身无碰撞结论。

训练模块位于 `qwen21_pro/{training,checkpoint,losses,offload}.py`。可选 `frozen_linear_cpu` 让原始冻结 Linear 权重保持CPU原dtype，反向仍计算输入梯度；task/LoRA和AdamW保持GPU FP32。新checkpoint保存完整task、显式LoRA、optimizer/LR、RNG及数据/源码/内存策略绑定；推理用 `load_adapter_weights()`，须在注册offload前调用。CPU训练相关77项通过、3项CUDA跳过；服务器固定32层小型Qwen的有/无LoRA两步resident/offload对照、CUDA 4=2+2精确续跑共3项通过（286.67s），新增hook guard后的snapshot再次通过CUDA精确续跑（214.10s）。完整模型显存门禁仍待执行，不能据小型测试推断峰值。

`conditions_bound1600_20261018T115624` 六例以male标签重新提取并通过缓存读回（84.02s）。typed pack现在绑定对应原生条件的sample、完整payload、prepared source与source VAE SHA，不能把另一条同尺寸pack混进训练；本轮使用新目录，原有缓存保留。

## 与其他实验的关系

上游为 `qwen21_union_dataset_adaptation_v1` 的数据锁与冻结缓存，设计见 `research/union_adapter_architecture_20261017/ARCHITECTURE.zh-CN.md`。新方案有独立代码、条件缓存、检查点和门禁；旧方案200步固定loss仅下降17.7868%，仍是失败，不作为新方案的训练结果。官方48张基线已补评，产物在 `qwen21_union_migration_v1/runs/baseline_reaudit_20261017_v2/`。
