# qwen21_union_migration_v1 —— 官方 Qwen-Image-2.1-Fun-Controlnet-Union 直接迁移与小样本验证

## 1. 实验目的

把阿里 PAI / VideoX-Fun 发布的 **Union 预训练控制分支原样搬到 sg 上跑通**，然后只补一层
"参考图输入"适配，验证**不训练**能否直接用于本项目的人物动作迁移。
第一轮不引入 target-only 注入、LoRA、新 Adapter 训练，也不与现有控制模块混合。

三关依次判定，前一关不过就先定位原因：

1. **官方复现**：权重完整加载 + Pose/Depth 控制有效响应 + Inpaint 白色=重绘语义正确。
2. **接入契约**：不传参考图时与原版逐位一致；控制强度 0 时与"不传控制图"逐位一致；
   参考图顺序、控制 token 数、输出裁剪正确。
3. **效果比较**：六组配对结果，指出哪些场景改善、哪些退化、退化发生在姿态/身份/接触哪一层。

结论只允许三种：可直接用官方权重 / 需要进一步输入适配 / 需要另立训练实验。

## 2. 版本与落点（全部固定，不随时间漂移）

| 项 | 值 |
|---|---|
| VideoX-Fun 源码 | commit `13bb3c625e8b1e457d7124d92207b49e00e4a6fe`，浅取到 `$WORK/source/VideoX-Fun` |
| Union 权重 | HF `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` @ revision `8a4702014d4dabb5f896fcba917e2ee0a961465f`，单文件 7,550,979,904 字节 |
| 基座 | `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`（复用现有，不重复下载） |
| 计算环境 | `$WORK/env`，由 `qwen21` 解释器建 `--system-site-packages` venv |
| torch / diffusers / transformers | `2.8.0+cu128` / `0.41.0.dev0` / `5.17.0`，装包前后逐字一致（见 `versions.json`） |
| 大产物根 `$WORK` | `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1` |
| 代码（服务器） | `experiments/6.qwen-controlnet union/qwen21_union_migration_v1/`（2026-10-19 被并入新的「6.qwen-controlnet union」族目录；旧路径 `experiments/2.qwen-test/qwen21_union_migration_v1` 保留为符号链接，本次已把它修好指向新位置） |
| 代码（本地） | `experiments/qwen21_union_migration_v1/` |

**没有往 `qwen21` 环境塞任何包**（红线 7）。venv 里只补了两个基座缺失项：`omegaconf`、`ftfy`。
`librosa` 不装：VideoX-Fun 的包 `__init__` 会连带导入 fantasytalking 等音频模型才需要它，
`union_env._stub_unrelated_optional()` 在 **diffusers/transformers 完成可用性探测之后**才挂空桩，
所以 diffusers 仍然正确认为 librosa 不可用（已实测 `_librosa_available == False`），
任何代码真去调用它会立刻抛 ImportError，不会静默出错。

## 3. 里面有什么

**脚本（`scripts/`，服务器同名目录）**

- `01_bootstrap.sh` —— 浅取固定 commit、建 venv、按固定 revision 下载权重并逐个记 SHA256、
  写 `versions.json`。日志在 `$WORK/logs/bootstrap_<时间戳>.log`。
- `02_device_check.py` —— 小张量 CUDA/BF16 matmul、8 GiB 分配、SDPA(head_dim=128) 实测；
  `nvidia_smi` 的失败**单独记录**，不作为可行性结论。
- `03_weight_audit.py` —— 180 个控制键的 key/shape/非零核验，并反证"加载后没被再次置零"。
- `04_official_samples.py` —— 官方复现关：Pose/Depth 各跑 scale 0 与 1，Inpaint 跑白色重绘，
  并把 mask 内外的平均变化做成数值判定。
- `18_control_response.py` —— 补上 04 缺的那一项量化证据：同 seed 下 scale 0 vs 1 的像素差，
  并出「控制图 | 官方卡面 | scale0 | scale1」面板，写 `response_check.json`。
- `05_freeze_inputs.py` —— 冻结六组输入，逐文件 SHA256 + 32 对齐检查，产出 `input_manifest.json`。
- `14_attach_provenance.py` —— 双人身份参照：从 `pair.json` 的 `hi4d_side` 人框（按
  `source_preprocess.json` 换算到画布）把每个槽位的源人物裁到
  `input/dual/<样本>/provenance/identity_slot*.png`，另存 `input_provenance.json`。
  只新增文件，不动既有图片与清单。
- `reference_union_pipeline.py` —— **本实验唯一的模型侧改动**：`ReferenceUnionPipeline`。
- `union_env.py` —— 固定版本引导、权重加载与审计、按官方顺序构造 pipeline。
- `06_contract_gate.py` —— 接入关四项检查（A/B/C/D，见第 5 节）。
- `07_run_matrix.py` —— 48 张矩阵。显存闸门默认 12 GiB（实测峰值 3.9 GiB，留三倍余量；
  首轮用 24 GiB 过保守，导致在同卡他人任务占卡时白等了三小时）；
  带续跑：读 `records.jsonl` 跳过已成功组合，不重复产出也不覆盖既有图片。
- `08_evaluate.py` —— 指标与面板，复用仓库既有 DWPose / PCK / 外观实现。
  单人姿态基准是 `keypoints_generation_px`（画布坐标），双人用 `canvas_points - content_box`；
  身份用 `src.pose_edit.appearance.judge_appearance`（姿态髋部配对 vs 外观最优配对，
  差超过 0.02/人才判「疑似互换」）。
- `15_report.py` —— 把 `metrics.jsonl` + 三关 JSON 汇总成 `report/METRICS.md`，含规则化判定；
  另直接比 SHA256 报「native 与 union_s0 逐字节相同」的配对数，作为对照公平性的硬证据。
- `19_collect_deliverables.py` —— 把某次运行的 48 张图、面板与指标复制进 `output/` 并写
  `deliverables_index.json`（逐文件 SHA256 + 来源）。图片永不覆盖（同名不同内容改名落地），
  `records/metrics` 这类派生视图允许刷新并清掉上一版 `.conflict` 副本。
- `20_queue_finish.sh` / `21_probe_ref_ablation.py` —— 前者等矩阵退出后自动跑 08+15+19；
  后者是控制通道的消融探针（详见第 9 节第 4 条）。
- `run.py` —— 统一入口，README 第 7 节那三条命令就是它（`report` = 08 + 15）。
- `16_gpu_probe.py` / `17_digest.py` —— 运维探针：前者实测 free/total 与占卡进程（本机
  `nvidia-smi` 因 NVML 与驱动版本不匹配不可用）；后者把各关 JSON 压成 README 用的关键数字。
- `09`–`13` —— 排队脚本与一次性诊断（`11_check_coordinates.py`、`12_check_guide_keys.py`
  就是发现「单人基准必须用 `keypoints_generation_px`」的那两个）。
- `probe_videox_imports.py` —— 一次性探针，列出 videox 包 `__init__` 为无关模型要求的缺失依赖。
- `source/`（本目录）—— 固定版本官方源码只读副本，便于离线对照阅读，不参与运行。

**数据与产物**

- `input/single/<样本>/`、`input/dual/<样本>/` —— 冻结的源图、骨架参考图、控制图、提示词、
  请求关键点 npz；目标 RGB 以 `*_eval_only` 命名，只给评估用；
  `dual/<样本>/provenance/` 是身份参照裁剪图与 `pair.json`/`dual_geometry.json`/`source_preprocess.json` 副本。
- `input_manifest.json` + `input_provenance.json` —— 冻结清单与身份出处，评估脚本只读这两个文件加实验目录内的图。
- `input/official_assets/` —— Union 仓库自带官方素材与卡面结果（对照用）。
- `$WORK/runs/<时间戳>/` —— 每次运行的图片、`records.jsonl`、`summary.json`、面板与指标。
- `$WORK/report/` —— `device_check.json`、`weight_audit.json` 等检查产物。

## 4. 关键实现约束（为什么必须这样接）

读固定版本源码得到的硬事实，决定适配层的形状：

1. 官方 control pipeline 把参考图路径**写死为 None**
   （`pipeline_qwenimage21_control.py:630`：`input_image_sizes, input_images, vae_images = [], None, None`），
   所以它的示例无法证明源人物身份能被保持——这正是要补的那一层。
2. `QwenImage21ControlTransformer2DModel.from_pretrained` 在加载后会**主动把
   `before_proj`/`after_proj` 归零并 `control_img_in.reset_parameters()`**
   （`qwenimage21_transformer2d_control.py:191-198`）。因此 Union 权重必须**之后**再加载，
   且加载完要复查非零——`03_weight_audit.py` 就是为这一步存在的。
3. 控制条件在**投影前**scatter 到图像 token 位（`:288-290`），长度必须等于
   `N_ref + N_target`。官方只覆盖目标段；一旦引入参考图，就要在 129 维输入前面补
   `N_ref` 行零。补零不等于该位置输出为零——`control_img_in` 带已训练 bias。
4. 纯 Pose 控制时后 65 通道（keep mask + masked-image latent）自然为零：
   无 mask 时 `mask_condition=1` → `mask_latent = 1-1 = 0`，无 inpaint 图 → latent 为零。
5. KV cache 在控制路径上本来就不生效（`:763` 的条件含 `control_context is None`），
   首轮显式关闭，不去声称拿到缓存加速。
6. 注入范围保持官方 joint 行为，**不加**残差门控、不屏蔽参考图区域残差（本轮范围外）。

## 5. 三关的判定口径与实测结果

**第 1 关（已过）**：`weight_audit.json` 的 `gate_pass`；Pose/Depth 在 scale 0 与 1 下产生可见差异；
Inpaint 满足"白色区域平均变化 > 黑色区域平均变化"。

- 权重：checkpoints 里 180 个控制键、模型里 180 个控制参数，全部 shape 相符；
  `from_pretrained` 先置零 34 个投影张量，Union 权重加载后 `restored_after_union_load = 180`、
  `still_zero_after_load = []`、shape/存在性不匹配 `[]`、逐位不同 `[]`；
  `loader_missing_keys = 297` 全是基座自有键（`img_in`、`modulation`、`norm_out`、`proj_out`、
  `time_text_embed` 等），`unexpected = 0`。权重 SHA256
  `65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd`，7,550,979,904 字节。
- 控制响应（`response_check.json`，40 步 seed 43 CFG 1）：Pose scale 0→1 平均绝对差
  **33.81 / 255**，变化像素占比 **87.08%**；Depth **69.23**，**90.85%**。
- Inpaint：mask 白色占比 54.61%，白区内平均变化 **79.71**、黑区 **2.25** → 白色=重绘语义成立。
- 提示词是我按控制图内容自拟的（`prompts_are_official: false`），模型卡没有随 demo 发布提示词，
  所以本关只验"控制通道有效"，不声称复现卡面像素。

**第 2 关（已过，`contract_gate_adapter_20261015T133049/contract_gate.json`）**，A/B/C/D 全绿：

- A 不传参考图 vs 原版官方 wrapper，单步 transformer 预测 `noise_pred` **逐位相等**
  （`bitwise_equal: true`，max/mean 绝对差 0.0，形状 `(1,1056,64)`，无 NaN/Inf）；
- B `control_context_scale=0` vs 完全不传控制图，同样**逐位相等**；
- C 抓取真实入参：控制条件 `(1, 9148, 129)` = 参考 8124 + 目标 1024 token、通道 129、
  `img_shapes` 三段覆盖 2 参考 + 1 目标、参考段 Leading 行全零、输出画布 512×512 与请求一致；
- D 交换参考图顺序：token 数不变，单步预测最大差 **5.0625**（输出确实随顺序改变）。

对照量选"单步 `noise_pred`"而不是最终图：1 步采样会把 latent 打成 NaN（早期踩到的坑，
`steps` 默认 2 并已复核有限），且单步预测才是"接入层有没有改变数值路径"的直接证据。

**第 3 关**：六组 × {native, 接入版 scale 0 / 0.5 / 1.0} × seed {42, 43} = 48 张。
姿态用仓库既有严格 PCK（漏检计为不命中，同时单报检测覆盖率）；双人先按髋部做人物分配，
近并列标为 ambiguous；外观报身份相似度与身份互换疑似；检测失败保留，拥抱等遮挡场景配人工面板。
数值与逐样本判定见 `report/METRICS.md`（由 `15_report.py` 生成，判定规则写在报告里）。

## 6. 当前状态（截至本文最近一次修改）

已完成并有产物：

- 固定版本源码与权重下载、独立环境、SHA256 清单：`$WORK/versions.json`、`models/*/_checksums.json`。
- 设备检查通过：BF16 matmul 164.9 ms、8 GiB 分配成功、SDPA 输出有限；
  `nvidia-smi` 返回码 18（`Driver/library version mismatch`，NVML 580.178）已单独记录，
  可行性以 torch 实际 CUDA 运算为准。矩阵运行期间 `nvidia-smi` 仍不可用，占卡情况改用
  `16_gpu_probe.py`（`torch.cuda.mem_get_info` + `/proc` 句柄）判断。
- **第 1 关通过**：180 键全加载、加载后置零残留为 0；Pose/Depth scale 0→1 分别改变
  87.08% / 90.85% 像素；Inpaint 白区变化 79.71 对黑区 2.25。产物
  `runs/official_smoke_20261015T131036`（8 步预检）与 `runs/official_formal_20261015T133449`
  （40 步正式，含 `response_check.json` 与两张控制响应面板）。
- **第 2 关通过**：A/B 逐位相等、C 序列对齐与画布正确、D 参考顺序影响输出。产物
  `runs/contract_gate_adapter_20261015T133049/contract_gate.json`
  （早一个时间戳的那次 `gate_pass: false` 是 1 步采样打出的 NaN，已改用 2 步 + 单步预测对照）。
- 六组输入全部冻结，画布单人 640×928、双人 672×896，全部 32 对齐；
  来源 run 目录与逐文件 SHA256 记在 `input_manifest.json`；双人身份参照与出处在
  `input/dual/*/provenance/` 与 `input_provenance.json`（`hug`、`dance` 两组 pair kind =
  `same_identity`，`cross_backhug` = `cross_pair`）。
  双人里 `dance_pair02_120` 与 `cross_backhug_pair02_037` 取的是
  `.skeleton_only.b_canvas` run（只有它们同时含 A/B 参考、`skeleton_both` 与提示词），已在清单里记明出处。
- **第 3 关已完成**：`runs/matrix_20261015T134829`，48/48 张全部产出并评估，`complete: true`，
  生成失败 0 次、DWPose 完全检不到人 0 张。耗时：native 157–197 s/张（峰值 3.75–3.90 GiB）、
  接入版单人 236–241 s/张、双人 295–300 s/张（峰值 4.68–4.71 GiB）。
  数值与逐样本判定见 `report/METRICS.md`（同一份内容也复制在 `output/`）。
- **对照公平性已量化**：native 与 union_s0 在 24/24 个 (样本, seed) 配对上**逐字节相同**
  （`15_report.py` 直接比 SHA256）。也就是说换后端本身零差异，scale 0.5 / 1.0 的所有差异
  都只来自 Union 控制通道。
- 主要数字（严格 PCK@0.2 / 身份分，两 seed 均值）：
  - `hard_02`：0.750 / 0.916 → s05 0.857 / 0.908 → s10 **0.964 / 0.585**
  - `hard_05_torso`：0.389 / 0.920 → s05 0.389 / 0.894 → s10 0.389 / **0.577**
    （这一例 PCK@0.5 从 0.667 升到 0.944、MPJPE 0.378→0.286、脸 MPJPE 0.085→0.028，
    但严格 PCK@0.2 卡在 0.389 不动——粗阈值看是改善，细阈值看没有）
  - `medium_02`：0.844 / 0.907 → s05 0.906 / 0.925 → s10 0.938 / 0.865
  - 双人三组在 scale 0.5 / 1.0 上**普遍退化**：`hug` 0.068→0.000、`cross_backhug` 0.456→0.000、
    `dance` 0.000→0.269(s05) 但 s10 又回 0.000。直接原因是检出人数从 native 的 1.5 / 1.0 / 3.5
    掉到 1.0——加了控制通道后两个人被合并成一个人，严格口径按定义记 0。
- 人工面板看到的机制（`output/panels/`、`output/review/`）：scale 1.0 时骨架线被**画进画面**
  （hard_02 两张 s10 上白色火柴人清晰可见），scale 0.5 干净。身份分在 s10 掉到 0.58 一部分就是
  这些线条与重绘的脸造成的。
- 消融探针 `runs/probe_skeleton_ref_ablation_20261019T222727`（不属于 48 张矩阵）：把骨架从
  参考图列表里去掉、只留源人物照片，scale 1.0 的火柴人**照样出现** → 排除「重复条件」这一解释。

一处操作教训值得记下来：首轮显存闸门设 24 GiB 过保守，而本机同卡另有 21 小时的 PE 重写任务
占着 ~62 GiB，矩阵在 21.3 GiB 空闲时白等了约 3 小时；降到 12 GiB 后又在一次 9.45 GiB 空闲处
卡住。实测本矩阵峰值 4.7 GiB，最终把闸门定在 8 GiB（仍留约 1.7 倍余量）并加了续跑
（跳过 `records.jsonl` 里已完成组合），已完成的图片不重跑也不覆盖。闸门只影响调度，不影响结果。

指标口径的一个坑（已修）：单人基准必须用 npz 里的 `keypoints_generation_px`（画布坐标）。
`keypoints_B_px` 是裁剪前的 B 帧坐标，画布 928 高时它的 y 能到 1184，用它比画布会把
工作流自己的 `final_canvas.png` 也算成 PCK@0.2 = 0；换成 `keypoints_generation_px` 后
同一张图是 PCK@0.2 = 0.923、MPJPE 0.0706。

## 7. 三条可复用命令

```bash
# 在 sg 上，仓库根 /home/shangguanrz/project/pic-edit
PY=/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/env/bin/python
CLI="experiments/6.qwen-controlnet union/qwen21_union_migration_v1/scripts/run.py"

# 1) 官方样例复现（先 --mode smoke 低分辨率短步数，再 formal 40 步 seed 43 CFG 1）
"$PY" "$CLI" official --mode formal

# 2) 六组批量比较（48 张；单卡串行，空闲显存 < 8 GiB 自动等待；重跑同一 --name 即续跑）
"$PY" "$CLI" matrix

# 3) 指标与面板生成（--run 用第 2 步打印的目录名）
"$PY" "$CLI" report --run matrix_20261015T134829
```

`experiments/2.qwen-test/qwen21_union_migration_v1` 这个旧路径仍可用（符号链接）。
本轮实际运行名：矩阵 `matrix_20261015T134829`（48/48）、官方复现
`official_smoke_20261015T131036` 与 `official_formal_20261015T133449`、
接入关 `contract_gate_adapter_20261015T133049`。

`matrix` 默认新建时间戳目录；要续跑已开的运行，直接
`07_run_matrix.py --name <既有运行名> --min-free-gib 8`（`run.py` 未透传该参数）。
辅助子命令：`freeze`（重生成冻结清单）、`audit`（180 键检查）、`gate`（接入关）、
`14_attach_provenance.py`（双人身份参照）、`18_control_response.py`（控制响应量化面板）、
`19_collect_deliverables.py`（把某次运行的图片/面板/指标复制进 `output/`）、
`20_queue_finish.sh`（等矩阵跑完自动出报告与交付物）、
`21_probe_ref_ablation.py`（只拿掉骨架参考图的消融探针）。
所有运行目录带时间戳且不覆盖既有产物；`matrix --only a,b` 可只重跑部分样本。

## 8. 与其他实验的关系

- 上游：`research/qwen21_controlnet_transfer_20261015/`（架构与权重审计，给出 16 控制块、
  129 通道、180 键、`[0,2,…,30]` 注入等结构结论）。本实验是它的**执行验证**，不重复其推导。
- 平行：`experiments/qwen21_dual_v8/`、`workflow-multiple/`（现有 Dual Adapter 与双人工作流）。
  本实验**不替换**它们，也不改其默认行为；矩阵里的 `native` 臂就是当前工作流的基线。
- 下游：若第 3 关显示需要输入适配或需要训练，才另立训练实验（方案 B），本目录只负责给出证据。

## 9. 结论（限定三选一）

**需要进一步输入适配**——不是「可直接用官方权重」，也还不到「必须另立训练实验」。

支持这个判断的事实，按证据强度排：

1. **接入层本身是对的**：24/24 个 (样本, seed) 配对上 native 与 union_s0 **逐字节相同**。
   所以 scale 0.5 / 1.0 上的任何差异都只来自控制通道，不可能是 pipeline 切换带来的混淆变量。
2. **控制通道在单人上确实有效**：严格 PCK@0.2 随强度单调上升——
   `hard_02` 0.750 → 0.857 → 0.964，`medium_02` 0.844 → 0.906 → 0.938；
   `hard_05_torso` 的 PCK@0.5 0.667 → 0.806 → 0.944、MPJPE 0.378 → 0.326 → 0.286、
   脸 MPJPE 0.085 → 0.038 → 0.028（但它的严格 PCK@0.2 卡在 0.389 不动，
   说明细阈值处仍有关键点没对上）。
3. **代价出在外观与人物可分性上，不是出在姿态上**：scale 1.0 时单人身份分集体下滑
   （0.916→0.585、0.920→0.577、0.907→0.865），面板里能直接看到**白色骨架线被画进画面**
   （scale 0.5 干净）。双人更糟：检出人数从 native 的 1.5 / 1.0 / 3.5 掉到 1.0，
   `hug` 与 `cross_backhug` 的严格 PCK 因此归零，只有 `dance` 在 0.5 档从 0.000 升到 0.269。
   也就是说双人退化发生在「两个人被并成一个人」这一层，不是关节角度层。
4. **机制已用消融排除掉一个想当然的解释**：原本怀疑是「同一张骨架既在 `reference_images`
   里、又作为控制图」造成重复条件。`21_probe_ref_ablation.py` 只拿掉参考图里的骨架、
   其余变量（提示词、控制图、seed、步数、画布）全部不变，结果 scale 1.0 下
   **骨架线照样被画进画面**（`output/probe_ref_ablation/single__hard_02__norefs_ctrl10__seed42.png`）。
   所以漏线不是重复条件，而是控制通道在「基座带参考图 + 我们这种骨架渲染」下满强度时的行为：
   我们的骨架带密集脸/手点云（见 `input/single/hard_02/control.png`），
   与官方 demo 的稀疏火柴人不同，很可能被当成可绘制的纹理。

下一步的适配方向（按优先级；第 1 条已被实测否掉）：

1. ~~开控制时不再把骨架图当参考图传~~ —— 实测无效，火柴人照样出现。
2. 强度上限取 0.5：单人在 0.5 已经是「姿态改善 + 身份不掉 + 无火柴人」；1.0 必须先解决漏线。
3. 换控制图的渲染形式：改成官方那种稀疏骨架（去掉密集脸/手点、线宽与配色对齐官方素材）。
   成本最低、最可能直接消除漏线的一档。
4. 双人必须处理人物可分性：按人分别注入，或在控制图里保留可区分的身份线索；
   本轮明确没做参考区域屏蔽与残差门控，正是这里欠的。
5. 以上做完仍拿不到收益，才转「另立训练实验」。

本轮结论不自动替换现有工作流：`native` 臂就是当前 workflow 的产物，Union 臂只作为对照存在。
