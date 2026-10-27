# workflow-v2：A+B+C 双源双人交互迁移

代码位于 `/home/shangguanrz/project/pic-edit/workflow-v2`。A 是人物 A 和场景，B 是独立人物 B，C 是双人动作参考；两个人的估计体型、服装与身份分别来自 A/B。原始 C 不进入 Qwen 生成参考。旧 `workflow-multiple` 的 `--pair` 模式保持原入口。

## 使用

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit/workflow-v2
bash run.sh --check --mode geometry_only

bash run.sh --source-a /path/A.jpg --source-b /path/B.jpg --interaction /path/C.jpg \
  --name abc_example_20261026 --mode geometry_only --resolution 512
bash run.sh --source-a /path/A.jpg --source-b /path/B.jpg --interaction /path/C.jpg \
  --name abc_example_20261026 --mode qwen_baseline --resolution 512
bash run.sh --source-a /path/A.jpg --source-b /path/B.jpg --interaction /path/C.jpg \
  --name abc_example_20261026 --mode native_v5 --resolution 512 --checkpoint /path/existing_v5.pt
```

同名、同输入、同几何设置的三次调用复用几何和 PE Prompt，两个生成模式的图片分别保存。`--user-prompt` 添加任务描述；默认 `--role-map A=left,B=right`，可改为 `A=right,B=left`。左右由 C 的投影骨盆排序决定。A/B 必须各检出一个主体，C 必须拟合两个人。手机照片先处理 EXIF 方向。

### Hi4D 优先使用官方骨架与接触

Hi4D 样例应增加 `--hi4d-record`，使用官方 SMPL 参数、标定相机和一一对应的接触标注，跳过 C 的 Multi-HMR 估计。可直接传入旧双人 pair 目录；只读取它的 C 帧元数据，不使用其中人物照片作为 A/B 来源。例如：

```bash
bash run.sh --source-a /path/A.jpg --source-b /path/B.jpg \
  --interaction ../workflow-multiple/pairs/hug_pair00_084_c4_r000/B_target.png \
  --hi4d-record ../workflow-multiple/pairs/hug_pair00_084_c4_r000 \
  --name abc_hug_official_20261026 --mode geometry_only --resolution 512
```

也可传 JSON，字段为 `official_smpl`、`meta`、`camera_file`、`rgb`、`camera_id`、`pair`、`action`、`frame_id`。C 的解码像素必须匹配该官方 RGB 帧。官方路径输出 body22 火柴人骨架与绿色接触点/连线；`geometry/interaction/official_stick_contact.png` 保留原始官方几何，Qwen 的三张骨架来自同一官方动作驱动后的 A′/B′，与目标 Depth/Normal 共享相机。

Hi4D 官方模型是 SMPL，目标人物是 SMPL-X，不能直接共用顶点编号。官方接触对完整保留；目标接触使用同部位、同骨骼局部坐标的表面附件转移，单独记录映射误差和不确定项。官方 SMPL 没有手指姿态，Native 中对应细手部几何标记为不可用；来源手部纹理仍可保留。普通 C 照片没有官方标注时才走估计路径，记录也会明确区分。

默认模式是 `native_v5`，必须显式提供既有 checkpoint。它会严格加载 V5 权重、建立两个独立来源的 Source64/脸手/Detail 绑定，并执行 Native forward；失败直接报错。此工具不启动训练，不生成新 checkpoint。

`--verify` 只读核对当前模式的输入、模型/代码指纹和全部阶段产物 SHA256，不启动估计器或生成器。完成阶段发生变化时使用新 `--name`；`--force` 自动追加时间戳建立新 run。失败的生成重试写入 `attempts/`，保留原失败证据，不覆盖旧图片。

更换 C 或修正接触约束时，可用 `--sources-from /data/.../runs/<existing_run>` 复用已经验证的 A/B mask、DWPose 与 SMPLest-X 参数。它只读旧 run 的 Source 产物，逐个核对 SHA 和实际 A/B 图像像素；C、重定向网格和生成结果仍写入新 run。A/B 交换时按实际图片匹配缓存并显式重新绑定逻辑人物。缓存分辨率必须一致。

## 产物与模块

默认输出在 `/data/shangguanrz/pic-edit-workflow-v2/runs/<name>/`，不向系统盘写生成缓存或权重。

- `geometry/scene.json`：A/B 独立来源、目标 SMPL-X 参数、相机、角色映射、估计器重建误差和接触/碰撞诊断。
- `geometry/render/`：独立与联合骨架、Depth、Normal、Part、Owner、Contact，及保留遮挡前后几何的 `geometry_arrays.npz`。
- `geometry/native/`：`geometry_only` 即可构建并校验的现有 Native 条件包，不加载 Qwen 或 Adapter 权重。
- `prompt/`：用户原文、PE 原始输出、角色编译后的 `generation_prompt.txt`。
- `generation/qwen_baseline/` 与 `generation/native_v5/`：`final_canvas.png`、裁去 A 画布留边的 `final_image.png` 和 `generation.json`。`qwen_source64.pt` 保存实际 Qwen 编码的两份独立来源张量及数值 SHA，不包含目标 RGB/latent。Native 目录另有双源特征来源记录。
- `evaluation/<mode>/`：逐人姿态、可见点覆盖、外观相似度、背景变化与叠加预览。纠缠姿态的人数检测可能合并，区域检测也可能重复检到同一个人，报告保留这类不确定性，不自动宣称质量通过。
- `run_<mode>.json`：运行状态；各阶段 `stage.json` 是可复验的证据。

`retarget/` 负责完整参数转换、同体型重建、共享相机放置和有限接触修正；`render/` 使用实际 SMPL-X 三角网格做 Z-buffer 合成。`abc_native_assets.py` 把这些几何适配到已有 V5；`abc_generate_worker.py` 将五张参考图按原生画布编码，依次加载文本编码器/VAE/Transformer，减少同时驻留显存。

五图顺序固定为 A、B、A′ 骨架、B′ 骨架、联合骨架。两张 Source 的画布和 VAE latent 保持独立，不拼接人物照片。Normal 复用已有 Native 路径；新增的是双源路由和五图协议，没有新增待训练的网络参数。

FCL 使用 `/data/shangguanrz/pic-edit-workflow-v2/deps/python_fcl` 中的 `python-fcl 0.7.0.11`，未向 `qwen21` 安装包。其[上游说明](https://pypi.org/project/python-fcl/)提供三角网格距离与碰撞查询；相交三角形数量用于限制接触平移步长，不等同于穿透体积。没有接触证据时明确记录不确定，不强制吸合最近顶点。

## 测试与当前验收

```bash
cd /home/shangguanrz/project/pic-edit
OMP_NUM_THREADS=2 /home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m pytest \
  workflow-v2/tests tests/test_qwen21_dualsource_v5.py tests/test_qwen21_native_multiref_v5.py \
  tests/test_qwen21_v5_real_tiny_multiref.py tests/test_qwen21_native_full_v5.py \
  workflow-multiple/tests/test_dual_geometry.py workflow-multiple/tests/test_dual_prompts.py -q
```

本轮完整回归为 **95 passed、2 skipped**；跳过的是旧测试中需要额外绑定外部真实资产任务的入口，另外已用本 Workflow 实际运行模型。真实矩阵包含站立、握手、拥抱的三种模式，以及来源交换、角色交换两个 Native 用例，**11/11 完成且 11/11 只读校验通过**；独立审计验证了双 Source64 数值 SHA、五图顺序、不读取目标 RGB、官方接触与转移附件的语义，以及未伪造官方手指姿态。

矩阵参数为 512 分辨率、12 步、seed 42；同一用例的基线/V5 共用 Prompt，且两份来源 latent 的 SHA 完全一致。站立/拥抱使用 Hi4D 官方参数和接触；握手参考是非 Hi4D 的历史上半身照片，明确走估计分支，其腿部不是标注真值。早期单独样例还实际跑过 30 步。

记录位于 `/data/shangguanrz/pic-edit-workflow-v2/validation/acceptance_20261026_v4/`：`summary.json` 是运行清单，`audit.json` 是独立条件/产物审计，`manual_review.json` 记录图像复查。同一 A/B 缓存的拥抱几何对照中，相交三角形计数从估计路线的 445 降为官方路线的 374；这不是穿透体积，也不代表所有动作改善。官方接触表面附件的不确定性保留在 `scene.json`。

质量尚未判定达标：本轮拥抱基线有骨架线条泄漏，部分 Native 手部有绿色残影或模糊/重影。流程通过不等于质量通过。任意 A+B+C 没有真实配对 D，不能拿 C 的 RGB 监督输出。

本轮验证的既有 checkpoint 为：

```text
/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-fixedpair-v5_20261025_v1/training/checkpoints/update_00000300_5821758668f84b5e8b4d99dc5499c83e.pt
```

文件 SHA256：`6a340c0a9bde9fbea943bfb26cbfce425a893ac2201d1b23651e0508e4bd20e2`。每次运行仍核对实际文件，不自动寻找或切换最新权重。

此流水线复用 `src/pose_edit`、SMPLest-X、Multi-HMR 和已有 Native V5，与单人 `workflow` 及旧双人 `workflow-multiple` 共存。
