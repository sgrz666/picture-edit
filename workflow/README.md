# 单人动作迁移 Workflow

部署位置：`sg:/home/shangguanrz/project/pic-edit/workflow`。

这条流程把 A 的人物外观和原场景保留下来，迁移 B 的动作。B 经 SMPLest-X 估计得到 SMPL-X，再将相机空间关键点投影成 DWPose 风格骨架；PE 同时读取 A、B、最终骨架并生成提示词；最后 Qwen 基座只读取 A 和骨架两张参考图。生成阶段没有 adapter、LoRA，也不会把 B 原图送给生成器。

测试 B 中的篮球不加入生成图。道具策略固定为 `source_only`：保留 A 已有物品，排除 B 独有物品。

审查记录与整改对应关系见 [`docs/workflow审查与整改_20261012.md`](../docs/workflow审查与整改_20261012.md)。

协作智能体在本目录的协作纪律（「标准六图面板」硬性收尾、跑批与共享 GPU、产物/缓存/评测约定）见 [`AGENTS.md`](AGENTS.md)。

## 运行

```bash
ssh sg
cd /home/shangguanrz/project/pic-edit/workflow
bash run.sh --check          # 资产路径、两个环境导入、CUDA、估计器 conf hook
bash run.sh                  # pictures/A + pictures/B 各一张，输出到 runs/demo
```

跑测试集里的一对，run 名默认取目录名。测试集位置两侧不同（sg 的 `experiments/` 是 5 族分组，Windows 侧平铺），`batch.sh` 与 `report.py` 现在都会自己探测两种布局；只写死平铺路径的命令在 2026-10-13 重构后指不到东西了：

```bash
PAIRS=../experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs   # sg 实测在位，7 对齐全
bash run.sh --pair $PAIRS/hard_05 --resolution 768
bash run.sh --pair .../pairs/hard_05 --resolution 768 --verify   # 同参数复验，不启动推理
```

跑整个测试集并逐对评测（单对失败不中断，日志在 `logs/`）：

```bash
RESOLUTION=768 bash batch.sh
RESOLUTION=768 bash batch.sh /absolute/path/to/other/pairs
```

指定图、种子、提示词与主体：

```bash
bash run.sh --name my_pair --a /abs/A.jpg --b /abs/B.jpg --seed 43
bash run.sh --name my_pair --prompt-file my_prompt.txt      # 手工提示词，见下
bash run.sh --pair .../pairs/hard_01 --subject largest      # 多人图里选最大的那个人
bash run.sh --name my_pair --subject 1                      # 或按面积优先排序取第 1 个
```

`--subject` 默认 `auto`：A、B 各只允许检出一个人，否则流程停止而不是猜。多人图必须显式选择；选了主体之后画面里其他人仍然在，生成器可能改动他们，这一点没有被评测覆盖。

`--framing` 决定骨架如何缩放到源人物身上：默认 `source_person` 把 B 的检测框内切进 A 的框；`source_torso` 用 B 的实测躯干长度去匹配"A 框高推出的源人物躯干"。两者框长宽比相反时，前者会被宽度主导、把姿态压得远小于源人物体格（见"实况"里的 `hard_05` A/B）。

分阶段检查或恢复：

```bash
bash run.sh --until geometry
bash run.sh --until prompt
bash run.sh
```

修改种子只重跑生成。改 `--prompt-file` 只重跑生成，PE 保持缓存且它的产物不会被覆盖。需要强制全部重跑时加 `--force`；建议新建 `--name` 来保留先前结果。原始 A/B 图片不会改写。

## 缓存怎么判定

每个阶段的 `stage.json` 记两样东西：`identity`（设置、workflow 代码哈希、模型文件身份、各输入的内容 SHA256）与产物清单（相对 run 目录的路径 + SHA256）。绝对路径只放在 `runtime` 里，不参与指纹，所以改 run 目录名、搬仓库、换机器都不会让内容失效；改设置、改代码、改输入内容、改产物内容才会。

失效时不再只说一句"stale"，而是逐条说明原因。下面是实测输出：改了 `run.py` 之后复验既有 run，会指名是哪个代码文件的哈希变了（`bash run.sh --name hard_02 --pair …/hard_02 --resolution 768 --verify`）：

```
WORKFLOW_FAILED: Prepared inputs are not reusable:
  request identity changed (bf07d16f... != a204fb7e...)
  workflow code changed: run.py (52794b78... -> d9809836...)
```

产物在阶段写完后被改动、或上游输入哈希变了，会分别报 `artifact content changed after the stage wrote it: <name> -> <path> (recorded …, on disk …)` 与 `inputs_sha256.<name> changed: …`（这两种措辞由 `tests/test_workflow_common.py` 断言）。

代码哈希在指纹里，意味着**任何一次 workflow 代码改动都会让全部既有 run 变成 stale**，即使产物一个字节都没变。这是设计取舍：报告里的数字必须绑定产出它的那份代码。要判断产物本身是否完好，看 `stage.json` 里的哈希，或读 `report.py` 的状态列。2026-10-13 把共享层（缓存与指纹、关节表、投影、估计器调用）提到仓库级 `src/pose_edit/` 供单人与双人两条线共用，因此该日之前产出的全部 run 现在都会报 `workflow code changed`。

收尾会对本轮涉及的每个阶段重跑一次校验，把实测结果写进 `run.json` 的 `verification.status`（`validated` / `mismatch`）并打印 `VERIFICATION …`；这句结论是量出来的，不是走到结尾就写上的。

PE 的产物 `prompt/pe_compiled_prompt.txt` 只读。生成阶段实际读的是 `prompt/generation_prompt.txt`，它要么等于 PE 产物，要么是 `--prompt-file` 指定的文本；`prompt/generation_prompt.json` 记录来源、来源哈希与生效哈希。所以"手工改提示词做对照实验"是显式输入，不再靠改一个被哈希跟踪的文件，也不会被下一次 PE 覆盖。

## 看哪些输出

| 文件，相对 `runs/<name>/` | 用途 |
| --- | --- |
| `generation/final_image.png` | 最终图，移除 A 预处理添加的留边 |
| `generation/final_canvas.png` | 未裁留边的完整生成画布 |
| `generation/generation.json` | 两张生成参考图、无 adapter/LoRA 记录、种子、提示词来源与哈希、显存/耗时 |
| `prompt/generation_prompt.txt` | 实际传给生成器的英文提示词 |
| `prompt/generation_prompt.json` | 该提示词来自 PE 还是 `--prompt-file`，及其哈希 |
| `prompt/pe_compiled_prompt.txt` | PE 输出经角色编译后的文本（只读，不被手工覆盖） |
| `prompt/pe_raw.txt`、`pe_result.json` | PE 原始回答、三图角色、采样设置及编译结果 |
| `prompt/pe_attempts.json` | JSON/图像引用验证及重试记录，最多两次 |
| `prompt/effective_system_prompt.txt` | 官方 system 全文 + 本流程的最终图像引用规则 |
| `geometry/B_smplx.pt` | B 的 SMPL-X 参数及估计器原始导出数据 |
| `geometry/B_keypoints.npz` | 137 个 SMPL-X 关节、134 个映射点、原图和生成画布坐标 |
| `geometry/B_3d_dwpose.png` | PE 与生成器实际读取的同一张骨架图 |
| `geometry/B_poseguide_overlay.png`、`A_poseguide_alignment.png` | 骨架叠加 B / 叠加 A，检查估计误差与尺度位置 |
| `geometry/B_camera.json` | 相机参数、等比对齐变换、重建/投影误差、检出人数与主体选择、主体一致性检查 |
| `geometry/estimator/` | 本次估计器暂存输入、原始参数和日志；权重链接到现有资产 |
| `evaluation/eval.json` | 生成图回估骨架与请求骨架的误差（分组 MPJPE / PCK） |
| `evaluation/eval_overlay.png` | 请求骨架与回估骨架叠加在生成图上 |
| `evaluation/final_keypoints.npz`、`final_smplx.pt` | 生成图的回估关键点与 SMPL-X 参数 |
| `run.json`、`outputs.json` | 运行状态、逐阶段证据、实测校验结论、产物绝对路径索引 |

每个阶段的 `run.log` 保留推理日志；`stage.json` 记录请求指纹与产物哈希。`latest_run.json` 指向最近一次执行，多个 run 并存时它只反映最后一个，横向比较请用 `report.py`。仅完成几何或 PE 时状态为 `partial`，全部生成完成后为 `complete`，失败时为 `failed` 并保留错误；进程被杀而没写失败状态的，`report.py` 会标成 `abandoned`。

离线实验脚本 `lowlatency_probe.py` / `check_case.py`（llama.cpp 低延迟探测，见「低延迟探测（llama.cpp / GGUF）」一节）只往调用方给的 `--out` 实验目录写产物，不会在 `runs/` 下留 run、也不碰历史产物。

## 评测

```bash
QWENPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
GEOPY=$($QWENPY -c 'import json;print(json.load(open("config.json"))["geometry_python"])')
$GEOPY evaluate.py --run hard_05          # 单个 run；已算过则 CACHE_HIT
$QWENPY evaluate2.py --all                # 无 GT 六轴补测，写 eval2.json，绝不覆写 eval.json
$QWENPY scorecard.py                      # 汇总 eval2 成横表
RESOLUTION=768 bash batch.sh              # 全测试集，跑完自动逐对评测
$QWENPY report.py                         # 全部 run 的索引、状态、逐对指标
```

`evaluate.py` 用与几何阶段同一个估计器回估生成图里的人，把 134 点与请求骨架逐点比对，按 body / feet / face / 左右手分组给出 MPJPE（像素与按躯干长度归一化）、P95、PCK@0.2 与 PCK@0.5，并记录生成图里检出的人数。归一化长度取请求骨架的颈到髋中心距离。

它只回答一个问题：生成器有没有把身体摆到骨架要求的位置。身份、服装、场景是否保持，以及多人图里"别人有没有被改动"，都不在指标内。指标两侧用同一个估计器，估计器自身的偏差会同时进入分子分母，没有与人工标注或第二套检测器交叉验证过。

`evaluate2.py` 与 `scorecard.py`（2026-10-14 落盘）补的就是上面这几条：DWPose 作第二读者的一致率、把"画外关节被排除"算成错的 strict PCK、骨长比漂移、ArcFace 身份余弦、背景与人体区域的 MAE/PSNR/SSIM、以及非主体人物的改动。它写 `evaluation/eval2.json` 与 `stage2.json`，不碰 `evaluate.py` 的产物，字节相同的输出图按 `duplicate` 只计一次。指标选择的完整推导见 `docs/workflow_无GT定量评测调研_20261013.md`，方向层的缺口与优先级见 `docs/workflow_大方向优化调研_20261014.md`。**仍然没有任何指标覆盖**：主观合理度、数据质量、双人交互合理性，以及合同验收写的那条"身高高低/体型胖瘦相对估计准确率"。`eval2.json` 目前只覆盖部分 run，`scorecard.py` 的表因此不完整。

主观读数（合同验收写的"合理比例"与"优胜率"）不在机器指标里，走 `human_eval/`：
出题（只出两臂都有图的题、随机左右）→ 单文件离线盲评 → `tally.py` 给 win_rate 与**按题** bootstrap CI95、
逐轴不合格率、Fleiss κ。用法与坑见 [`human_eval/README.md`](human_eval/README.md)。
合同的参照臂（不给任何姿态条件的纯 Qwen-Edit）尚未实现——它要改 `run.py`（在缓存指纹 `INPUTS_CODE` 里），
排在批量收尾的同一个窗口。

## 低延迟探测（llama.cpp / GGUF）

2026-10-16 新增的离线实验线，用 llama.cpp 在 tiktok runs 上跑 PE 式提示词重写，和 HF 全精度 PE 基线做**同输入三方对比**：HF PE / GGUF PE / GGUF 4B。两个脚本都不碰生产代码与历史 run，只往调用方给的 `--out` 实验目录写产物。

- `lowlatency_probe.py`：起一个 llama.cpp server（`--server` 二进制 + `--model` GGUF 权重 + `--mmproj` 投影权重，`-ngl 99 -c 32768 -fa on`，默认 `--port 18777`，可选 `--mtp` 开 draft-mtp 投机解码：`--spec-type draft-mtp --spec-draft-n-max 2`），对 `workflow/runs/tiktok` 下所有 `tt_*` run 发请求（可用 `--runs` 限定、用 `--tag` 打标签）。两个 arm：
  - `--mode pe`：**质量臂**。直接喂官方 PE-I2I 契约——各 run `prompt/effective_system_prompt.txt` 的 system + `pe_result.json` 的 instruction，回答按官方三段式 JSON 解析校验，再用 `pose_transfer.prompts.compile_prompt(rewritten_prompt, 'source_only')` 编译；解析/角色校验不过则按 pe_worker 的方式重试（默认 `--attempts 1`）。
  - `--mode describe`：**4B arm**。模型未经 PE 训练，不要求它出 PE 契约，只给一段短 system（`DESCRIBE_SYSTEM`）和短 ask（`DESCRIBE_ASK`，要求 90 词内描述目标身体的腿/双臂(viewer-left/right)/头/表情/手持），返回简短动作描述；身份/服装/背景保留条款由脚本用 `PREAMBLE`/`SUFFIX`（镜像 `compile_prompt` 的前缀与后缀契约）补上，拼成 `PREAMBLE + "Apply this pose: " + desc + SUFFIX` 再落盘。
  - 两臂都从各 run 的 `prompt/pe_result.json` 读出 HF 基线 `output_tokens` / `seconds` / `seed`，记进每条 record，保证三方在**完全相同输入**上比。
  - 输出（`--out` 根下）：`records/<run>.json`（逐 case 的 tokens/seconds/tok_per_s、状态、HF 基线）、`raw/<run>_attemptN.txt`（原始回答）、`prompts/<run>.txt`（编译后提示词）、`summary.json`（schema `lowlatency_pe_probe_v1`：均值、相对 HF 的 `speedup_vs_hf`；`describe` 模式还带 `content_check_pass` 的 arm/head/legs/left_right 命中统计）。
- `check_case.py`：冒烟入口。对 tiktok 各 run 调 `lowlatency_probe.load_case()` 校验输入能否正常读出，打印前 3 个 run 的图像数、HF 基线 tokens/seconds/seed、instruction 与 system 字数等诊断；其文件 docstring 还写到「再跑单 case 的 4B arm」，但当前代码体只做了 load_case 冒烟部分（待补充）。

注意：这条线和「评测」「实况」里的全精度 HF PE 基线不是一回事——它是把 PE 式重写搬到 GGUF 上的延迟/质量探针，产物落在调用方指定的实验目录，不会进 `runs/`、也不会改写任何历史 run。

## 测试

```bash
cd /home/shangguanrz/project/pic-edit/workflow
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m unittest discover -s tests -v
cd .. && /home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest workflow/tests -q
```

单元测试覆盖相机投影、关节语义映射、主体选择、手脸连接、缺失点处理、等比对齐、PE 图像角色、生成端引用守卫、缓存失效与失效原因、run 目录搬迁后仍可复验。其中关节映射的用例直接解析安装版 SMPLest-X 的 `joints_name` 与 `flip_pairs`，校验 137→134 每个槽位的语义，而不是把常量表重抄一遍。

`B_3d_dwpose.png` 是估计 3D 关节的 **2D RGB 投影**，不是生成器能直接理解的三维参数或精确硬约束。被遮挡的点是模型推测；有效点标记仅代表坐标有限且位于画面内，不是遮挡置信度。对齐使用同一缩放系数，保持骨架比例，输出保持 A 的画面比例。它与训练侧 `add/tools/SMPLest-X/scripts/render_iper_v1_dwpose_wholebody.py` 产出的 DWPose 条件图只是"同名同点数"，关键点来源（SMPL-X 投影 vs 真实检测器）、配色与用途都不同，不能互换喂给在训练侧条件上训练的 adapter。

PE 提示词中的源图 `<image1>` 保持为生成图 1，骨架 `<image3>` 转换为生成图 2。若回答还依赖 B 的 `<image2>`，会拒绝并重写；`--prompt-file` 提供的文本同样要过这道守卫（只允许 `<image1>`/`<image2>`）。角色和 JSON 校验不能保证 PE 的动作理解完全正确，详细动作仍需看 `generation_prompt.txt`。当前环境缺少 PE 的可选快速线性注意力内核，会使用正确但较慢的 PyTorch 回退实现。

## 实况

`config.json` 指向 sg 已有模型与两个 Python 环境（几何走 `add/envs/smplestx_bw`，PE 与生成走 `~/nihaochen/envs/qwen21`），默认 seed 42、30 步、1024 分辨率预算、true CFG 1、KV cache 关闭。`bash run.sh --help` 查看全部参数，不需要另行安装模型。

测试集共 7 对（sg 上 `experiments/2.qwen-test/qwen-2.1-test/data/testset/pairs/`：hard_01~hard_05、medium_01、medium_02）。覆盖率与逐对指标以 `report.py` 的输出为准（2026-10-14 修掉它三处崩溃后实测：19 个 run 全部索引进来、7/7 对都有 run；`medium_01` 实际嵌在 `runs/test/demo/medium_01/`，按一层遍历的旧版本看不见它），下表是整改后那次全量批处理的快照：768 分辨率、seed 42、30 步、提示词全部来自 PE（无手工干预），每对单次运行，指标含义与局限见"评测"一节。

| run | 对齐 | 主体 | body PCK@0.2 | body MPJPE（归一化） | hand | face | 生成图检出人数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hard_01` | source_person | largest（A 有 3 人） | 1.000 | 0.034 | 0.055 | 0.029 | 3 |
| `hard_02` | source_person | auto | 0.944 | 0.057 | 0.179 | 0.021 | 1 |
| `hard_03` | source_person | auto | 0.944 | 0.076 | 0.065 | 0.031 | 1 |
| `hard_04` | source_person | auto | 1.000 | 0.064 | 0.127 | 0.020 | 1 |
| `hard_05` | source_person | auto | 0.056 | 1.476 | 0.921 | 1.685 | 1 |
| `hard_05_torso` | source_torso | auto | 0.389 | 0.328 | 0.341 | 0.029 | 1 |
| `hard_05_manual` | source_torso | auto（提示词手工） | 0.667 | 0.196 | 0.359 | 0.052 | 1 |
| `medium_01` | source_person | largest（A 有 2 人） | 1.000 | 0.033 | 0.060 | 0.014 | 2 |
| `medium_02` | source_person | auto | 1.000 | 0.040 | 0.070 | 0.024 | 1 |

九次运行收尾自检全部 `VERIFICATION validated`。这些数字分属三份代码状态，按各 run 的 `geometry/stage.json` 所记 `pose_transfer/geometry.py` 哈希区分：`6603a819…` 是首轮批处理（hard_02、hard_03、hard_04、hard_05、medium_02），`6a57ff88…` 是 hard_01 与 medium_01 用 `--subject largest` 的复跑，`5ac98986…` 是两个 hard_05 变体（当前版本，含 `--framing`）。因此现在对它们执行 `--verify` 都会报 `workflow code changed`，产物本身完好；全测试集尚未在同一份代码状态下重跑过。

`hard_05` 的三行是同一对图的两个变量对照。定标：A 框长宽比 0.39、B 框 0.88，`source_person` 的内切缩放被宽度主导（尺度 0.410），请求骨架躯干只有 95.9px，而生成人物躯干 225.6px，错配 2.35 倍；`source_torso` 把尺度提到 0.977，躯干比回到 1.02，指标随之改善 4~7 倍。其余四对 B 框都比 A 框窄，两种定标差不多，所以默认值没有改。提示词：`hard_05_manual` 用的是 `--prompt-file` 引入的一段手写提示词（与 legacy run `testset_hard05_promptv2` 里那份逐字节相同，来源哈希 `17b9c608…`），几何与 PE 阶段照常跑，PE 自己的产物同时保留在 `prompt/pe_compiled_prompt.txt`；在同一份躯干定标下，手写提示词的 body PCK 0.667 高于 PE 编译版的 0.389。即便修好尺度与提示词，`hard_05` 的 body PCK 仍明显低于其余各对，它依然是这套基线最难的一对。

schema v1 的旧 run（`demo`、`testset_hard05*`、`testset_medium02_768`）是整改前的冻结证据：记录格式已变，不再可 `--verify`，`report.py` 会标为 `legacy`；其中 `testset_hard05_promptv2` 与 `testset_hard05_fitgeo` 的提示词是手工写入的，几何与 PE 阶段记录逐字节相同，两者只差提示词文本。整改前的整树备份在 `/data/shangguanrz/pic-edit-archive/workflow_runs_backup_20261012/`（含 `SHA256SUMS`）。

## source_torso 改用实测 A 躯干（2026-10-14）

**改了什么**：`src/pose_edit/geometry.py` 的 `align_to_source_bbox(points, bbox_B, bbox_A, framing,
a_torso_px=None)`。`source_torso` 现在把 B 的实测躯干缩放到 **A 的实测躯干**（neck→髖心），只有拿不到 A 的
关键点时才退回 `TORSO_TO_PERSON_BOX_HEIGHT × A框高`，且**在产物里写明走了哪条路**：`B_to_generation.torso_reference`
取 `measured_A` / `constant_fallback`，`source_person` 取 `not_applicable`。

**为什么必须改**：常数原来的注释写着"在 5 对上把尺度还原到约 4% 以内"，这句不成立。`workflow-multiple/torso_ratio.py`
按 图片哈希+框 去重量过：单人线 8 张源帧的 躯干/框高 从 **0.259 跨到 0.524**，换算成本流水线自己的单位（迁移后躯干
= 0.31 × A框高 ⇒ 人被画成 `0.31 / 比值` 倍）是 **0.59×–1.20×，只有 2/8 落在 ±10% 内**。框高不含躯干信息，所以修法是
换代理量，不是再调一个常数。

**一格真实重跑**（三个数必须在同一张画布上比）：`runs/torso_measured_hard02_20261014`
（`run.py --until geometry --framing source_torso`，输入沿用已验收 hard02 的 A_rgb/B_rgb，不改旧 run，
默认 `--resolution 1024` ⇒ 画布 832×1248）：A 实测躯干 **585.7px**、A 框高 1118.2px ⇒ 比值 **0.524，
正好是上面区间的上界**（这条线自己的两个代理量给出几乎同一个数，见下）：

| 同一格的口径 | scale | 相对实测 |
|---|---|---|
| `source_torso` + 实测 A 躯干（本次改动） | **1.526** | 1.00 |
| `source_torso` + 0.31 常数（旧行为） | 0.903 | 只有应有尺寸的 **59%**，即把人**做小 41%** |
| `source_person`（框内接，本线默认） | 0.928 | 61%，做小 39% |

也就是说在这一帧上 0.31 与框内接给出的数几乎相同（差 2.7%），而 A 的真实躯干要求的是 1.65×——
两个代理量都在猜，常数并不比框内接更"接近真相"，它只是错得差不多。
`VERIFICATION validated`。

**回归说明（不要误读成我改了默认路径）**：`runs/torso_check_person_20261014` 是同输入的默认 framing 重跑，
A/B 检出框与已验收 run 数值一致（记到 0.1px：A `[8.6, 131.7, 664.7, 1278.6]`、B `[260.0, 31.9, 639.3, 1237.0]`，
score 0.931/0.954，persons 1），`scale` 从旧 run 的 0.690 变成 0.928 是**画布不同**造成的：旧 run 是 768 分辨率
（画布 640×928、A 框高 831.5px），新 run 是 1024（A 框高 1118.2px），0.928/0.690 = 1118.2/831.5 = 1.345 正好等于
框高之比；`source_person` 的公式一行没动。要与已验收数字对齐必须同样传 `--resolution 768`。

**A 的关键点从哪来**：新增 `src/pose_edit/dwpose.py`，一条约定只放一处——ONNX 会话走 CPU、
`inference_detector` 这一版只返回 xyxy 且自带 0.3 阈、模型输出 **133** 点转成本仓库渲染/打分用的 **134** 点、
1 号 neck 是派生点所以分数取两肩较弱者、按 IoU 选主体而不是按检测顺序（检测顺序不稳定）。
**双人线也改从这走**：`workflow-multiple/evaluate_dual.py` 的 `detect_people` 变成转调它，删掉本地那份重复实现；
重跑 `hug_pair00_084…skeleton_only.b_canvas` 与 `dance_pair02_120…skeleton_only.b_canvas` 的评测，
新写的 `eval_dual.json` 与原存文件**逐字段相同**（含 per-person 细节，只差 `seconds`），所以这次委托不改数，
只是把两份 133/134 约定合成一份。

**失败要响**：`source_torso` 下若 DWPose 在 A 里找不到与主体框有重叠的人，`smplx_worker.py` 直接抛
RuntimeError，**不**悄悄退回常数——那会在不留记录的情况下改变这条 run 的迁移尺度。

**指纹影响（重要，别误判成缓存坏了）**：`src/pose_edit/geometry.py` 属 SHARED_CODE，出现在两条线所有阶段的
代码清单里，所以**本次改动起，两边已有的阶段缓存都不再等于当前代码**；已有 run 的产物与 `stage.json` 保持原样
（它们诚实记录了当时是谁算的），不 `--force` 就不会重算。`dwpose.py` 已加入单人线 geometry 阶段与双人线评测的
代码清单；当 `pose_framing == source_torso` 时，几何指纹还额外包含两个 DWPose 权重的 `model_signature`（权重内容
变了指纹就变），其余 framing 不为此付哈希成本。

**测试**：`workflow/tests/test_workflow_geometry.py` 从 30 条涨到 33 条（新增实测路径、
`nan/0/负值` 一律视为不可测量而落回兜底、另一 framing 不声称躯干来源；原来那条"由 A 框高与 B 躯干导出"
改名为 `..._falls_back_to_the_box_height_only_without_a_measurement` 并加了来源断言），新增
`workflow/tests/test_dwpose_helper.py` 8 条（IoU、按重叠而非检测顺序选人、133→134、neck 分数规则、
量不到人返回 None）。单人线 33 条与双人线 11 个模块全绿。
