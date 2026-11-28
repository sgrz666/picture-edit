# Contact Hand Adapter V3

独立的 Qwen-Image 2.1 手部控制器，用于单手手势、同人双手和双人接触。默认冻结 Qwen DiT、Qwen3-VL 和 VAE。单独的 checkpoint 保存 Adapter 权重、架构、冻结基座身份和可选的训练状态。

## 架构

V3 继承 V2.1 的 19 维 Joint GCN、MANO 编码与多尺度接触关系。单手最多 109 个结构和外观 token，另外可为有证据的相邻手各加一个有向接触 token。手槽采用人物、解剖侧和部位编码，可靠的被遮挡节点仍参与 Anatomy Encoder。缺测时继续使用 2D 条件。

CNN/FPN 使用 21 张关节热图、五指骨链、16 类分区，以及可用的深度和法线。关节、手指和表面锚点在四个尺度各采样四个可学习偏移。3 层单手和 4 层跨手 Transformer 融合手级、部位级和稀疏表面接触。可选的拟合、法线、跨视角、穿透四项质量与源置信度相乘，再用 8 mm 带宽的高斯距离核形成软接触分。未核实的表面关系保留 Unknown，不生成硬接触监督。 [Deformable DETR](https://arxiv.org/abs/2010.04159)

每次 DiT forward 从静态几何记忆初始化。V3 在重叠 ROI 中保留四手槽的独立空间特征。当前 Qwen 图像特征结合实例 coverage、深度、遮挡和接触证据选手槽，再更新动态状态。跨手深度相对顺序只在共同相机已核实时进入路由。Hand K/V 分支继续复用原生 RoPE Query。Hand、Joint、Surface、Contact、Appearance 五类门控的早期、中期和后期初值可随训练调整。宽度 768、8 个注入层配置下，compressed、low_rank、direct 的可训练参数分别为 94.69M、90.57M、107.08M。默认保留 compressed。 [OminiControl](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_OminiControl_Minimal_and_Universal_Control_for_Diffusion_Transformer_ICCV_2025_paper.html) · [V1 条件与评测背景](../../../docs/superpowers/specs/2026-11-27-contact-hand-adapter-design.md)

源外观使用 256×256 裁片。全局分支提供与像素位置无关的掩码颜色统计，并可加冻结 DINOv3 的 CLS 和掩码池化 patch 特征。局部分支按源图解剖分区池化多尺度 CNN 特征，映射到目标图同名关节或表面锚点。缺少部位证据时保留全局外观 token。离线缓存可用 `FrozenDinoAppearance(local_model_dir, expected_sha=...)` 和 `attach_dino_appearance(condition, extractor, ordered_source_hashes)` 生成，绑定模型、裁片、掩码和源图 SHA。是否有助于保持外观，需要在目标条件固定、源手姿态变化的对照中测量。 [DINOv3](https://github.com/facebookresearch/dinov3)

## 训练输入

每行 JSONL 对应一个真实帧对。资产项使用 `{"path":"相对于清单的路径","sha256":"64 位小写哈希"}`。必填键如下。

| 字段 | 内容 |
| --- | --- |
| `id`, `group`, `split` | 唯一帧对 ID，预先合并同人物、视频与近重复的组 ID，以及 `train` / `val` / `test`。程序检查 group 和 RGB SHA 跨 split 冲突。 |
| `mode` | `global` 或 `local`，可在同一清单混用。 |
| `source_refs`, `prompt` | 有序参考图与编辑提示。独立双源先放 A、再放 B。附加几何参考使用显式顺序。 |
| `target`, `condition` | 监督 RGB 和独立目标几何包。目标 RGB 可以是 CanvasSpec 原尺寸，或已放到声明画布。 |
| `context` | Local 样本的全图上下文，需在声明的目标画布上，例如同机位源帧，或同条件的粗生成图。训练和推理都将裁片放在已有参考图之后。 |
| `context_kind`, `current_support` | Local 可标 `source`、`coarse` 或 `synthetic`。可选当前手包为 `{"canvas_sha": "...", "support": Tensor[4,H,W]}`。合成样本需要当前手包和同一有序源图，程序只对该上下文的手部局部作可复现扰动。 |
| `person_swap_safe`, `prompt_on_person_swap` | 独立双源换位需声明 prompt 可交换或提供换位后的 prompt。通过资格门时同步交换整人、姿态、源引用、接触关系和区域监督，手的解剖侧保持原值。存在未声明结构的 RGB 标签时跳过换位。 |
| `slots` | 可选的 Local 选手槽整数列表，顺序为 A-L、A-R、B-L、B-R。相连手部自动加入联合 ROI。 |
| `regions`, `rgb_labels` | 可选、使用最终训练画布的监督包。资产独立于推理条件；省略 regions 时从手和腕部几何构造 FM 区域。 |

条件包由 `torch.save(condition_record(condition), path)` 写出。2D 按 CanvasSpec 映射到目标画布。最小条件继续使用 `HandCondition(canvas, person_ids, joints, visible)`，四个槽分别对应 A-L、A-R、B-L、B-R。3D 输入使用相机坐标系和米，`joint_3d_valid`、`joint_2d_valid`、实际可见 `visible`、已知遮挡 `joint_occluded`、`hand_present` 互相独立。6D 旋转和 MANO 包有单独有效位，指尖旋转设为缺失。部位接触的 `-1/0/1` 对应未知、分离、接触。可选外观记录对应源图 SHA。字段、尺寸、单位和缺失规则见 [conditions.py](conditions.py)。

regions 包是包含 `canvas_sha`、逐手 `hands[4,H,W]` 和可选 `wrists`、`contacts` 的字典，`contacts` 只放独立标注的正接触。可重复传 `--objective-weight NAME=VALUE` 扩展区域 FM。姿态、解剖、接触、遮挡、穿模使用有标签的输出头。姿态可使用现有合格 2D/3D 目标标注，其他头依赖独立 `rgb_labels` 的 `(target, valid)`。启用这些头需提供独立校验的可微 TorchScript evaluator。Source Appearance 按可见同名部位比较源手和解码结果，Perceptual 通过冻结本地 DINO 比较有效手区与目标。首个实际使用的监督会记录有效覆盖和对 clean latent、Adapter 的梯度。拟合出的未知表面关系不生成接触标签。

## 使用与验证

从仓库根目录运行。在 sg 使用项目 qwen21 环境及本机已有 Qwen 基座。先运行 **CPU 数据预检**，逐条校验 SHA、划分、图片和条件画布。

### 从 InterHand2.6M 准备帧对

转换器使用官方相机和 42 点关节规范，把毫米世界坐标投影到目标 Canvas，重排为每手腕点开头、手指按根到尖的 21 点。同一人物、同一视角和序列只在隔离后的 split 内配对。提供 MANO 模板后，可生成通过拟合质量门的 Mesh、深度、法线、分区和交互遮挡。Mesh 推导的表面关系只作为软条件。 `--images` 可读解压目录，默认也支持官方 5 fps 分卷。 [InterHand2.6M 项目页](https://mks0601.github.io/InterHand2.6M/) · [官方数据读取代码](https://github.com/facebookresearch/InterHand2.6M/blob/main/data/InterHand2.6M/dataset.py)

```bash
python scripts/data/prepare_interhand_contact_hand_adapter.py \
  --dataset-root /data/shangguanrz/datasets/hand/InterHand2.6M \
  --split train --max-pairs 16 \
  --mano-model-dir /data/shangguanrz/models/single/wilor/mano_data \
  --out /path/to/new-interhand-pilot
```

输出 `manifest.jsonl`、条件包和 `audit.json`。正式训练用 `train` 帧对。分卷未齐时，`--allow-partial-archive` 仅用于明确指定 split/capture 的读取验证。通过预检和 16 对过拟合后再扩大单手与同人双手训练，跨人握手由独立双人数据补足。

2026-11-27 在 sg 对 `test/Capture1/cam400375/ROM09_Interaction_Fingers_Touching` 做 CPU 冒烟。3 对帧通过清单预检和小型 Adapter 的条件编码，6 只手通过 MANO 拟合门，产出 3 个遮挡关节、17 条单向记录的候选表面边。候选接触均为 Unknown。此次归档核验可读取前 8 个连续分卷，后续 `partai` 在校验时未下载完整。

### 数据预检、训练和出图

```bash
python scripts/train/train_contact_hand_adapter.py --manifest /path/to/pairs.jsonl --check-data
```

用有训练授权的真实帧对做首步训练，下列命令创建独立权重。正式运行按清单锁定权重路径、输入 SHA 和训练阶段，产物放在 /data/shangguanrz/pic-edit-experiments/ 的独立实验目录。

```bash
python scripts/train/train_contact_hand_adapter.py \
  --manifest /path/to/pairs.jsonl --model /data/shangguanrz/pic-edit-models/Qwen-Image-2.1 \
  --updates 1 --accumulate 1 --out /path/to/run/hand-step1.pt
```

训练退出前输出 checkpoint SHA256。中间权重由 `--save-every` 保存。相同数据和配置用 `--resume` 恢复优化器与采样随机数，换数据阶段用 `--init-adapter` 继承同版本权重。整人换位默认概率 0.5，可用 `--person-swap-probability 0` 关闭。`--local-fraction 0.7` 固定局部训练配额，`--local-context-weights 1 1 1` 在三类上下文之间等权采样。已有注入层、表面 token、接触级别和可选 LoRA 消融仍可从 `--help` 查到。

消融用 `--spatial-mode instance|average`、`--gate-mode typed|uniform`、`--residual-mode compressed|low_rank|direct`、`--appearance-mode part|global|off` 和 `--no-appearance-dino`。独立 Hand Prior 用 `--hand-prior FILE --hand-prior-sha SHA256` 在训练和推理中加载。该权重保持冻结，Q/K/V/O 先验只向目标手部 query 写入直接修正，权重随去噪推进增加。可比较原生 Qwen、仅 Adapter、仅 Prior 和联用。推理传 `--strength 0` 保留 Prior 单独工作，传 `--hand-prior-strength 0` 保留 Adapter 单独工作。现有官方 Prior 只完成过 1 次更新及重载冒烟，质量需要独立留出集验证。 [Hand Prior 运行记录](../../../experiments/qwen21-hand-prior-lora/README.md)

同一份权重支持 `global`、`local` 和 `global-local`。Global-Local 用第一遍生成的全图作上下文。Local 默认遍历所有手部连通组，合并膨胀后相交的编辑区，在各自的 512×512 画布去噪并只按编辑掩码贴回。传 `--current-support old.pt` 可将上下文画布上已知的旧手掩码一起纳入。该文件为 `torch.save` 存的 `[4,H,W]` 张量，顺序为 A-L、A-R、B-L、B-R。省略时按目标手部范围编辑。`--local-dilation` 默认 8 个全图画布像素，也可用 `--slot` 限定待编辑组。

训练和推理使用同一套参考图尺寸规则。小局部画布仍按指定尺寸生成，参考图单独满足 Qwen 视觉编码器的最小面积，保证 VLM 与 VAE 使用一致的参考 token 网格。

```bash
python scripts/run/run_contact_hand_adapter.py \
  --model /data/shangguanrz/pic-edit-models/Qwen-Image-2.1 --adapter /path/to/hand.pt \
  --condition /path/to/geometry.pt --reference /path/to/source.png \
  --prompt 'Make the two people shake hands.' --mode global-local --out /path/to/result.png
```

运行 `python -m pytest -q tests/test_contact_hand_v3.py tests/test_contact_hand_appearance_v3.py tests/test_contact_hand_prior_v3.py tests/test_contact_hand_training_v3.py` 验证 V3。真实图使用同条件多 seed 和人工盲评，报告单手、同人双手和跨人双手三组。姿态评分统一使用 [SDPose 项目规范](../../../docs/SDPOSE_POSE_EVAL_SPEC.md)。

### V3 首步记录

2026-11-28 在 sg 将 8 对 InterHand2.6M 训练帧转换成 256 像素源手条件，8 只手通过 MANO 拟合。使用官方冻结 Qwen、冻结的 I2I Hand Prior（SHA256 `0c5c2553e72a4c6ed378e052d559a24c4f3ac22251a27e243511d945bdbcea54`）和随机初始化的 V3 做 1 次更新。Region FM 与同名部位外观联合损失为 1.887792，外观原始损失为 0.066362，外观对 Adapter 的梯度范数为 0.030869。运行路径为 `/data/shangguanrz/tmp/contact-hand-v3-stage0-20261128T053653873Z`，含输入清单、源码快照、训练日志和 `hand-v3-step1.pt`。这些数字用于验证训练链路。

重载该 checkpoint（SHA256 `df6e7fba044be6beb6d4f2492f75b586ff4b9481b730487b74f6a044354797f5`）后，用独立验证主体的同人双手帧完成 BF16 CPU Global-Local。seed 42，全图及每个 256px 局部各 2 步。输出 `val-global-local-seed42.png` 为 352×512 RGB，SHA256 `f7a3e79c5dafa158ca34a8263d71d607140de71f197ecf36681b7b592f133d1d`。验证脚本和日志在 `validation-source/` 和 `val-global-local-cpu-retry.log`。

另外用冻结 DINOv2 ViT-S/14 在真实手部裁片上验证 Perceptual 对 RGB 的反向梯度。实测使用的权重、配置和预处理聚合 SHA256 为 `06a1d57f57bf693d630ab229627bca29f13faf16a0c04bd7ca8bb819415c4173`。目前 sg 尚未核实预训练 DINOv3 权重。解剖、接触等输出头仍需要有可靠标签的可微评估器和独立验证。
