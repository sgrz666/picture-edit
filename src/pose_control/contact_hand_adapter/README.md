# Contact Hand Adapter V2

独立的 Qwen-Image 2.1 手部控制器，用于单手手势、同人双手和双人接触。默认冻结 Qwen DiT、Qwen3-VL 和 VAE。单独的 checkpoint 保存 Adapter 权重、架构、冻结基座身份和可选的训练状态。

## 架构

每手最多 108 个 token，含整手 1 个、手指与手掌 6 个、关节 21 个、MANO 表面 64 个和源手外观 16 个。Joint/Mesh GCN 编码拓扑。3 层单手和 4 层跨手 Transformer 融合手级、部位级和稀疏表面级接触，CNN/FPN 将目标手部几何对齐到空间位置。缺测项通过有效位和三态标签表示。

每次 DiT forward 从静态几何记忆初始化动态手部 token。指定层的目标 ROI 图像特征更新手部状态，静态几何再对它作锚定。Hand K/V 分支复用带原生 RoPE 的 Qwen Query，单独归一化，再通过共享投影和逐层低秩增量写回原生 Attention 残差。默认宽 768，8 层版约 87.20M 参数，16 层版约 88.45M。原生处理器对象保留，集成支持非重入 activation checkpointing。 [OminiControl](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_OminiControl_Minimal_and_Universal_Control_for_Diffusion_Transformer_ICCV_2025_paper.html) · [V1 条件与评测背景](../../../docs/superpowers/specs/2026-11-27-contact-hand-adapter-design.md)

## 训练输入

每行 JSONL 对应一个真实帧对。资产项使用 `{"path":"相对于清单的路径","sha256":"64 位小写哈希"}`。必填键如下。

| 字段 | 内容 |
| --- | --- |
| `id`, `group`, `split` | 唯一帧对 ID，预先合并同人物、视频与近重复的组 ID，以及 `train` / `val` / `test`。程序检查 group 和 RGB SHA 跨 split 冲突。 |
| `mode` | `global` 或 `local`，可在同一清单混用。 |
| `source_refs`, `prompt` | 有序参考图与编辑提示。独立双源先放 A、再放 B。附加几何参考使用显式顺序。 |
| `target`, `condition` | 监督 RGB 和独立目标几何包。目标 RGB 可以是 CanvasSpec 原尺寸，或已放到声明画布。 |
| `context` | Local 样本的全图上下文，需在声明的目标画布上，例如同机位源帧，或同条件的粗生成图。训练和推理都将裁片放在已有参考图之后。 |
| `slots` | 可选的 Local 选手槽整数列表，顺序为 A-L、A-R、B-L、B-R。相连手部自动加入联合 ROI。 |
| `regions`, `rgb_labels` | 可选、使用最终训练画布的监督包。资产独立于推理条件；省略 regions 时从手和腕部几何构造 FM 区域。 |

条件包由 `torch.save(condition_record(condition), path)` 写出。先按 CanvasSpec 将坐标映射到目标画布，再构造 `HandCondition(canvas, person_ids, joints, visible)`。最小条件含四个固定手槽、每槽 21×3 的 x/y/置信度，以及布尔可见位。未观测点留为不可见。可选的 `HandMesh` 含 778 顶点、固定模板邻接与 64 个锚点及其投影，锚点按每个语义部位连续四个存放。可靠共相机的手对可加表面边。部位接触的 `-1/0/1` 分别表示未知、分离、接触。可选外观从对应 Source 裁取，给有效手槽记录参考源图 SHA。两人共用第一个源参考时设 `shared_appearance_source=True`。字段、尺寸、单位和缺失规则见 [conditions.py](conditions.py)。

regions 包是包含 `canvas_sha`、逐手 `hands[4,H,W]` 和可选 `wrists`、`contacts` 的字典，`contacts` 只放独立标注的正接触。rgb_labels 包包含 `canvas_sha`，其余键为 `joint`、`part`、`contact`、`anatomy` 中实际有监督的项，每项是 `(target, valid)` 两个张量。RGB evaluator 接收冻结 VAE 解码后的 `[1,3,H,W]` 浮点 RGB，返回各项预测。Part 输出带类别轴，Contact/Anatomy 输出 logits。程序在每个有效头首次使用时检查到 clean latent 的有限非零梯度。首轮可只用区域 FM。

## 使用与验证

从仓库根目录运行。在 sg 使用项目 qwen21 环境及本机已有 Qwen 基座。先运行 **CPU 数据预检**，逐条校验 SHA、划分、图片和条件画布。

```bash
python scripts/train/train_contact_hand_adapter.py --manifest /path/to/pairs.jsonl --check-data
```

用有训练授权的真实帧对做首步训练，下列命令创建独立权重。正式运行按清单锁定权重路径、输入 SHA 和训练阶段，产物放在 /data/shangguanrz/pic-edit-experiments/ 的独立实验目录。

```bash
python scripts/train/train_contact_hand_adapter.py \
  --manifest /path/to/pairs.jsonl --model /data/shangguanrz/pic-edit-models/Qwen-Image-2.1 \
  --updates 1 --accumulate 1 --out /path/to/run/hand-step1.pt
```

训练退出前输出 checkpoint SHA256。中间权重由 `--save-every` 保存。相同数据和配置用 `--resume` 严格恢复优化器与采样随机数，使用新的输出文件名。换数据阶段用 `--init-adapter` 继承权重并重新初始化优化器。训练消融有 `--surface-tokens 32`、`--injection-layers 16`、`--no-image-feedback`、`--contact-level hand|part|surface`，以及单独打开 `--hand-lora-rank 16` 的原生 Attention 输出 LoRA。可用 `--local-fraction 0.7` 固定局部训练配额。

同一份权重支持 `global`、`local` 和 `global-local`。Local 需要声明画布上的上下文，Global-Local 用第一遍生成的全图作上下文。Local 在独立的 512×512 画布上执行 Qwen 去噪，从未拉伸的联合手部裁片恢复结果，只在掩码内贴回。多组相隔较远时可用 `--slot` 指定要优化的手和其接触邻居。以下示例执行两个生成阶段。

```bash
python scripts/run/run_contact_hand_adapter.py \
  --model /data/shangguanrz/pic-edit-models/Qwen-Image-2.1 --adapter /path/to/hand.pt \
  --condition /path/to/geometry.pt --reference /path/to/source.png \
  --prompt 'Make the two people shake hands.' --mode global-local --out /path/to/result.png
```

运行 `python -m pytest -q tests/test_contact_hand_adapter.py` 验证工程行为。该套件使用小型真实 Qwen DiT/VAE 检查原生 Attention、零强度旁路、跨层状态、梯度重算、ROI 合成和独立权重重载。真实图效果需用同条件多 seed 和人工盲评验收，姿态评分统一使用 [SDPose 项目规范](../../../docs/SDPOSE_POSE_EVAL_SPEC.md)。
