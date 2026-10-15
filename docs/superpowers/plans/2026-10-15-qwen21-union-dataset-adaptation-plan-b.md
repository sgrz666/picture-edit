# 方案 B：Qwen 2.1 Union 数据集适配与小规模训练 Implementation Plan

> **For agentic workers:** 实施时使用 `superpowers:executing-plans`，按任务顺序推进。本文只制定计划；没有实现新模块、下载权重、修改环境或启动训练。用户已选择“官方结构接入＋小规模训练与消融”。

**Goal:** 用现有 iPER、TikTok、Hi4D 数据，验证官方预训练 Union 在接入参考图后能否通过小规模训练改善姿态控制，并测出联合序列注入与 target-only 注入的差异。

**Architecture:** 保留官方 16 个 Qwen 控制块和 129 通道输入，增加参考图输入、统一数据协议与控制分支 LoRA。基座、VAE、Qwen3-VL 和原始 Union 参数冻结；两组训练只改变残差注入范围，其余输入、初始化、采样轨迹与更新次数完全一致。

**Tech Stack:** Python、PyTorch、VideoX-Fun QwenImage21 Control Transformer/Pipeline、当前 Qwen21 Diffusers/Transformers、NumPy/PIL、safetensors、pytest、JSON/JSONL。

## 1. 本方案回答什么问题

方案 A 是“原样官方复现＋六组工作流小样本直接接入”，以不训练为范围。方案 B 是独立的代码与数据集实验，复用 A 的官方加载和参考图接入成果，但不要求 A 的 48 张图全部完成后才能开发。

三个主问题：

1. **直接迁移是否有效？** 同样的源图、骨架参考图、提示词和随机噪声，增加官方 Union 是否比原生 Qwen 更遵守姿态？
2. **小规模训练是否有收益？** 仅训练控制分支 LoRA，能否改善我们的单人、双人数据，同时保留身份和服装？
3. **控制应写到哪里？** 官方向整个 joint sequence 写残差；只向目标图 token 写残差是否减少身份干扰，是否损害双人控制？这是待测假设，不预设 target-only 胜出。

第一轮不加入 DeepGen 控制核、BFS LoRA、Face/Hand Adapter、接触语义新网络或全量 Union 微调。现有人物对应、接触、遮挡信息用于抽样、条件准备和评估；避免一次改变多个结构因素。

历史 Full/Dual Adapter 检查点仅作为附录参考。只有能证明训练数据、基座、预处理和评测集关系的检查点才可加入附录；不能据本实验宣称“官方架构在等预算下全面优于旧 Adapter”。这个结论需要另做旧 Adapter 的同数据训练。

## 2. 2026-10-15 实测依据

### 2.1 数据清单与可用性

以下数量来自本次 `ssh sg` 读取实际清单；本方案所需 RGB、Pose、Depth、关键点/几何路径已按数据集检查存在性，未发现缺失。这里只证明文件存在，没有声称全部图像已解码、所有几何均质量合格。

所有相对数据路径均以 `/home/shangguanrz/project/pic-edit` 为根。

| 数据集 | 使用的清单 | train / val / test 行数 | 分组情况与用途 |
|---|---|---:|---|
| iPER | `datasets/iPER/iper_sampled_6src64tgt/splits/{dev_train,dev_val,test}_pairs.jsonl` | 26496 / 3840 / 7680 | 69 / 10 / 20 个 appearance，三者无 appearance 重叠；单人主训练与评测 |
| TikTok | `datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/{train,val,test}_pairs.jsonl` | 8574 / 1120 / 1156 | 20 / 3 / 3 个 source_video_group，组间无重叠；舞蹈、脸手、小人物 |
| Hi4D | `datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_{train,val,test}.jsonl` | 1312 / 822 / 606 | 6 / 2 / 2 个人物 pair；双人姿态、遮挡与接触 |
| 3DPW | `datasets/3dpw_single_final_v2/{train,validation,test}_pairs.csv` | 14508 / 600 / 600 | 14 / 7 / 9 个 sequence，无序列重叠；本轮只用 test 做域外评估 |

必须保留的区别：

- iPER 的 `train_pairs.jsonl` 有 30336 行，包含 dev_val；训练必须用 dev_train。appearance 的人物编号前缀在 train/val、train/test、val/test 分别有 7、9、5 个交集，因此报告称“appearance 留出”，不称“完全未见身份”。
- TikTok 用 `pairs_strict_v1`，不混入 `pairs/` 的较宽松版本。`sequence_id` 不等于独立视频组，防泄漏按 `source_video_group`。
- Hi4D 训练人物为 pair01/09/10/12/14/16，验证为 pair22/28，测试为 pair00/02。三组唯一目标分别为 774/274/330；多条 source 对同一 target 不能伪装成独立目标样本。
- Hi4D 的 native_interaction_v4_2 清单数量相同，但生产训练入口读取的是 final_assets_v4_2。首个样本 RGB/geometry 是 **940×1280**，目录名不能当成 512 分辨率依据。
- Hi4D NPZ 中关键点生成器为 `Hi4D_mask_bbox_DWPose_official`：条件准备历史上利用了目标人物 mask。生成器不直接读目标 RGB/mask，不等于上游条件完全没有 GT 辅助。报告单列“现成数据资产协议”；与真实 workflow 结果分开解释。
- Hi4D 的 `target_contact=true` 不保证接触顶点数大于零。抽样和接触评估使用实际顶点数，零接触记录单列。

### 2.2 代码和环境

- sg Git HEAD 为 `17991d3`，本地为另一条工作树，双方还有未提交修改。实施前重新记录 Git 状态和文件 SHA256；不能整仓覆盖或用本地文件替换远端更新。
- `src/data/hi4d_qwen21_dataset.py` 的几何转换和人物字段可参考，但不把其整套 DeepGen/V6 条件作为 Union 输入。
- `src/pose_control/qwen21_full/qwen_runtime.py` 已有 RGBA/VAE、缓存和 reference＋target latent 拼接实现；目前缓存主要面向单参考图，不能直接承载本方案的多参考图协议。
- `src/pose_edit/geometry.py::render_poseguide`、`src/evaluation/no_gt_metrics.py`、`src/pose_edit/appearance.py` 可复用。旧 workflow 评估脚本会写 run 目录，新实验只复用计算函数，不直接对旧 runs 执行写入型入口。
- `qwen21_full/joint_training.py` 本地与 sg 哈希不同；新训练器独立新增，不依赖覆盖该文件解决兼容问题。
- 官方代码固定 VideoX-Fun `13bb3c625e8b1e457d7124d92207b49e00e4a6fe`；Union HF revision 固定 `8a4702014d4dabb5f896fcba917e2ee0a961465f`。基座复用 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`，执行时记录实际配置和权重索引哈希。
- 当前 qwen21 是 torch 2.8.0+cu128、diffusers 0.41.0.dev0、transformers 5.17.0。使用独立环境副本；现有环境不装包。NVML 曾报版本不匹配，但 torch 查询 CUDA 可用，尚不能代替真实分配/计算测试。

来源：[官方模型卡](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)、[固定版本控制模型](https://github.com/aigc-apps/VideoX-Fun/blob/13bb3c625e8b1e457d7124d92207b49e00e4a6fe/videox_fun/models/qwenimage21_transformer2d_control.py)、[固定版本训练入口](https://github.com/aigc-apps/VideoX-Fun/blob/13bb3c625e8b1e457d7124d92207b49e00e4a6fe/scripts/qwenimage21_fun/train_control.py)、[已有审计](D:/codeplus/pictureedit/research/qwen21_controlnet_transfer_20261015/REPORT.zh-CN.md)。

## 3. 实验规模、抽样和变量

### 3.1 默认 pilot 数据量

| 数据集 | train | val | test | 作用 |
|---|---:|---:|---:|---|
| iPER | 512 | 24 | 32 | 单人动作和 appearance 保持 |
| TikTok | 512 | 24 | 32 | 舞蹈、脸手与不同视频域 |
| Hi4D | 512 | 24 | 32 | 双人交互、遮挡、人物对应 |
| 3DPW | 0 | 0 | 32 | 完全不参与训练与调参的域外检查 |
| 合计 | **1536** | **72** | **128** | 首轮预算固定，不自动扩大 |

抽样 seed=20261015，排序键为 `SHA256("union_b_v1:" + sample_id)`；不能使用进程间不稳定的 Python `hash()`。

- sample_id 由 dataset、split、source 相对路径、target 相对路径、camera/person 对应生成。CSV 的数字 pair_id 不是全局唯一键。
- iPER 按 appearance 轮转分配配额，并覆盖 standard/motion source；TikTok 先按 source_video_group，再按 sequence 轮转；Hi4D 按 pair、动作族、same/cross-action、实际接触数量分层；3DPW 按 sequence 和已有 pose_bin 轮转。
- 每个 split 中，优先只保留一个 source 对应一个 target；用 canonical target path、图像内容哈希防重复。余量按稳定哈希继续轮转，不复制样本凑数。
- 需要间隔的序列以 target frame 排序后做确定性去近邻：iPER/TikTok 同序列至少 8 帧；Hi4D 在相同 action/camera 内至少 2 帧；3DPW source/target 至少相隔 8 帧。若筛选后不足配额，数据构建报错并列明不足分层，不自动降低阈值或借用其他 split。
- 对 Hi4D 先按 `(pair,target_action,camera_id,target_frame_id)` 去重，再在候选 source 中按 same/cross-action 配额选择。相同目标的多个 source 不进入 bootstrap 的独立样本计数。
- 本次只读去重/间隔 dry-run 后，iPER 的 train/val/test 候选目标为 3626/526/1067，TikTok 为 3380/443/437，Hi4D 为 594/161/240，3DPW test 为 346，均足够覆盖上述配额。该数量尚未包含图像内容哈希去重和质量排除，正式构建仍须核验。
- 训练使用每个数据集 512 个样本；每次 optimizer update 恰好各取 iPER、TikTok、Hi4D 一条，micro-batch=1，gradient accumulation=3。512 次更新正好完成这三个锁定子集各一遍。
- 额外用 train 中每个数据集两条组成 overfit6；这些实验的权重不作为正式训练初始化。验证挑选和测试名单只读，不能看生成结果后换样本。

构建时输出 `lock.json`，包含原清单/选择清单/必需资产的 SHA256、排除原因、组交集、实际配额、代码与模型版本。条件清单和监督清单物理分开。

### 3.2 主实验组

| arm | 后端 | 控制分支 | 残差范围 | 训练 |
|---|---|---|---|---|
| `native` | 当前原生 Qwen | 无 | 无 | 无 |
| `union_frozen` | 官方＋参考图接入 | 原始预训练权重 | joint | 无 |
| `union_lora_joint` | 同上 | 原始权重＋LoRA | joint | 512 updates |
| `union_lora_target` | 同上 | 同一初始权重＋同一初始 LoRA | target-only | 512 updates |

另做三个诊断，不当成额外主训练组：

1. `union_zero`：控制强度 0，在验证集每数据集前 4 条上校准后端差异。
2. `union_target_frozen`：关闭 LoRA 的 target-only，在同一 12 条上观察“只改注入范围”的即时影响；不能把它与训练后的 joint 比较来归因训练收益。
3. `wrong_pose`：同一 12 条中，用同 dataset、同人数、不同 target 的控制图替换正确控制图，保留参考骨架和提示词不变。统计输出偏向正确/错误姿态的程度，检查直接控制通道是否实际生效。置零控制图不等于关闭控制，不能代替 scale=0。

正式推理统一 512×512 画布、40 steps、CFG=1、control_scale=1、seeds=42/43、BF16、关闭 KV cache。源图、骨架参考图、提示词和初始噪声跨 arm 相同。方案 A 的强度探索不与本轮结构消融混合；本轮不做自动 scale 搜索。

原生 `native` 的实现差异用 `union_zero` 单独报告。比较训练收益以同后端的 `union_frozen` 为基线，比较注入范围以两组同训练轨迹的 LoRA 为主。

## 4. 代码边界与目录

代码在仓级共享；实验入口、协议、数据锁与报告属于实验目录。

| 新建位置（相对项目根） | 责任 / 必须暴露的接口 |
|---|---|
| `src/data/qwen21_union_dataset.py` | 解析四套清单；`read_records`、`select_pilot`、`UnionInferenceDataset`、`UnionTrainingDataset`；监督与条件分离 |
| `src/pose_control/qwen21_union/contracts.py` | `ConditionRecord`、`SupervisionRecord`、`CanvasTransform`、`CachedCondition`、`UnionRunConfig` |
| `src/pose_control/qwen21_union/conditions.py` | 画布转换、Pose 渲染、人物对应、129 通道拼接、joint/target mask 构建 |
| `src/pose_control/qwen21_union/pipeline.py` | `ReferenceUnionPipeline`、`ScopedUnionTransformer`；接入参考图与注入范围；独立原生 baseline 后端 |
| `src/pose_control/qwen21_union/weights.py` | 固定版本加载、180-key 审计、130 个 Linear 的精确 LoRA 白名单、训练参数审计 |
| `src/pose_control/qwen21_union/cache.py` | `UnionConditionCache` / `UnionSupervisionCache`；多参考图 VLM/VAE 缓存和内容指纹 |
| `src/pose_control/qwen21_union/training.py` | timestep/sigma、目标流损失、确定性采样轨迹、参数冻结与梯度检查 |
| `src/pose_control/qwen21_union/checkpoint.py` | LoRA、优化器、LR、RNG、采样游标及协议签名的保存/恢复 |
| `src/pose_control/qwen21_union/evaluation.py` | 输出关键点检测、人物匹配、外观/接触代理指标、配对汇总 |
| `tests/test_qwen21_union_{data,layout,lora,training,checkpoint}.py` | CPU 契约测试和标记为 GPU 的集成测试 |
| 实验内 `scripts/run.py`、`configs/protocol.json`、`README.md` | 统一 CLI、四组配置、复现说明；报告生成也由此入口调度 |

实验代码归属：

- 本地：`D:/codeplus/pictureedit/experiments/qwen21_union_dataset_adaptation_v1/`。
- sg：`/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_dataset_adaptation_v1/`。
- 大产物：`/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/`，下设 data、cache、runs、report；每个 runs 子目录带时间戳。公开官方源码原样放独立 source checkout，不修改其已固定提交。

新增包有中文 README；主树的旧 Adapter、训练器、workflow 不为本实验改变默认行为。不存在的路径和函数在本文中均为待实现接口，不代表当前已经可用。

## 5. 数据接口与条件转换

### 5.1 对外契约

最小接口如下；路径在清单落盘前解析为绝对路径，模型接口只接收条件对象。

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetName = Literal["iper", "tiktok", "hi4d", "3dpw"]

@dataclass(frozen=True)
class CanvasTransform:
    original_wh: tuple[int, int]
    resized_wh: tuple[int, int]
    offset_xy: tuple[int, int]
    canvas_wh: tuple[int, int] = (512, 512)

@dataclass(frozen=True)
class ConditionRecord:
    sample_id: str
    dataset: DatasetName
    split: Literal["train", "val", "test"]
    group_id: str
    source_rgb: Path
    reference_pose_images: tuple[Path, ...]
    control_image: Path
    prompt: str
    person_order: tuple[int, ...]
    source_transform: CanvasTransform
    target_transform: CanvasTransform
    provenance: str

@dataclass(frozen=True)
class SupervisionRecord:
    sample_id: str
    target_rgb: Path
    target_masks: tuple[Path, ...]
    requested_keypoints: tuple[Path, ...]
```

`UnionInferenceDataset.__getitem__` 返回 ConditionRecord 派生的输入；没有 target_rgb/target_latent 字段。`UnionTrainingDataset` 在训练阶段按 sample_id 联接监督。评估单独加载 SupervisionRecord。清单构建器可读取目标文件检查尺寸/哈希，但生成条件的图像内容不能来自目标 RGB。

### 5.2 分数据集映射

- iPER：`appearance/source.rgb_1024` → source；target.dwpose_512 → 控制与骨架参考图；target.rgb_1024 → supervision。NPZ 取 `keypoints_134_px`、`scores_134`、`image_width`、`image_height`，已有 134 点，不再次插入 neck。本次样本 RGB/关键点坐标为 1024×1024、Pose 图为 512×512；评估点按前者缩放。
- TikTok：source_rgb、target_dwpose、source/target_dwpose_json 显式映射；target_rgb 只进入监督/评估。JSON 的 `keypoints_norm` 与 `scores` 已为 134 点，x/y 分别乘 image_width/image_height 后应用画布变换；出画点保持失效，不 clamp 到边界。本次样本 RGB/Pose 同为 604×1080。暂不要求加载 v6_conditions、ArcFace 或 Face/Hand 训练缓存。
- Hi4D：source_rgb 为双人参考；target_dwpose_A/B 的 xy_native/scores 渲染两张人物骨架；按原人物索引分别保存，再在黑底合成 combined Pose 供 Union。渲染使用 score≥0.25 且在画布内的点，失效点用 NaN 排除，不能画到 (0,0)。source mask 只用于确定 A/B 在源图中的左右对应和外观裁剪。
- 3DPW：CSV 相对路径接数据集根，target_dwpose_render 作为控制；target_rgb 仅评估。现有目录未发现可直接使用的关键点数值资产；评估阶段用同一冻结 DWPose detector 从 target RGB 提取参考点并写入实验评估缓存，报告标为“GT RGB 估计姿态”，不能冒充控制图的精确关节。无需读取其 train/validation RGB 训练或选参数。

所有 RGB 与控制图分别根据其原画布等比缩放至 512×512，居中补边，保存精确 resized_wh 和 offset_xy。RGB 补白、Pose 补黑；mask 最近邻；关键点坐标使用实际 resize 后 x/y 比例，不用未经取整的理论 scale。若资产本身已经 letterbox，必须读取其坐标元数据或从匹配坐标关键点重绘，禁止二次猜测缩放。每个 provider 用真实样本检查 RGB、Pose、关键点叠图。

源与目标原画布不同也分别记录 transform；不把 source 的裁切偏移套到 target。Hi4D 全部人物使用同一 target transform；合成图不得分别缩放人物。

双人 source 左右顺序由 source mask 中心决定，再同步重排 pose reference、person_order 和评估映射。中心相同按中心 y、原人物编号确定稳定顺序，同时标记 ambiguous；不把这种样本用于“左右提示绑定正确”的强结论。合并骨架不能用彩色 person mask 直接替代官方 Pose。

### 5.3 提示词与参考图

单人参考列表固定 `[source_rgb, pose]`；双人固定 `[source_rgb, pose_person_left_source, pose_person_right_source]`。native 与三个 Union 组完全相同。

单人固定模板：`Change the person in image 1 to the pose shown in image 2. Preserve the person's identity, clothing, and background.`

双人固定模板：`Move the person on the left in image 1 to the pose in image 2, and the person on the right in image 1 to the pose in image 3. Preserve each person's identity and clothing and preserve the background. Keep the two people distinct.`

不调用 PE，不让提示词生成器看目标 RGB。若 source 人物左右难分，仍按协议生成但单列统计，不能在看完结果后人工改 prompt。训练和推理用同一文本及图像编号。

## 6. 官方接入与 target-only 的确切改法

### 6.1 ReferenceUnionPipeline

新增接口：

```python
pipe(
    prompt=prompt,
    reference_images=reference_images,
    control_image=control_image,
    control_context_scale=1.0,
    injection_scope="joint",  # "joint" | "target"
    generator=generator,
    num_inference_steps=40,
    guidance_scale=1.0,
    use_kv_cache=False,
)
```

原官方 `image` 继续保留 Inpaint 含义。该 pilot 只用纯 Pose；不把 source_rgb 偷接成 inpaint_image。参数名差异由 wrapper 归一到所固定的官方函数签名。

源图及骨架参考图同时走 Qwen3-VL 与 VAE，统一 RGBA opaque 输入、原 VAE 归一化和图像 slot 语义。目标由高斯噪声初始化。

若 reference latent token 总数为 Nref、目标为 Ntgt：

```python
# source/reference latent: [B, Nref, 64]
# noisy_target: [B, Ntgt, 64]
hidden_states = torch.cat([reference_latents, noisy_target], dim=1)

# control_latents: [B, Ntgt, 64]
target_control = torch.cat(
    [control_latents, control_latents.new_zeros(B, Ntgt, 65)], dim=-1
)
control_context = torch.cat(
    [target_control.new_zeros(B, Nref, 129), target_control], dim=1
)
```

参考图控制输入补零不表示控制投影输出也为零：保留官方 affine bias 和 before_proj 行为。完整 img_shapes 描述所有参考图和最后的目标；img_mask 每个 VLM image slot 对应 4 个 latent token，不直接拿 slot mask 当 latent mask。

### 6.2 ScopedUnionTransformer

继承固定官方控制 Transformer，仅覆盖 `forward_control`。复用官方算出的 target_token_mask，避免 hook、全局变量或自己猜 joint 序列前缀长度。

```python
class ScopedUnionTransformer(OfficialControlTransformer):
    injection_scope = "joint"

    def forward_control(self, control_joint, x_joint, kwargs):
        hints = super().forward_control(control_joint, x_joint, kwargs)
        if self.injection_scope == "joint":
            return hints
        if self.injection_scope != "target":
            raise ValueError("unknown injection_scope")
        mask = kwargs.get("target_token_mask")
        if mask is None or mask.ndim != 1 or mask.numel() != x_joint.shape[1]:
            raise ValueError("target-only requires official target metadata")
        keep = mask.to(x_joint.dtype)[None, :, None]
        return tuple(hint * keep for hint in hints)
```

实施时 `OfficialControlTransformer` 明确绑定官方 `QwenImage21ControlTransformer2DModel`，并在初始化、checkpoint metadata 中固定 scope。保持 causal_condition=True、单卡、无 KV cache，和官方 mask 契约一致。

只屏蔽**注入残差**，不屏蔽控制链读取文本和参考图，不改变 attention、RoPE、时间调制或 control blocks 的计算过程。因官方残差同时影响文本/参考图，本消融有明确行为意义；但屏蔽残差本身也产生预训练分布变化，所以两组必须分别训练并与冻结组比较。

## 7. LoRA、缓存与训练实现

### 7.1 权重加载与可训练参数

加载顺序固定为：基座与控制结构 → 官方 Union 180 个 tensor → 校验 → 安装 LoRA → 恢复本实验 LoRA（如有）。strict=False 只允许缺少来自另一文件的预期基座键；控制键缺失、额外控制键、shape 不符均报错。不得在加载后执行 zero reset。

LoRA 只覆盖下面的精确模块，不用一个可能误匹配基座的 `to_q` 后缀规则：

```python
suffixes = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out", "after_proj",
)
targets = ["control_img_in", "control_blocks.0.before_proj"]
targets += [f"control_blocks.{i}.{name}" for i in range(16) for name in suffixes]
assert len(targets) == 130
```

rank=16、alpha=16、dropout=0；A 用 Kaiming 初始化，B 为零；LoRA 参数和 AdamW 状态使用 FP32，冻结权重 BF16，计算在 BF16 autocast 下完成。输出投影的原始 weight/bias 保持官方值，不训练 bias/norm。PEFT 使用精确全路径正则或完整白名单，并核对实际 matched modules。

根据官方 safetensors shape 计算，新增参数为 **23,267,344**。运行时数量必须一致，所有 requires_grad 参数均属于这 130 个模块的 LoRA A/B。两组正式训练读取同一个初始 LoRA 文件，避免仅靠相同 seed 推测初始化一致。

### 7.2 两类缓存

`UnionConditionCache` 包含 prompt embedding/mask、全部参考图 latent、参考图 slot mask、control latent、img_shapes、人物映射和 transform；`UnionSupervisionCache` 只存训练 target latent 与有效内容区 mask。

缓存键包含：模型权重/配置指纹、上游代码 SHA、图像内容 SHA、有序参考图列表、prompt、预处理版本、画布/变换、dtype、VAE 编码策略。不能直接重用现有单参考图缓存，也不能只按文件名或路径作为缓存键。

VAE 编码策略固定为 `latent_dist.mode()`，按 `(latent - latents_mean) / latents_std` 归一化，训练 target、reference 和 control 全部一致，写入 lock。本次已核实官方推理 `_encode_vae_image` 使用 argmax/mode；官方训练入口采用 posterior sample，本实验为固定缓存显式改用 mode，不声称逐行复刻官方训练分布。缓存命中与否不能消耗 denoising RNG。缓存加载后 clone 成普通 tensor，避免 inference_mode tensor 参与反向保存时报错。

预计算阶段冻结 Qwen3-VL/VAE，完成后释放编码模型，再加载 Transformer 训练。训练只读取缓存；验证推理进程没有 target latent 缓存入口。估算缓存体积时先测每数据集各两条，报告按选定样本计算的上限，不先全量缓存整个数据集。

### 7.3 Flow loss 与 scheduler

采用官方 flow-matching 数据流，明确区分 sampler 的 timestep 和 sigma；不能直接照搬旧训练器里 `sigma=torch.rand()` 后把 sigma 当全部时间语义的写法。

- 从固定基座 scheduler 配置构建训练 schedule，1000 个时间位置，按目标 token 数计算 dynamic shift；不把参考图 token 计入目标分辨率 shift。
- 均匀抽取 schedule index，读取对应 sigma 和 timestep；Transformer 接收 `timestep / 1000`。
- 模型输入 reference latent 始终干净；只对 target latent 加噪。reference/text 使用官方 t=0 路径。
- 以下损失固定为内容区均值 flow MSE，不使用官方训练脚本的 diff>50 截断；这是显式实验选择。首轮不附加 ID、LPIPS、接触或脸手训练损失。

```python
sigma = sigma.to(z_target.dtype).view(-1, 1, 1)
z_noisy = (1 - sigma) * z_target + sigma * noise
flow_target = noise - z_target
inputs = cached_condition.transformer_inputs(z_noisy, timestep / 1000)
prediction = transformer(**inputs, return_dict=False)[0][:, -z_target.shape[1]:]
square = (prediction.float() - flow_target.float()).square()
weights = valid_target_tokens.float().unsqueeze(-1)
loss = (square * weights).sum() / (weights.sum() * square.shape[-1]).clamp_min(1)
```

valid_target_tokens 是 target letterbox 的内容区经 area 下采样到 latent 的覆盖率，范围 [0,1]；不是 GT 人物分割，只用于避免补边面积支配 loss。`CachedCondition.transformer_inputs` 返回 hidden_states、prompt embeddings/masks、img_shapes、img_mask、control_context 和固定 scope 下的模型输入，不包含监督图路径。

冻结基座参数，但训练 forward 不包 `torch.no_grad()`，不 detach hints：反向必须穿过主干才能更新早期控制 LoRA。官方控制链和主干均启用 non-reentrant gradient checkpointing。

### 7.4 固定训练配置

| 配置 | 值 |
|---|---|
| train seed / sampler seed | 3407 / 20261015 |
| 总更新数 | 每 arm 512，两个正式训练 arm |
| micro-batch / accumulation | 1 / 3，一次更新各一个数据集 |
| optimizer | AdamW，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01 |
| LR schedule | 32 updates 线性 warmup，之后 constant |
| gradient clipping | 全部 LoRA 参数 global norm=1 |
| 数据增强 / condition dropout | 均关闭 |
| checkpoint | 0、128、256、512；异常时只写新的 interrupted checkpoint |
| 生成验证 | step 0、256、512；seed 42、40 steps、72 条 |
| 最终 checkpoint | 固定 step 512，不根据 test 或临时印象选择 best |

采样顺序在训练前写入 lock；noise 与 timestep 使用两个独立 Generator，其 seed 从 train_seed、update、microstep、sample_id 和 stream 名经 SHA256 派生为 63 位整数。两个 arm 使用同一派生规则，验证生成使用另一个 Generator，不能改变训练轨迹。

正式初始化与 overfit6 分离。先跑 10 microsteps 显存/耗时探针，再跑 overfit6 200 updates；overfit 使用 micro-batch=1、accumulation=1、六条稳定循环，其余 optimizer 配置相同。评价固定 schedule indices=250/500/750，固定 noise seed=43，各样本三种噪声位置共 18 条；平均 loss 相对 step0 下降至少 20% 才继续正式两组训练。这是实现可学习性门禁，不是生成质量承诺。未达到则检查输入、梯度与时间语义，不自动延长训练掩盖问题。

训练进程仅驻留冻结 base+Union 和 LoRA，VAE/VLM 不驻留 GPU。BF16 模型本体约 20.29 GiB，不能当作峰值显存预测。第一次 10 microsteps 记录 max_memory_allocated/reserved 和耗时，推算余下时间；超过总显存 90% 或 OOM 则保存探针报告并停止，不静默减分辨率、换量化或改变模型。推理使用官方 group offload，训练不直接复用推理卸载 hook。

### 7.5 恢复契约

checkpoint 包含 LoRA tensors、optimizer、LR scheduler、Python/NumPy/torch CPU/CUDA RNG、optimizer_step、microstep、下一样本位置、实际 loss/gradient 日志游标、锁文件 SHA、base/Union 指纹、scope、rank/alpha、130 模块列表和三数据集顺序。

在 optimizer update 边界保存；中途收到停止请求时完成当前累积组后保存。恢复时检查全部签名，任何数据/模型/scope/预处理不一致均拒绝；不能只恢复权重却把 optimizer_step 标成原步数。输出检查点采用新文件名，旧文件保留。

## 8. 测试、指标与停止条件

### 8.1 必须实现的契约测试

| 测试 | 输入 / 断言 |
|---|---|
| `test_manifest_split_overlap` | 小型假清单中制造相同 appearance/video_group/pair 跨 split；构建失败；iPER 的人物前缀交集仅记录，不冒充 appearance 冲突 |
| `test_inference_cannot_read_target` | 将监督 RGB/mask 路径设为不可打开；正常条件加载和推理输入构造仍成功；对目标图的意外 open 立即失败 |
| `test_letterbox_coordinates` | 940×1280 转 512×512，resize 为 376×512、left=68；点/图/mask 使用同一实际仿射；双人重排映射可逆 |
| `test_multiple_reference_slots` | 至少三张参考图、文本夹在图像 slot 间、带 padding；token 数与 img_shapes 对齐，target mask 只覆盖最后目标 |
| `test_control_129_layout` | 前 Nref 输入全零，目标 64+1+64 顺序正确；无 inpaint 时后 65 通道全零；错误 token 数报错 |
| `test_scope_masks_only_residual` | joint 返回原 hints；target 保留目标位置数值，其他位置精确为零；控制链输入与权重未变化 |
| `test_lora_inventory` | 实际匹配 130 个 Linear、23,267,344 参数；无 transformer_blocks 主干匹配；B=0 初始化与加载前同输出 |
| `test_flow_schedule_and_gradients` | 同一 index 的 timestep/sigma 来自同一 schedule；加噪公式一致；LoRA-B 首步有非零有限梯度，第二步 A 可获得梯度；冻结参数无梯度且 checksum 不变 |
| `test_resume_equivalence` | 小模型连续训练 4 updates，对比 2 updates 保存重载再 2 updates；采样/噪声轨迹相同，状态在规定容差内一致 |
| `test_official_compatibility` | no-reference、joint、LoRA disabled 的单步预测，与原版同后端一致；scope 错配、控制 key 缺失、缓存版本不同均报错 |

浮点比较：CPU FP32 小模型使用 rtol=1e-5、atol=1e-6；BF16 同后端使用 rtol=1e-2、atol=1e-2，并记录 max/mean error。最终图像不能以像素哈希相同作为跨实现等价的唯一条件。真实 GPU 的恢复测试首先要求相同采样与噪声，再报告参数差异，不跨硬件承诺 bitwise 相同。

### 8.2 指标和报告

- 姿态：复用严格 PCK@0.1/@0.2、躯干归一化关键点距离，分 body/hands/face；请求点的 confidence≥0.25 为分母依据，生成点漏检或请求点出画记失败，另报仅画内指标和 detector coverage。不能删掉难点后称为严格 PCK。
- 双人：先按目标几何做人物 assignment，再计算逐人指标；记录人物合并/丢失/多生、hip 相对位置、遮挡分组和身份交换疑似。匹配不能用 GT 外观先选“最像的结果”来提高姿态分数。
- 身份/外观：ArcFace 有效检测覆盖率与 cosine、DINO 人物裁剪相似度、衣着/背景保持。没有有效人脸不填 0 当成真实相似度；身份指标与覆盖率同时展示。DINO 是外观代理，不能作为身份真值。
- 重建：PSNR/SSIM/LPIPS 仅用于同一数据样本、相同坐标、具有合理 GT 的图像；背景或视角发生变化时按数据集说明局限，不把全图像素误差作为主要排名。
- 接触：按现有 geometry 的 contact/occlusion 标签分层，人工审查交互位置和四肢归属。二维距离只称“接触代理指标”，不声称恢复了真实三维接触。
- 统计：每样本先平均两个 seed，再对 target 去重、按 appearance/video_group/pair/sequence 聚合，给每数据集宏平均、逐组结果和最差案例。Hi4D test 只有两个 pair，报告逐 pair 数值，不作人口级显著性结论。
- 性能：每 arm 保存峰值显存、平均步耗时、总时间及 offload 配置；推理和训练分别报告。

输出必须包含四组同样本面板、原参考图、实际模型控制图、目标 GT（只在评估面板）、逐样本 JSONL、分组 CSV、检测失败清单和执行协议。测试集一次锁定评估，不用它调 scale、prompt、checkpoint。

### 8.3 下一步决策

正式报告按事实给出结论，不强行选赢家：

- frozen 已有稳定收益且训练改善不明显：推荐沿用方案 A 的直接接入。
- joint LoRA 增益更好：保留官方注入范围，后续扩大数据。
- target-only LoRA 姿态与身份同时改善：保留该分支，下一轮独立复测，不直接替换生产。
- 姿态提高但身份交换/人物合并增加：报告 tradeoff，下一轮再设计人物绑定或接触模块；本轮不临时加网络。
- 出现加载、梯度、泄漏或坐标契约错误：停在工程问题，不将其失败结果作为架构优劣证据。

## 9. 按任务实施与验收

每项先增加上表对应测试，确认当前缺少能力的失败，再实现接口并运行该项测试。测试只针对有实际风险的模型/数据契约；README 和报告排版不额外制造测试。

### Task 1 — 数据协议与锁定清单

- [ ] 实现 contracts.py 和 qwen21_union_dataset.py 的四个 provider，区分条件与监督。
- [ ] 实现分组/去重/配额/时间间隔策略，生成 train1536、val72、test128 与 overfit6。
- [ ] 保存元清单 hashes、缺失/排除原因、split 检查和 pose provenance。
- [ ] 执行 `pytest tests/test_qwen21_union_data.py -q`；预期全部契约通过，真实数据构建数量严格符合表格。

### Task 2 — 控制图、坐标和缓存

- [ ] 实现 conditions.py、cache.py，完成多参考图 VLM/VAE 编码与 train-only 监督缓存。
- [ ] 每数据集导出两组叠图人工检查；所有 overlay 写新实验 output，不改源资产。
- [ ] 执行 `pytest tests/test_qwen21_union_layout.py -q`；验证 129 维、slot 展开、mask、Hi4D 人物重排。
- [ ] 先对 smoke 样本预计算，重跑应命中同一 key，改变参考图顺序/prompt/transform 应失效。

### Task 3 — 官方权重与参考图 Pipeline

- [ ] 固定上游提交和环境，加载 base＋Union；输出键、shape、控制层和数值有限性报告。
- [ ] 实现 ReferenceUnionPipeline；no-reference 路径与原版对照，reference 路径跑一条单人和一条双人。
- [ ] 实现 ScopedUnionTransformer，只在 hints 输出处乘官方 target mask。
- [ ] 真 GPU 单步比较通过后，生成 12 条诊断样本，核对控制关闭和错误控制响应。

### Task 4 — 控制 LoRA 与 Flow 训练

- [ ] 精确安装 130 个 LoRA，保存公共初始化；实现 schedule、loss 和冻结审计。
- [ ] 执行 `pytest tests/test_qwen21_union_lora.py tests/test_qwen21_union_training.py -q -m 'not gpu'`。
- [ ] 单卡执行 10 microsteps 探针，再执行 overfit6 200 updates；门禁失败则定位，不开始全量 pilot。
- [ ] 两个 arm 分别从公共 step0 初始化训练 512 updates；记录同一 sample/noise/schedule 轨迹签名。

### Task 5 — checkpoint 与恢复

- [ ] 实现 checkpoint.py 的完整状态和签名校验，测试 scope、模型指纹、数据版本错配拒绝。
- [ ] 执行 `pytest tests/test_qwen21_union_checkpoint.py -q -m 'not gpu'`。
- [ ] 在正式训练前完成小模型连续/恢复对照和真实模型 2+2 更新短测；通过后再运行 Task 4 的正式两组。

执行依赖顺序为 **1→2→3→Task 4 的 CPU/梯度部分→5→Task 4 正式训练→6**。任务编号按模块归属，不表示可以跳过恢复测试。

### Task 6 — 评估、消融与报告

- [ ] 72 条 val 做阶段诊断；四组均达到工程契约后，锁定 step512 运行 test128×4 arms×2 seeds，最多 1024 张正式测试输出。
- [ ] wrong_pose、union_zero、target_frozen 只用于 val 诊断，正式 test 不追加选择性生成。
- [ ] 分别汇总 iPER/TikTok/Hi4D/3DPW；输出中文报告、面板与失败清单。
- [ ] README 写清已完成阶段、真实样本数、训练步数、数据来源、复现命令与结论；未运行项明确标记未运行。

## 10. 实现完成后的使用方式

以下是**拟实现 CLI**，现在不要执行。`--workspace` 为大产物根，命令必须拒绝向数据集源目录写入。`--name` 唯一；已有运行目录拒绝覆盖。audit/select/cache 都有独立输出签名；train/infer 显式指定产物目录，不自动查找“最新模型”。

```bash
cd /home/shangguanrz/project/pic-edit
UNION_PY=/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/env/bin/python
UNION_CLI='experiments/2.qwen-test/qwen21_union_dataset_adaptation_v1/scripts/run.py'
UNION_WORK=/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1
UNION_SESSION=$(date +%Y%m%dT%H%M%S)

# 1. 审计和选数；默认协议即本文的 1536/72/128。
"$UNION_PY" "$UNION_CLI" audit --workspace "$UNION_WORK"
"$UNION_PY" "$UNION_CLI" select --workspace "$UNION_WORK" --protocol pilot_b_v1

# 2. 缓存与工程门禁。supervision 只预计算 train。
"$UNION_PY" "$UNION_CLI" cache --workspace "$UNION_WORK" --conditions train,val,test --supervision train
"$UNION_PY" "$UNION_CLI" smoke --workspace "$UNION_WORK" --name "smoke_$UNION_SESSION"
"$UNION_PY" "$UNION_CLI" overfit --workspace "$UNION_WORK" --name "overfit6_$UNION_SESSION" --updates 200

# 3. 两组同初始化训练，按机器空闲情况串行运行。
"$UNION_PY" "$UNION_CLI" train --workspace "$UNION_WORK" --arm union_lora_joint --name "joint_$UNION_SESSION" --updates 512
"$UNION_PY" "$UNION_CLI" train --workspace "$UNION_WORK" --arm union_lora_target --name "target_$UNION_SESSION" --updates 512

# 4. 显式绑定两组 checkpoint，生成四组同条件测试。
"$UNION_PY" "$UNION_CLI" infer --workspace "$UNION_WORK" --split test --name "test128_$UNION_SESSION" \
  --arms native,union_frozen,union_lora_joint,union_lora_target --seeds 42,43 \
  --joint-checkpoint "$UNION_WORK/runs/joint_$UNION_SESSION/checkpoint_000512" \
  --target-checkpoint "$UNION_WORK/runs/target_$UNION_SESSION/checkpoint_000512"
"$UNION_PY" "$UNION_CLI" evaluate --workspace "$UNION_WORK" --run "test128_$UNION_SESSION"
"$UNION_PY" "$UNION_CLI" report --workspace "$UNION_WORK" --run "test128_$UNION_SESSION"
```

实际 runs 目录精确使用调用者传入的 name；上述命令将时间戳放入 name，checkpoint 参数是实际绝对路径。CLI 不自动改名，不解析“最新模型”，重名立即报错。

smoke 命令包含：CPU 契约汇总、CUDA/BF16 小张量计算、全权重加载、single/dual 两步生成、一次训练 backward、freeze audit、保存重载、连续/恢复短测和 10 microsteps 资源估计。未通过 smoke/overfit 的 workspace 拒绝 train 正式 arm。

最终交付：可复现代码与配置、独立环境清单、数据 lock、公共初始化、两个 LoRA checkpoint、训练轨迹、配对结果与中文报告。原始媒体、旧 run、现有 qwen21 环境保持原位置和内容。方案本身不授权后台长程训练；后续实施由用户发起。
