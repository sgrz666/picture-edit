# PE-I2I × Body/Face Adapter 八组实验：代码实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在现有原始尺寸 Body＋Face 联合训练代码上，实现逐样本 PE 提示词的八组对照、1024 次新增更新、严格隔离的推理评估，以及可复制到本地的中文结果网页。

**Architecture:** 复用现有联合损失、冻结 BFS 加载器、原始尺寸数据加载器和评估器；新增一个小型 prompt_study 包管理实验协议、数据锁、提示词、采样轨迹和结果汇总。现有 pilot 脚本通过显式 study 参数进入新协议，旧实验的 256 步限制和数据校验继续有效。PE 离线重写与 Qwen 训练/推理分别运行；每条最终文本同时用于 Qwen 文本编码和 Adapter 的文本条件。

**Tech Stack:** Python、PyTorch、Diffusers QwenImage21Pipeline、Transformers PE-I2I、PIL/NumPy、pytest、JSON/JSONL、离线 HTML。

---

## 0. 计划状态与实施入口

本文件是代码设计和实施任务，**尚未实现下述新接口，也没有启动本次训练**。本文依据 2026-10-10 本地工作区实际代码及 sg 上已有数据制定。工作区已有 Hand、native、恢复日志等改动，实施前记录这些改动，保留其行为；不能以本计划为理由回退已有修改。

本地代码目录：[D:/codeplus/pictureedit](D:/codeplus/pictureedit)。远端代码目录：/home/shangguanrz/project/pic-edit。远端 Python：/home/shangguanrz/nihaochen/envs/qwen21/bin/python。

代码开发按任务 1→12 执行；实验运行按第 5 节执行。实现每个任务后运行该任务列出的测试。不要在实现阶段直接运行完整八组训练。

前期证据：[几何与提示词研究记录](D:/codeplus/pictureedit/docs/experiments/2026-10-09-geometry-prompt-coordination-study.md)。PE 读取几何的四条研究探针可复用模型加载和解码流程，但它们不是这次实验的正式提示词缓存。

## 1. 已核对的代码事实与需要改变的位置

| 现有文件 / 函数 | 当前行为 | 本次改动 |
|---|---|---|
| [qwen_runtime.py](D:/codeplus/pictureedit/src/pose_control/qwen21_full/qwen_runtime.py:117) / Qwen21IPERCache.pair | 已支持 prompts 序列，source 缓存键包含最终文本 | 直接复用，不重新设计另一套文本缓存 |
| [joint_training.py](D:/codeplus/pictureedit/src/pose_control/qwen21_full/joint_training.py:208) / run_joint_loss | 调用 cache.pair 时没有 prompts | 增加可选 prompt 参数，并记录实际文本哈希 |
| 同文件 / validate_locked_split | 只允许 common256 或 branch256，要求旧 lock.json | 旧函数保留；study 使用新的 512/24/24/6 数据锁校验 |
| [train_qwen21_joint_pilot.py](D:/codeplus/pictureedit/scripts/train/train_qwen21_joint_pilot.py:36) / arguments、main、save | steps≤256；单次遍历；sample_order 是 range(step)；固定 PROMPT；还预缓存 val 的 GT latent | 增加显式 study 模式、全程 1024 更新轨迹、stop-after-step、逐样本 prompt；study 模式只缓存训练 GT |
| [eval_qwen21_joint_pilot.py](D:/codeplus/pictureedit/scripts/eval/eval_qwen21_joint_pilot.py:201) / generation_arguments | 参考 RGB 只有 A，默认高斯初始化，没有 target/latents 参数 | 保持这个边界；增加逐样本文本和数值 study_step 元数据 |
| 同文件 / main | config、cache.source、pipe 三处都写死 PROMPT；stage 仅接受旧字符串；当前ROI先转补边坐标而输出再unpad | 三处统一使用同一条编译后文本；用 study_step 标识 0…1024；明确裁图采用原图ROI坐标 |
| 同文件 / validate_base_provenance | 旧 mid/final 要求训练和推理 LoRA 一致；hotload 才允许切换 | 正式 step0 先生成带当前 arm 签名的新 checkpoint；不把原 J checkpoint 冒充已在 Head 上适配的模型 |
| [data_runtime.py](D:/codeplus/pictureedit/src/pose_control/qwen21_full/data_runtime.py:32) / pair_metadata | TikTok 的 source/target appearance 都取 sequence_id；跨人物审查行不能直接按原格式送入 | 对显式 TikTok source_appearance/target_appearance 增加支持，sequence_id 绑定 B 的几何 |
| 同文件 / FaceConditionStore | 需要 geometry_valid 与 metadata.geometry_valid；只取 A 的 ArcFace/DINO，B 只取 landmarks/jaw/expression/box | 新测试数据必须准备兼容此格式的缓存，不能误用另一种轻量 Face 缓存 |
| [letterbox.py](D:/codeplus/pictureedit/src/data/letterbox.py:96) / NativePad | 原 RGB 不缩放，仅补齐到可打包尺寸；604×1080 → 608×1088 | 固定 resolution=native、align_mode=native；保存输出时 unpad |
| [tiktok_dataset.py](D:/codeplus/pictureedit/src/data/tiktok_dataset.py:865) / _native_item | normal、Body25、part、global26、mask；低分/出画 Body25 置零；已有 part_512 会最近邻对齐至原画布 | 本轮保持这一控制协议；报告明确几何资产对齐可能重采样，不宣称所有几何图都未经重采样 |
| [frozen_bfs_lora.py](D:/codeplus/pictureedit/src/pose_control/qwen21_full/frozen_bfs_lora.py:204) / load_frozen_bfs | 验证权重、加载一个未融合 LoRA、冻结 Transformer，保留输入梯度 | 复用；每 arm 独立进程、新 pipeline；禁止在 Transformer forward 外包 no_grad |
| 旧报告 / summarize、build_report | 固定旧 variants/stages；部分文本英文 | 新报告按 arm、整数 step、split、seed 动态组织；旧 HTML 作为历史参考 |

需要特别区分：Body 的 26 维 global 并不包含所有面部/手部细节；Face 分支另读目标面部几何。当前 Body25 图只有少量小点是热图可视化，不能据此认定条件只有几维。报告必须同时展示实际 Body25 热图和 PE 使用的 DWPose 骨架。

当前global包含B的beta/体型，normal也来自B人体网格。本轮保持这项输入条件，不能把“保留A衣服/身份”的文本理解为已经消除了B体型约束；跨人物结果需要单独标注这种潜在冲突。

## 2. 锁定的实验条件

### 2.1 八组与初始化

| arm | 冻结底座 | prompt protocol |
|---|---|---|
| qwen_p0 | Qwen-Image-2.1 | 固定短文本 |
| qwen_pg | Qwen-Image-2.1 | PE 看 A＋Depth＋Normal＋过滤后 DWPose |
| qwen_pb | Qwen-Image-2.1 | PE 看 A＋B |
| qwen_pbg | Qwen-Image-2.1 | PE 看 A＋B＋Depth＋Normal＋过滤后 DWPose |
| head_p0 | Qwen-Image-2.1＋冻结 BFS Head V1.1，strength=1 | 固定短文本 |
| head_pg | 同上 | PG |
| head_pb | 同上 | PB |
| head_pbg | 同上 | PBG |

所有组从同一个 J256 Adapter 权重初始化，**各自新建 optimizer，从本轮 step0 开始训练**。Head 从本轮第一步已经加载，不先在纯 Qwen 训练再切换。该设计回答“共同的已有 Adapter 换成不同底座/提示词后继续适配会怎样”；不回答“从随机初始化训练谁最好”。

共同初始化：

~~~
/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/J/checkpoint_step_000256.pt
SHA256 = 9e0f048c585937ab74be792eb6d9730b0621eccda16e1744a6837b0eb9d24651
~~~

Qwen：/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1，revision=790c92633540aa0cb11d9abf19eb46d861714758。PE：/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I。Head：/home/shangguanrz/models/bfs/bfs_head_v1.1_qwen_2.1.safetensors。

BFS Head 官方的两张 RGB 换头流程与本项目的 A RGB＋Adapter 条件不同。报告要写明这一差异。本轮不加入 BFS Body；后续在相同协议下扩展 Body 底座，继续服务于此前确定的底座对照主线。

### 2.2 训练、分辨率与损失

- 每 arm **新增 1024 次 optimizer update**，batch=1，无梯度累积，正式共 8192 次。J256 的历史更新单独显示。
- Body＋Face 联合训练；Hand、Interaction、shared Detail payload 关闭；legacy detail gate=1，因为 Face 也依赖这个门。
- Body 保持 normal、Body25、part、global26、mask。Adapter use_depth=False。本次 PE 能看 depth/DWPose，不代表给 Adapter 新增了这两个通道。
- 原始 RGB 宽高，NativePad 补边。禁止统一 resize 到 512/1024；输出裁回 B 所定义的原始目标画布。
- AdamW：betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01；参数与 optimizer FP32，forward autocast BF16。
- Body heads/encoders/core LR=2.5e-5/1.25e-5/2.5e-6；Face=5e-5/2.5e-5/5e-6；warmup20，然后恒定；grad clip=1。
- 沿用 body flow＋0.5 face ROI flow＋0.10 identity＋0.05 perceptual；aux 请求每4更新一次，仅低噪声 σ≤0.35 且 Face/监督有效时生效。
- 推理30步、true_cfg_scale=1、KV cache=True；geometry=1、face=1、hand=0；不额外添加文本 dropout、调强度或换损失。

提示词不要写“Adapter 帮你调整几何细节”。模型得到的是文本 token 和已注入的控制条件，这句话不构成控制协议。写清编辑动作与保留要求即可；其是否和几何一致要通过样本审查及实验验证。

### 2.3 GT 使用边界

| 路径 | B/GT 可以做什么 | 禁止的输入 |
|---|---|---|
| PE 的 PG | 使用 B 对应的几何可视化；不读 B RGB、不读人工 B 动作描述 | target_rgb、action_description、source_preserve_description 等旁路进入 PG 请求 |
| PE 的 PB/PBG | B RGB 是目标动作、头向、表情、手势的信息来源 | 从 B 迁移服装、身份、场景到 A |
| Qwen 训练 | A 是唯一 RGB 参考；GT latent 经加噪形成 x_t，并提供 flow target；GT 像素用于限定的辅助监督 | B 作为 encode_prompt 的 image、reference_latent 或身份向量 |
| Qwen 测试/验证/监控 | A＋编译后的文本＋目标数值几何；从新高斯噪声采样 | cache.target、cache.pair、run_joint_loss、GT latent 初始化、backward、optimizer.step |
| 离线统计/可视化 | B 尺寸、目标几何、B 参考图用于审查，成对同人物集可算 GT 图像误差 | 用测试结果改 prompt 模板、改训练步数、选 checkpoint |

因此“GT 只给 PE”指 RGB 参考条件边界。标准训练监督仍使用加噪 GT；测试完全不走该训练路径。PB/PBG 是额外 B 信息条件，不可把收益全部解释为“长提示词更好”。

### 2.4 数据、轨迹与结果数量

原 pilot data 根目录：
/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/data。

| 文件 | 数量 / 用途 | 锁定 |
|---|---|---|
| train512_pairs.jsonl | 原有512条训练样本，字节与顺序原样复制 | SHA256 854133776d2067aa392442553f3003ef9e6d45107ccc9cffe5a4af8600fb929f |
| val24_pairs.jsonl | 原有24条验证样本，字节与顺序原样复制 | SHA256 cece883b96dbbc9270fd77d4bb1e973470f52f941af85c2a0746947f5d91e10a |
| test24_pairs.jsonl | 新独立3个原视频组，每组8条 | 从严格 test 候选确定性选取 |
| diagnostic6_pairs.jsonl | 原 native_value_audit 的 tiktok_00…05 | 跨人物诊断；B 不是 A 的重定向真值 |

test 候选：/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/test_pairs.jsonl，已核对1156条。三个组：seq_00010_00028、seq_00241_00244、seq_00289_00307，与 train512 和旧 Face3k train1000 的视频组均无交集。新 lock 时再次校验，不只相信本文。

选择规则：先检查文件和必要几何可加载，再按 SHA256("pei2i_20261009:"+pair_id) 排序；每组8条、目标帧唯一、同 sequence 的目标帧距离至少8。Face 无效可以入测试并标记；不能用输出质量、能否检测到生成脸作为筛选标准。条件不足选满24条就报错，不自动放宽间距。

epoch0=原512顺序；epoch1=random.Random(1709) 产生的512排列，写入 lock；共1024条采样轨迹。第 u 次更新（u 从0开始）的 noise_seed=42+256+u。各 arm 轨迹完全一致。

监控训练索引：[0,64,128,192,256,320,384,448]。val 小监控：[0,4,7,8,13,14,16,19,22]。

| 评估 split | step | seed | 每 arm 图数 |
|---|---|---|---|
| train_monitor 8条 | 0,128,256,384,512,640,768,896,1024 | 42 | 72 |
| val24 | 0,256,512,768,1024 | 42,43 | 240 |
| val_monitor 9条 | 128,384,640,896 | 42 | 36 |
| diagnostic6 | 0,512,1024 | 42 | 18 |
| test24，全部训练结束后统一运行 | 0,1024 | 42,43 | 96 |
| 合计 | | | 462 |

八组正式计划共3696张输出。梯度 preflight、自检的小型推理/恢复试验单独计数。无需在同一 step 重复生成 val 小监控与完整 val 重叠项；用结果索引复用相同任务。

## 3. 文件责任与产物协议

代码路径均相对本地/远端代码根目录。新增文件集中在下列位置；训练与评估主逻辑继续留在现有脚本。

下表及任务中 prompt_study/... 均为 src/pose_control/qwen21_full/prompt_study/... 的缩写。

| 文件 | 责任 |
|---|---|
| src/pose_control/qwen21_full/prompt_study/__init__.py | 包标记，导入时不加载 GPU 模型 |
| prompt_study/protocol.py | 常量、arm、路径、文件哈希、原子 JSON、配置校验 |
| prompt_study/pairs.py | 数据锁、独立 test 选择、诊断行规范化、资产清单 |
| prompt_study/geometry.py | 过滤后的 DWPose 渲染与固定目标手部 ROI |
| prompt_study/prompts.py | PE 请求白名单、原始回答解析、文本编译、PromptBank |
| prompt_study/sampling.py | 固定1024轨迹、更新种子、warmup、轨迹校验 |
| prompt_study/results.py | 任务枚举、记录对齐、有效计数、成对差值、导出校验 |
| scripts/data/prepare_qwen21_prompt_study.py | CPU 数据锁和几何预览 CLI |
| scripts/data/prepare_qwen21_prompt_study_faces.py | 兼容的 A 身份/B 几何 Face 缓存；保持固定样本数量 |
| scripts/precompute_qwen21_pei2i_prompts.py | 独立 PE GPU 进程、重写/恢复/失败记录 |
| scripts/precompute_qwen21_prompt_study_cache.py | 逐文本 source 缓存；仅 train target latent 缓存 |
| scripts/run/run_qwen21_pei2i_study.py | 单阶段调度、依赖校验、每128步训练/推理分进程 |
| scripts/reports/build_qwen21_pei2i_study_report.py | 中文静态网页、缩略图与原图链接 |
| tests/test_qwen21_prompt_study_*.py | 数据/提示词/缓存/轨迹/GT隔离/调度/报告测试 |
| docs/experiments/2026-10-10-pei2i-eight-arm-runbook.md | 实现后填写真实 CLI 与运行结果清单 |

修改现有文件仅限：joint_training.py、data_runtime.py、train_qwen21_joint_pilot.py、eval_qwen21_joint_pilot.py 及其对应回归测试。qwen_runtime.py 现有 prompts 接口可以满足本轮要求，优先不改。

远端新结果根目录：
/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009。

本地结果根目录：
D:/codeplus/pictureedit/research/pei2i_body_face_8arm_20261009。

~~~
pei2i_body_face_8arm_20261009/
  data/
    lock.json
    train512_pairs.jsonl  val24_pairs.jsonl
    test24_pairs.jsonl    diagnostic6_pairs.jsonl
    assets.jsonl         rejected_test_candidates.jsonl
    face_cache/          face_supervision_native.json
    geometry/<pair_id>/dwpose_original.png dwpose_filtered.png
  prompts/
    instruction_v1.json
    qa8.json
    pg/<split>.jsonl  pb/<split>.jsonl  pbg/<split>.jsonl
    p0/<split>.jsonl
    attempts/<split>/<pair_id>/<protocol>/attempt_0.json
    timing.json
  cache/
    p0/  pg/  pb/  pbg/
  preflight/
    <arm>/gradient_only.json
    resume_check/ result.json
    selfcheck.json
  training/<arm>/
    run_manifest.json  lora_loading.json  metrics.jsonl
    checkpoint_step_000000.pt … checkpoint_step_001024.pt
    checkpoint_last.pt resume_reconciliation.json summary.json
  evaluation/<arm>/step_000000/<split>/seed_42/
    config.json selected_pairs.jsonl per_sample.jsonl scores.json
    images/ crops/ conditioning/
  report/
    index.html summary.json errors.json thumbnails/
  export/
    files.json export_status.json
~~~

同一 protocol 的四份 split 提示词共566条；PG/PB/PBG 共1698次成功重写。P0也保存逐样本条目，但文本相同、无需调用 PE。Qwen/Head 共享同一份 prompt 记录及缓存。

## 4. 逐项代码实施任务

### 任务 1：建立协议常量、配置锁与安全的 CPU 导入

**新增：** prompt_study/__init__.py、protocol.py、sampling.py。
**测试：** tests/test_qwen21_prompt_study_protocol.py。

- [ ] 先增加以下测试，首次运行应因包或函数不存在而失败：

~~~python
import pytest
from src.pose_control.qwen21_full.prompt_study.protocol import ARMS, validate_arm
from src.pose_control.qwen21_full.prompt_study.sampling import make_order, noise_seed

def test_eight_arms_and_two_complete_epochs():
    assert ARMS == tuple(f"{base}_{p}" for base in ("qwen", "head")
                         for p in ("p0", "pg", "pb", "pbg"))
    order = make_order()
    assert order[:512] == list(range(512))
    assert sorted(order[512:]) == list(range(512))
    assert make_order() == order
    assert (noise_seed(0), noise_seed(1023)) == (298, 1321)

def test_invalid_arm_rejected_before_loading_models():
    with pytest.raises(ValueError, match="arm"):
        validate_arm("head_punknown")
~~~

- [ ] 在 protocol.py 添加以下协议核心。后续字段统一使用这里的常量；路径写入 lock 后不能悄悄改：

~~~python
from pathlib import Path
import hashlib
import json
import os

SCHEMA = "qwen21_pei2i_body_face_study_v1"
PROTOCOLS = ("p0", "pg", "pb", "pbg")
P0_PROMPT = "Change the person pose to match the target pose. Preserve identity, clothing, and background."
ARMS = tuple(f"{base}_{p}" for base in ("qwen", "head") for p in PROTOCOLS)
PROMPT_ROLES = {
    "p0": (),
    "pg": ("source_A", "target_depth", "target_normal", "target_dwpose"),
    "pb": ("source_A", "target_B"),
    "pbg": ("source_A", "target_B", "target_depth", "target_normal", "target_dwpose"),
}
EVAL_STEPS = tuple(range(0, 1025, 128))
FULL_VAL_STEPS = (0, 256, 512, 768, 1024)
TRAIN_MONITOR = (0, 64, 128, 192, 256, 320, 384, 448)
VAL_MONITOR = (0, 4, 7, 8, 13, 14, 16, 19, 22)
INIT_SHA256 = "9e0f048c585937ab74be792eb6d9730b0621eccda16e1744a6837b0eb9d24651"
TRAIN_SHA256 = "854133776d2067aa392442553f3003ef9e6d45107ccc9cffe5a4af8600fb929f"
VAL_SHA256 = "cece883b96dbbc9270fd77d4bb1e973470f52f941af85c2a0746947f5d91e10a"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n", encoding="utf-8")
    temp.replace(path)

def validate_arm(arm):
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}")
    return arm.split("_", 1)
~~~

- [ ] 在 sampling.py 写入确定性轨迹，禁止依赖已加载 LoRA 造成的 RNG 消耗：

~~~python
import random

def make_order():
    second = list(range(512))
    random.Random(1709).shuffle(second)
    return list(range(512)) + second

def noise_seed(update_index):
    if not 0 <= update_index < 1024:
        raise ValueError("update_index must be in 0..1023")
    return 42 + 256 + update_index

def warmup_scale(update_index):
    return min((update_index + 1) / 20, 1.0)

def validate_trajectory(saved, metrics, order):
    step = int(saved["step"])
    if saved["sample_order"] != order[:step] or len(metrics) != step:
        raise ValueError("Resume sample_order/metrics differs")
    for u, row in enumerate(metrics):
        if (row["step"], row["pair_index"], row["noise_seed"]) != (
            u + 1, order[u], noise_seed(u)
        ):
            raise ValueError(f"Resume trajectory differs at update {u}")
~~~

- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_protocol.py -q。预期全部通过，且执行过程中没有模型加载、CUDA 分配、SSH 或训练进程。
- [ ] 后续 protocol 的 config 校验检查 exact8 arms、native/native、max_updates=1024、eval_every=128、Body/Face 开、其余关。不同实验配置必须使用新 run 目录。

### 任务 2：数据锁、新 test24 与跨人物 TikTok 元数据

**新增：** prompt_study/pairs.py、scripts/data/prepare_qwen21_prompt_study.py。
**修改：** data_runtime.py 的 pair_metadata。
**测试：** tests/test_qwen21_prompt_study_pairs.py。

- [ ] 写测试：打乱候选顺序后 test24 选择不变；每组8条且帧距≥8；train/val/旧 Face 训练组与 test 组相交时报错；样本不足时报错；跨人物 metadata 不能把 A 改成 B。

~~~python
from src.pose_control.qwen21_full.prompt_study.pairs import select_test24
from src.pose_control.qwen21_full.data_runtime import pair_metadata

def test_test24_is_order_independent_and_spaced():
    groups = ("seq_00010_00028", "seq_00241_00244", "seq_00289_00307")
    rows = [dict(pair_id=f"g{g}_{i}", source_video_group=group,
                 sequence_id=f"{g:05d}", target_stem=f"{i*16:04d}")
            for g, group in enumerate(groups) for i in range(12)]
    picked = select_test24(rows)
    assert picked == select_test24(list(reversed(rows)))
    assert len(picked) == 24
    for group in groups:
        frames = sorted(int(r["target_stem"]) for r in picked
                        if r["source_video_group"] == group)
        assert len(frames) == 8
        assert all(b-a >= 8 for a, b in zip(frames, frames[1:]))

def test_cross_tiktok_metadata_preserves_two_appearances():
    row = dict(pair_id="tiktok_00", sequence_id="00232",
               source_appearance="00130", target_appearance="00232",
               source_stem="0032", target_stem="0223",
               source_rgb="/a.png", target_rgb="/b.png")
    meta = pair_metadata(row)
    assert (meta["source_appearance"], meta["target_appearance"]) == ("00130", "00232")
    assert meta["cross_person"] is True
~~~

- [ ] 在 pairs.py 实现这个选择器；传入的 eligible_rows 必须已通过资产检查，原样记录所有排除原因：

~~~python
from collections import defaultdict
from .protocol import text_sha256

TEST_GROUPS = ("seq_00010_00028", "seq_00241_00244", "seq_00289_00307")

def select_test24(eligible_rows):
    selected = []
    for group in TEST_GROUPS:
        candidates = sorted(
            (r for r in eligible_rows if r["source_video_group"] == group),
            key=lambda r: (text_sha256("pei2i_20261009:" + r["pair_id"]), r["pair_id"]),
        )
        chosen, frames, targets = [], defaultdict(list), set()
        for row in candidates:
            seq, frame = str(row["sequence_id"]), int(row["target_stem"])
            key = (seq, str(row["target_stem"]))
            if key in targets or any(abs(frame-old) < 8 for old in frames[seq]):
                continue
            chosen.append(dict(row))
            frames[seq].append(frame)
            targets.add(key)
            if len(chosen) == 8:
                break
        if len(chosen) != 8:
            raise ValueError(f"Cannot lock eight spaced targets for {group}")
        selected.extend(chosen)
    return selected
~~~

- [ ] 修改 TikTok 分支的 appearance 赋值及返回，保留其余旧字段：

~~~python
appearance = str(row["sequence_id"])
source_app = str(row.get("source_appearance", appearance))
target_app = str(row.get("target_appearance", appearance))
if target_app != appearance:
    raise ValueError("TikTok sequence_id must select target geometry")
source_stem = str(row["source_stem"] if "source_stem" in row else row["source"]["stem"])
target_stem = str(row["target_stem"] if "target_stem" in row else row["target"]["stem"])
return {
    "pair_id": str(row.get("pair_id", f"{source_app}_{source_stem}__{target_app}_{target_stem}")),
    "source_appearance": source_app, "target_appearance": target_app,
    "source_stem": source_stem, "target_stem": target_stem,
    "source_rgb": str(row["source_rgb"]), "target_rgb": str(row["target_rgb"]),
    "cross_person": source_app != target_app, "dataset_type": "tiktok_v3",
}
~~~

上面是 pair_metadata 函数内的 TikTok 分支完整替换块，return 位于函数内；iPER 分支保留原逻辑。验证代码片段时需要将此块作为函数体解析。

- [ ] diagnostic6 只提取白名单字段，不把旧人工 action_description、嵌套 source_original_row 带入 PE 正式请求。规范化时设置 sequence_id=target_appearance；从实际 target RGB 的文件 stem/appearance 构建 target_dwpose_json、part_512、v6_conditions、smplestx 的路径，存在后才写锁。
- [ ] 数据锁 CLI 接收 --source-pilot、--face-source-run、--test-candidates、--diagnostics、--output。train/val 用 shutil.copyfile 保持字节哈希；另保存原 lock 的哈希。写入 epoch_order=make_order()、独立视频组审计、每条资产 hash/尺寸、初始化 SHA、模型 revision/config/system_prompt/Head SHA。
- [ ] 新 test 只依据必要资产有效性筛选。检查 RGB、mask、normal、depth、DWPose JSON/PNG、part、v6_conditions 中目标 stem、global[26]、Body25[25,3] 有限且可见点≥8，以及 Face 几何可构建；**不要求 Face identity_valid=True 才入 test**。逐条记录缺资产和无效几何，单独公开每组候选/排除数量。
- [ ] 若新 test 的 part/v6_conditions 尚未生成，数据锁阶段列出缺失 sequence/stem 后停止，不偷偷换组。可先用已存在的 scripts/build_tiktok_v6_conditions.py 按具体 sequence＋原test候选manifest准备资产，再重做锁定。它的路径参数是 --raw_root/--assets_root/--smplx_path（下划线），与其他脚本的 --raw-root 不同；不传 --overwrite，不修改原训练/验证资产。先运行 --help 核对环境与模型文件：

~~~bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/build_tiktok_v6_conditions.py --help
# sequence取缺资产审计里的实际序列；例如确实缺00012时：
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/build_tiktok_v6_conditions.py \
  --sequence 00012 \
  --manifest /data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/test_pairs.jsonl
~~~
- [ ] lock.json 写入全部文件清单后，计算文件自身哈希供训练签名引用；不要把 lock 自身哈希写回 lock 内形成循环。资产清单按 path 去重，复用同文件哈希。
- [ ] 再次执行 --verify 时只校验现有锁及资产，不重选、不覆盖。首建若目录已有不一致文件即失败。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_pairs.py tests/test_qwen21_native_resolution.py -q。预期新旧元数据、原尺寸行为均通过。

### 任务 3：DWPose 清洗渲染与固定手部 ROI

**新增：** prompt_study/geometry.py。
**修改：** 数据准备 CLI，调用渲染器。
**测试：** tests/test_qwen21_prompt_study_geometry.py。

- [ ] 用真实 JSON 结构写测试：keypoints_norm[134,2]、scores[134]、image_width/height。出画点及低于0.5的点不画，连线的两端都必须有效；所有点无效也应生成原尺寸黑图并记录。

~~~python
import numpy as np
from src.pose_control.qwen21_full.prompt_study.geometry import visible_points, hand_boxes

def test_visibility_does_not_clip_out_of_frame_points():
    xy = np.full((134, 2), .5)
    scores = np.ones(134)
    xy[1] = (-.1, .3)
    scores[2] = .49
    visible = visible_points(xy, scores)
    assert visible[0] and not visible[1] and not visible[2]
    assert tuple(hand_boxes(xy, scores)) == ("left", "right")
~~~

- [ ] visibility 的代码固定如下；不要先 clamp xy 再检查，否则出画点会变成有效边缘点：

~~~python
import numpy as np

def visible_points(xy, scores):
    xy, scores = np.asarray(xy, dtype=float), np.asarray(scores, dtype=float)
    if xy.shape != (134, 2) or scores.shape != (134,):
        raise ValueError("Expected DWPose134 xy/scores")
    return (np.isfinite(xy).all(axis=1) & np.isfinite(scores)
            & (scores >= .5) & (xy >= 0).all(axis=1) & (xy <= 1).all(axis=1))

def hand_boxes(xy, scores):
    xy = np.asarray(xy, dtype=float)
    good = visible_points(xy, scores)
    result = {}
    for name, start in (("left", 92), ("right", 113)):
        ids = np.flatnonzero(good[start:start+21]) + start
        if len(ids) < 4:
            result[name] = None
            continue
        lo, hi = xy[ids].min(axis=0), xy[ids].max(axis=0)
        side = max(float((hi-lo).max()) * 1.5, .02)
        center = (lo+hi) / 2
        result[name] = np.concatenate((np.maximum(center-side/2, 0),
                                       np.minimum(center+side/2, 1))).tolist()
    return result
~~~

- [ ] render_dwpose(record) 返回 PIL RGB 原尺寸图与 diagnostics。身体使用 OpenPose18 的边；脸24:92画有效点；双手92:113和113:134按五指各4节连接。134布局已经由 hand/geometry.py 和 build_iper_v6_detail_conditions.py 核对；不要套用 COCO17 下标。

~~~python
BODY_EDGES = ((1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
              (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
              (1,0),(0,14),(14,16),(0,15),(15,17))
HAND_EDGES = tuple((0 if n == 1 else 4*f+n-1, 4*f+n)
                   for f in range(5) for n in range(1,5))
FOOT_EDGES = ((13,18),(18,19),(13,20),(10,21),(21,22),(10,23))

def render_dwpose(record):
    from PIL import Image, ImageDraw
    xy = np.asarray(record["keypoints_norm"], dtype=float)
    good = visible_points(xy, record["scores"])
    width, height = int(record["image_width"]), int(record["image_height"])
    if min(width, height) <= 0:
        raise ValueError("Invalid DWPose canvas")
    image = Image.new("RGB", (width,height), "black")
    draw = ImageDraw.Draw(image)
    def point(i):
        return (round(xy[i,0]*(width-1)), round(xy[i,1]*(height-1)))
    palette = ((255,80,80),(80,255,80),(80,160,255),(255,220,80),(220,80,255))
    for n,(a,b) in enumerate(BODY_EDGES + FOOT_EDGES):
        if good[a] and good[b]:
            draw.line((point(a),point(b)), fill=palette[n%5], width=3)
    for start in (92,113):
        for a,b in HAND_EDGES:
            if good[start+a] and good[start+b]:
                draw.line((point(start+a),point(start+b)), fill=palette[(b-1)//4], width=2)
    for i in np.flatnonzero(good):
        x,y = point(i)
        radius = 1 if 24 <= i < 92 else 2
        draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill="white")
    diagnostic = {
        "threshold":.5, "original_size":[width,height],
        "render_version":"dwpose134_filtered_v1",
        "filtered_indices":np.flatnonzero(~good).tolist(),
        "visible_body":int(good[:18].sum()), "visible_face":int(good[24:92].sum()),
        "visible_left_hand":int(good[92:113].sum()),
        "visible_right_hand":int(good[113:134].sum()),
    }
    return image, diagnostic
~~~

- [ ] 原 DWPose PNG 原样复制；过滤图另存。报告列出原图、过滤图和 Body25 热图三项。过滤器只影响 PE 可视化，不改原 JSON、Adapter Body25、训练 mask 或 Face landmark。
- [ ] 手部 ROI 只从目标几何定义，保存原图坐标。即使 Hand Adapter 关闭也保留手部裁图用于观察。测试中生成手检测失败不能更换 ROI。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_geometry.py -q。预期不画越界点/低分线，左右手 ROI 能在原图与补边图之间正确转换。

### 任务 4：PE 请求白名单、文本编译与 PromptBank

**新增：** prompt_study/prompts.py、scripts/precompute_qwen21_pei2i_prompts.py。
**测试：** tests/test_qwen21_prompt_study_prompts.py。

先在 prompts.py 定义记录类型；后续训练、推理和报告全部使用这些字段：

~~~python
from typing import TypedDict

class PromptInput(TypedDict):
    role: str
    path: str
    sha256: str
    size: list[int]
    asset_relative_path: str

class PromptRecord(TypedDict):
    schema: str
    pair_id: str
    split: str
    protocol: str
    status: str
    input_roles: list[str]
    inputs: list[PromptInput]
    instruction_version: str
    instruction_sha256: str
    system_prompt_sha256: str | None
    model_revision: str | None
    model_files_sha256: dict[str,str]
    seed: int | None
    attempt: int | None
    raw_path: str | None
    rewritten_prompt: str
    compiled_prompt: str
    compiled_prompt_sha256: str
    tokens: int
    seconds: float
~~~

schema固定qwen21_pei2i_prompt_v1。hash运行时计算真实64位SHA256。PE revision从本地checkpoint元数据/缓存snapshot获取；无法取得远端commit时model_revision=None并记录实际权重/config/processor/system文件哈希。P0的inputs/input_roles为空，PE模型/seed/attempt/raw字段为None或空字典，rewritten/compiled均为固定P0，tokens/seconds=0；不伪造PE调用。

- [ ] 写泄漏和编译测试，要求构造请求只读白名单，而不是把整行 JSON 放入 user message：

~~~python
import pytest
from src.pose_control.qwen21_full.prompt_study.prompts import (
    request_roles, parse_answer, compile_prompt,
)

def test_pg_never_requests_B_or_manual_caption():
    row = dict(source_rgb="a.png", target_rgb="SECRET_B.png",
               target_depth_vis="d.png", target_normal="n.png",
               filtered_dwpose="p.png", action_description="SECRET_ACTION")
    assert request_roles(row, "pg") == [
        ("source_A", "a.png"), ("target_depth", "d.png"),
        ("target_normal", "n.png"), ("target_dwpose", "p.png")]
    assert "SECRET" not in str(request_roles(row, "pg"))

def test_compiled_text_has_no_unprovided_image_tokens():
    raw = '<image1>: Raise both arms. Follow <image2>. Preserve clothing.'
    result = compile_prompt(raw, ("source_A", "target_B"))
    assert "<image" not in result and "Raise both arms" in result
    with pytest.raises(ValueError, match="image"):
        compile_prompt("Use <image9>", ("source_A", "target_B"))
    assert parse_answer('thinking</think>{"rewritten_prompt":"Raise both arms"}')[
        "rewritten_prompt"] == "Raise both arms"
~~~

- [ ] prompts.py 的核心函数按下面实现；仅解析 JSON 对象，不用 eval，不把模型输出当可执行指令：

~~~python
import json
import re
from .protocol import PROMPT_ROLES

ROLE_FIELDS = {
    "source_A": "source_rgb", "target_B": "target_rgb",
    "target_depth": "target_depth_vis", "target_normal": "target_normal",
    "target_dwpose": "filtered_dwpose",
}
ROLE_WORDS = {
    "source_A": "the source image",
    "target_B": "the target pose described in this instruction",
    "target_depth": "the target depth structure described in this instruction",
    "target_normal": "the target surface orientation described in this instruction",
    "target_dwpose": "the target joint arrangement described in this instruction",
}

def request_roles(row, protocol):
    return [(role, row[ROLE_FIELDS[role]]) for role in PROMPT_ROLES[protocol]]

def parse_answer(raw):
    _, sep, answer = raw.partition("</think>")
    answer = (answer if sep else raw).strip()
    if answer.startswith("\x60\x60\x60"):
        lines = answer.splitlines()
        if lines[-1].strip() != "\x60\x60\x60":
            raise ValueError("Incomplete JSON fence")
        answer = "\n".join(lines[1:-1])
    result = json.loads(answer)
    text = result.get("rewritten_prompt") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing nonempty rewritten_prompt")
    return result

def compile_prompt(text, input_roles):
    aliases = {i+1: ROLE_WORDS[role] for i, role in enumerate(input_roles)}
    def replace(match):
        index = int(match.group(1))
        if index not in aliases:
            raise ValueError(f"Unknown image reference: {index}")
        return aliases[index]
    text = re.sub(r"<image\s*(\d+)>", replace, text, flags=re.I)
    text = re.sub(r"\bimage\s*(\d+)\b", replace, text, flags=re.I)
    text = " ".join(text.split())
    if not text or re.search(r"<image|\bimage\s*\d+", text, re.I):
        raise ValueError("Unresolved image reference")
    return text
~~~

这是文字引用替换，不会让未提供的 B/guide 真正进入 Qwen。PE user instruction 必须要求输出**自足的明确姿态描述**，不可只写“match image2”。首批人工 QA 检查这一点；若不满足，先修订统一 PE 指令并重生成开发样本，不能从 B 手动给 PG 补动作。

- [ ] instruction_v1.json 由固定 common text＋role-specific text组成，各 protocol 每个样本同模板，不插入手工 B 动作。使用下列 common text：

~~~text
Write one standalone English image-edit instruction for changing the pose of the person in the source photograph. The source photograph supplies the person's facial identity, hair, clothing design, garment colors and material texture, accessories, and background. Describe the desired body pose, head direction, facial expression, and hand configuration only to the extent they are observable in the permitted target references. Preserve the source identity and outfit while allowing head orientation, expression, limb position, cloth folds and pose-dependent shadows to adapt. Keep the source scene, camera framing and overall illumination. Use viewer image-left and image-right for spatial directions. Omit uncertain or invisible joint and finger details. Never transfer another person's identity, clothes or scene, and never render skeleton lines or normal/depth visualization colors. The final prompt must state the visible pose changes explicitly and be understandable when the editor receives only the source RGB and numerical geometry controls.
~~~

PG 的附加说明分别标明 depth/normal/skeleton 不是照片，不能从 normal RGB 推断衣服颜色；表情不可辨认则不强行描述。PB 的 B 只提供姿态/头向/可观察表情/手形；PBG 若几何与 B 有可见冲突，明确描述共同可靠部分，省略矛盾细节并在原始 PE 答案审查中保留冲突记录，不发明新的几何。

附加说明使用固定字典，image序号由已锁定角色顺序产生；不从样本旁路描述拼接：

~~~python
ROLE_INSTRUCTIONS = {
    "source_A": "This source photograph supplies the person's identity, hair, outfit, accessories and scene.",
    "target_B": "This target photograph supplies only desired body pose, head direction, observable expression and hand configuration.",
    "target_depth": "This grayscale visualization encodes target depth and occlusion; it supplies no clothing or identity appearance.",
    "target_normal": "These RGB colors encode target surface orientation; they are not skin, fabric or lighting colors.",
    "target_dwpose": "This confidence-filtered skeleton encodes visible target joints; black regions and omitted joints supply no hidden anatomy.",
}

def build_instruction(protocol, template):
    roles = PROMPT_ROLES[protocol]
    if protocol == "p0":
        raise ValueError("P0 does not invoke PE")
    parts = [template["common"]]
    parts.extend(f"Image {i+1}: {ROLE_INSTRUCTIONS[role]}"
                 for i,role in enumerate(roles))
    if protocol == "pbg":
        parts.append("When the target photo and guides visibly disagree, state only reliable observable pose details and omit contradictory details.")
    return "\n".join(parts)
~~~

template["common"] 就是上一段common text，存于instruction_v1.json并参与哈希；模板包含version、common、role_instructions、sampling、compiler_version，保证改写和编译规则一起锁定。

- [ ] PromptBank(path, expected_protocol, expected_pair_ids) 在 CPU 上加载 JSONL。检查 pair_id 唯一、状态 complete、protocol 匹配、compiled hash 正确、allowed roles 完全匹配、expected IDs 集合相等；prompt_for(pair_id) 只返回 compiled_prompt。不允许 missing→P0 fallback。
- [ ] PE runner 采用研究 probe 的 AutoProcessor、AutoModelForImageTextToText、官方 system_prompt.txt、enable_thinking=True、BF16、max_new_tokens=24000、do_sample=True、temperature=1、top_p=.95、top_k=20。图片原样读取，processor 自己的视觉编码预处理单独记载。
- [ ] 每条请求 seed42；解析/字段校验失败最多再试2次，seed43/44；每次 raw、解析错误、角色/图像哈希、耗时都原子保存。已有成功条目只有输入/模板/model/system/hash完全一致才复用。不按效果自动挑选三个不同成功 prompt。
- [ ] 首先只重写 train 索引0…7的三种 PE protocol，生成 qa8.json 供人工标记头向/双臂/双手/脸/身份衣服背景/不可见点/自足文本。模板只能在这一阶段调整；正式 train/val/diagnostic 重写前锁定统一版本。
- [ ] cache 命中以 pair_id＋protocol＋完整输入/模型/模板签名为依据；重复 epoch 不再次运行 PE，Qwen/Head 两底座不重复运行 PE。test 的 PE 文本在全部八组正式训练结束后生成，禁止据此反调模板。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_prompts.py -q。预期角色泄漏、未知 image 引用、损坏/重复/缺失 prompt 均被阻止；所有完成条目可重复加载。

PromptBank 的最小实现直接加入 prompts.py；input_roles 对 PE protocols 严格匹配，P0 不调用 PE：

~~~python
from pathlib import Path
from .protocol import text_sha256, P0_PROMPT

class PromptBank:
    def __init__(self, path, expected_protocol, expected_pair_ids):
        self.path = Path(path)
        self.protocol = expected_protocol
        self.records = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["pair_id"]
            if key in self.records:
                raise ValueError(f"Duplicate prompt: {key}")
            if row["protocol"] != expected_protocol or row["status"] != "complete":
                raise ValueError(f"Invalid prompt record: {key}")
            if tuple(row["input_roles"]) != PROMPT_ROLES[expected_protocol]:
                raise ValueError(f"Prompt roles differ: {key}")
            if tuple(item["role"] for item in row["inputs"]) != PROMPT_ROLES[expected_protocol]:
                raise ValueError(f"Actual prompt inputs differ: {key}")
            if not isinstance(row["compiled_prompt"],str) or not row["compiled_prompt"].strip():
                raise ValueError(f"Empty compiled prompt: {key}")
            if expected_protocol == "p0" and row["compiled_prompt"] != P0_PROMPT:
                raise ValueError(f"P0 differs from fixed baseline: {key}")
            if text_sha256(row["compiled_prompt"]) != row["compiled_prompt_sha256"]:
                raise ValueError(f"Prompt hash differs: {key}")
            self.records[key] = row
        if set(self.records) != set(expected_pair_ids):
            raise ValueError("Prompt manifest does not cover the locked split")

    def prompt_for(self, pair_id):
        return self.records[pair_id]["compiled_prompt"]
~~~

### 任务 5：准备兼容 Full Face 的缓存，禁止测试筛选

**新增：** scripts/data/prepare_qwen21_prompt_study_faces.py。
**复用：** scripts/prepare_tiktok_qwen21_face.py 的 assemble_cache、extract_source，以及已有 FaceSupervisionStore/native 转换器。
**测试：** tests/test_qwen21_prompt_study_faces.py。

这里有一个已核对的格式陷阱：scripts/precompute_qwen21_face_features.py 输出的是 source/target_geometry 等字典，并非 FaceConditionStore 所需的 ArcFace/DINO＋逐帧 tensor 格式；scripts/precompute_tiktok_face_features_v65.py 的基础格式也没有本项目 Full Face 所需的 geometry_valid 字段。不要直接换用这两个 CLI 的产物。

- [ ] 先增加最重要的角色隔离测试。下面用真实 assemble_cache 的数据结构，确认目标独有帧的身份/纹理为零，同时保留目标几何：

~~~python
import torch
from scripts.prepare_tiktok_qwen21_face import assemble_cache

def test_target_only_frame_has_geometry_but_no_identity():
    geometry = [dict(face_landmarks=torch.zeros(72,3),
                     face_boxes=torch.tensor([.2,.1,.6,.4]),
                     jaw_pose=torch.ones(3), expression=torch.ones(10),
                     geometry_valid=True) for _ in range(2)]
    source_features = {"0001": (torch.ones(512), torch.ones(256,1536))}
    cache = assemble_cache(["0001","0002"], geometry, source_features,
                           {"schema_version":1, "appearance":"00001"})
    assert cache["valid"].tolist() == [True, False]
    assert cache["geometry_valid"].tolist() == [True, True]
    assert cache["metadata"]["geometry_valid"] == [True, True]
    assert not cache["arcface"][1].any()
    assert not cache["dino_patches"][1].any()
    assert cache["jaw_pose"][1].any()
~~~

- [ ] 新 CLI 接收 --run、--splits、--source-features、--assets-root、--raw-root、--device。按 pair_metadata 的 source/target appearance 分别汇总角色，不能按一个 sequence_id 给 A/B 强行归到同组：

~~~python
from collections import defaultdict
from src.pose_control.qwen21_full.data_runtime import pair_metadata

def requested_frames(rows):
    groups = defaultdict(lambda: {"stems": set(), "sources": {}})
    for row in rows:
        meta = pair_metadata(row)
        for side in ("source", "target"):
            seq = meta[f"{side}_appearance"]
            stem = meta[f"{side}_stem"]
            groups[seq]["stems"].add(stem)
            if side == "source":
                old = groups[seq]["sources"].get(stem)
                path = meta["source_rgb"]
                if old is not None and old != path:
                    raise ValueError("Source frame resolves to two different files")
                groups[seq]["sources"][stem] = path
    return groups
~~~

- [ ] 对旧 train/val/diagnostic，优先验证并复用源 Full Face 缓存：
/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/data/face_cache。
将被请求的序列复制到新 run 的 data/face_cache，记录原文件 SHA 与 metadata.cache_fingerprint；旧文件只读。检查请求的所有 stem 都存在，source.valid、target.geometry_valid 与原锁一致。
- [ ] 新 test 缺失帧的序列，按 prepare_tiktok_qwen21_face.py 中的实际过程构建：加载 params/frame_map；project_tiktok_face72_to_rgb_normalized；compute_face_box；取 jaw_pose/expression；仅对 requested_frames.sources 的 RGB 调用 extract_source；assemble_cache；atomic_save_cache。使用与已有 helper 相同的 box 扩展、DINO crop、ArcFace 模型和 source 阈值。不要调用原 helper 主函数中最后的 kept/exclusions 筛选流程。
- [ ] 已有缓存需要补充 stem 时，从旧缓存保留既有帧并构建缺失帧，合并后重算 metadata/source_hashes/geometry_valid/fingerprint；如果模型或既有特征值不匹配即报错。新 test 无效 source/target face 记录为零/False 并保留这一条样本，不重新抽样。
- [ ] 只给 train512 准备辅助监督门。优先复制并验证旧 face_supervision_native.json 的 train512 所需 records；records 对所有训练 target 都完整，align_mode=native。缺记录时调用现有 face_supervision.build_cache，align_mode=native，而不是沿用512门再假定等价。
- [ ] val/test/diagnostic 不传 FaceSupervisionStore 到生成器；B 的数值 landmarks/jaw/expression/ROI 是生成条件，目标 RGB 检测只用于离线审查/评分。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_faces.py tests/test_qwen21_joint_training.py -q。预期 B 身份隔离、Full Face schema 和已存在 Hand 模式均通过。GPU Face precompute 只放在后续正式准备阶段，不属于单元测试。

### 任务 6：打通逐样本 prompt 与缓存，保留 legacy 行为

**修改：** joint_training.py/run_joint_loss。
**新增：** scripts/precompute_qwen21_prompt_study_cache.py。
**测试：** tests/test_qwen21_prompt_study_cache.py。

- [ ] 用 fake cache 断言新文本实际送到 cache.pair；当 prompt=None 时保持旧三参数调用。另用 Qwen21IPERCache._path 测试同一 A 不同文本缓存键不同。

~~~python
from PIL import Image
from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache

def test_source_cache_is_sensitive_to_compiled_prompt(tmp_path):
    cache = Qwen21IPERCache(tmp_path, resolution="native", preprocessing="native_v1")
    a = Image.new("RGB", (608,1088), "red")
    assert cache._path(a, "source", prompt="Raise arms") != cache._path(
        a, "source", prompt="Lower arms")
    assert cache._path(a, "target", prompt="Raise arms") == cache._path(
        a, "target", prompt="Lower arms")
~~~

- [ ] run_joint_loss 签名末尾增加 prompt=None；只替换 pair 构建块，噪声与损失计算保持原样：

~~~python
# 新签名追加 keyword-only 参数 prompt: str | None = None。
if prompt is None:
    pair = cache.pair(pipe, batch, device)
else:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Joint prompt must be nonempty")
    pair = cache.pair(pipe, batch, device, prompts=[prompt])
# report 中追加；默认旧调用也写出实际默认文本的哈希：
from .prompt_study.protocol import text_sha256
report["prompt_sha256"] = text_sha256(cache.prompt if prompt is None else prompt)
~~~

- [ ] 不向该函数加入第二个 adapter_prompt。当前控制上下文从相同 Transformer text embeddings 经过 text_bridge/pooled_bridge 取特征；给两者使用不同文本会再增加实验变量。
- [ ] 独立 cache CLI 接收 --run、--protocols、--splits、--model。对每 split/protocol 先完整校验 PromptBank；以该样本经过 QwenFullDataset(native/native) 的 **补边后 A** 为 source，而不是直接缓存原 A 再交给训练时补边后的 A。
- [ ] source 逐样本按 compiled_prompt 预计算；target 只允许 split=train。cache 目录按 p0/pg/pb/pbg 隔离，与旧 native_cache_tiktok_native 无关。两底座共享该缓存，因为 BFS LoRA 只改 Transformer，source text encoder/VAE 不变。
- [ ] precompute 循环使用如下调用；不要复用 execution.precompute 的固定文本和双侧遍历：

~~~python
from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil

def cache_split(pipe, cache, dataset, bank, *, split):
    counts = {"source_calls": 0, "target_calls": 0}
    for index in range(len(dataset)):
        sample = dataset[index]
        pair_id = sample["pair_metadata"]["pair_id"]
        cache.source(pipe, dataset_image_to_pil(sample["src_image"]),
                     prompt=bank.prompt_for(pair_id))
        counts["source_calls"] += 1
        if split == "train":
            cache.target(pipe, dataset_image_to_pil(sample["tgt_image"]))
            counts["target_calls"] += 1
    return counts
~~~

- [ ] 单元测试 fake dataset 包含两条验证行，FakeCache.target 调用即 pytest.fail；断言 source prompt 正确、target_calls=0。缓存 CLI 导入不加载模型。
- [ ] formal train 如果缺 source/target 缓存，应报清晰错误并让用户返回 cache 阶段补齐；不能在训练开始时自动给 val/test 编码 GT。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_cache.py tests/test_qwen21_joint_training.py -q。预期 source 文本哈希区分，训练损失数值/梯度/Face 门的旧测试通过。

### 任务 7：扩展联合训练器，1024更新与分段恢复

**修改：** scripts/train/train_qwen21_joint_pilot.py 的 arguments、main、save、preflight、训练循环、summary。
**使用：** prompt_study/sampling.py、PromptBank。
**测试：** tests/test_qwen21_prompt_study_training.py。

- [ ] 增加 CLI 参数：

~~~python
parser.add_argument("--study-lock", type=Path)
parser.add_argument("--arm", choices=ARMS)
parser.add_argument("--prompt-manifest", type=Path)
parser.add_argument("--stop-after-step", type=int)
# study_mode = args.study_lock is not None
# legacy 仍要求 1<=steps<=256；study 要求 steps==1024。
# study 要求 arm/prompt_manifest、native/native、noise_offset==256、
# seed==42、with_hand==False、固定本计划 LR/loss/aux/warmup。
# stop_after_step 默认为 args.steps；必须 0<=stop<=steps。
# stop==0 允许新建 step0 full checkpoint 后正常退出，不执行更新。
~~~

总预算始终用 --steps 1024；--stop-after-step 只控制这次进程结束边界，**不能把 --steps 分别改成128、256…**。stop_after_step、输出路径、resume checkpoint 路径、gradient-only 不进入不变的训练配置签名；数据/文本/模型/LoRA/代码哈希及1024总预算要进入。

- [ ] 从 data/lock.json 校验 train512/val24 字节哈希、1024轨迹、初始化SHA；study 不调用旧 validate_locked_split，legacy 继续调用。study 读取 train512，不拼接训练 val。拒绝 train 中 cross_person=True。
- [ ] joint_signature 在旧字段基础上增加 study_schema、study_lock_sha256、arm、protocol、prompt_manifest_sha256、instruction_sha256、sample_order_sha256、max_updates=1024、initial_history_steps=256、native preprocessing、代码哈希集合。P0 的文本也通过 PromptBank 取，不能绕过记录。

sample_order_sha256=text_sha256(json.dumps(order,separators=(",",":")))；study签名的prompt_manifest_sha256指**train.jsonl**。评估另记录所用val/test/diagnostic的manifest哈希，并验证它们与相同protocol/template对应；不能要求val文本文件哈希与train文件相等。Face资产另写data/face_assets.json及SHA进入训练签名，准备Face资产/正式prompt时不改已经冻结的data/lock.json。
- [ ] source cache 预计算块：legacy保留现有行为；study 只核验已准备的 train source/target 项存在，不创建 val 的 target。train、val、test 的 PromptBank 路径在训练器里不能互相代用。
- [ ] 加载流程保持顺序：新 pipeline→冻结 Qwen邻接模块→load_frozen_bfs(Head 或 None)→开启 Transformer checkpointing→新 Adapter→load J256 weights→joint_parameter_groups→enable_joint_training→**全新 AdamW**。如果 --resume，再恢复本轮 checkpoint 的 optimizer/RNG，而不是 J 的 optimizer。
- [ ] fresh study 先 reset seed_all(42+256)，写 step0，保存实际 lora_loading.json。加载 LoRA 所消耗的随机数不能改变后面的轨迹。
- [ ] initial的stop0必须能完整结束：在已有fresh-output冲突校验之后创建空metrics.jsonl；summary对空日志使用last_loss=None、steps=0、completed=False，不能读取all_rows[-1]或缺失文件。resume从step0恢复空日志/空sample_order是合法状态。
- [ ] 用以下study循环替换“step同时等于样本索引”的假设；legacy循环保留现有行为。检查与更新沿用当前实现：

~~~python
# order 来自锁；legacy 则 list(range(args.steps))。
end_step = args.steps if args.stop_after_step is None else args.stop_after_step
if not 0 <= start_step <= end_step <= args.steps:
    raise ValueError("Invalid resume/stop interval")

for step in range(start_step, end_step):
    index = order[step]
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * warmup_scale(step)
    prompt = bank.prompt_for(train.samples[index]["pair_id"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        row = run_joint_loss(
            pipe, adapter, cache, train, store, index,
            seed=noise_seed(step), prompt=prompt,
            auxiliary=step % 4 == 0, evaluators=evaluators,
            supervision=supervision, face_weight=.5, with_hand=False,
        )
    grads = joint_group_gradients(adapter,names)
    active_branches = ("body","face") if row["face_active"] else ("body",)
    validate_group_gradients(grads,branches=active_branches)
    norm = float(torch.nn.utils.clip_grad_norm_(
        [p for p in adapter.parameters() if p.requires_grad],1.))
    if not math.isfinite(norm) or norm <= 0 or not math.isfinite(row["loss"]):
        raise FloatingPointError("Invalid study gradient/loss")
    assert_frozen_gradients(
        adapter,transformer=pipe.transformer,vae=pipe.vae,text_encoder=pipe.text_encoder,
        **{f"evaluator_{i}":model for i,model in enumerate(evaluators)},
    )
    optimizer.step()
    assert_fp32_training_state(adapter,optimizer)
    optimizer.zero_grad(set_to_none=True)
    sample_order.append(index)
    row.update(step=step+1, epoch=step//512, epoch_position=step%512,
               protocol=protocol, arm=args.arm,group_grad_norms=grads,grad_norm=norm,
               elapsed_seconds=time.perf_counter()-started,
               peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
    with metrics_path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(row)+"\n")
        stream.flush()
    if (step+1)%128 == 0 or step+1 == end_step:
        save(args.output/f"checkpoint_step_{step+1:06d}.pt",
             adapter,optimizer,args,step=step+1,signature=signature,sample_order=sample_order)
        save(args.output/"checkpoint_last.pt",
             adapter,optimizer,args,step=step+1,signature=signature,sample_order=sample_order)
~~~

上面是 **study 分支的替换片段**；legacy 使用原 noise_offset/steps/aux逻辑。step 是已完成的更新计数，epoch 是本轮0/1。

- [ ] save 保留现有 architecture、adapter/optimizer、joint_signature、sample_order、初始化来源、所有 capture_rng 字段。study 追加下列字段：

~~~python
if args.study_lock is not None:
    payload.update(
        study_schema=SCHEMA, arm=args.arm,
        protocol=args.arm.split("_", 1)[1],
        max_updates=1024, initial_history_steps=256,
        study_lock_sha256=signature["study_lock_sha256"],
        prompt_manifest_sha256=signature["prompt_manifest_sha256"],
        next_update_index=step,
    )
~~~

- [ ] 每128步及进程停止边界保存 checkpoint_step 和 checkpoint_last，使用临时文件＋replace。不要保留9份重复 last；last只保留当前一份。正式新增步数来自成功提交的 metrics/checkpoint，不把preflight或中断后重算更新记成正式进度。
- [ ] resume 先执行现有 reconcile_resume_metrics：尾部未提交日志归档，不丢弃。再用 validate_trajectory 检查 sample_order=order[:step]、step序列、noise_seed、pair_id、prompt_sha 与对应锁定文本一致，不能再要求 sample_order==range(step)。
- [ ] summary.completed=(actual_step==1024)，stopped_at_checkpoint=(actual_step<1024)；steps保存actual，max_updates=1024；active_seconds分进程记录并累计；不能在128停止时写 completed=True。
- [ ] preflight 使用同 arm、同初始化、同 prompt/cache/签名；低噪声σ=.2检查 Body＋Face各optimizer组非零；高噪声σ=.8确认Body有效、Face时间门关闭；aux eligible时两个辅助损失对Face bridge梯度非零；所有基干/LoRA/Hand/Interaction冻结未改。
- [ ] 增加纯 CPU 的512边界测试：

~~~python
import pytest
from src.pose_control.qwen21_full.prompt_study.sampling import (
    make_order, noise_seed, validate_trajectory,
)

def test_resume_at_epoch_boundary_uses_locked_permutation():
    order = make_order()
    saved = {"step": 512, "sample_order": order[:512]}
    rows = [dict(step=u+1, pair_index=order[u], noise_seed=noise_seed(u))
            for u in range(512)]
    validate_trajectory(saved, rows, order)
    assert order[511] == 511
    assert order[512] == make_order()[512]
    rows[-1]["noise_seed"] += 1
    with pytest.raises(ValueError, match="trajectory"):
        validate_trajectory(saved, rows, order)
~~~

- [ ] 参数解析测试：legacy --steps1024失败；study --steps256失败；允许stop0/128/1024；更改 prompt_manifest、Head权重、数据、总预算或不变训练代码后 resume 失败。仅改 stop_after_step 不导致签名失败。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_training.py tests/test_qwen21_joint_training.py tests/test_qwen21_joint_pilot_resume.py -q。预期多轮采样通过，旧严格恢复和Hand测试通过。

### 任务 8：逐样本推理、GT 隔离与原图/局部图保留

**修改：** scripts/eval/eval_qwen21_joint_pilot.py。
**使用：** PromptBank、geometry.hand_boxes。
**测试：** tests/test_qwen21_prompt_study_eval.py。

- [ ] 增加 --study-lock、--arm、--prompt-manifest、--study-step，study_step允许0,128,…1024；旧 --stage 不变。study wrapper 显式传 canonical --variant Qwen 或 BFSHead，避免 infer_variant_stage 把 head_pg 名称识别错。
- [ ] 任意 study_step 的正式评估都要求 checkpoint.joint_signature 与本 arm/LoRA/hash相符。新step0是任务7创建的当前arm签名checkpoint，可以按 strict mid/final provenance 验证；原J256只列历史来源，不用 hotload 覆盖正式签名。
- [ ] main 的 config 追加 schema、study_step、arm、protocol、prompt_manifest_sha256、lock_sha；prompt记录改成 per_sample 字段，不再写 config.prompt=PROMPT 冒充实际文本。
- [ ] 进入每个 case 后只取一次：

~~~python
compiled_prompt = bank.prompt_for(meta["pair_id"])
prompt_record = bank.records[meta["pair_id"]]
row.update(
    arm=args.arm, protocol=bank.protocol, study_step=args.study_step,
    prompt_sha256=prompt_record["compiled_prompt_sha256"],
    prompt=compiled_prompt,
    prompt_record_pair_id=prompt_record["pair_id"],
    initialization="fresh_gaussian",
)
# conditioning：
entry = cache.source(pipe, source, prompt=compiled_prompt)
# generation：
kwargs = generation_arguments(
    source, args, prompt=compiled_prompt,
    generator=torch.Generator(device="cuda").manual_seed(args.seed),
    target_size=target_size,
)
if kwargs["image"] != [source] or "latents" in kwargs or "target" in kwargs:
    raise AssertionError("Study generation violated source-only/noise initialization")
result = pipe(**kwargs)
~~~

实际生成调用保留在现有 torch.inference_mode＋autocast＋Qwen21FullControlContext 内。pipeline 与 Adapter 均 eval/requires_grad=False。cache.source 的作用是 Adapter 的 A latent/text缓存；pipe 内部编码同一 A 与同一 compiled_prompt，不让二者文本不同。

- [ ] 对 cross6，任务2的 normalized sequence_id绑定B，source_rgb仍A，因此 TikTok native loader自然补边A，读取B几何；测试 QwenFullDataset/sample.meta.source_appearance 仍A，FaceConditionStore参考只读A。target_body25的查找使用target_appearance。
- [ ] face crop继续使用**固定目标几何框**裁生成图，但修正study坐标边界：当前eval先对框做padding变换，随后image被unpad；不能直接拿padded-normalized框裁original图。额外保存两个手ROI；invalidFace也保存整图、原条件、失败记录，不能只保留face_valid=True。

~~~python
# 在 Face cache读框的循环中，在 box 被 transform.points 之前记录：
row[f"{name}_face_box_original_normalized"] = box.tolist() if box is not None else None
# transform.points之后记录另一份，供条件审查：
row[f"{name}_face_box_padded_normalized"] = box.tolist() if box is not None else None
# 所有图像unpad之后，save_case_artifacts之前，study模式统一改成：
if args.study_lock is not None:
    for side in ("source","target"):
        row[f"{side}_face_box"] = row[f"{side}_face_box_original_normalized"]
~~~

双手hand_boxes从原始DWPose得到的框已经是原图normalized坐标，保存输出crop时不再次变换；要显示Adapter补边坐标则另存NativePad.points后的副本。增加一个有明显彩色块的原尺寸图测试，要求Face裁图始终取到预定原图像素；604×1080和736×1024两种尺寸都覆盖。
- [ ] conditioning/ 下保存 normal、实际Body25、PE depth、原DWPose、过滤DWPose。generated、A、B整图unpad回原始尺寸；网页缩略图独立生成，不覆盖原图。
- [ ] 新评估契约可直接用旧 generation_arguments 测试：

~~~python
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest
from PIL import Image

@pytest.fixture
def evaluation():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prompt_study_eval",root/"scripts/eval/eval_qwen21_joint_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_study_generation_keeps_A_and_original_target_size(evaluation):
    a = Image.new("RGB", (608,1088))
    args = SimpleNamespace(resolution="native", align_mode="native",
                           steps=30, use_kv_cache=True)
    kw = evaluation.generation_arguments(a, args, prompt="Raise both arms",
                                        generator=None, target_size=(736,1024))
    assert kw["image"] == [a]
    assert kw["prompt"] == "Raise both arms"
    assert (kw["width"],kw["height"]) == (736,1024)
    assert not {"latents","target","gt_image"} & kw.keys()
~~~

evaluation pytest fixture 只加载脚本定义，不创建 Diffusers pipeline。

- [ ] 再写一个 mock eval case：monkeypatch Qwen21IPERCache.target、pair、joint_training.run_joint_loss、Tensor.backward、Optimizer.step 为立即抛错；FakePipeline记录 encode_prompt 的image列表与调用kwargs，验证没有B RGB、默认高斯、prompt完全相同、上下文30次forward。为GPU端记录运行审计，而不是仅靠文本grep证明无泄漏。
- [ ] 改写推理 config 的 target_B_rgb 文案：Qwen RGB condition=Source A only；B may be read by PE for PB/PBG、numeric geometry、dimensions、offline scoring。不能继续用“offline only”混淆PE的B输入。
- [ ] 失败结果不在同目录静默覆盖；要复跑固定样本，保存 attempts/新目录和 rerun_reason，再由结果索引指向新尝试，同时保留原失败率/记录。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_eval.py tests/test_qwen21_joint_pilot_visuals.py tests/test_qwen21_native_resolution.py -q。预期onlyA、无target编码、原尺寸、Face/Hand关门与旧报告测试通过。

### 任务 9：任务枚举与分阶段调度，防止重复生成和提前测试

**新增：** prompt_study/results.py、scripts/run/run_qwen21_pei2i_study.py。
**测试：** tests/test_qwen21_prompt_study_runner.py。

- [ ] 先写纯 CPU 任务数量测试。这是固定预算和可视化覆盖的验收，不需要 GPU：

~~~python
from src.pose_control.qwen21_full.prompt_study.results import eval_jobs
from src.pose_control.qwen21_full.prompt_study.protocol import ARMS

def test_fixed_generation_budget_and_no_duplicate_jobs():
    jobs = eval_jobs(include_final_test=True)
    assert len(jobs) == 3696
    keys = {(j["arm"],j["step"],j["split"],j["pair_index"],j["seed"])
            for j in jobs}
    assert len(keys) == len(jobs)
    assert len(eval_jobs(include_final_test=False)) == 2928
    for arm in ARMS:
        assert sum(j["arm"] == arm for j in jobs) == 462
~~~

- [ ] results.py 的任务枚举：

~~~python
from .protocol import ARMS, EVAL_STEPS, FULL_VAL_STEPS, TRAIN_MONITOR, VAL_MONITOR

def eval_jobs(*, include_final_test=False):
    jobs = []
    def append(arm, step, split, indices, seeds):
        for seed in seeds:
            for index in indices:
                jobs.append(dict(arm=arm, step=step, split=split,
                                 pair_index=index, seed=seed))
    for arm in ARMS:
        for step in EVAL_STEPS:
            append(arm,step,"train",TRAIN_MONITOR,(42,))
            if step in FULL_VAL_STEPS:
                append(arm,step,"val",range(24),(42,43))
            else:
                append(arm,step,"val",VAL_MONITOR,(42,))
            if step in (0,512,1024):
                append(arm,step,"diagnostic",range(6),(42,))
        if include_final_test:
            for step in (0,1024):
                append(arm,step,"test",range(24),(42,43))
    return jobs
~~~

- [ ] runner 参数 --run、--phase、--splits（nargs+）、--arms（nargs+）、--max-updates=1024、--eval-every=128、--resume、--dry-run。phase固定：

~~~
lock, rewrite-qa, rewrite, faces, cache, selfcheck,
preflight, initial, train, finaltest, report, export
~~~

--dry-run 只打印当前阶段 argv、任务数量、输出目录、依赖状态，不导入 GPU 库、不启动子进程、不写完成标志。--phase为必填，脚本没有“默认跑全套”的行为。

- [ ] invoke 使用 subprocess.run(list_of_args, check=True, cwd=ROOT)，不拼接 shell 字符串。示例正式 train 调用按以下结构，具体 arm 的 lora 由 validate_arm 返回的 base决定：

~~~python
argv = [
    sys.executable, str(ROOT/"scripts/train/train_qwen21_joint_pilot.py"),
    "--study-lock", str(run/"data/lock.json"), "--arm", arm,
    "--train-pairs", str(run/"data/train512_pairs.jsonl"),
    "--val-pairs", str(run/"data/val24_pairs.jsonl"),
    "--prompt-manifest", str(run/f"prompts/{protocol}/train.jsonl"),
    "--features", str(run/"data/face_cache"),
    "--face-supervision", str(run/"data/face_supervision_native.json"),
    "--cache", str(run/f"cache/{protocol}"),
    "--output", str(run/f"training/{arm}"),
    "--steps", "1024", "--stop-after-step", str(stop),
    "--resolution", "native", "--align-mode", "native",
    "--noise-offset", "256",
    "--gradient-report", str(run/f"preflight/{arm}/gradient_only.json"),
]
# 新建：追加 --init-checkpoint 与锁中的共同初始化路径。
# 恢复：追加 --resume run/training/<arm>/checkpoint_last.pt。
# base=head：追加 --lora 与锁中的 BFS Head 路径；base=qwen不追加。
# study 参数校验时同时检查锁中 model 路径，并在 argv 中显式追加 --model。
~~~

- [ ] 新 eval 参数再增加 --study-split（train/val/diagnostic/test），与 pairs、prompt-manifest 路径交叉验证。train_monitor仍使用train512 manifest和对应完整PromptBank，只传8个indices。推理 argv 固定 --resolution native --align-mode native --steps30 --seed42/43 --variant Qwen/BFSHead --stage final --study-step整数值；输出目录按任务枚举划分。
- [ ] initial 阶段为所有 arm 新建各自 step0 full checkpoint（optimizer无历史），运行计划的step0训练监控/val/diagnostic。不要在 initial 跑新 test；其 step0 结果等全部训练完成后再生成。
- [ ] train 阶段逐 arm 运行128→256→…1024。每个进程结束释放训练pipeline；然后只生成这个 arm/step 的预定任务，运行独立CPU评分/报告增量构建；接着恢复训练。--resume 从checkpoint和评估任务状态接着执行，不能用已保存图像作为重新初始化训练的依据。
- [ ] 同一组checkpoint/sample/seed的完成输出只在config/hash/PNG均匹配时复用。目录有记录而PNG缺失、签名不同或重复pair索引，直接写错误并退出。不能把缺失视作完成。
- [ ] finaltest 的前置条件：所有8个training summary均 actual_step=1024、completed=True、冻结检查通过；test prompt已生成且模板版本与训练一致；test source cache完备。任一组没完成不能先跑部分正式test。
- [ ] rewrite --splits test、cache --splits test 也检查8组完成门。数据锁/Face数值资产可提前准备，正式test的PE文本/生成结果不得参与训练模板开发。
- [ ] CPU调度测试 monkeypatch invoke：初始阶段不含test；stop参数只改变边界；恢复后不重算已完成任务；缺prompt/preflight/校验失败不dispatch训练；正式test提前请求报错；dry-run零调用。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_runner.py -q。预期任务预算3696、分进程参数和test门均通过。

### 任务 10：恢复、梯度和输入隔离自检

**修改：** runner 的 selfcheck/preflight 阶段，results.py 的检查结果汇总。
**测试：** tests/test_qwen21_prompt_study_selfcheck.py。

以下检查在专属 preflight 目录做，不能占用正式训练输出或污染正式 step 计数。

- [ ] CPU selfcheck 汇总任务1…9的测试、1024轨迹与512边界、原图尺寸/补边/ROI坐标、四种PromptBank覆盖、PG角色白名单、sourcecache哈希、eval仅A/noGT/noBackward、模型资产/数据锁哈希、正式test门。验证protocol.P0_PROMPT与qwen_runtime.PROMPT字节完全一致。逐角色校验实际PE输入path/hash绑定锁定sample的A/B/guide，不能仅检查role名称。输出 selfcheck.json，任何强制项失败 passed=False 并退出非零。
- [ ] 每 arm 梯度preflight使用共同J权重和本arm实际prompt，但只backward、不optimizer.step。生成同signature的gradient_only.json，报告每组梯度范数与Face时间门。输入图/caches缺失不能用别的样本替代。
- [ ] GPU恢复检验采用 qwen_pg 私有轨迹：总预算签名仍1024，只运行更新0…3。direct先stop2写字节确定的checkpoint，再resume至4；从同一个step2 checkpoint fork出resume、repeat1、repeat2，分别续到4。这样总共10次私有optimizer更新；把分段边界是否影响状态与CUDA自然漂移分开检查。
- [ ] fork使用shutil.copyfile或同盘hardlink复制step2 checkpoint及前2行metrics，复制run_manifest；末尾保存checkpoint采用atomic replace，避免改到其他hardlink。恢复必须先用 assert_exact_state 检查刚加载的Adapter、optimizer、RNG与step2完全一致。
- [ ] 复用旧 runner 的数值诊断函数，不直接调用旧 compare_resume 的目录/256协议假设：

~~~python
from scripts.run.run_qwen21_joint_pilot import _state_drift, _loss_drift
from src.pose_control.qwen21_full.joint_training import assert_exact_state

def compare_private_continuations(left_state, right_state, left_rows, right_rows):
    errors = []
    if left_state["step"] != 4 or right_state["step"] != 4 or len(left_rows) != 4 or len(right_rows) != 4:
        raise ValueError("Private continuation must end at four updates")
    for key in ("step", "sample_order", "joint_signature",
                "joint_initialization_sha256"):
        assert_exact_state(left_state[key], right_state[key], key)
    assert_exact_state(left_rows[:2],right_rows[:2],"forked_metrics_prefix")
    for a, b in zip(left_rows, right_rows):
        for key in ("step", "pair_index", "pair_id", "noise_seed",
                    "prompt_sha256", "sigma", "face_active"):
            assert_exact_state(a[key], b[key], f"metrics.{key}")
    if len(left_rows) != len(right_rows):
        raise ValueError("Continuation metric lengths differ")
    return {
        "adapter": _state_drift(left_state["adapter_state_dict"],
                                right_state["adapter_state_dict"], "adapter", errors),
        "optimizer": _state_drift(left_state["optimizer_state_dict"],
                                  right_state["optimizer_state_dict"], "resume.optimizer", errors),
        "losses": _loss_drift(left_rows, right_rows, "losses", errors),
        "errors": errors,
    }
~~~

- [ ] result.json 分开记录 mandatory_exact_gate（恢复时state/optimizer/RNG、签名、离散计数、数据/文本/噪声一致）与 numerical_diagnostics（continuation权重/loss相对独立repeat漂移）。不得用一个任意固定阈值掩盖元数据错误；超出repeat观察范围先调查并记录，不能把结果标成“完全可复现”。
- [ ] 全量逐128步进程恢复额外记录 sampler/warmup 校验；CPU测试造511/512/513 checkpoint，检查epoch1样本及noise不重复。GPU不额外跑512更新只为测试边界。
- [ ] 正式训练器的 summary 验证冻结Transformer/LoRA参数版本与冻结Adapter digest；LoRA参数requires_grad=False、grad=None；Body/Face属于optimizer，Hand/Interaction不属于。评估进程没有optimizer/GT损失入口。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_selfcheck.py tests/test_qwen21_frozen_bfs_lora.py tests/test_qwen21_joint_pilot_resume.py -q。预期强制门损坏时失败，数值漂移独立记录，旧BFS冻结与恢复测试通过。

### 任务 11：指标汇总和中文离线网页

**新增：** scripts/reports/build_qwen21_pei2i_study_report.py。
**修改：** results.py 的统计/配对/产物校验；eval脚本离线评分增加背景/同人物误差。
**测试：** tests/test_qwen21_prompt_study_report.py。

- [ ] 结果主键固定为 (arm,study_step,split,pair_id,seed)。按此对齐，不用目录排序或pair_index跨split硬拼。检查locked A/B binding、prompt hash、checkpoint hash、LoRA hash、尺寸与生成参数一致；任何重复或冲突结果列为provenance_error。
- [ ] 沿用 joint_pilot_metrics 的 identity_A、pose_pck_002、pose_nme_diagonal 等。identity只有A和输出均唯一检测脸时有效；无脸/多脸单独计数。pose数值只在相应检测有效时计算，失败不能填0。
- [ ] score_saved_outputs已经向PoseMetric传生成图的实际width/height，但scores.json目前仍写metric_resolution=512。study评分记录每行image_size=[width,height]，总metadata使用metric_resolution="per_sample_native"，避免网页把原尺寸结果标成512实验；legacy保持旧字段兼容。
- [ ] 保存背景保持误差 bg_l1_A：在A与目标人体mask的并集以外，再排除一个固定8原像素的边缘带，比较生成图与A；A/B画布不同尺寸或无有效背景则None＋reason，不能resize后混算。使用PIL MaxFilter(17)对mask并集扩张即可，不加新学习模型。
- [ ] 同人物val/test可另存pixel_l1_GT辅助描述；cross diagnostic禁止整图B GT误差。身体/衣服/手质量仍需整图和固定ROI人工核对，PCK不足以代表身份和纹理质量。
- [ ] 两个推理seed先在每个pair内求有效均值，再按3个原视频组求组均值/组宏平均；保留各seed明细。不能把24×2视作48个独立人物/独立训练重复。这里只做描述性统计，不宣称单个训练seed的结论有统计显著性。
- [ ] 配对函数用有效交集计算差值，并公开缺失数量。核心片段：

~~~python
import math
from collections import defaultdict

def paired_metric(left, right, metric):
    # 调用前已按pair汇总seed；每row含split/pair_id/group/metrics。
    right_map = {(r["split"],r["pair_id"]): r for r in right}
    values, group_values = [], defaultdict(list)
    def finite(value):
        return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
    for row in left:
        other = right_map.get((row["split"],row["pair_id"]))
        a = row["metrics"].get(metric)
        b = None if other is None else other["metrics"].get(metric)
        if finite(a) and finite(b):
            values.append(b-a)
            group_values[row["group"]].append(b-a)
    means = {g:sum(v)/len(v) for g,v in group_values.items()}
    return {
        "expected_n":len(left), "valid_pair_n":len(values),
        "missing_or_invalid_n":len(left)-len(values),
        "mean_difference":sum(values)/len(values) if values else None,
        "group_macro_mean_difference":sum(means.values())/len(means) if means else None,
        "group_means":means,
    }
~~~

- [ ] 主要对照：同底座PG/PB/PBG分别对P0；同prompt的Head对Qwen；step1024对本arm step0。val曲线用于观察学习与过拟合；正式test只展示预定0/1024，不选择“测试最优checkpoint”。
- [ ] 网页全部导航、表头、说明、失败原因标题用中文；原PE英文输出和日志如实显示。首屏至少有：
  - 当前完成状态：正式更新数/8192、PE成功数/1698、生成任务完成数/3696；preflight另列。
  - 八组条件表、共同J256初始化SHA、各arm真实LoRA加载摘要、Body/Face/Hand/Depth通道状态。
  - 原尺寸/补边说明、GT流向表、PB/PBG额外B信息、训练/验证/独立测试/跨人物诊断区别。
  - split、step、seed、pair筛选；8列arm并排图；全身/脸/左手/右手切换；原PNG链接。
  - 实际Body25、normal、PEdepth、原/过滤DWPose；每protocol真正喂给PE的图片列表、instruction、raw回答、compiled文本与哈希。
  - 验证loss/identity/PCK/背景曲线，正式test的valid_n/expected_n/各组均值/配对交集。
  - 缺图、无脸、Face无效、几何冲突、OOM/评分失败与失败尝试归档。

- [ ] 历史J/Qwen/BFSHead图放单独“历史实验”区，保留其旧prompt/预算/热加载条件；不把历史图片拿来填本次step0缺图。只有provenance完全相同才可去重引用。
- [ ] 每个case保留原始PNG；网页用PIL另生成宽度192或256缩略图，点击打开原图。报告只引用相对路径，不能让本地浏览器依赖服务器绝对路径。
- [ ] PE所有请求图片按文件字节SHA去重复制到 prompts/input_assets/<sha><原扩展名>；inputs记录追加 asset_relative_path，原path/hash/尺寸也保留。这样包括未进入8个train-monitor的训练样本，都能在网页核对PE实际输入。复制原字节，不用缩略图作为PE输入。
- [ ] 文本输出用html.escape，不内嵌未经转义的PE/日志/文件名。构建一个含script标签的假prompt和含引号pair_id的测试，要求可显示原文而不执行脚本：

~~~python
from html import escape

def text_block(text):
    return "<pre>" + escape(text, quote=True) + "</pre>"

def test_raw_prompt_is_html_escaped():
    assert "<script>" not in text_block("<script>alert(1)</script>")
    assert "&lt;script&gt;" in text_block("<script>alert(1)</script>")
~~~

- [ ] fixtures 包含8arm、两step、两个seed、缺图、失败、重复记录、PG/PB不同原输入；测试中文标题、相对链接存在、失败保留、两seed先合并、cross不算GT误差。report可以在部分完成时构建，但必须标“进行中”并显示未执行。
- [ ] 运行：python -m pytest tests/test_qwen21_prompt_study_report.py tests/test_qwen21_joint_pilot_summary.py tests/test_qwen21_joint_pilot_visuals.py -q。预期新8列报告和旧报告均通过。

### 任务 12：本地导出、最终回归与使用说明

**修改：** results.py 的导出清单、runner的export阶段。
**新增：** docs/experiments/2026-10-10-pei2i-eight-arm-runbook.md。
**测试：** tests/test_qwen21_prompt_study_export.py。

- [ ] 导出只允许data/prompt文本与输入图、preflight JSON/日志、training元数据/metrics、evaluation、report。所有目录排除 .pt/.safetensors/.bin/.onnx 和 cache；不下载模型、optimizer、latent缓存。face_cache的schema/hash摘要要导出，完整.pt留服务器。
- [ ] 打包前从报告的href/src遍历核验每个相对文件存在；错误/未运行记录可无generated，但网页必须有中文状态。输入照片、geometry和raw PE回答必须在包内。记录每个导出文件相对路径、bytes、sha256到export/files.json；包装report/statistics/data的同一版本。
- [ ] 用名单复制到export/staging后制作export/results.tar.gz。名单禁止 ..、绝对路径或解析后逃逸run根目录；不能跟随指向cache/模型目录的外部symlink。明确估算空间再复制，远端预留500GiB用于checkpoint/cache/输出，本地预留50GiB用于导出。
- [ ] 测试生成临时run，含evaluation/png、report/html、training/checkpoint.pt、cache/latent.pt、prompts/raw.json；要求png/html/raw进入包，所有.pt/cache排除；缺相对图链接让验收失败。
- [ ] 回归命令在远端环境执行，避免本地Python未安装GPU依赖造成误判：

~~~bash
cd /home/shangguanrz/project/pic-edit
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest \
  tests/test_qwen21_prompt_study_*.py \
  tests/test_qwen21_joint_training.py \
  tests/test_qwen21_joint_pilot_resume.py \
  tests/test_qwen21_joint_pilot_visuals.py \
  tests/test_qwen21_joint_pilot_summary.py \
  tests/test_qwen21_native_resolution.py \
  tests/test_qwen21_frozen_bfs_lora.py -q
~~~

- [ ] 在runbook写入实际CLI --help、最后测试摘要、实现提交/代码SHA、数据锁SHA、preflight状态及以下运行顺序。此阶段不自动开8192更新。
- [ ] 代码提交若需要，逐任务只git add该任务文件；先查看staged diff，不能git add -A把当前清理/Hand/native的其他工作一并提交。用户自行实施可按任务独立提交；本次制定计划不产生代码commit。

## 5. 实现完成后的逐步运行手册

以下命令调用的是**计划新增的CLI**，须先完成任务1…12并通过回归。不能现在直接运行后期待接口存在。Linux命令均在sg上运行；Python环境固定。

### 第一步：锁定输入，检查资源，准备开发样本

~~~bash
ssh sg
cd /home/shangguanrz/project/pic-edit
export STUDY_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
export STUDY_RUN=/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009

$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase lock
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase rewrite-qa
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase report
~~~

lock完成后应有512/24/24/6、1024order、无视频组交叉审计、模型/初始化哈希与几何预览。rewrite-qa只生成8训练样本×3protocol=24条PE开发记录；人工在中文网页核对，按实际情况填写qa8.json，固定instruction_v1和review记录。正式输入模板变更应重建对应prompt/cache/preflight，不能只改模板文件继续resume。

### 第二步：正式 PE 重写、Face资产与缓存

~~~bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase rewrite \
  --splits train val diagnostic
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase faces \
  --splits train val diagnostic test
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase cache \
  --splits train val diagnostic
~~~

正式train/val/diagnostic共542条sample×3=1626条PE记录，开发样本若签名一致计入复用；P0同时生成对应索引。test24的72条PE记录留到第六步。Face test资产可提前准备，生成文本与生成图像稍后做。

开始时记录前20条正式PE请求速度、tokens、失败率/重试数、GPU峰值，写timing.json，估算剩余耗时。原研究探针单条约89～117秒，仅供资源预估，不能保证全量耗时。PE任务分split断点续写；完成后进程结束释放GPU，再开Qwen缓存/训练。

### 第三步：自检与八组梯度检查

~~~bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase selfcheck
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase preflight
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase initial --dry-run
~~~

强制输入/梯度/恢复门失败则先修复；不删除失败记录。preflight不会更新正式checkpoint；GPU恢复试验的10次更新单独列。检查Head四组实际LoRA哈希相同、Qwen四组无LoRA，Body/Face训练，Hand关闭，Depth Adapter关闭。

### 第四步：保留本轮step0结果

~~~bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase initial
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase report
~~~

预期每arm新签名step0 checkpoint、8训练监控＋24val×2seed＋6diagnostic=62张，共496张初始输出。此时不生成正式新test24。

### 第五步：逐128步保存，完成1024更新

~~~bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase train \
  --max-updates 1024 --eval-every 128 --resume
~~~

initial已生成checkpoint_step_000000，因此此命令从本轮step0恢复。也允许指定 --arms qwen_p0 先验证一个arm的执行流程，但正式test仍等待全部8组完成。任意中断后重复相同--resume命令；已有完整结果去重，未完成更新从最新checkpoint/RNG/metrics恢复。

应观察：
每组metrics最终1024行、两轮各512个样本、Body/Face有效梯度、冻结状态不变、step512之后sample来自锁定epoch1排列、warmup不重启。每128步有新增整图与Face/双手局部图；评估只推理，不反传。

资源参考：原全checkpoint约4.1GiB，8组×9step加last会占数百GiB，因此新run放/data。原native单图约14.4秒，3696张仅推理约15小时量级；PE全量可能需要几十小时；8192更新另加模型加载/保存/缓存/报告/评分开销。记录实测，不用上述估值作为训练成功标准。

### 第六步：全组结束后生成独立test提示词与推理

~~~bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase rewrite --splits test
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase cache --splits test
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase finaltest
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase report
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase export
~~~

正式test共24×2seed×2step×8arm=768张；0和1024都在此时评估。同一张test的PG/PB/PBG分别按允许信息重写，GT不进入Qwen参考条件/测试latent。报告主要看预定final1024；不能把测试曲线/结果反用于挑训练步数。

### 第七步：复制完整图片与中文网页到本地

远端export生成results.tar.gz后，在本地PowerShell执行：

~~~powershell
$studyLocal = "D:\codeplus\pictureedit\research\pei2i_body_face_8arm_20261009"
New-Item -ItemType Directory -Path $studyLocal -Force
scp sg:/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/export/results.tar.gz "$studyLocal/results.tar.gz"
tar -xzf "$studyLocal/results.tar.gz" -C "$studyLocal"
Start-Process "$studyLocal/report/index.html"
~~~

导出实现须让tar成员从data/、prompts/、evaluation/、training元数据/、report/直接开始，不多套staging目录。核对export/files.json的文件数量/hash，网页所有相对图片链接都应可本地打开。服务器保留完整checkpoint和缓存，便于后续恢复或新增推理。

## 6. 实施验收和允许得到的结论

- [ ] 8组均记录共同初始化、各自新optimizer、第一步已加载的底座/LoRA和全部1024新增更新。
- [ ] PG完全没有B RGB或人工B描述；PB/PBG的B只给PE与允许的离线/数值条件路径。
- [ ] 每个训练/验证/测试/诊断sample都有P0/PG/PB/PBG条目；PE记录与训练/推理实际text哈希一致，无缺项静默fallback。
- [ ] 原RGB像素未统一缩放；补边、几何资产对齐、内部视觉encoder预处理及最终unpad分别可审计。
- [ ] train的加噪GT监督与eval的高斯初始化被测试分开；验证/测试没有backward或optimizer更新。
- [ ] checkpoint/resume保留optimizer、RNG、锁定order、文本签名、LR进度和日志归档；eval不会改变训练随机状态。
- [ ] 正式更新8192、PE成功重写1698、计划图像3696，完成/失败/未执行分别计数；失效条件不被替换。
- [ ] 中文网页能查看8组各step、原图、固定Face/双手ROI、实际条件、PE原始输入/回答/最终prompt，以及有效样本数。
- [ ] 本地能完整打开导出网页，所有原PNG保留；没有依赖远端绝对路径才能显示的图片。

可以比较：在已有J256初始化、固定Body/Face架构/数据/预算下，不同提示词信息和冻结Head先验对身份、动作、背景及局部细节的影响。

不能由本轮直接推出：Head按官方两RGB协议的最优效果、BFS Body优劣、Depth/Hand Adapter收益、从零训练Adapter的优劣、长prompt必然更好，或几何必然压过文本。只有3个独立test原视频组、单个训练seed，结论应保留这一适用范围。

## 7. 本计划的自审记录

代码层面的主要风险已落实到对应任务：逐文本缓存与Transformer/Adapter一致（任务4/6/8）；训练器256限制/多轮索引/512边界/stop签名（任务1/7/10）；旧val GT latent预缓存（任务6/7）；跨人物TikTok A/B绑定及ROI补边坐标（任务2/8）；Full Face geometry_valid格式与测试不筛选（任务5）；test提前使用/任务重复（任务9）；中文网页、图像/原文转义、离线导出（任务11/12）。

任务中示例测试与补丁是实施起点。GPU路径须用相同项目环境完成preflight与恢复小试验后再进入正式训练；仅文档存在不代表新增CLI已实现或实验已完成。
