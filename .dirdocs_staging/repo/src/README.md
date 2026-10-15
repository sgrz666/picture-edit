# src/ —— 核心算法库：姿态 Adapter 网络、控制信号生成、数据集读取与管线桥接

> 工作区路径：`/home/shangguanrz/project/pic-edit/src`　|　本级：6 个子包 / 3 个本级文件 / 3.3 MB　|　子树规模：127 个 `.py`（实测 `find src -name '*.py' -not -path '*__pycache__*' | wc -l`），全部被 git 跟踪（`docs/架构审查_20261012.md` 统计表）
> 数据来源：本文件数字为 2026-10-12 对远端工作区只读实测；行数取自 `wc -l`。

## 一、这一层在干什么

`src/` 是全项目的"算法心脏"：所有可复用的神经网络模块（姿态 Adapter、Token 编码器、面部监督）、3D 控制信号生成器（SMPL-X 渲染骨架/法线/接触图）、PyTorch Dataset 读取协议、以及把 Adapter 残差挂进扩散 Pipeline 的桥接代码都在这里。它**不含任何训练入口**——入口统一在 `scripts/`（2026-10-12 起 `scripts/` 为唯一入口层，见 git 提交 `3836d19`）。上游由 `configs/` 提供架构超参契约，下游被 `scripts/`、`tests/`、`workflow/`、`bfs_explore.py` 等以 `from src.xxx import ...` 方式消费（无打包安装，依赖 `sys.path` 注入，全仓 229 处，属已登记的 P2 债务）。

## 二、层级地图

| 子包 | 装什么 | 内容实证（实测行数） | 与其他子包的关系 |
|---|---|---|---|
| `adapter/` | 最早期 CNN 姿态适配器基线（`PoseConditionAdapter`，零卷积残差注入） | `pose_adapter.py` 165 行，含 `ResnetBlock2D`/`Downsample2D` | 历史基线，被 `v6` 架构取代，保留供论文对照 |
| `controls/` | SMPL-X → 控制图（骨架/法线/深度/接触图）生成器与相机模型 | `generator.py` 344 行，`Camera`+`ControlSignalGenerator` 两类 | 为 `pose_control/*` 准备输入几何，被 `scripts/data/*` 预计算脚本调用 |
| `data/` | iPER / TikTok / Hi4D 的 PyTorch Dataset 与 letterbox 变换记录 | 4 个实现共 1,781 行（`tiktok_dataset.py` 999、`iper_dataset.py` 297、`hi4d_qwen21_dataset.py` 283、`letterbox.py` 202） | 被 `pose_control/qwen21*` 训练运行时消费；数据实体在 `datasets/` |
| `evaluation/` | 评测器：DWPose ONNX 推理 + 四臂身份/姿态指标 | `joint_pilot_metrics.py` 89 行（自 `eval_iper_four_arm.py` 追踪复制，头部有 SHA256 记录）；`dwpose_onnx/`（`onnxdet.py`/`onnxpose.py`/LICENSE） | 服务于 `scripts/eval/`，指标口径与 `experiments/*/summary.json` 一致 |
| `integration/` | CHAMP 样本校验 + DeepGen 无控制基线调用 | `champ_deepgen.py` 206 行（`load_champ_sample`/`call_deepgen_without_control` 等） | 服务 `deepgen_no_control_ablation` 实验与 `third_party/champ_guidance` |
| `pose_control/` | 姿态控制核心，三条演进线 `qwen21` / `qwen21_full` / `v6` | 见其本级 README；`qwen21_full` 10,795 行、`v6` 7,715 行、`qwen21` 801 行 | 本层的主体；被全仓大量导入（src+scripts+tests 内 `qwen21_full` 44 处、`v6` 35 处、`qwen21` 18 处） |

`__pycache__`、`__init__.py` 为 Python 机制文件，不在说明范围。

## 三、版本谱系与取代关系

- **DeepGen 线**：`adapter/`（CNN 基线）→ `pose_control/v6/`（V6.3 统一控制，24 层 MMDiT 注入，见 `configs/adapter_v6_architecture.json` 的 `version: "6.3-dynamic-deepgen-interface-v1"`）→ V6.5/V6.6 面/手能力作为 `v6/face/`、`v6/hand/` 子包生长在同一树内。
- **Qwen 线**：`pose_control/qwen21/`（轻量 Body/Face/Dual 适配，当前产出 PSNR 25.86 dB、PCK 98% 的主线之一）与 `pose_control/qwen21_full/`（把整套 V6 架构字节级复制后加 Qwen 边界适配层）。`docs/架构审查_20261012.md` P1-1 已实测确认：`v6/` 的 47 个 `.py` 与 `qwen21_full/` 同名文件**逐字节相同**（本次抽样 `control_core.py`、`deepgen_adapter.py`、`face/adapter.py` md5 一致），`qwen21_full` 仅新增 `qwen_adapter.py`(660)、`execution.py`(832) 等 9 个边界文件。改注入层内核必须两处同改。

## 四、注意事项 / 新人阅读入口

1. **先读**：`src/pose_control/README.md` → 按你在跟的实验线选读 `qwen21/` 或 `qwen21_full/`；做数据/评测则先看 `src/data/`、`src/evaluation/`。
2. **不要单独修改 `v6/` 或 `qwen21_full/` 的复制内核**，两者是 fork 镜像（P1-1），任何修复需 `diff -rq --exclude=__pycache__ src/pose_control/v6 src/pose_control/qwen21_full` 复核。
3. `adapter/` 与 `integration/` 是历史/消融用途，日常 Qwen 训练链路不经过它们，勿当作现役实现。
4. 本包路径契约由 `tests/` 60 个测试锁定（见 `tests/README.md`）；`configs/adapter_v6_architecture.json` 是 V6 结构单一事实源，改通道数/层索引要同步该 JSON。
5. 全仓叙述口径以根 `README.md`（人工手写，禁改）为准；其中 3.1 节树状图个别文件名（如 `base_dataset.py`、`metrics.py`）与工作区实况不符，本 README 的清单以实测为准。
