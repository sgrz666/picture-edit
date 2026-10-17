# fulladapter —— 现有 Qwen-Image-2.1 适配器代码快照（融合改造工作区）

## 里面有什么

2026-10-17 从 `src/pose_control/` 原样拷来的两套适配器代码，`diff -r`（排除 `__pycache__`）实测与源目录**逐字节一致**。

| 目录 | 规模 | 是什么 |
|---|---|---|
| `qwen21_full/` | 56 个 `.py` + 6 个 `README.md`，656K | 单人全量适配器 `Qwen21FullAdapter`。顶层入口 `execution.py`、`joint_training.py`、`controlled_pipeline.py`、`qwen_adapter.py`、`control_core.py`、`condition_injector.py`、`checkpoint.py`；下挂 `detail/ face/ hand/ interface/ reasoning/` 五个分支 |
| `qwen21/` | 7 个 `.py` + 1 个 `README.md`，64K | 双人/身体/面部轻量适配器线：`body_adapter.py`(8904B)、`dual_adapter.py`(11082B)、`face_adapter.py`、`dual_inference.py`、`runtime.py`、`action_prompts.py` |
| `SNAPSHOT.sha256` | 70 个文件指纹 | 快照落地当时的逐文件 SHA256，用来回答"我到底改了哪几个文件" |

## 与上下层的关系

- **出处**：远端 main 分支 HEAD `17991d3`（2026-10-14 13:19 `fix(dual-appearance): make the square crop exactly square...`）。拷贝前 `git status` 确认这两个包的 `.py` 相对 HEAD 无改动（只有 19 个未跟踪的 `README.md`）。
- **正式代码仍在 `src/pose_control/`**。这里是只读参考 + 改造草稿区：所有 `import` 保持源样未改，因此本快照**不能独立运行**。这是有意的——避免出现"改了 union-adapter 里的副本，实验却仍跑 `src/` 原版"的分叉（AGENTS.md 红线 4）。改完要么回写 `src/`，要么显式做包名替换再跑。
- **与 Union 的分工**：`../Qwen-Image-2.1-Fun-Controlnet-Union/` 是外挂控制分支（`control_img_in` + 16 个 `control_blocks`，180 tensor 全 BF16，残差式注入）；这两套 fulladapter 是把条件写进 transformer 内部的适配器线。融合点大概率在 `qwen21_full/control_core.py`、`condition_injector.py` 的条件注入路径与 Union 的 `control_blocks` 输出相加处。
- **上游实验**：`workflow/experiment/qwen-union/`（Union 六组单样本姿态重定向，T2a Body PCK 0.8889 / T4b 0.6111，对照组 Native Qwen 0.3333）与 `workflow/experiment/qwen-union-hybrid/`（双轨融合原型）。

## 注意与坑

- **没拷进来的依赖**（快照按纯代码处理，跑的时候仍要命中 `src/` 与 `third_party/`）：
  - `qwen21_full` → `src.data.{iper_dataset, letterbox, tiktok_dataset}`、`third_party.v65_face.{mcld, stableanimator, visual_persona, xdyna}`
  - `qwen21` → `src.data.hi4d_qwen21_dataset`、`src.pose_control.v6.{condition_bridge, condition_injector, reasoning, token_encoders}`（双人线直接复用 `v6/`  internals，只拷 `qwen21/` 是跑不起来的）
- **同名陷阱**：`qwen21_full/` 的顶层文件名与 `v6/` 高度对应（`conditions.py` / `control_core.py` / `condition_injector.py` / `checkpoint.py` / `deepgen_adapter.py` / `controlled_pipeline.py` / `experts.py` / `geometry_encoder.py` / `token_encoders.py`），它是复制后迁移的版本；`detail/face/hand/interface/reasoning` 五个分支与 `v6/` **逐文件字节一致**。在 fulladapter 里改这些分支，`src/` 下的 `v6/` 和 `qwen21_full/` 两份都不会变。
- **dual v8 不在这里**：main 分支上没有 v8 实现代码，它在 `origin/codex/qwen21-dual-v8` 分支；v8 只有 568MB 权重与评测产物留在 `experiments/4.two man /qwen21_dual_v8/pilot_pair14_v8_300_v1/`。本次未拷 v8。
- **`smplx_global` 字段名不副实**：实际 26 维 = betas[10] + root rot6d[6] + transl[3] + cam[7]，不含任何关节姿态，只做全局形状/朝向/位移/相机条件。
- **Git**：根 `.gitignore` 里 `/union-adapter/*` 挡住了权重目录，但 `!/union-adapter/fulladapter/` 单独放行，所以这份代码快照是**可入库**的；包内若之后落了 `.pt` / `.safetensors`，仍被全局规则忽略。
