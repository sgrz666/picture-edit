# pose_transfer/ —— 单人姿态迁移的核心 Python 实现

本目录是 `workflow/` 两阶段基座流水线的"引擎"：把目标人物 B 的姿态估计、与源人物 A 的对齐、骨架渲染、提示词编译等纯 Python 逻辑集中在此。`run.py` 与各 stage worker 通过 `import` 调用本目录，不依赖任何训练代码或 adapter/LoRA。

## 里面有什么

- `__init__.py`：模块 docstring，定义"用原生 Qwen + SMPL-X 姿态引导做单人姿态迁移"。
- `common.py`：阶段血缘（stage provenance）、缓存指纹（sha256）、画布预处理（letterbox / canvas_size）与 estimator 沙箱封装。核心概念是把一次请求拆成 `identity`（设置、代码哈希、模型身份、输入内容哈希——可跨运行复用）与 `runtime`（本 run 绝对路径——移动 checkout 不应使其失效）两部分；stage 产物相对 run 目录记录。顶层函数有 `sha256_file`、`fingerprint`、`atomic_json`、`make_request` 等。
- `estimator.py`：唯一调用 vendored SMPLest-X 估算器的入口。`REQUIRED_PARAMETER_KEYS` 校验 12 个 SMPL-X 参数（global_orient / body_pose / betas / transl / focal_length_xy 等）；`estimate_single` 在 run 本地沙箱里跑单图估计并返回已校验参数；`project_camera`、`reconstruct` 负责相机重建。共享估算器根只读，权重以 symlink 复用。
- `geometry.py`：纯 CPU 的 SMPL-X 姿态引导几何辅助。`_BODY18_FROM_SMPLX137` 把 SMPL-X 137 点降到 body-18，`_FACE68_FROM_SMPLX137` 取面部 68 点；`project_camera` 用针孔内参投影（深度分母采用与 SMPLest-X 一致的 `Z+1e-4` 约定，非有限/非正深度返回 NaN）。另有 `align_to_source_bbox`、`render_poseguide`、`smplx137_to_dwpose134`、`valid_points`。`align_to_source_bbox` 的 `framing` 参数有两种定标：`source_person`（默认，按 A 的人体检测框内切缩放——B 服装轮廓撑大检测框时会被宽度主导把骨架缩小，hard_05 实测尺度 0.410）与 `source_torso`（按 B 的实测躯干映射到 A 的期望躯干，hard_05 尺度 0.977，躯干错配从 2.35× 回到 1.02×）；当前默认值未改，因为其余测试对两种定标差别不大，选哪种由调用方按 `--framing` 决定。
- `prompts.py`：PE-I2I 提示词编译。`PE_ROLE_SYSTEM_SUFFIX` 规定最终 `rewritten_prompt` 不得出现 `<image2>` 或对 target 的引用，只能引用 `<image1>`（source）与 `<image3>`（骨架），下游把 `<image3>` 映射成 `<image2>`；`prop_policy` 仅支持 `source_only`（测试 B 的篮球等道具不进生成图）。顶层 `pe_instruction`、`parse_pe_answer`、`compile_prompt`。

## 与上下层的关系

- 上游：`../run.py` 用 `common.make_request` 生成请求，`config.json` 提供默认设置。
- 下游：`../smplx_worker.py`、`../pe_worker.py`、`../generate_worker.py` 分别 import `estimator`/`geometry`、`prompts`、`common`。
- 产物消费：每次运行的 `runs/<run>/geometry|prompt|generation/stage.json` 由本目录指纹逻辑写入与校验。
- 相关：`../README.md`（基座流水线正稿，2026-10-12 审查整改后的实况与逐对指标以其为准）、`../runs/README.md`。

## 注意与坑

- `identity` 侧不含绝对路径，`runtime` 侧含绝对路径；改路径不应让缓存失效。
- 提示词契约强制"不引用 target"，若 PE 输出违反会被 `validate_generator_prompt` 拦截。
- estimator 沙箱保证共享 SMPLest-X 根不被写入，重跑安全。
