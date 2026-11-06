# PromptHMR Hi4D 配置补齐与四图运行

用户已在本聊天批准补齐配置并实际运行。目标是在原四模型拼图的相同 4 张 Hi4D 图上补出 PromptHMR 的真实双人 SMPL-X、投影与 normal，并保留所有旧结果。

源码使用服务器 `/data/shangguanrz/models/two-man/M1_PromptHMR/PromptHMR`，权重使用其 `data/phmr/checkpoint.ckpt`，不修改模型目录。运行环境新建于 `/data/shangguanrz/pic-edit-runtime/prompthmr-hi4d-20261106/env`，继承 mamma 的 PyTorch 2.7.1+cu128，只在新环境安装匹配的 xformers 0.0.31；不修改 mamma、qwen21 或 MultiHMR2 环境。

实验代码本地放 `experiments/mete-test/hi4d-prompthmr-comparison-20261106`，服务器对应 `experiments/9.hi4d/mete-test/hi4d-prompthmr-comparison-20261106`。大产物位于 `/data/shangguanrz/pic-edit-experiments/mete-test/hi4d-prompthmr-comparison-20261106`。

4 张样本固定为 `809de42848f7edd5`、`be3143ff3423db7c`、`3da3afa1d3751ba5`、`09fb91c00c68d3cc`。输入从既有 GT-free `fit_records.json` 取 RGB、SHA256、原图尺寸和真实 K，从既有自动 `selected.json` 加原始 MultiHMR 预测取两个不同人物框。明确标注为“PromptHMR + MultiHMR 自动框”，不宣称 PromptHMR 自带检测的端到端性能。GT 框、mask、关节不进入推理。

固定 text=None、masks=None、use_mean_hands=True。串行跑 `box-only`（interaction=False）与 `interaction`（True），各 4 张，分别记录产物和耗时；不依据样例结果改超参数。严格加载完整 checkpoint，允许跳过构造阶段重复的 CLIP 预训练下载，前提是完整 checkpoint 的 strict=True 加载零缺失/零额外键。

每帧保存 SMPL-X 顶点、关节、pose、betas、平移、原图 K、模型 K、2D 原图投影、模型 faces、输入/配置/模型 SHA256、成功或错误状态。独占创建，失败保留分母；重跑换名称。保存正确的缩放和 pad 逆变换，不将模型画布坐标直接叠到原图。

验证分三层：环境实际 CUDA attention 小测试、输入白名单/框/投影契约测试、4 张图的完整模型加载与有限值/正深度/投影一致性验证。只用这 4 张做可视化试跑，不将其称为完整 Hi4D 泛化评测。最终补齐四模型同图对照，并交付配置路径、复现命令与限制。
