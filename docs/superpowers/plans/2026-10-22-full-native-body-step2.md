# Full Native Body 第二步实施计划

> 执行方式：当前智能体使用 `superpowers:executing-plans` 顺序实现与自查，不派subagent。承接第一步 `b10904cb`；用户已批准方案C并要求继续。

**Goal:** 使原生512维几何特征通过Qwen主干当层4096维hidden产生动态target残差，验证零输出等价、冻结主干梯度和checkpoint重算。

**Architecture:** 共享query下投影、几何局部融合和时间条件，32个注入位置使用6组零输出head。只在原生Qwen block完成后更新target；保留原生forward、参数对象及原有state key。逐实例接入，不改安装包或全局类；本次执行对象显式捕获进checkpoint函数，避免反向重算读取过期上下文。

**Tech Stack:** 现有PyTorch/Diffusers；本地及服务器小型32层原生Qwen模型CPU测试。首轮single-card、B=1、KV cache关闭；不加载Union，不启用Depth，不开始正式训练。

## Task 1：动态控制模块（先红后绿）

**Create:** `tests/test_qwen21_native_body.py`、`src/pose_control/qwen21_full/native_body.py`。

- [x] 先验证模块缺失导致测试失败。
- [x] 实现版本化 `NativeBodyConfig`：hidden4096、geometry512、width512、layers32、groups6、默认全部层；正整数、唯一有效注入位置明确校验。
- [x] `prepare_geometry(features)` 把已训练geometry通过独立norm/projection和depthwise局部卷积变为共享512维上下文。每个transformer.forward计算一次，保留梯度。
- [x] 每层使用 `q=Down(Norm(H_target))`；融合geometry、geometry对q的调制、独立层号和sigma编码，经分组 `ZeroHead` 返回hidden维残差。初始只有最终head为0，上游正常初始化。
- [x] 输出head与前端参数保持FP32计算；隐状态可BF16。错误dtype要明确拒绝，避免整体dtype转换静默损伤新参数。
- [x] 验证零head等价、更新head后query/geometry梯度非零、改变hidden/geometry/layer/sigma会改变输出、valid画布mask有效。target-only不误作body-only。

## Task 2：原生Qwen逐实例接入

**Create:** `src/pose_control/qwen21_full/native_integration.py`。

- [x] 只接受原生 `QwenImage21Transformer2DModel`，校验层数/宽度/causal；拒绝Union、重复安装、旧Full上下文并用。
- [x] `install_native_body(model, adapter)`冻结原主干后注册根级`native_body`；不重新构造block，不改变原参数对象/state key。逐实例mixin在原block输出后执行target更新。
- [x] `forward(..., native_geometry=features, native_enabled=True, native_valid_mask=None)`根据实际`img_shapes/img_mask`调用原生metadata构建target mask；正负CFG独立验证布局。
- [x] 为每次forward建立不可变execution，包含target mask、sigma、几何上下文、有效画布。checkpoint调用封装显式捕获execution；退出forward后清理临时上下文和checkpoint封装。先后两次forward再backward不得串条件。
- [x] 首轮明确拒绝KV cache与B>1。所有新分支关闭恢复原生模型；prefix直接写入为零，后续prefix因果计算也不受target影响。

## Task 3：真实原生block结构验证

- [x] 用32层小宽度 `QwenImage21Transformer2DModel`，保留真实图文交错slot、RoPE和block-causal attention。
- [x] 与同权重未接入模型比较zero/disabled输出逐值相同；新分支非零时prefix保持相同、target改变。
- [x] 比较启用/关闭gradient checkpointing的两个条件forward及梯度，验证返回后backward、上下文清理和异常清理。
- [x] 验证原模型state key/参数对象未变；所有原主干参数无梯度，所有新分组head能学习。
- [x] 版本化adapter state；不宣称旧Full checkpoint可以整包strict恢复新Body接口。

## Task 4：服务器复核与记录

- [x] 新增脚本 `scripts/audit/audit_qwen21_native_body.py`，CPU小型原生Qwen真实结构审计，输出版本、参数量、zero/prefix差、梯度和源码SHA。
- [x] 本地测试通过后，scp新增模块/测试/脚本。修改文档前保存第二步时间戳快照，产物写到同一Full Native实验的数据盘新run。
- [x] 在服务器现有Qwen环境执行相同测试和审计，不安装依赖、不更新主干或旧实验权重。
- [x] CUDA最小核验通过且显存充足后，使用`verify_qwen21_native_body_runtime.py`对现有7B/512做一次有界前向/反向；不建optimizer、不采样图片，结果与CPU小模型报告分开记录。
- [x] 更新README/验收记录并提交本步代码；第一步实权重前端＋本步真实Qwen结构共同建立连接基础，后续完整7B采样/训练仍须独立验收。
