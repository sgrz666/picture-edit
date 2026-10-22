# CHAMP 式 Native Depth 实施计划

> 当前智能体按 `superpowers:executing-plans` 顺序实施、自查，不使用subagent。用户已批准 `research/qwen21_depth_champ_20261023/REPORT.zh-CN.md` 的独立Depth与残差融合方向并明确要求实施。

**Goal:** 在现有Full frontend和Native Body之间接入可训练、可关闭、有明确输入语义的Depth分支，核验CHAMP有效路径权重与真实Qwen接口。

**Architecture:** 新数据契约固定近1远0，显式valid/canvas/source/camera与尺度状态；原始camera Z与relative PNG分开读取，无隐式fallback。CNN复用CHAMP16/32/96/256节奏，单层256维空间attention，支持1/16默认适配及1/8官方位置对照。新的零投影到512后与原Native geometry残差相加；旧Full fusion、主干和默认路径保持原样。

**Tech Stack:** 现有PyTorch SDPA / Diffusers QwenImage21；CPU契约及真实小型Qwen测试；服务器官方Depth权重、Full权重与7B CUDA有界验证。不安装依赖、不开始正式长训练。

## 1. Typed Depth（先红后绿）

**Create:** `src/pose_control/qwen21_full/depth_condition.py`, `tests/test_qwen21_depth_condition.py`。

- [x] 先测模块缺失；再实现`DepthCondition`的B×1×H×W有限0..1、bool有效区域、canvas、available及元数据检查。
- [x] `camera_depth_condition`只根据有效camera Z生成近1远0；保留canvas上的raw Z、范围、单位/尺度；平面为0.5，全无效明确缺失。多个人共享同场景范围，最近表面合成。
- [x] `relative_depth_condition`必须声明输入方向；PNG内0可为有效前景；禁止按灰度>0推valid。缺失/尺度未知不冒充metric。
- [x] `load_depth_record`显式选择NPY或PNG及独立valid资产，读取后按同一canvas记录做masked resampling与padding；不猜原始尺寸、不拉伸至方形。
- [x] 验证错误方向、有限性、错误metadata、背景哨兵、常量、空条件、矩形canvas与多人的共享范围。

## 2. CHAMP active path 与512残差（先红后绿）

**Create:** `third_party/champ_guidance/spatial_attention.py`, `src/pose_control/qwen21_full/native_depth.py`, `tests/test_qwen21_native_depth.py`。

- [x] 单帧移植CHAMP有效空间Transformer的GN/proj、self-attention、LayerNorm/GEGLU/残差，保留匹配state key；用PyTorch SDPA，不引入旧diffusers依赖。
- [x] `NativeDepthConfig(attention_scale=16)`，可显式8做权重原位置对照；CNN后一次stride2到Qwen的1/16网格，无任意resize。
- [x] official loader仅白名单读取CNN/最后attention，精确验证key/shape/finite；三RGB核求和映射为depth核，valid核初始0；旧320输出head不迁入新512head。其余仅允许已知未使用模块与旧head。
- [x] 新512投影零初始化，融合系数非零；缺失条件最终硬归零；版本化state/config。关闭分支精确返回原features，interaction/person_valid不变。
- [x] 测初始等价、训练后Depth敏感性、矩形和错误网格、有效性/参数梯度、FP32新参数与BF16输入、严格state roundtrip、错误权重拒绝。

## 3. 数据与训练接线

**Create:** `src/pose_control/qwen21_full/native_depth_runtime.py`, `scripts/train/train_qwen21_native_depth.py`, `tests/test_qwen21_native_depth_runtime.py`。

- [x] 使用独立Depth manifest按pair_id绑定深度、valid、camera/units，按dataset给出的target_canvas变换；不修改并行开发的旧loader，也不消费其可能反向的depth字段。
- [x] 专用入口明确使用`Qwen21NativeGeometry`+`Qwen21NativeBody`+新Depth；现有名为native_full的入口仍是旧Full控制器，不在该入口静默改结构。
- [x] 入口提供off/relative_png/camera_z、attention scale、可选官方初始化、noDepth Native warm start、CPU preflight、独立新输出目录。
- [x] 真正训练使用原生Qwen cache和flow目标noise−target，canvas coverage加权；B=1、gradient checkpointing、KV关闭；保存新schema的组件、配置与训练来源，不宣称旧Full整包resume。
- [x] 验证pair对齐、metadata和显式缺失、可执行CPU preflight、配置保存/恢复与小型真实Qwen反向。

## 4. 核验与部署

**Create:** `scripts/audit/audit_qwen21_native_depth.py`, `scripts/audit/verify_qwen21_native_depth_runtime.py`, `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。

- [x] CPU核验官方active path与移植attention数值，加载真实10.7MB Depth权重；真实资产统一语义、参数量、初始无损、训练模拟梯度、state恢复。
- [x] 本地与服务器定向回归，远端新增Python全部py_compile。部署前核对git log并在/data保存时间戳快照；通过scp送新文件，既有文档按hash避免覆盖并行修改。
- [x] 真7B/Qwen4096、矩形target、原Full frontend与Depth分支做有界前向/反向；验证prefix不变、目标有效、冻结基座、Depth及Body梯度。
- [x] 在既有Full Native实验族记录第三步报告和复查命令；更新中文README及第三方来源。测试通过后提交本步新文件，不混入并行canvas修改。

## 验收边界

产物是可用的Depth模块、专用训练入口与架构实测证据。正式训练、生成质量与基线优劣另开run验证。两个attention位置是显式配置；移动attention后仅声称权重映射兼容，不声称与原CHAMP整网功能等价。官方权重不进入Git。

完成记录：本地109项、服务器116项通过；官方attention数值差0；7B/44×24前后向及3次真实更新通过，checkpoint可独立重建。详见experiments/qwen21_full_native_v1/report/STEP3_VERIFICATION.md。
