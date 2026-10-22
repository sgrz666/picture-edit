# Full Native Geometry 第一步实施计划

> 执行方式：遵从用户“不派subagent”，由当前智能体使用 `superpowers:executing-plans` 顺序执行。用户已批准 Full 方案 C，并要求逐步验证、方向问题及时沟通。

**Goal:** 从现有 Full 权重独立恢复几何前端，产出与旧高低尺度融合完全一致的原生512维token，作为下一步主干实时控制接口的输入。

**Architecture:** 保留condition injector、reasoner和geometry/interaction两张高分辨率downsample卷积；不构造SD3控制核，不加载Union，不增加Depth。新出口严格要求条件网格与目标网格相同，不静默resize；原Full入口和权重格式不变。

**Tech Stack:** 现有PyTorch、pytest；本地CPU开发，服务器现有Qwen环境做CPU实权重核验；不安装依赖。

## 本批边界

本批交付可用的几何前端API、严格继承器、测试、实权重对照工具。下一批才接4096→512→4096主干反馈模块；再下一批接真实模型采样、训练和Depth。几何一致性通过不代表新生成效果已经通过。

## Task 1：恢复并确认基线

- [x] 检查主工作区改动，创建独立工作副本与 `codex/full-native-geometry` 分支。
- [x] 阅读Full encoder/reasoner/bridge及已有测试。初始定向测试31通过、1因本地缺配置失败。
- [x] 从已核对服务器复制 `configs/adapter_v6_architecture.json`，SHA256必须为 `361fd62561d665b1a4469f207cd8cfd41a06bfe9f49f0399a914d2405b1324fb`，不修改配置内容；重跑失败测试。

## Task 2：先写契约测试

**Create:** `tests/test_qwen21_native_geometry.py`

- [x] 新模块尚不存在时，先用 `importlib.util.find_spec` 断言入口存在，运行并观察预期失败。
- [x] 用真实Full tiny encoder/reasoner/bridge构造单人、双人、混合batch和矩形网格；验证新token恢复成map后，经旧128维projection得到与旧bridge相同的结果。
- [x] 验证高/低两路梯度都可达；单人interaction严格为零；改变高分辨率输入会改变输出。
- [x] 验证网格尺寸不匹配、非法shape、非有限数拒绝；不把人体mask误作整个目标图输出mask。
- [x] 继承器拒绝缺键、额外前端键、错误shape及错误architecture；只读取四个允许前缀。验证复制后的参数不与旧模型共享存储。

主要测试公式：

```python
legacy_output = legacy.condition_bridge(state)
features = native.project_state(state, target_token_hw=state.geometry_feature.shape[-2:])
geometry_map = features.geometry.transpose(1, 2).reshape_as(state.geometry_feature)
torch.testing.assert_close(legacy.condition_bridge.geometry_projection(geometry_map), legacy_output.geometry_scene, rtol=0, atol=0)
```

## Task 3：实现独立几何前端

**Create:** `src/pose_control/qwen21_full/native_geometry.py`

- [x] `NativeGeometryFeatures` 明确geometry/interaction `[B,N,D]`、`target_token_hw`、`person_valid`；默认D=512，测试允许小型reasoner配置。
- [x] `Qwen21NativeGeometry`持有原injector、reasoner、两张stride2卷积。前向调用原reasoner，计算：

```python
geometry = state.geometry_feature + self.geometry_high_downsample(state.geometry_highres)
interaction = state.interaction_feature + self.interaction_high_downsample(state.interaction_highres)
interaction = interaction * state.interaction_valid[:, None, None, None]
```

- [x] 两个map按原生行优先顺序flatten成tokens。要求实际网格等于调用方传入目标网格，不做插值、latent转换或1536维计算。
- [x] `from_full_adapter`独立加载对应模块的FP32 CPU参数；`from_full_checkpoint`严格读取 `qwen21_full_v1` payload，以四个明确prefix加载；包含加载/忽略键数报告。恢复本阶段原生权重时用独立schema，配置需匹配。
- [x] Depth及非native stem目前明确拒绝，防止隐式换输入约定；不修改旧Full类/默认构造。
- [x] 运行新增测试至全部通过，再运行任务1的基线测试。

## Task 4：服务器实权重核验

**Create:** `scripts/audit/audit_qwen21_native_geometry.py`

- [x] 工具使用显式 `--checkpoint`，CPU、`torch.load(..., mmap=True)`；读取已有Full权重，按前缀构建新前端，不触碰optimizer或写checkpoint。
- [x] 在同一权重下用旧injector/reasoner/bridge作为参照，比较新输出再经原128投影后的结果。输入为固定合成有效条件，明确不声称真实图片效果。
- [x] 对32×32和32×64条件分别输出特征shape、max_abs差、加载参数量、schema、实际源码根。JSON仅在明确新路径下以独占创建写出。
- [x] 通过scp投递有清单的新增源码/测试/审计脚本，先记录远端HEAD和时间戳；旧源码不覆盖。远端先语法编译，再CPU契约与实权重核验。

## Task 5：文档、复核和交付

**Modify:** `src/pose_control/qwen21_full/README.md`、`scripts/audit/README.md`、`tests/README.md`、`docs/superpowers/plans/README.md`。

- [x] 写出API使用、checkpoint继承边界、网格要求、测试和审计命令。
- [x] 当前智能体独立复核加载完整性、旧模型参数未变、零/非法输入、梯度路径和实际import来源。
- [x] 记录测试与远端审计结果，更新本计划；仅提交本批文件，保留用户主工作区其他改动。

## 后续阶段

下一阶段以这里的 `[B,N,512]` 为输入实现主干当层反馈、target metadata、checkpoint重算和零输出等价验证。只有完整采样能保持现有Full控制/外观，才进一步比较注入层数并增加Depth。若旧前端严格继承失败、画布契约冲突或实际效果回退无法定位，暂停相关方向并向用户沟通。
