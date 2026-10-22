# 第一步验收记录（2026-10-22）

现有Full几何前端已经独立提取，当前生成入口仍使用旧Full。新增模块尚不负责Qwen残差注入。

## 实权重对照

- checkpoint：`/home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt`。
- SHA256：`6aa320cf7e718f824c047594ad8ad04e44385c3249328406e79a75baf5d28b2d`；4,009,974,391字节；审计前后文件stat不变。
- 从四个允许前缀加载329个张量、75,966,284个前端参数；旧SD3控制核、Face/Hand等648项被明确排除。
- 单人32×32、单人32×64、双人32×64三组固定合成条件：原512维融合特征和再经旧128维投影的最大绝对差均为0。
- 原始报告：[body_checkpoint_audit_20261022.json](body_checkpoint_audit_20261022.json)。这验证权重/特征迁移，不是自然图像效果评测。

## 测试

本地Python3.13 / torch2.12.0+cpu：

```bash
python -m pytest tests/test_qwen21_native_geometry.py tests/test_condition_bridge_contract.py tests/test_adapter_reasoning_v1.py tests/test_qwen21_native_resolution.py -q
```

结果：**58 passed**。其中基线缺失的`configs/adapter_v6_architecture.json`从服务器恢复，内容SHA256为`361fd62561d665b1a4469f207cd8cfd41a06bfe9f49f0399a914d2405b1324fb`，配置未改写。

远端现有Qwen环境Python3.12 / torch2.8.0+cu128，全部在CPU执行：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m pytest tests/test_qwen21_native_geometry.py tests/test_qwen21_full_bridge.py -q -p no:cacheprovider
```

结果：**35 passed**，另有现有Swig弃用与NVML初始化警告。

扩大到`tests/test_qwen21_full_preservation.py`时为**43 passed、1 failed**。失败为旧`tests/test_v66_hand_core.py:16`仍导入`scripts.precompute_iper_hand_features_v66`，实际文件已迁到`scripts/data/precompute_iper_hand_features_v66.py`；本次没有改这些旧测试或手部代码。故不宣称全仓回归全部通过。

## 调试与边界

混合batch曾在旧128投影后出现`2.384185791015625e-7`差异。定位为token还原map后的channels-last布局触发不同CPU卷积路径；512维特征本身差为0。核验恢复原NCHW布局并直接捕获旧projection输入比较，保持`atol=rtol=0`，没有放宽容差。

另补充meta-context恢复测试，防止在空权重上下文内`load_state_dict`无效复制；权重继承器显式建立FP32 CPU参数。

服务器部署前快照：`/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/geometry_step1_20261022_v1/source_before_20261022.tar.gz`。本轮新增源码、测试与审计入口；原模型源码/权重未覆盖。README同步了新接口说明。
