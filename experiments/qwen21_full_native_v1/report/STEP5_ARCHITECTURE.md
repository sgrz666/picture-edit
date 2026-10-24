# 第五步：Native Depth V2 架构已接通，尚未运行验收

2026-10-24，沿用 sg 当前已部署 Body21/Depth V1；源码快照记录 UTC 时间、HEAD `11a5361`、未提交状态及 SHA256。本地继续使用 `codex/full-native-geometry` 工作树。本轮本人实施，没有派 subagent。

## 完成内容

- DepthConditionV2 与 DepthFusionContext：独立Z/Relative/Geometry有效性、相机、FK pelvis深度、同beta身高、Normal/Part/SMPL mask、confidence可用性及来源；支持to、index_select、混合模态stack。
- 源网格几何：Root-relative Z、归一化坐标差分、Laplacian、jump、轮廓、相机射线、仅连续有效区域的Depth–Normal关系；Normal基显式转换，PNG物理通道关闭。米制路径额外要求同Pose记录的尺度标定系数。
- 模型：原CHAMP2通道分支保留；23通道Residual CNN输出256，concat512→Mixer256，29维上下文生成512维逐位置/通道Gate，只有最终256→512输出投影置零。输出仍为 `[B,N_target,512]`。
- 数据读取、collate、runtime、推理和训练配置接通。七组参数可独立配置，Qwen/Body21/Native Body接口和主干序列保持原方式。
- runtime v3＋Depth v2＋condition v2；旧checkpoint按已保存版本重建，升级须显式调用。迁移检查旧参数，仅继承CHAMP Encoder，重新初始化融合模块并重置输出头。Full/Depth-off源要求明确的CHAMP权重。
- Camera-Z导出工具直接读取已有目标mesh_cam，不重新估计人体；仅创建新NPY与版本化manifest。旧数据和权重没有被覆盖。

固定维度：CHAMP `2→256`，Detail `23→16→32→64→128→256`，Mixer `512→256`，Gate `(64+64+32)=160→128→512`，投影 `256→512`，空间均在目标1/16网格衔接。

## 本轮做了什么核对

`depth_geometry_step5_20261024_v1/report/static_architecture_final_v4.json` 保存stdlib AST与语法核对，`remote_syntax_delivery*.json` 保存远端逐文件SHA与语法编译结果。静态声明统计为23 Detail通道、29 Gate上下文通道。源码对照确认原 `native_depth.py`、`depth_condition.py`、`native_geometry.py`、`native_body.py`、`body_pose.py` 字节未变。AdamW构造、velocity loss AST未变；训练循环除允许冻结时记录None梯度的两项日志赋值外，AST相同。

这些结果只证明源码/声明/语法层面的核对。没有import项目、实例化模型、执行前向/反向、运行测试、创建optimizer、训练、渲染补缓存或执行权重迁移；没有新权重或生成图。本轮不引用第四步的152/59项测试作为Depth V2的运行验收。

## 使用与后续

完整契约、维度、命令和未执行验收清单见 `src/pose_control/qwen21_full/NATIVE_DEPTH_V2.md`。后续先准备版本化Camera-Z/PNG manifest，再进行数据、模型、checkpoint与梯度验收；当前不作生成质量、耗时或显存结论。P4几何损失和S0–S4训练日程未纳入本轮。
