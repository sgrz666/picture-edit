# 人物任务 Adapter：推荐架构草案

本目录是2026-10-17的设计入口，**当前没有可运行的新Adapter实现或训练权重**。完整调研、原始证据、控制层公式与验证清单见 [架构报告](../../research/union_adapter_architecture_20261017/ARCHITECTURE.zh-CN.md)。

建议以官方Qwen-Image-2.1 Union的16个原生控制块和发布权重为主干，保留source图像的原生Qwen3-VL/VAE路径；把旧Adapter中核实可用的任务组件迁到control stream：

1. 以源体型重定向目标动作，提供同camera下的geometry、owner、visibility和真实接触条件。
2. 标准Pose/Depth继续走原129通道；新条件用typed encoder，不覆盖inpaint通道语义。
3. 当前control target ROI hidden查询自己人物的source appearance/Face ID bank，使用显式归属与null/background处理。
4. 在控制链内以独立零输出融合任务信息，保持官方投影、RoPE、时间调制和block-causal规则。
5. 全身控制后，对脸和双方接触手做有几何约束的局部精修，分别报告身份、表情、姿态、接触与背景变化。

旧DeepGen/SD3残差核、六组输出重复注入32层、相加式人物绑定、占位手部条件不会原样复制。已有单人前端/Face编码器仅在权重、输入分布和梯度证据都核对后作初始化候选。

用户选择效果优先、先在85GB单卡验证。四个任务融合位置是初始消融设置，完整16个Union块保留；新分支关闭应恢复同references的官方Union。不能把冻结网络包进no_grad截断新分支，也不能串联两个全零边界。

复核与后续使用：阅读报告第4节的控制层顺序及第7节的测试矩阵；通过零分支等价、梯度、person bank交换、坐标和恢复测试后，才进入小规模训练。旧200步LoRA门禁只下降17.7868%，仍为失败，不拿它充当本新方案结果。正式实现放共享`src/`，实验按现有Qwen控制族登记，大产物存`/data/shangguanrz/`，本目录只保留设计与实现索引。
