# 人物任务 Adapter：实现与使用入口

共享实现位于 [`src/pose_control/qwen21_pro/`](../../src/pose_control/qwen21_pro/)，包含条件准备、人物source bank、官方control stream接入、独立训练/checkpoint、coarse推理与768局部精修。复跑命令、已执行证据和待完成实验见 [`qwen21_pro_adapter_v1/README.md`](../../experiments/6.qwen-controlnet%20union/qwen21_pro_adapter_v1/README.md)。本目录保留设计入口；2026-10-17的完整调研、原始证据、控制层公式与验证清单见 [架构报告](../../research/union_adapter_architecture_20261017/ARCHITECTURE.zh-CN.md)。

实现以官方Qwen-Image-2.1 Union的16个原生控制块和发布权重为主干，保留source图像的原生Qwen3-VL/VAE路径；人物任务组件接到control stream：

1. 以源体型重定向目标动作，提供同camera下的geometry、owner、visibility和真实接触条件。
2. 标准Pose/Depth继续走原129通道；新条件用typed encoder，不覆盖inpaint通道语义。
3. 当前control target ROI hidden查询自己人物的source appearance/Face ID bank，使用显式归属与null/background处理。
4. 在控制链内以独立零输出融合任务信息，保持官方投影、RoPE、时间调制和block-causal规则。
5. 全身控制后，对脸和双方接触手做有几何约束的局部精修，分别报告身份、表情、姿态、接触与背景变化。

旧DeepGen/SD3残差核、六组输出重复注入32层、相加式人物绑定、占位手部条件不会原样复制。已有单人前端/Face编码器仅在权重、输入分布和梯度证据都核对后作初始化候选。

用户选择效果优先、先在85GB单卡验证。四个任务融合位置是初始消融设置，完整16个Union块保留；新分支关闭应恢复同references的官方Union。不能把冻结网络包进no_grad截断新分支，也不能串联两个全零边界。

现有完整发布模型已通过零分支等价、冻结权重、因果prefix、任务分支梯度及4=2+2精确续跑工程门禁，真实单/双人coarse与768精修已出工程探针图。六对样本的200步主训练已完成：工程门禁通过，全局MSE下降5.67%、未过20%门槛；旧200步LoRA为17.7868%，同样失败。按用户最新要求已停止消融与后续出图队列。详见 [改造与样本训练结果](../../experiments/6.qwen-controlnet%20union/qwen21_pro_adapter_v1/report/RESULTS_20261018.zh-CN.md)；权重、缓存和图像留在服务器`/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/`。

使用现有B环境运行 `scripts/run/qwen21_pro.py status --output-root /data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1` 查看记录；训练入口为 `scripts/train/qwen21_pro.py`，本轮不再追加训练或消融。手动测试推理按实验README选择完整冻结代码根、新name与实际checkpoint；当前模型需显式 `--engineering-probe`。每个命令均有 `--help`；正式1536例训练和1024张test门禁尚未实现，不能把六例pilot当作全量训练。


远端pilot复跑使用完整冻结根 `/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1/code_task7b_20261018T161055/`。远端主仓已有较新的torso几何实现，本次部署保留；不要把主仓不同字节的依赖与冻结checkpoint混用。实验族入口见上面的README，里面记录代码签名与当前运行。
