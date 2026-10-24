# Native Dual V3 实施记录

用户确认：完整双人闭环、一张双人参考图、原生Qwen64通道latent、FCL真实表面距离、旧单人路径独立。按后续“实施”指令写源码；保留不安装、不运行模型／测试／渲染／缓存／迁移／训练的边界。本人实施，不使用subagent。

- D0–D1：按模型家族验证的Body21、pelvis中心相机变换、来源绑定、独立表面和Typed条件。
- D2–D3：Scene CHAMP一次、共享Detail/Mixer/Gate、8通道Raster、196→28关系Token、12维上下文Interaction融合。
- D4：原生64通道源latent、身体32＋头部16个Token、稳定人物映射、14维上下文身份融合。
- D5：独立runtime/schema、12组参数、严格V2迁移／V3重建、数据／推理／训练入口。
- 交付核对限语法、AST、SHA和静态接口；后续验收不在本轮执行。

完整数据契约、架构尺寸、使用命令和未执行验收清单见 `src/pose_control/qwen21_full/NATIVE_DUAL_V3.md`。
