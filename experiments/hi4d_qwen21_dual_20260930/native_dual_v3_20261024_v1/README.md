# Native Dual V3：源码实现，尚未运行验收

本轮在Body21＋Depth V2上接通双人独立几何、共享Scene CHAMP、Relation和源人物身份绑定，保持512维Native Body入口。原Full和单人V1/V2保留。没有运行模型、训练、测试、渲染、缓存构建或迁移，也没有安装依赖。

`source/baseline.json` 与 `baseline_source.tar.gz` 保存sg实施前源码、完整HEAD、UTC和SHA256。`scripts/` 是只做快照／静态审计／源码投递的stdlib工具，`report/` 保存语法、AST、接口和交付SHA记录。

远端同名run位于 `/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/`。架构尺寸、后续环境准备／缓存／迁移／推理命令见 `src/pose_control/qwen21_full/NATIVE_DUAL_V3.md`，其中的示例不是已执行记录。

上游是qwen21_full_native_v1的Body21与Depth V2；本目录属于双人实验族，不覆盖旧Hi4D训练、图像或权重产物。
