# Full Native第三步：Depth编码与融合验收

## 目的

验证CHAMP式独立Depth分支能以明确输入语义接入现有512维Native几何，并在冻结Qwen7B上前向/反向和真实训练。原Full继续固定为对照。

## 里面有什么

- `data/pairs.jsonl` / `depth.jsonl`：2个固定TikTok pair及明确的Depth/SMPL有效区域来源，不拷贝媒体。
- `preflight_v1/preflight.json`：2个pair的CPU画布与Depth来源检查。
- `report/cpu_depth_audit_v2.json`：官方active权重、原版空间attention数值、参数、初始化和梯度。
- `report/gpu_depth_7b_v2.json`：真实条件的44×24矩形网格、7B prefix/target和冻结/梯度检查。
- `report/checkpoint_smoke_v1.json`：真实3次更新及新checkpoint的weights_only读取、独立重建。
- `training_smoke_v1/`：3次更新的metrics、cache和新checkpoint；只是工程冒烟，未作为质量基线。
- `source/runtime_and_canvas_20261022T174927Z.tar.gz`与`delivery_20261022T174927Z.json`：51份实际文本源码和SHA256，包含当前canvas依赖，不含媒体/权重/cache。其他before快照保存部署前源码或文档。

## 结论与怎么用

本地109项、服务器116项定向回归通过。官方33个有效张量已映射，空间attention与原类最大差0；新分支2,254,656参数。真实7B上初始/zero-Depth/prefix差均0；激活新head后Depth与Body都有梯度。3次真实更新实际抽到1个pair，第2步Depth head开始学习、第3步CNN开始学习。checkpoint可不依赖原Full文件重建。

使用说明在服务器仓库`src/pose_control/qwen21_full/NATIVE_DEPTH.md`，完整验收在父实验`report/STEP3_VERIFICATION.md`。先执行CPU `--preflight-only`；所有output必须选新目录，重复审计也换新JSON文件。正式对照要固定pairs、canvas、loss、seed和预算，不能把3步冒烟当作效果证据。

## 与上游的关系

继承第一步Full frontend和第二步Native Body；新增Depth数据契约、CHAMP式encoder与512残差融合。训练入口使用服务器现有canonical canvas代码；单独检出旧Native分支需要集成该依赖，快照供独立checkout复现。旧Full默认训练入口与checkpoint未改动。
