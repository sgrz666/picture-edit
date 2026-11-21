# 新区域 FM 目标 v1 的设计与 CPU 参考

## 目的

先把新空间目标定义完整：减少背景对 loss 的主导，增强身体关节及每只手的监督。后续主对照是 **无空间加权 uniform vs 新 region_balanced_v1**。本目录不是旧 cap=2.5 的训练结果，也没有把新目标接入真实 Trainer。

## 里面有什么与如何使用

- [完整设计](../../../../docs/superpowers/specs/2026-11-20-region-normalized-fm-design.md)：独立人物/手槽、区域来源与质量系数、软面积下限、像素交集、窗口分母、接入互斥和后续主对照边界。
- [loss_spec.json](loss_spec.json)：两个模式、五个家族系数及 Token 面积下限。标注 `trainer_config_compatible=false`，不能当成现有训练 CLI 配置。
- [CPU 参考脚本](../../scripts/check_region_objective_design.py)：完整计算新目标的图内 ROI 均值和窗口内有效样本均值；不加载模型、不 decode、不执行 optimizer。
- `checks-v1.json`：初始定义的数学检查；后续补充边界检查以新文件保存，不覆盖旧结果。
- [checks-v2.json](checks-v2.json)：当前 15 组 CPU 数值检查，包括补充的零质量不计数、非法零通道拒绝；均已通过。

在仓库根运行，输出文件名必须尚不存在：

```powershell
python experiments/bilibili-single-spatial-train-20261107/scripts/check_region_objective_design.py `
  --output experiments/bilibili-single-spatial-train-20261107/report/region-objective-v1/my-recheck.json
```

成功时 JSON 的 `passed` 为 true。`unverified` 列出真实数据绑定、区域覆盖、Trainer/GPU 和生成效果仍未验证的部分。CPU 参考不能替代这些门禁。

## 已固定的设计与关键结论

新目标系数为 Global .10、Person .30、Joint .15、Hands .35、Background .10。uniform 只有 Global=1。区域模式替换旧 velocity 空间目标；旧像素加权、旧 Face/Hand velocity 与 V7 默认 Person/Contact 项不能重复叠加。其他辅助目标由外层独立配置。

每只手独立保存；图内按有效手槽求均值，窗口内再按有效手图求均值。质量系数放在 ROI 均值外；空家族不把剩余权重自动分给背景。面积下限按完整 Token 计：Person 4、Joint 1、Hands .5、Background 16。

现有缓存把手区合并为 `max(hand_a,hand_b)`，不能恢复逐手归属，必须增加独立 sidecar。既有 fine 区与 coarse 区采取同手优先级选择，不当作两个 ROI。像素归属交集先完成再池化；同一 Token 内两半不相交的支持，池化后再取 min 会制造 .5 的假交集。

第一版 Joint 支持沿用已有 12 个身体点 `(2..13)`，不把它标成 17 点监督；评价可继续使用 Body17。左右手指解剖侧别和稳定人物 ID，不能按画面左右排序。粗可见手区不强制打开缺失的精细 Hand 条件。

## 与已有实验的关系

上游是 Bilibili 空间权重消融、`research/spatial-loss-region-budget-20261117/` 和当前 V7 接口。下游是新的逐 ROI 监督 sidecar、正式 loss 接入以及 uniform/RN 训练对照。原实验缓存、图片和检查点均保持原样；本目录只增加新的设计、JSON 和 CPU 检查结果。
