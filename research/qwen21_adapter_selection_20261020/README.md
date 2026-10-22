# Qwen 2.1：几何控姿与人物身份保持的架构复核

研究日期：2026-10-20。目标是核对 Full、Pro 与 Alibaba PAI / VideoX-Fun Union 的代码和实验，判断如何同时改善姿态与身份保持。参数量作为成本因素，不作为首要筛选条件。

## 里面有什么

- [研究报告](REPORT.zh-CN.md)：Qwen原生机制、三种实现、已有实验、候选改造与公平验证顺序。
- `evidence/`：服务器Pro的18位置固定评测、200步gate、样本FaceID覆盖、512/1024旧Face实验汇总、既有骨架参考消融图、本次核算结果，以及研究期间已有作业产出的首张Pro验证图。图片是原产物的副本，没有做编辑。
- `source/`：固定版本官方Transformer/control pipeline和旧DeepGen配置，只供审阅，不是独立运行包。
- `scripts/audit_evidence.py`：用标准库重算指标并校验SHA256；可选用torch meta统计当前Full/Pro参数，不加载权重。
- `manifest.json`：采集来源、日期、版本与文件SHA256，区分本次材料与仓内引用。

## 关键结论与使用

建议以Pro/Union接入为实验平台，分开几何与身份路径并补齐有效身份监督，尚不能宣称它效果优于Full。Pro 200步global flow误差下降5.67%，face区域下降31.86%，但8个人物仅2人有有效FaceID；没有把这两个loss下降解释为身份指标改善。旧512四臂共同30脸中，Body→加源脸裁图/角色提示的ID为0.3953→0.4719，再加Face为0.4802。

当前Full去重参数373,099,568，另有226,492,428个buffer元素；Union增量3,775,479,808参数；Pro在完整Union上再加14,646,784个任务参数。不能混用checkpoint元素总数、训练参数与推理总规模。

从仓库根目录运行：

```powershell
python -X utf8 research/qwen21_adapter_selection_20261020/scripts/audit_evidence.py --verify
```

需要参数复核时增加`--parameters`，使用已有torch/diffusers，无需GPU。架构与训练下一步按报告第7节开展；本轮完成的是研究与证据核验，没有修改模型代码、训练配置或启动GPU作业。

## 与其他研究的关系

上游包括 `research/union_adapter_architecture_20261017/`、`research/adapter_value_diagnosis_20261011/`、官方 Union 迁移实验与服务器 Pro 200 步 pilot。本次以当前代码和原始记录更新旧文档中已经过时的状态。
