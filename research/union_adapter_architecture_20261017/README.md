# Union 与人物 Adapter 架构研究（2026-10-17）

## 研究目的

比较“按官方Union机制重构旧Adapter”与“在预训练Union上移植人物任务能力”，给出适合单人/双人姿态、身份、表情和接触控制的完整设计。用户选择效果优先、85GB单卡验证。

## 里面有什么

- [ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md)：推荐架构、两路线取舍、代码迁移边界、10项论文机制与分阶段验证。新结构是待验证建议，不是已实现或已优于基线的模型。
- `source_snapshot/official/`：固定VideoX-Fun版本的控制模型、原生模型、control pipeline与配置，供核对原生接口和行号。
- `source_snapshot/adapter/`：远端Full/轻量Adapter的63个Python文件与说明快照；`SNAPSHOT.sha256`提供上游70项指纹，`README.remote.md`是远端总说明，`README.md`是本地已有索引。不含权重或媒体，保持原import，不作为独立运行入口。
- `evidence/`：只读复制的smoke/overfit门禁、两时刻固定loss、旧48张矩阵记录/现有31项指标、Body20k与Face3000质量汇总等。文件里的绝对路径仍指原服务器产物，不随复制改写。
- `manifest.json`：上述证据/源码的SHA256、源路径、版本与外部来源索引。用于复核字节，没有伪装成新的训练结果。
- `SHA256SUMS`：再包含manifest本身的标准校验清单；在本目录运行`sha256sum --check --quiet SHA256SUMS`可只读复核。manifest的文件列表不包含自身和这个校验清单，避免自引用。

## 关键结论

主推完整16块Union作为控制骨架，在control target state中接入typed geometry/contact、按人物隔离的source bank和ROI精修。保留旧任务输入与已核实的前端经验，替换DeepGen残差核心及不完整的绑定接口。

原小样本200updates loss下降17.7868%，未达到20%门槛；队列已停止。18/18固定位置均改善，不代表结构失效。既有官方移植矩阵48张生成已齐、指标31/48，不能作完整双人优劣结论。现有强姿态控制有身份/外观代价，详见主报告逐项数值与边界。

本轮没有启动新训练、修改源模型或搬动/删除媒体。使用方式：先读主报告第1、3、4节确定结构，再按第7节生成实现任务与验证清单；无需运行快照代码。复核证据可运行下面的只读校验：

```bash
python -c 'import hashlib,json,pathlib; r=pathlib.Path("research/union_adapter_architecture_20261017"); m=json.loads((r/"manifest.json").read_text()); bad=[x["path"] for x in m["files"] if hashlib.sha256((r/x["path"]).read_bytes()).hexdigest()!=x["sha256"]]; print({"files":len(m["files"]),"mismatches":bad}); raise SystemExit(bool(bad))'
```

## 与其他工作的关系

上游是`research/qwen21_controlnet_transfer_20261015`及已验证的官方迁移/数据集适配实验。本报告用2026-10-17实测补正了旧报告的完成状态、mock hybrid效果边界与单样本baseline混杂。下游设计入口为`union-adapter/pro-adapter/README.md`；正式源码仍应放`src/`，大产物继续落`/data/shangguanrz/`。
