# runs-face-lora/ —— 预留给 face LoRA 的运行（当前为空）

runs-face-lora/ 是为「face LoRA 相关运行」预留的目录，与 [`../runs/`](../runs/README.md)（单人动作迁移基座流水线的运行产物）并列。当前目录为空，尚未有任何实验产物落入。

## 里面有什么

- 空目录（仅 `.` 与 `..`，2026-10-13 创建后一直未写入）。

## 与上下层的关系

- 与 [`../runs/README.md`](../runs/README.md) 同级、并列：基座流水线不带 adapter/LoRA（见 [`../README.md`](../README.md)），一旦引入 face LoRA 对照实验，运行产物应放这里以区别于无 adapter 的基线。
- 工具与脚本另见 [`../scripts/README.md`](../scripts/README.md)；若后续计划有变，以 [`../README.md`](../README.md) 的「实况」为准。

## 注意与坑

- 占位目录，不要往里塞非 face-LoRA 的运行；普通单人迁移结果仍归 `../runs/`。
- 若确认不再需要该目录，删除前先核对是否已有 face-LoRA 计划文档引用了此路径。
