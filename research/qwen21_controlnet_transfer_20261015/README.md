# Qwen 2.1 ControlNet Union 移用研究（2026-10-15）

## 研究目的

通过 `ssh sg` 核对当前简化版 / Full Adapter 与已安装的 Qwen 2.1 源码，对照 Alibaba-PAI 发布的 ControlNet Union，判断如何加强人物姿态、身份与接触控制。本目录是调研证据包，未实现新 Adapter、未训练或生成图片。

## 里面有什么

- [REPORT.zh-CN.md](REPORT.zh-CN.md)：架构差异、推荐移用方案、文件落点与分阶段实验方案。
- `remote_audit.json`：服务器 Git 状态、代码 SHA256、包版本、模型配置、基座权重文件头统计及 NVML 检查结果。
- `source_snapshot/remote_20261015/`：服务器 28 个项目源码文件、已安装的 Transformer / Pipeline 和模型配置副本。用于审阅，不是运行入口。
- `source_snapshot/upstream/`：固定 VideoX-Fun revision 的模型、管线、训练脚本、配置、示例；Union 模型卡和仅 20,280 字节的 safetensors JSON 文件头。没有下载 7 GiB 权重本体。
- `scripts/audit_remote.py`：在服务器用标准库读取源文件、配置及模型文件头；不 import torch、不加载张量、不初始化 CUDA。
- `scripts/verify_audit.py`：离线核对代码哈希、源码语法、权重 shape 与参数预算；默认只打印结果，不写文件。
- `verification.json` / `manifest.json`：本次核对结果与证据文件摘要。

## 关键结论

Full Adapter 的 6 组残差重复注入 32 层；官方由 16 个原生 Qwen 控制块各自产生残差。Union 文件头包含 180 个 BF16 张量、3,775,479,808 个参数；其中 16 × 9 = 144 个控制块核心张量的 shape 与当前基座 block 对应。这个检查支持结构移用，不能证明直接接入后的生成质量。

建议优先复用官方预训练控制分支，同时保留本项目源图身份输入，逐步接入 SMPL-X / 接触 / 脸手条件；先做 LoRA 或新增小模块适配，再决定是否需要压缩成 4～8 个块。官方原样管线不提供我们现有的源图 reference 路径，不能直接替换。

## 使用与复核

从本地仓库根目录执行（仅需 Python 标准库）：

```powershell
python research/qwen21_controlnet_transfer_20261015/scripts/verify_audit.py
```

本次输出 `PASS`：28 个项目源码哈希匹配、37 个 Python 文件可解析、144 个控制核心张量 shape 匹配。它不是 GPU 推理测试。出图对比的具体分组和判据见报告第 7 节；新的训练 / 推理命令尚未实现。

服务器本次证据目录为 `/data/shangguanrz/pic-edit-research/qwen21_controlnet_transfer_20261015_005429/`。原始审计脚本和 JSON 在其根目录，完整本地证据包镜像在其 `bundle/` 下。

## 与其他研究的关系

继承 `qwen21_dual_upgrade_20261011/` 对人物绑定、接触关系和目标区域注入的要求。本次重新读取 10-15 的服务器代码与公开预训练权重文件头，重点补充 Union 的实际参数量、源图路径缺失、缓存开关与移用成本，未执行或替代旧研究中的候选设计。
