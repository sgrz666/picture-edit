# Native V7 单配对真实训练计划

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐项完成；用户授权“补齐一下训练试试”，直接在 main 工作。实验不扩展到大规模训练。

**Goal:** 补齐真实 Hi4D 4Ref 的监督与 repair，完成有界 V7 单配对训练、同人物观察、生成与评价。

**Architecture:** 复用已有 typed reader、训练侧车、Canvas、ArcFace 和 V7 Trainer；所有新资产落新的实验目录。从生产初始化权重显式迁移，保留完整冻结、数据与状态门禁。任何不具备的监督保持缺失，不用跳过门禁冒充成功。

**Tech Stack:** PyTorch/Diffusers、既有 InsightFace/ONNX/TorchScript、SDPose；不安装或修改 qwen21 依赖。

---

- [x] 审计 GPU/内存、两个真实样本、Source Mask、五点检测资源、生产初始化及其训练来源；保留旧服务。
- [x] 逐人物匹配 Source/Target 真实人脸检测，输出资格、叠图和拒绝原因；核对 ArcFace embedding 与 TorchScript 一致性。
- [x] 通过既有 `training_sidecar()` 和 `write_record(..., schema='native_pose_repair_v61')` 写新的训练监督与源位置 repair；创建训练 1 行、观察 1 行的 manifest 和 group 文件，重新校验 SHA/Canvas/稳定 ID。
- [x] 显式执行 V5→V6→V6.1→V6.2→V7 初始化，登记来源和每版读回，不继承旧 optimizer/RNG。
- [x] 使用真实模型准备新的四图冻结缓存，绑定模型清单、Prompt、图像角色、Canvas 与条件 SHA。记录 50% 显存上限。
- [ ] 完成 data/grad preflight 和未训练生成。确有 bug 才改共享代码，改前先读、添加失败回归，再修复、测试并快照同步。
- [ ] 按 Body 30→Head 30→Full 120→Joint 120 的首轮预算逐阶段运行，阶段转换采用用户本次原文授权和真实技术门禁；不伪造 gradient/body acceptance 报告。
- [ ] 比较固定噪声 Loss、两人 ID、SDPose、背景及训练/观察生成；如资源或监督确实受限，明确记录实际已完成步骤与阻碍，不宣称过拟合成功。
- [ ] 更新实验 README、保存可复现命令、提交本次脚本/文档/必要修复，向用户说明结果与如何使用。

当前状态：用户明确继续原生尺寸并已暂停 ComfyUI。三种 50% 预算失败记录均保留；新 `pilot300-native-fullgpu-v2` 使用 90% 预算，S0 完整 ID 真实梯度预检已通过（304.8 秒，peak allocated 54.17GiB、reserved 62.38GiB），开始生成未训练对照，optimizer updates 暂为 0。数据别名失效后，逐项 SHA 校验并仅重绑本实验路径，grounded 条件与缓存 Tensor 保持一致。阶段 CLI 丢失已绑定 parent SHA 的缺陷已用失败回归复现并修复。其余训练与评价以新 run 的实际产物更新。
