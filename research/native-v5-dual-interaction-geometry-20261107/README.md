# Native Full V5 双人交互几何审查（2026-11-07）

核对 `sg:/home/shangguanrz/project/pic-edit` 当前工作树中的双人 Adapter，判断用户提供的 Relative / Contact Geometry 建议哪些已经实现、哪些值得继续补强。本次交付是代码审计、CPU 探针和方案研究；没有修改 Adapter、训练配置或已有实验产物。

## 里面有什么

- [审查与改造建议](Native-V5-双人交互几何审查与改造建议.md)：当前数据流、已证实的表达缺口、9 项论文/方法对照、三种路线比较，以及分步消融方案。
- [audit-evidence-v2.json](audit-evidence-v2.json)：2026-11-07 01:30 左右的远端源码 SHA256、两个 Hi4D target 缓存的实际统计、编码器 CPU 探针。`audit-evidence.json` 是 01:28 的首轮快照，保留原样。
- [source-snapshot/](source-snapshot/)：9 份从远端直接 `scp` 回来的源码证据，文件原名与 SHA 的对应见审计 JSON。这些是审查快照，不能拿来覆盖主树。
- [audit-probe.py.txt](audit-probe.py.txt)：只读 CPU 探针的文本存档。可运行版本在远端 `/data/shangguanrz/pic-edit-audits/native-v5-interaction-20261107T0127/audit_evidence_v2.py`。
- [verification.json](verification.json)：8 项既有 CPU 回归测试的命令、退出状态和结果，以及源码快照校验。

## 关键结论

当前 V5 已有 Relative11、双人独立 Depth/Normal/Part、接触图、196 个部位对和独立 Interaction 残差，不能从“尚未实现交互几何”重新设计。一个击掌缓存含 2616 条有方向的顶点接触记录、3 个正接触部位对；模型家族为 SMPL，每人 6890 个顶点，尺度声明为 `model_unit/model_scale`。

CPU 探针发现旧 8 条 Contact Token 的加法编码存在人物—部位绑定歧义；另一个 PartRelation 分支能保留有序部位对，但只用最近点差向量，没有显式编码接触位置、接触面法线和穿插状态。探针是模块级证据，不证明训练后整套 Adapter 的效果。

## 怎么复核与使用

先读报告中的“推荐路线”和“如何验证”。在 `sg` 上复跑只读探针：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B \
  /data/shangguanrz/pic-edit-audits/native-v5-interaction-20261107T0127/audit_evidence_v2.py
```

输出 JSON；不读取 target RGB、不加载 checkpoint、不运行估计器、不训练、不使用 GPU，也不改写缓存。输出中的源码 SHA 随主树变化，需要和本目录快照比较，不能跨版本直接套用结论。回归测试复跑命令见 `verification.json`。本报告中的候选模块、阈值、训练消融均是后续方案，尚未实施。

本目录接续 `research/qwen21_dual_upgrade_20261011/` 的调研主题，但以 11 月 7 日远端 Native Full V5 为审查对象；不替代旧证据，不修改旧报告。
