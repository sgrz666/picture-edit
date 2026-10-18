# third_party/ —— 纳入仓库、经审校的第三方源码改编

本目录是项目对上游第三方代码的"定点改编"存放区。它与 `add/`（外部原始资产与工具链）不同：这里的是被源码直接 `import`、锁定上游 commit 并做了最小改动的模块，属于项目依赖的一部分。

## 里面有什么

- `__init__.py`：模块 docstring，说明本目录是被打包进项目的、经审计（audited）的第三方源码改编。
- `v65_face/`：V6.5 人脸路径的外部依赖封装（StableAnimator、X-Dyna、MCLD、Visual Persona 的改编）。见 `v65_face/README.md`。
- `champ_guidance/`：CHAMP 姿态引导特征编码器（GuidanceEncoder）的单帧改编。见 `champ_guidance/README.md`。

## 与上下层的关系

- 被 `src/` 与各训练/推理路径 `import`（如 `v65_face` 供 V6.5 人脸路径，`champ_guidance` 供相关引导特征编码）。
- 许可证见各子目录的 `licenses/` 与 `LICENSE`，并汇总于根 `THIRD_PARTY_NOTICES.md`。
- 相关：本目录的两个子 README，以及根 `THIRD_PARTY_NOTICES.md`。

## 注意与坑

- 每个子目录都记录上游 commit 与改动范围（如 `v65_face/UPSTREAM_SOURCES.md`、各文件头注释）。升级上游需同步更新这些记录，否则接口漂移难追溯。
- 与 `add/` 的边界：`third_party/` 是被 import 的、随仓库走的源码改编；`add/` 是不可再生的原始资产/环境，整体被 `.gitignore` 忽略。
