# home/ —— HOME 运行器统一入口（符号链接目录）

本目录是「家目录长驻运行脚本」在仓库内的统一入口：4 个符号链接指向 `/home/shangguanrz/` 下的原始脚本，让人从仓库位置也能找到它们，而不把散落脚本复制进 Git。目录内没有任何实体代码。

## 里面有什么（4 个软链）

| 链接 | 指向 | 作用 |
|---|---|---|
| `run_smplestx.sh` | `/home/shangguanrz/run_smplestx.sh` | 对 `pA`/`pB` 循环跑 SMPLest-X 导出（不传 `--retarget_cam`，保持导出相机与原图一致，否则 `mesh_cam` 无法与照片对齐）；依赖 `add/envs/smplestx_bw` 与 `add/tools/SMPLest-X` |
| `run_stage1.sh` | `/home/shangguanrz/run_stage1.sh` | `experiments/16-test` stage1 双基座基线：DeepGen 与 Qwen 必须各用各的 conda 环境（transformers 大版本不同） |
| `run_stage2_both.sh` | `/home/shangguanrz/run_stage2_both.sh` | 16-test stage2 双适配器（deepgen_adapter + qwen_adapter）批量运行 |
| `watch_and_run.sh` | `/home/shangguanrz/watch_and_run.sh` | 看门狗：等待指定 eval 进程退出后再启动 16-test 基干实验，避免抢卡 |

## 与上下层的关系

- 上级是 [`scripts/run/`](../README.md)（实验编排与收尾运行入口）；这些脚本操作的实验对象在 `experiments/16-test/` 与 `workflow/`。
- 与 `add/` 外部工具链（SMPLest-X、conda 环境）配合完成几何与基线运行。

## 注意与坑

- 链接目标位于家目录、不在 Git 内；换机器或清理家目录时这些链接会断，脚本本体需另行保全。
- 脚本里写死了实验目录与 conda 环境绝对路径，复用前先核对路径仍有效。
