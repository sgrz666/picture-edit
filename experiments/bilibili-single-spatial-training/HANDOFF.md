# 交给训练监管 agent

本次用户最后要求只写好训练流程，由其他 agent 监管。当前没有训练作业；不要把流程交付当作已经完成 20k。启动授权和预算保存在运行根 `authorization.json`，直接执行 README 的一次性启动命令即可。

服务器 `sg`，运行根 `/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1`，正式独立代码 `source/runtime_repo_v4`，入口脚本位于该代码内 `experiments/bilibili-single-spatial-training/scripts`；Python `/home/shangguanrz/nihaochen/envs/qwen21/bin/python`。主工作区其他未提交修改保留。实际交付验收记录见 `source/delivery_verification_v4.json`。早期根 `scripts/` 和 `source/runtime_repo/` 留作历史版本，不用于此次启动。

原划分为 9,520/1,490/1,146 对，全保留；使用 Sapiens2 v1 和 1,960 帧 `visible_region_v2`。原空间监督 full v2 作业已经结束，不要重启。原准备作业留下的 561 对及全量 Source/物理特征由新流程核验复用。剩余装配、12,156 对实际读取、冻结 Qwen 缓存、各阶段真实梯度报告仍由正式主管完成，任何一项失败都不能宣称开训就绪。

运行步骤、控制命令和产物位置见 README。首次 `launch.sh run`，中断后 `resume`；不用 cron/systemd/Linger。关注 `status` 的 `owner_alive` 和日志；资源不足等待，不杀其他任务。明确 `failed` 时不能删记录后重试，须报告错误和最后 committed checkpoint。修改代码/数据/变量必须使用新版本和新的匹配验收，不能原地漂移。

固定 S0/S1/S2/S3 = 10,000/2,000/4,000/4,000 成功更新，B1 累积4，512长边，seed42，S3 LR倍率.1，Interaction关闭。禁止额外 optimizer 冒烟运行或旧300步实验 optimizer。粗手部 loss 监督不代表精细3D手指条件变得有效。

完成需要 `status=completed`、20,000 committed 更新、全量最终 val/test loss、固定 test 图和像素指标，以及 `export/` 的权重/摘要/README。摘要记录每阶段更新、尝试耗时、最后验证结果与权重位置；技术验收不能冒充人工画质验收。
