# 交给训练监管 agent

用户已追加授权等价提速和代码SHA更新。2026-11-07实施时，旧v5主管正在prepare_data、optimizer更新0；新v6在数据完整验收后自动接续同一run，不要另开run或重复启动主管。原训练授权和预算在authorization.json，本次更新授权在source/performance_update_authorization.json。

服务器 `sg`，运行根 `/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1`，正式独立代码 `source/runtime_repo_v7`，入口脚本位于该代码内 `experiments/bilibili-single-spatial-training/scripts`；Python `/home/shangguanrz/nihaochen/envs/qwen21/bin/python`。主工作区其他未提交修改保留。实际交付验收记录见 `source/delivery_verification_v7.json`。早期根 `scripts/` 和 `source/runtime_repo/` 留作历史版本，不用于此次启动。

原划分为 9,520/1,490/1,146 对，全保留；使用 Sapiens2 v1 和 1,960 帧 `visible_region_v2`。原空间监督 full v2 作业已经结束，不要重启。原准备作业留下的 561 对及全量 Source/物理特征由新流程核验复用。剩余装配、12,156 对实际读取、冻结 Qwen 缓存、各阶段真实梯度报告仍由正式主管完成，任何一项失败都不能宣称开训就绪。

运行步骤、控制命令和产物位置见 README。首次 `launch.sh run`，中断后 `resume`；不用 cron/systemd/Linger。关注 `status` 的 `owner_alive` 和日志；资源不足等待，不杀其他任务。明确 `failed` 时不能删记录后重试，须报告错误和最后 committed checkpoint。修改代码/数据/变量必须使用新版本和新的匹配验收，不能原地漂移。

固定 S0/S1/S2/S3 = 10,000/2,000/4,000/4,000 成功更新，B1 累积4，512长边，seed42，S3 LR倍率.1，Interaction关闭。禁止额外 optimizer 冒烟运行或旧300步实验 optimizer。粗手部 loss 监督不代表精细3D手指条件变得有效。

完成需要 `status=completed`、20,000 committed 更新、全量最终 val/test loss、固定 test 图和像素指标，以及 `export/` 的权重/摘要/README。摘要记录每阶段更新、尝试耗时、最后验证结果与权重位置；技术验收不能冒充人工画质验收。

性能结果见 source/throughput_audit_v6.json（真实Adapter形状的合成梯度审计对照，无optimizer更新），代码接续结果见control/code_update_completed_*.json，等待日志见logs/code_update_*.log。实际版本必须查v6 supervisor status的active_repo_root。只修改重复样本读取、梯度审计回传和重复参数枚举，全部训练变量及数据/模型/cache/checkpoint SHA保留；每阶段仍必须新production gradient门禁。


v7采用按需冻结输入。所有12,156对Native数据已经完整真读验收；不再等待11,010对VLM/VAE缓存全部生成。固定描述文件绑定全部输入，每pair缓存独立SHA/commit，checkpoint绑定已使用缓存，缺失时按原采样顺序等待编码，不跳样本。S0标准Body资格复用已通过Joint资格并逐项对照；S1/S2/S3保留各自真实资格/梯度门禁。编码器仅编码时上GPU，随后退CPU；40GiB资源门槛保证显存余量，训练精度/loss/LR/B1累积4及20k预算均不变。实际开训以正式metrics中optimizer_updated=true为准。
