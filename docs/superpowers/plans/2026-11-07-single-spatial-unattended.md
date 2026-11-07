# 单人 Native V5 空间加权 20k 实施清单

用户已批准完整计划。目标：S0→S1→S2→S3 成功 optimizer 更新 10000/2000/4000/4000，自动技术验收推进，完成后退出。只保证本次服务器持续运行；断 SSH/聊天继续，服务器重启后以恢复命令启动。不创建循环任务或修改 Linger，不做冒烟训练。

- [x] 建立托管隔离工作树，保留原工作区未提交变更；固定服务器部署基线并同步空间监督代码。
- [x] 核查现有划分、1960 帧物理/Source 特征、561 对配对缓存，以及另一路正在生成的全量 v2。
- [x] 核心训练接口：独立梯度案例预算、有限验证采样、自动验证 guard 与状态恢复、正式自动 Body gate、分组件加载及可恢复冻结缓存。
- [x] 实现数据准备程序：复用旧产物，核验全量v2，支持补齐12156对Native数据并输出固定SHA清单；标准lossless NPZ压缩。全量装配尚待正式主管执行。
- [x] 主管与阶段worker：单实例、资源等待、最多2次瞬时故障重试、committed恢复、20k预算；完整最终val/test loss、有限出图与模型/摘要导出。
- [x] CPU契约回归、SCP部署和语法检查；实际交付记录在服务器source/delivery_verification_v4.json，正式代码source/runtime_repo_v4保留旧快照。
- [ ] 全量12156对Native最终数据验收、冻结Qwen缓存及每阶段真实grad：由正式主管执行。
- [ ] 启动脱离SSH的主管并监督20k运行：最后用户要求只交付流程，由其他agent接管，故本次不启动。

交付边界以用户最后一次指令为准：不再使用subagent，写好并部署训练流程，提供启动、查看、停止、恢复、验收和HANDOFF。训练更新数为0；CPU测试不能替代真实梯度门禁，也不宣称20k已完成。

新run：`/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1`。

固定配置：迁移初始化 native_v5_init_v2；Sapiens2 v1、512等比画布；seed42、B1、累积4；protect_inherited，S3倍率.1；不额外解冻共享继承Geometry；Spatial系数Body=.5/Joint=.5/Hand=1/max=2.5，velocity=1，旧face/hand/ID/perceptual=0。

每100次成功更新检查固定验证loss并保存；每1000次及阶段末输出固定10对验证图（5视频各2）。Loss非有限或连续3次超过本阶段基线2倍时停止，保留上个提交断点；不静默更改batch、分辨率、loss或数据选择。原图、视频、旧产物不删移覆盖。
