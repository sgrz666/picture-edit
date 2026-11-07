# Native V5 训练等价提速实施清单

用户已批准实现建议中的前两项。本任务单独执行，不使用 subagent。目标为消除采样时的完整样本重复读取、合并梯度审计的 GPU→CPU 同步；固定20k成功更新、B1累积4、精度、样本顺序、loss、LR、所有检查和保存/验证频率。

- [x] 先写失败回归：完整样本每micro只读一次，sigma/noise序列不变；审计结果与旧实现逐项一致、每device仅一次结果取回、异常及冻结检查保留。
- [x] 修改training/trainer.py：采样仅用预检bucket中的scene kind，前向仍完整读取并检查绑定，不保留四份大样本。
- [x] 修改training/validation.py：每参数同样计算finite/nonzero/norm，在原device汇集标量，一次回传；不改变gradient或optimizer。parameters.py的重复全模型枚举同时缩为每次审计一次，绑定/冻结检查不删。
- [x] CPU回归177通过、2跳过；服务器独立CUDA测试7通过。3对真实样本逐字段bitwise一致；真实Adapter形状的四阶段合成梯度审计输出一致、梯度字节SHA不变；实际完整生产梯度仍由正式门禁执行。
- [x] 新建独立runtime v6，保留v5产物与代码。主管增加仅在训练前、无断点、显式授权时的版本切换记录；预算/数据/初始化不变。用户追加同意更新旧代码SHA约束，其他资产SHA继续严格。
- [x] 实现并部署一次性后台交接程序，PID696694、SID696694，等待prepare_data完整验收后自动接续。Linux独立控制测试验证真实锁、plan revision、resume、资源阻塞与stop，不构建模型或optimizer。
- [x] 文档写明使用、验收和当前生效状态，落盘实测数据；不承诺未测量的整程倍速。
- [ ] 生产切换触发：当前仍在prepare_data，0次optimizer更新；后台会在完整数据验收通过后执行，尚未宣称v6已生效。

审计单环节中位耗时：S0 1.398→0.0192秒，S1 1.267→0.00496秒，S2 1.789→0.0424秒，S3 2.038→0.0622秒。这不是完整训练的耗时或倍速。证据：实验report/*summary.json及服务器source/throughput_audit_v6.json、source/code_update_control_v6.json。后台日志为logs/code_update_20261107T090137Z_696686_2026.log；代码切换完成标志为control/code_update_completed_*.json。

当前磁盘证据：run处于prepare_data，成功optimizer更新0；代码在source/runtime_repo_v5。不得原地编辑已固定代码。原run_plan.json不可覆盖；版本切换必须追加独立新plan及journal事件，所有历史引用继续可验证。若生产grad/train已开始或出现checkpoint，自动交接拒绝，报告需另行设计保留optimizer的迁移。
