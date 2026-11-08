# 资源错误分类与500步恢复

2026-11-08 01:02，S0实际更新570、committed 500时，在准备下一批冻结编码的Qwen3-VL视觉分支抛出torch.OutOfMemoryError，消息仅为“Allocation on device ”。原字符串分类漏识别，错记terminal，自动重试未执行。500步验证loss0.2171275、冻结摘要、optimizer/RNG及2010个已使用缓存SHA均通过，未发现数值发散。

- [x] 读取完整trace、typed worker result和失败journal；只读校验500 checkpoint、缓存、指标。
- [x] 先写失败回归，再按error_type识别OOM；ValueError/SHA/数值/资格错误仍terminal。
- [x] 用追加证据事件纠正此特定错误，不删除失败记录或重置重试次数；其余terminal不放行。
- [x] 原v7训练源码和worker完全保留。修复单独部署控制器，显式调用已固定v7 worker；strict resume的数据/模型/配置/代码身份保持一致。
- [x] 启动独立后台恢复程序：主管停止期间用像素输入计划单独补齐剩余冻结VLM/VAE缓存，避免与14GiB DiT共占显存；不加载3D条件、不构建optimizer、不降低分辨率或改变loss。
- [ ] 缓存补齐并释放编码器后，从checkpoint500恢复原optimizer/RNG与Sampler，确认后续实际更新；20k预算不变，未提交501–570按原策略回滚。

恢复PID3209724，日志logs/resource_recovery_20261108T030554Z_3209718_13533.log；控制代码source/resource_controller_20261108_v1。原failed状态在缓存准备结束且证据追加前保留，不宣称已恢复。

11:16状态：11010对缓存输入全部核验，3542对可复用、7468对待补齐；可用显存24.64GiB，未达到独立编码预留28GiB，恢复进程等待资源。无训练进程，不能声称恢复成功。修复回归186通过、2跳过。
