# 按需冻结输入与正式开训

用户最新授权：先把全量正式训练跑起来，输入逐步加载；最终以训练效果和正常进行为准。保持原数据、20k成功更新、采样顺序/RNG、精度/loss/LR、真实梯度门禁。单人缓存只含冻结VLM/VAE产物，训练特征不缓存。

- [x] 定位启动瓶颈：DataPlan全量Native解码、cache binding再次解码、S0后端无用局部分支约定扫描；此前V6只优化正式训练循环。
- [x] 实现依赖已通过完整资格报告的像素输入计划，按需验证RGB/Guide/Prompt/Canvas及SHA，不全量解码Native数组。
- [x] 实现固定描述文件+逐pair不可变commit；缺失输入按原Sampler顺序生成，不跳样本或重新采样；冻结编码器退CPU，GPU反向时不占编码器显存。
- [x] S0标准Body资格由已通过Joint资格精确投影；真实样本逐项对照。S1/S2/S3各自的数据与真实梯度检查仍在阶段入口执行。
- [x] checkpoint绑定已使用缓存的文件SHA并保留optimizer/RNG。全量数据12,156对此前已真读通过，不丢弃任何原始行或标签。
- [x] CPU回归本地184通过、2跳过；部署回归183通过、2跳过。真实3对样本像素/条件/绑定/S0状态精确一致，缓存source identity保持不变，S0计划构建实测17.4秒。
- [x] 自动接续同一正式run；S0真实production gradient四个必需组合通过，无optimizer创建或step。21:48已观察13次正式成功optimizer更新，loss/梯度有限，无额外冒烟更新；首个100步checkpoint尚未到边界。

运行目录保持不变，代码source/runtime_repo_v7，主管PID3394445。首批验证baseline=0.2251354665；近期正式更新约26.5–27.9秒/次（包含按需冻结编码，B1累积4），不据此承诺整程ETA。证据在report/formal_start_v7_summary.json、report/streaming_startup_summary.json；正式日志stages/S0/segments/train_848fcf1883e94df6b83e9900a9f2b91a/metrics.jsonl。

若数值/资格/冻结/SHA检查不通过，停止并报告；不降低分辨率、batch、loss或预算。显存门槛40GiB用于同时容纳冻结DiT和临时编码模型，属于资源准入，不改变训练数学。
