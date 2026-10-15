# tiktok_overfit_sanity_full/ —— V6.5 面部适配器两阶段门禁的完整裁决（FAIL，推动路线转向）

实验目的
- 对 V6.5 Face Adapter 在 TikTok 00001 上执行严苛两阶段过拟合协议：Phase A（前 200 步，固定 t=150）测确定性记忆；Phase B（201~1500 步，全轨迹随机时间步）测生成泛化，用 C1~C6 六大门禁做科学裁决。

里面有什么
- `checkpoints/`、`condition_cache/`：1500 步全量训练产出。
- `summary.json`（15.5 KB）：门禁裁决主文件（phase_a_steps=200，max_steps=1500，v6.5）。
- `gen_eval_history.jsonl`、`gen_evals/`、`previews/`、`sanity_curves.png`：评估历史与可视化。
- `train.log`、`*.nohup.log`、`train_metrics.jsonl`：训练与日志。
- `input_source.png`、`target_gt.png`：源/目标参考图。

关键结论
- 裁决 `verdict = FAIL`。C1 接口不变量 PASS（0.0）；C2 记忆收敛 PASS（0.4196≤0.7×0.6505）；C3 全轨迹下降 FAIL（tail 0.2383 未达 0.6×init 0.1521）；C4 生成显著优于初始 PASS（+7.95 dB）；C5 绝对拟合 PASS（20.04 dB）；C6 适配器正向贡献 FAIL（ON 比 OFF 低 -0.044 dB）。best_gen_psnr_on 20.65 dB。
- 技术根因：V6.5 在 MMDiT 中间层直接注入面部局部特征，在全轨迹去噪中晚期强行覆盖特征，扰乱背景与服装全局流，使最终 ON 未超越原生基座（C6 失败）。这一结论直接终结单模型内单层揉合面部的方案，确立转向 Qwen-Image-2.1 或后置两阶段面部精修（BFS/Face Retargeting）的决策。

与其他实验的关系
- DeepGen 技术线面部适配器门禁族的核心裁决实验（[../README.md](../README.md) 第 4.1 节）。对照 `tiktok_overfit_sanity_smoke` 的短步版，并为 `tiktok_v65_*` 系列提供失败证据。见 [../../README.md](../../README.md)。
