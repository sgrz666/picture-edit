# Pro 实验脚本

共享训练与推理入口是仓级 `scripts/train/qwen21_pro.py` 和 `scripts/run/qwen21_pro.py`；本目录只保存该实验的资产审计和受控 pilot 工具。几何、缓存与完整模型审计的命令见上一级 README。

`render_pilot.py` 读取冻结 V4 的真实代码、发布权重和 legacy B 六例条件，分别为主 run 的 step0、step200 生成6张图。固定40步、seed43、CFG1、KV cache关闭；源条件不变，GT RGB不进入生成。step0先检查零分支与原生 Union 逐值等价。输出新 run 的 `report.json`、逐样本 `sampling.json`、`coarse/coarse.json` 及图片；这是训练集诊断，不是独立验证集结论。

```bash
P=/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1
B=/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1
C="$P/code_task7b_20261018T161055"
M="$P/runs/pilot_main200_v4_20261018T161055"
"$B/env/bin/python" "$P/scripts/render_pilot.py" \
  --code-root "$C" --name train6_step0_<新名称> --checkpoint "$M/checkpoint_000000" \
  --feature-run "$P/runs/conditions_pilot1600_20261018T142046" \
  --source-geometry-index "$P/runs/train_geometry_20261018T142046/source_geometry_index.json" \
  --contact-vertex-parts "$P/data/smpl_male_v6_parts_20261018.npz"
```

step200改为 `checkpoint_000200` 并换新 name。`--code-root` 必须是完整冻结 V4，不能混用服务器更新过的依赖；脚本会校验实际导入路径与46文件签名。

`summarize_pilot.py` 汇总主200步与 no_dense/no_appearance/no_contact/full8 四个独立25步 run。它核验实际检查点、工程门禁、原始18条评测、初始化和训练协议，重新计算质量门禁；真实失败保留为失败。JSON保留每例loss、区域覆盖与参数统计，中文Markdown便于阅读。训练未捕获noise张量SHA，CUDA重放证明明确标记 `replayed_not_captured`，不会冒充训练时捕获。

```bash
"$B/env/bin/python" "$P/scripts/summarize_pilot.py" --code-root "$C" \
  --name pilot_summary_<新名称> --main-run "$M" \
  --no-dense-run <no_dense运行目录> --no-appearance-run <no_appearance运行目录> \
  --no-contact-run <no_contact运行目录> --full8-run <full8运行目录> \
  --noise-replay-device cuda
```

缺失运行时报告 `pending`；缺少同后端重放证明时为 `pending_noise_proof`。数值/资产错配直接报错。CPU回归：`python -m pytest tests/test_qwen21_pro_pilot_tools.py -q`；本轮34项通过。两个工具均通过spec与quality审查，未修改冻结模型代码。
