# Full Native：几何前端迁移验证

## 实验目的

从现有Full Body checkpoint继承几何前端，核验原生512维输出与旧Full高低尺度融合一致。当前只做CPU结构/权重验证，不训练、不生成图片、不验证图像质量。

## 里面有什么

本地本目录保存说明和轻量验证报告。远端按身体控制目标归入 `experiments/3.singel man/qwen21_full_native_v1/`，持久产物放 `/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/`。

- 生产代码：`src/pose_control/qwen21_full/native_geometry.py`；接口用法见同目录 `NATIVE_GEOMETRY.md`。
- 核验入口：`scripts/audit/audit_qwen21_native_geometry.py`，从显式Full checkpoint读取对应前端张量；用合成单人/双人、方形/矩形条件比较旧bridge与新出口。
- 测试：`tests/test_qwen21_native_geometry.py`，含梯度、独立权重、加载拒绝、网格和不覆盖输出契约。
- 数据盘中本轮目录 `geometry_step1_20261022_v1/` 保存部署前源码快照、核验输出与执行记录；不存新训练权重。

复核命令（服务器项目根目录，现有Qwen环境）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m pytest tests/test_qwen21_native_geometry.py -q -p no:cacheprovider
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B scripts/audit/audit_qwen21_native_geometry.py \
  --checkpoint /home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt
```

如需保存JSON，传 `--output` 指定一个不存在的文件，父目录须已存在。脚本拒绝覆盖已有输出；源checkpoint仅只读。

## 关键结论

本阶段的验收是原始512维特征及重新经过旧128维projection后的数值一致，不用投影后相同代替原始特征相同。真实权重结果在执行后记录于本目录的验证报告；没有据此声称新主干接口或生成效果已完成。

## 与其他实验的关系

上游是现有Full Body和 `research/qwen21_full_native_geometry_20261022/` 的方案C研究。下一步消费这些token实现Qwen主干实时反馈；Depth、稀疏层数和Face/Hand迁移各自独立验证。无需Union模型或Pro checkpoint。
