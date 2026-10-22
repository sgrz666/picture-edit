# Full Native：几何前端迁移验证

## 实验目的

从现有Full Body checkpoint继承几何前端，核验512维输出，再接Qwen主干实时反馈与独立Depth。目前完成CPU/7B核验及3次真实更新冒烟；尚未做正式训练和生成质量对照。

## 里面有什么

本地本目录保存说明和轻量验证报告。远端按身体控制目标归入 `experiments/3.singel man/qwen21_full_native_v1/`，持久产物放 `/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/`。

- 生产代码：`src/pose_control/qwen21_full/native_geometry.py`；接口用法见同目录 `NATIVE_GEOMETRY.md`。
- 核验入口：`scripts/audit/audit_qwen21_native_geometry.py`，从显式Full checkpoint读取对应前端张量；用合成单人/双人、方形/矩形条件比较旧bridge与新出口。
- 测试：`tests/test_qwen21_native_geometry.py`，含梯度、独立权重、加载拒绝、网格和不覆盖输出契约。
- 数据盘中本轮目录 `geometry_step1_20261022_v1/` 保存部署前源码快照、核验输出与执行记录；不存新训练权重。
- 第二步 `body_step2_20261022_v1/` 保存主干实时反馈接口的源码快照、CPU结构审计及有界7B前向/反向报告。接口说明见 `src/pose_control/qwen21_full/NATIVE_BODY.md`。
- 第三步 `depth_step3_20261023_v1/` 保存CHAMP式Depth的源码/画布依赖快照、2个pair预检、CPU/7B核验与3次真实更新的checkpoint。接口及训练用法见 `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。
- `scripts/prepare_depth_step3.py` 固定少量pair与显式Depth来源；`check_depth_smoke.py` 重建并检查训练checkpoint；`record_depth_delivery.py` 备份文档并封存实际源码依赖。均不复制、删除或移动媒体。

复核命令（服务器项目根目录，现有Qwen环境）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m pytest tests/test_qwen21_native_geometry.py -q -p no:cacheprovider
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B scripts/audit/audit_qwen21_native_geometry.py \
  --checkpoint /home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt
```

如需保存JSON，传 `--output` 指定一个不存在的文件，父目录须已存在。脚本拒绝覆盖已有输出；源checkpoint仅只读。

## 关键结论

第一步见 [前端验收记录](report/STEP1_VERIFICATION.md)：现有Full权重的329项前端张量被严格继承，原512维特征及旧128投影后的差均为0。

第二步见 [主干接口验收记录](report/STEP2_VERIFICATION.md)：本地82项、服务器59项定向测试通过；真实7B/512分辨率零输出及prefix差为0，六组head有有效梯度，原基座冻结。新增主干接口为16,541,184参数，旧前端75,966,284参数另计。尚未据此声称生成质量改善。

第三步见 [Depth验收记录](report/STEP3_VERIFICATION.md)：本地109项、服务器116项通过；官方空间attention最大差0；新增Depth为2,254,656参数。实际704×384条件接入7B，初始/zero-Depth/prefix差均0，Depth与Body均有梯度；3次真实更新后的新checkpoint可独立重建。尚未做质量对照。

## 与其他实验的关系

上游是现有Full Body和方案C研究；现已完成Native主干反馈及独立Depth接口。下一步固定预算训练noDepth/Depth对照，再考虑稀疏层数和Face/Hand迁移。无需Union模型或Pro checkpoint。
