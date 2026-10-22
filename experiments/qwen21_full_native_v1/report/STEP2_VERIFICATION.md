# 第二步验收：Qwen主干实时反馈（2026-10-22）

实现已通过原生32层结构测试及现有7B权重的CUDA前向/反向核验。新Body head仍未训练，现有默认出图入口仍使用原Full；不能用本记录声称生成效果改善。

## 实现

新增 `native_body.py`：4096→512当前主干query、512维几何局部编码/调制、layer/sigma条件、六组512→4096零head；默认32个接入位置。生产维度新增接口共 **16,541,184** 个参数，原Full前端 **75,966,284** 个参数另计；不含7B基座、Face/Hand或其他提取器。

新增 `native_integration.py`：在本模型实例的原生block输出后写target；原主干参数对象和已有state key保持。pipeline通过 `bind_native_body` 提供条件，每次forward重新构建真实layout；checkpoint显式保留本次execution。

第一版B=1、KV cache关闭、任务参数FP32/主干BF16；拒绝会丢梯度的reentrant checkpoint配置。Depth、训练和采样效果均未混入本步。

## 测试

- 本地torch2.12.0+cpu：新增Body/Geometry及前端相关回归 **82 passed**。
- 服务器torch2.8.0+cu128，CPU执行新Body/Geometry与旧Full bridge：**59 passed**，有现有Swig/NVML警告。
- CPU小型32层真实Qwen结构＋现有Full已训练前端：[cpu_body_audit_20261022.json](cpu_body_audit_20261022.json)。zero、disabled、prefix差均0；checkpoint开关的forward及gradient最大差均0；6组head均有梯度，几何输入有梯度，原主干参数冻结且值不变。
- 跨两次forward改变了geometry、sigma、文本长度、参考图数量及目标网格，再退出绑定后backward；梯度与非checkpoint参考一致。另验证BF16原生Qwen的零输出等价和异常后的上下文清理。

第一步记录的旧hand测试导入迁移问题仍单独存在，见 [STEP1_VERIFICATION.md](STEP1_VERIFICATION.md)；未把本次定向通过说成全仓回归全部通过。

## 现有7B CUDA核验

原始报告：[gpu_7b_runtime_audit_20261022.json](gpu_7b_runtime_audit_20261022.json)。

- 基座：`/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`，32层、4096维、7,115,124,736个参数。
- 前端：现有 `body_512_100/checkpoint_last.pt`，SHA256仍为 `6aa320cf7e718f824c047594ad8ad04e44385c3249328406e79a75baf5d28b2d`。
- 分辨率512、target1024 token；合成latent/text与有效合成几何，不加载VLM/VAE，不采样图片。
- 安装新模块前后的zero差 **0**，prefix差 **0**。
- 标准non-reentrant checkpoint下，一次backward的6组head梯度norm依次为 **2.43094、2.07525、2.05424、2.44353、1.79822、1.36114**。
- 原主干无梯度、参数version不变；源Full checkpoint的stat不变；执行/绑定上下文已清理。
- 本次有界探针峰值CUDA allocated **14.75055 GiB**，总耗时约 **41.67秒**。这不包含VLM/VAE/optimizer，不等于正式训练/采样显存，也不是相对旧Full的公平速度对比。
- 没有创建优化器、更新权重或写checkpoint。

复核命令：

```bash
cd /home/shangguanrz/project/pic-edit
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m pytest tests/test_qwen21_native_body.py tests/test_qwen21_native_geometry.py tests/test_qwen21_full_bridge.py -q -p no:cacheprovider
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B scripts/audit/verify_qwen21_native_body_runtime.py --base-model models/Qwen-Image-2.1 --full-checkpoint /home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt --resolution 512
```

如要存JSON，`--output`必须指向新文件。部署前快照位于 `/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/body_step2_20261022_v1/source_before_step2_20261022.tar.gz`。

## 环境观察与下一步

内核驱动为580.173.02，NVML用户库链接到580.178.04，`nvidia-smi`因版本不匹配失败。CUDA计算和上述7B核验可运行，显存统计来自PyTorch CUDA；本次没有修改系统驱动。

下一阶段在锁定的现有数据、参考尺寸、seed、scheduler和采样步数下，进行新接口迁移训练与完整出图对照。先确认姿态/身份/服装/背景保持，再比较注入层数和Depth；不能从工程通过直接跳到效果通过。
