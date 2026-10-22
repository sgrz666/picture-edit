# 第三步：CHAMP式Depth编码、512融合与真实Qwen验收

2026-10-23（北京时间）。在已有Full frontend与Native Body上新增独立Depth；由当前智能体独立实施，没有派subagent。新模块与入口已投递`sg:/home/shangguanrz/project/pic-edit`，旧Full默认路径与原checkpoint保留。

## 完成了什么

`DepthCondition`统一近1远0，区分camera Z、relative PNG、depth valid和canvas valid；显式来源、相机、单位/尺度、归一化范围与资产SHA。新manifest绑定pair_id、target_stem和target_canvas，无PNG/NPY隐式fallback；运行中资产或元数据漂移报错。

新`Qwen21NativeDepth`采用CHAMP的16/32/96/256 CNN与有效空间Transformer。默认在1/16做attention，显式8为原位置对照；新零head输出512维，与原geometry残差相加。分支共 **2,254,656参数**；不扩旧fusion、不添加Qwen图像sequence token、不加载Union或Pro网络。

`NativeDepthRuntime`接入原Native Body，提供组件checkpoint与独立重建。专用训练入口是`train_qwen21_native_depth.py`，CPU预检、原生VLM/VAE cache、flow目标、coverage加权loss、冻结基座与geometry、只训练Body/Depth，已走完真实更新和保存/恢复。

## 验收结果

| 检查 | 实测 |
|---|---|
| 本地定向回归 | **109 passed**，新Depth三模块、原Geometry/Body与native resolution |
| 服务器定向回归 | **116 passed**，上述内容加现有canvas coverage测试 |
| 官方Depth初始化 | 33个有效源张量，1,533,136源元素；RGB核求和后迁入1,532,992元素；未知key/shape/nonfinite拒绝 |
| 与原版attention数值比较 | 固定官方原类、无mask时最大差 **0** |
| 初始Depth几何差 | **0**；Body head激活时，新增零Depth仍使Qwen输出差为 **0** |
| 真实7B配置 | 32层、hidden4096，实际TikTok目标canvas **704×384 → 44×24 = 1056 token** |
| 激活Depth后的prefix差 | **0**；目标输出最大变化0.125，用于敏感性工程检查 |
| Depth梯度 | CNN约9.75e-6、attention约8.11e-3、projection约2.22e-2，均有效 |
| Body梯度 | 六组head全部非零；原Qwen参数冻结且版本未变 |
| GPU峰值allocated | 14.545GiB，工程probe；不含VLM/VAE/optimizer完整训练峰值 |
| 真实训练冒烟 | 3次更新，22×12网格；2个pair通过预检，实际随机抽到同一pair做3次更新 |
| 保存恢复 | 529,641,677字节的新checkpoint；`weights_only=True`可读，无原Full文件可重建 |

真实训练第1步Depth head/CNN梯度为0；第2步head梯度0.185211、CNN仍0；第3步head梯度0.051273、CNN梯度3.31566e-7。验证了两个串联零head的梯度延迟，未把首步0误判为永久死分支。

原Full checkpoint SHA256仍为 `6aa320cf7e718f824c047594ad8ad04e44385c3249328406e79a75baf5d28b2d`。官方CHAMP权重为 `4da437759e68cfe1c6361432305cc1ce02e1d04651db0d813b01eaaabd4665a1`；权重均不进入Git。

## 证据和复现

- `step3_cpu_depth_audit_20261023.json`：CPU对齐、参数量、初始化、梯度与state roundtrip。
- `step3_gpu_depth_7b_20261023.json`：真实矩形条件、7B前向/反向、源码hash、prefix和冻结断言。
- `step3_checkpoint_smoke_20261023.json`：3次真实更新、实际pair数、梯度与已保存checkpoint重建。
- `step3_preflight_20261023.json`：2个pair的实际target canvas及Depth资产来源。
- `step3_delivery_sources_20261023.json`：服务器HEAD、51份实际文本源码与当前canvas依赖的SHA256、文档修改前后hash及source archive。

服务器本轮根目录：

`/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_step3_20261023_v1/`

其中`source/runtime_and_canvas_20261022T174927Z.tar.gz`固定实际运行依赖；含并行canvas工作、本步代码、文档，**不含图片/视频、模型权重或cache**。恢复到独立checkout核验，不覆盖正在工作的主树。`source/native_before_20261023T0130.tar.gz`保存部署前Native代码；每次文档更新也先保存原稿。

在服务器项目根目录使用现有环境复测CPU契约：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -B -m pytest \
  tests/test_qwen21_depth_condition.py tests/test_qwen21_native_depth.py \
  tests/test_qwen21_native_depth_runtime.py tests/test_qwen21_native_geometry.py \
  tests/test_qwen21_native_body.py tests/test_qwen21_native_resolution.py \
  tests/test_qwen21_canvas_coverage.py -q -p no:cacheprovider
```

数据CPU预检、depth manifest格式、显式off与warm start、checkpoint恢复代码见仓库`src/pose_control/qwen21_full/NATIVE_DEPTH.md`。重复运行需换新的output目录；所有审计JSON也拒绝覆盖。

## 范围与后续

默认1/16 attention会改变官方特征计算次序，新降采样也要训练；这里证明权重映射与attention原语正确，不证明迁移后质量优于随机初始化。下一轮应锁定pairs、预算、canvas和loss，对比训练后的noDepth Native、1/16 Depth和官方1/8位置，原Full继续固定为对照。

7B audit使用真实几何/Depth和合成text/latent，并手工激活新head以检查敏感性和反向；没有采样图片。3步真实训练只证明训练入口可学习，不构成生成质量、速度或收敛结论。

当前服务器的canonical canvas工作尚有独立修改，本步没有覆盖这些loader/cache文件。训练入口依赖这些接口，已包含在source archive；仅使用旧Native分支时应先集成该依赖。核心Depth、单测及新checkpoint重建独立于该loader。
