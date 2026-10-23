# 第四步：Global Geometry v2 验收

2026-10-24，以 `sg` HEAD `11a5361` 和当时未提交的 canvas 修改为基线，先快照再实施。源码位于 `src/pose_control/qwen21_full/`；接口、维度与使用命令见 [GLOBAL_GEOMETRY_V2.md](../../../src/pose_control/qwen21_full/GLOBAL_GEOMETRY_V2.md)。本人完成，没有派 subagent。

## 完成内容

Native 默认新增 Body21：63 维局部轴角 → 前两列 rotation6D → 11→256→256 MLP＋Joint Embedding＋两层 Skeleton Attention。完整 22 节点树先算距离，再移除 pelvis token。原 4 个 Global＋1 个 Task Attention 后新增独立 Joint Cross-Attention；只在最后 512 维增量投影零初始化。空间 Gaussian 使用目标像素 cell 中心、同源 SMPL mask bbox 高度；projection valid 与 FK valid 分开，全不可投影行跳过并记录。

条件、双人子批次、collate、设备/类型迁移、canonical 数据读取、推理/审计及既有训练入口均已接线。新增 Pose 参数独立暴露，不随旧 Geometry 的默认冻结一起冻结。全程保留原 Full 和显式 Native v1 legacy，未触碰原 `src/pose_control/v6`。

| 参数/接口 | 数值 |
|---|---:|
| 原 Geometry 参数 | 75,966,284 |
| 新 Pose 参数 | 3,100,672 |
| v2 Geometry 参数 | 79,066,956 |
| Joint Tokens | 21×256 |
| Global Tokens / Task | 4×256 / 1×256 |
| Joint Attention | 512 维、8 heads |
| Native 输出 | B×N_target×512 |
| 增加的 Qwen token | 0 |

## 数据和兼容性证据

持久产物根目录：`/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1/`。

- TikTok 00001 的 0001–0008 共 8 帧；iPER 001_8 共 8 帧，覆盖 source_standard、source_motion、target。使用已有参数，未重跑估计。FK/mesh_cam 最大误差分别 `1.90735e-6`、`3.81470e-6`，低于 `1e-4`。16 帧相同 canonical 输入下，新 Pose 零增量与旧 Geometry 最大差均为 0。证据：`data/fk_v1/audit.json`、`report/data_audit_v1.json`。
- iPER 两个平移槽位改为 raw 相机的 `transl_raw`。本批与旧缓存最大差 `9.4317627`；这是输入语义修正，与零增量结构误差分开记录。旧缓存原样保留。
- iPER 512 条件图明确记录恢复到 1024 原图的 transform。Depth manifest 新增可选 `asset_to_original`，错误/重复尺度拒绝；未声明仍拒绝尺寸差。Depth 的神经编码与 512 维接口不变。
- 真实第三步 Native v1 checkpoint 分别重建 v1 和迁移 v2。389 个继承张量逐值一致，最大差 0；v2 仅新增 41 个声明的 Pose state key。两份输出都没有 optimizer 状态，可独立 strict 重建。证据：`report/checkpoint_audit_v1.json`；新文件位于 `checkpoints/native_legacy_rebuild_v1.pt` 和 `checkpoints/native_body21_v2.pt`。原 checkpoint 未覆盖。

## 前反向与调用链证据

服务器 152 项定向测试通过；Windows 工作树 59 项核心测试通过。服务器包含原 v6/Full legacy 数值与梯度保全、rotation 0/近 π/6D 往返、拓扑、21/256/512、空间响应、越界/全 mask、双人、strict checkpoint、缓存漂移和真实小型 Qwen 32-block 前反向。原 47 个 v6 源文件继续按历史 manifest 核对；5 个允许扩展的拷贝文件在 `native_v2_migration_manifest.json` 明确列出新旧 SHA，而非放宽任意文件检查。

最终真实 7B＋CHAMP Depth 验收：

| 检查 | 结果 |
|---|---:|
| 目标画布 / 网格 | 704×384 / 44×24 |
| 目标 token | 1056 |
| 检查的全部 prefix 位置 | 260，最大差 0 |
| 基座原输出与零 Body 输出差 | 0 |
| 总输出序列 | 1316，和同输入基座相同 |
| 激活新 head 后目标响应最大差 | 0.16015625 |
| Pose MLP 梯度 norm | 2.748e-5 |
| Skeleton 两层梯度 norm | 1.610e-4 / 1.645e-4 |
| Joint Cross-Attention / Projection 梯度 norm | 7.440e-4 / 2.302e-3 |
| Depth CNN / Attention 梯度 norm | 1.214e-5 / 4.545e-3 |
| 六组 Body head 梯度 | 全部非零、有限 |
| Qwen/重算期间的 Pose 编码次数 | 0 |
| 基座及旧 Geometry | 冻结，无梯度/更新 |
| 峰值分配显存 | 14.655 GiB |

Qwen 自身会把文本/参考位置展开，所以检查了目标前全部 260 个位置，未只检查前 256 个参考图位置。两次条件编码各执行一次关节编码器，之后 Qwen 的 32 个 block 和 backward checkpoint 重算只读取特征。证据：`report/7b_audit_final_v2.json`。

训练入口仅执行两套各 8 pair 的 `--preflight-only`，Depth 均启用，TikTok 网格 44×24、iPER 32×32。推理入口完成 v2 checkpoint 的 Depth-off TikTok 和 Depth-on iPER 条件前向预检。`report/training_policy_audit_v1.json` 确认原训练循环、AdamW 构造及 velocity loss 的 AST 完全相同。

本轮没有创建 optimizer 或执行训练更新；7B 梯度验收只在内存扰动新 head，没有保存其扰动权重。迁移工具写的是新的、未训练 Pose 初始化 checkpoint。没有执行完整采样或生成质量评测，不能据此声称动作控制质量改善。环境已有 NVML、Swig/Pillow 提示仍存在，未安装或替换任何包。

## 复查

在服务器项目根目录执行：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest \
  tests/test_qwen21_global_pose.py tests/test_qwen21_pose_cache.py \
  tests/test_qwen21_full_preservation.py tests/test_qwen21_native_geometry.py \
  tests/test_qwen21_native_depth_runtime.py tests/test_qwen21_native_depth.py \
  tests/test_qwen21_native_body.py tests/test_canvas_contract.py \
  tests/test_geometry_pair_dataset_v2.py -q
```

缓存构建、checkpoint 迁移、推理预检和配置消融的可复制命令见接口文档。重新执行写文件的命令必须使用新 output 名称；权重、FK cache 与图像不进入 Git。
