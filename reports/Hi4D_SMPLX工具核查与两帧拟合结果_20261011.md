# Hi4D SMPL-X 工具核查与两帧试拟合结果

## 完成内容

已检查远端 `ssh sg` 的工具、模型资产与 Hi4D 已解压标注；随后用已有 Multi-HMR 初始化，在真实相机与原 SMPL 几何约束下完成 `pair14/talk14/frame32`、`pair14/hug14/frame62` 两帧的双人 SMPL-X 拟合。

输出四个人体的参数和网格、八视角的三维全身关键点及投影图。这里的关键点使用 DWPose 的排列方式，但来自 SMPL-X；另有多视角 DWPose 三角化的稀疏三维观测。**结果是拟合伪标注，不是官方 SMPL-X 真值。**

本轮是两帧的数据拟合实验，没有启动 Adapter 训练，也没有将结果自动覆盖到现有 Hi4D 数据集或接入其训练入口。

## 现成资源核查

远端项目根目录：`/home/shangguanrz/project/pic-edit`。

| 项目 | 核查结果 |
| --- | --- |
| Multi-HMR | 已有 SMPL-X 专用 demo、896_L 权重和运行环境；本轮实际使用 |
| SMPLest-X | 已有完整代码、Huge 权重和运行环境；本轮确认存在，未另跑其预测 |
| SMPL-X 模型 | 中性、男性、女性资产均存在；本次与 Multi-HMR 对齐，拟合中性模型的体型参数 |
| DWPose | 已有官方 ONNX 模型、全身检测、绘图代码；本轮复用姿态模型 |
| Hi4D 已有 SMPL-X | 对 Hi4D、实验、研究目录共 14,269 个 NPZ 检查字段，未找到符合 SMPL-X/三维全身参数模式的旧结果；搜索不包括所有磁盘和未打开的压缩包 |
| Hi4D 三维身体骨架 | 有 `joints_3d [2,24,3]`，属于 SMPL 身体关节 |
| Hi4D DWPose | 现有 `dwpose_A/B.npz` 是二维 `xy_native [134,2]` 和置信度，不含三维全身骨架 |
| 官方模型转换对应表 | 检查 add/tools、add/models、third_party、archive 后未找到 smpl2smplx/deftrafo 对应资产；本轮不冒用官方拓扑对应关系 |

[DWPose 官方仓库](https://github.com/IDEA-Research/DWPose)提供二维全身姿态估计。三维化需要标定多视角三角化，或通过 SMPL-X 拟合生成三维关节，不能给二维坐标补一列零就称为三维骨架。[Hi4D 官方结构说明](https://github.com/yifeiyin04/Hi4D/blob/main/dataset.md)列出的注册人体模型是 SMPL。

工具、模型大小、SHA256、字段与搜索范围保存在远端 `research/hi4d_smplx_pilot_20261011/asset_audit.json`。

## 实际拟合流程

1. 从 `/data/shangguanrz/pic-edit-datasets/Hi4D-origin/pair14.tar.gz` 按需提取两帧各八视角，共 16 张图。只写入新的研究目录，未全量解包。
2. 对每张原图运行 Multi-HMR，使用 Hi4D 标定 K，并按实际缩放、填充转换到模型画布。没有使用旧测试的假设 FOV 60 度。
3. 根据原 SMPL 关节的投影建立固定人物 ID 0/1 与检测结果的对应。多人位置交换不会自动交换 A/B；漏检保留缺失槽位。
4. 复用现有 DWPose 姿态模型，在 SMPL 投影的人物区域上估计二维全身点，重新检查对应，剔除缺失或不可靠检测；使用相机标定进行鲁棒三角化。
5. 以 Multi-HMR 的身体、手、脸参数初始化 SMPL-X，用原 SMPL 前 22 个身体关节、迭代最近身体表面顶点、可靠的多视角关键点、原接触对距离及参数先验进行 500 次优化。两帧的同一人物共享拟合 betas；SMPL betas 没有直接复制到 SMPL-X。
6. 保存标准 SMPL-X 参数，分别在各相机重新构建网格和面部轮廓点，生成叠图、骨架及 OBJ。

Multi-HMR 的头部平移、标准 SMPL-X 的加性平移需要转换；相机旋转也必须进入全局姿态。已通过真实模型网格重建和相机空间对照检查。`flat_hand_mean=True`，`use_pca=False`，手部参数为每只手 45 维 axis-angle；眼球姿态默认零并标明未观测。

## 数值结果与实际意义

下表是相对原 SMPL 前 22 个身体关节的平均距离；它是身体拟合一致性指标，不能解释为完整 SMPL-X 手脸的真实精度。

| 帧 | 人物 | 初始化身体误差 | 拟合后身体误差 | 拟合身体表面至原 SMPL 最近顶点平均距离 |
| --- | --- | ---: | ---: | ---: |
| talk14 / 32 | A / ID 0 | 34.81 mm | 10.44 mm | 11.97 mm |
| talk14 / 32 | B / ID 1 | 50.64 mm | 12.48 mm | 11.37 mm |
| hug14 / 62 | A / ID 0 | 52.72 mm | 11.00 mm | 12.01 mm |
| hug14 / 62 | B / ID 1 | 97.89 mm | 13.13 mm | 10.89 mm |

拥抱帧抽取 160 对原 SMPL 接触表面对应点：原平均距离 12.40 mm，拟合后平均距离 16.50 mm，逐对距离差的平均绝对值 6.13 mm。SMPL-X 点通过最近表面估计对应，**不是官方跨模型顶点对应表，也不是独立接触质量评估。**

真实参数重新构建保存网格成功。16 个视角中，相机空间网格转换最大坐标差约 `6.89e-7 m`；这个数值检查说明坐标转换一致，不代表模型拟合达到亚微米精度。

## 手部、脸部与旧标注的限制

- 源 talk 帧，两人的双手各有 21 个点达到三视角观测条件。拟合手关节与这些三角化点平均距离分别约 5.42 / 8.45 / 15.39 / 8.66 mm；这是相对于自动关键点的误差。
- 目标 hug 帧，A 左手有 18 个两视角三角化点，但没有达到三视角拟合条件；其余手缺少可靠三维观测。目标双手主要依靠网络先验和有限二维约束，图中用灰色标明，不能当可靠手势真值。
- 人脸 17 个轮廓点随视角变化，排除于固定三维关键点拟合，导出时为每个相机单独生成。其余面部点参与置信度和几何筛选。
- 旧 cam4 的 DWPose A/B 身体点相对 SMPL 投影的加权偏差约 76.68 / 47.18 像素，存在遮挡或绑定不一致风险。本轮没有直接沿用旧二维文件，也没有修改它们；仅凭该指标不能证明身份错标。
- 尚无专门的 SDF 穿透优化。网格图可用于检查整体贴合和交互，但手指接触、细小穿模仍需要进一步处理。

## 输出结构与使用

远端工作目录：

```text
/home/shangguanrz/project/pic-edit/research/hi4d_smplx_pilot_20261011/
  asset_audit.json
  extraction.json
  inputs/pair14/{talk14,hug14}/images/{camera}/...
  initialization/             # 16 次 Multi-HMR 输出及模型版本
  observations/               # 二维 DWPose、三角化 xyz/valid/inlier_views
  fit_v1/
    smplx_world.npz
    wholebody134_by_camera.npz
    result.json
    visuals/
      *_mesh.png
      *_projection.png
      *_skeleton.png
      *_3d_skeleton.png
      *_actor{0,1}.obj
      *_keypoints.npz
      pilot_contact_sheet.jpg
      export_checks.json
```

`smplx_world.npz` 的前两项是源 A/B，后两项是目标 A/B。包含 `betas [4,10]`、`global_orient [4,3]`、`body_pose [4,63]`、左右手各 `[4,45]`、`jaw_pose [4,3]`、`expression [4,10]`、`transl [4,3]`、`vertices_world [4,10475,3]`、`joints_world [4,144,3]` 与人物、帧标识。单位为米，Hi4D 世界 Y 轴向上。

`wholebody134_by_camera.npz` 保存 `[4,8,134,3]` 的世界/相机关键点及观测状态。布局是 body18、feet6、face68、左手21、右手21；17 个面部轮廓点以各自相机为参考，其余解剖点可跨视角对照。使用 `observed_multiview` 与 `inlier_views` 区分自动观测与模型估计；三视角可靠条件为 `observed_multiview & (inlier_views >= 3)`。

OBJ 可导入 Blender 或 MeshLab 检查三维人体。参数可供后续重新渲染，但当前 Hi4D converter 仍有 SMPL 的 6890 顶点、部位和接触索引假设；接入 Adapter 前应升级转换器并重新生成控制资产。

本地已经下载参数、关键点、JSON 和主要图板到 `D:/codeplus/pictureedit/research/hi4d_smplx_pilot_20261011/`。

## 如何测试与复现

登录 `ssh sg` 后执行。核查现有结果：

```bash
cd /home/shangguanrz/project/pic-edit
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -m unittest discover \
  -s research/hi4d_smplx_pilot_20261011 -p 'test_*.py'
```

覆盖 7 项相机/映射/匹配/三角化检查和 5 项真实导出检查。真实模型重建、身体对齐改善、人物体型共享、缺失手部标记与 16 视角导出均检查；测试不宣称已验证未知手指真值。

重新核查资产：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python \
  research/hi4d_smplx_pilot_20261011/audit_existing.py \
  --output /tmp/hi4d_smplx_asset_recheck.json
```

使用已保存初始化和观测重新拟合到新目录：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python \
  research/hi4d_smplx_pilot_20261011/fit_smplx.py \
  --workdir /home/shangguanrz/project/pic-edit/research/hi4d_smplx_pilot_20261011 \
  --output-name fit_recheck --steps 500
/home/shangguanrz/nihaochen/envs/qwen21/bin/python \
  research/hi4d_smplx_pilot_20261011/export_visuals.py \
  --workdir /home/shangguanrz/project/pic-edit/research/hi4d_smplx_pilot_20261011 \
  --fit-name fit_recheck
HI4D_SMPLX_FIT=fit_recheck /home/shangguanrz/nihaochen/envs/qwen21/bin/python \
  -m unittest discover -s research/hi4d_smplx_pilot_20261011 -p 'test_*.py'
```

从已有权重重新生成初始化和观测，可分别运行 `initialize_multihmr.py`（`add/envs/multihmr/bin/python`）及 `prepare_dwpose.py`（`nihaochen/envs/qwen21/bin/python`），两者均传入同一 `--workdir`。它们固定处理这两帧；用于其他 Hi4D 帧前需要扩展帧清单并重新验证。

## 下一步建议

这组结果可用于双人身体、相对位置与遮挡的几何接口试验。进入批量训练前应优先处理接触帧的逐人二维关键点、隐藏手部质量标记、SMPL-X 部位/接触映射和穿透检查；不把所有输出参数存在视为所有细节都可信。
