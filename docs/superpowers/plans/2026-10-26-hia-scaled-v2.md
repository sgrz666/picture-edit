# HIA 规模化 V2 实施计划

用户已选择可见部位清晰口径并授权更密抽帧，保留video split。目标是从数千视频生成实际规模的清晰数据，不以96帧小集结束；实际数量由质量门和验收决定，初始工程目标为万帧级主集。

1. GPU模型模块：复用已在盘YOLOX/DWPose/SCRFD，CUDA显式启用，真实模型计时及provider核验，禁止静默CPU回退。保留134点raw输出，主人物框与独立face结果分开。
2. 独立手定位：MediaPipe原生/腕部裁剪检测与身体手腕匹配；检测不到不能自动叫遮挡。依证据区分visible_clear、visible_blurry、occluded、out_of_frame、unknown，后两种无效但unknown不得自动通过。
3. 质量校准：去除V1三部位同时Lap60固定假设；脸看眼鼻口特征区，手看真实手ROI/指部边缘。初始门须对正负、未知、不可见等分层样本原生复核后冻结。独立校准视频不进入评估集。
4. 密集GPU流水线：先在至少20个视频验证，随后对既有数千视频按4FPS起步，必要动作窗8–15FPS加密。输出只按动作/时间近重复去重，不设每视频8帧或全库人工逐帧放行瓶颈；原生PNG、SHA/PTS和raw pose溯源保留。
5. V2独立契约：face_usable、每手status/usable、loss_mask、valid_keypoint_mask/effective_confidence；raw坐标响应不改。主集与严格子集分别manifest，严格子集继承主split。可见糊手整帧拒绝，遮挡/出画手关闭监督。
6. 流式/有界配对与视频组划分：冻结V1已公布视频归属及CHAMP源视频归属，源视频/RGB/确认近重复不得跨集合。身体、可用手、脸动作准入，禁止无效手点参与距离。
7. 分层验收与发布：对每个实际放行层从独立视频抽样查看原生ROI，记录误接收及覆盖；失败/未覆盖层不能发布。全路径/PNG/来源/PTS/有效点/分组检查，源像素回验抽样与全量文件hash范围明确记录，最终提供主集/严格子集的train/val/test规模与使用验证命令。

并行职责：模型worker仅负责v2_models.py/test_models.py；质量worker仅负责v2_quality.py/test_quality.py；导出worker仅负责v2_finalize.py/test_finalize.py；主任务负责独立手模块、GPU流水线、校准/复核及真实运行集成。所有worker不得改V1或互相撤销编辑。
