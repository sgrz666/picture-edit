# Global Geometry v2：Body21 接线与架构验收

本轮在 `sg` 的 Native＋CHAMP Depth＋canvas 代码上加入 21 个关节条件。只验证编码、接口、兼容性和梯度，不执行训练更新，不以生成质量作为结论。

远端产物根目录为 `/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1/`。`source/source_20261024.tar.gz` 和 `source/baseline.json` 保存实施前源码、SHA256、HEAD `11a5361` 和未提交状态；每次 `source/delivery_*` 保留源码投递记录，并检查远端文件是否发生并发漂移。`scripts/prepare_pose_acceptance.py` 固定 8 个 TikTok 帧及 8 个 iPER 帧，后者覆盖 source_standard、source_motion、target 三个 role。`data/fk_v1/` 保存原始轴角、22 关节 FK、原图投影和来源，不保存可学习特征；数据与模型权重留在服务器，不进入 Git。

已核验的 16 帧最大 FK 网格误差：TikTok `1.90735e-6`，iPER `3.81470e-6`，门槛 `1e-4`。相同 canonical 输入下，新增 Pose 零投影与旧前端输出差为 0。iPER 两个平移槽位改用 `transl_raw`，本批最大输入变化 `9.43176`，不属于结构数值误差。

7B 验收使用 704×384 画布、44×24 网格、1056 个目标 token，目标区前的全部 260 个位置差为 0，基座及旧 Geometry 冻结。Pose、CHAMP Depth 和 Body 均有梯度；Qwen 的 32 个 block 及 checkpoint 重算不再执行关节编码器。峰值分配显存 14.66 GiB。新 head 只在内存中扰动以验证梯度，没有 optimizer、训练更新或基座权重写入。

上游是前三步 Native Geometry/Body/Depth 与远端 canvas 合同。正式接口和复查命令见仓级 `src/pose_control/qwen21_full/GLOBAL_GEOMETRY_V2.md`，最终验收汇总见本实验族 `report/STEP4_VERIFICATION.md`。后续训练与质量对照另行开展。
