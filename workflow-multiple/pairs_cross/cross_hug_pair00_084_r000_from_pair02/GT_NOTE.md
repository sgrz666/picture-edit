# cross_hug_pair00_084_r000_from_pair02

- 类型：cross_pair（A 取自 pair pair02 的源帧 `dance02`，B 取自 pair pair00 的目标帧 `hug00`）
- 人物对应关系：无 GT，评测按两种分配取较优
- A：`taichi00` 帧 9；B：`hug00` 帧 84，相机 4，relation `CROSS_ACTION`
- 骨架 slot 顺序：按 B 中两人髖心 x 从左到右 → B, A（Hi4D 侧标记）
- 逐人关键点：`B_person_1_keypoints.npz`、`B_person_2_keypoints.npz`（DWPose-134，940×1280 原生坐标）
- 可用 GT：逐人掩码/深度/接触图、`A_occluded_by_B`、SMPL 参数与 8 相机内外参
- 不可用：手脸来自 SMPL（只有 24 关节）；被遮挡点的分数是检测器猜测；A 与 B 外观相同，这一类对测不出外观泄漏
