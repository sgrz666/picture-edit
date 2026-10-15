# backhug_pair02_037_c64_r545

- 类型：same_identity（Hi4D 原生配对，A 与 B 同演员同相机，只换动作）
- A：`backhug02` 帧 7；B：`backhug02` 帧 37，相机 64，relation `SAME_ACTION`
- 骨架 slot 顺序：按 B 中两人髖心 x 从左到右 → B, A（Hi4D 侧标记）
- 逐人关键点：`B_person_1_keypoints.npz`、`B_person_2_keypoints.npz`（DWPose-134，940×1280 原生坐标）
- 可用 GT：逐人掩码/深度/接触图、`A_occluded_by_B`、SMPL 参数与 8 相机内外参
- 不可用：手脸来自 SMPL（只有 24 关节）；被遮挡点的分数是检测器猜测；A 与 B 外观相同，这一类对测不出外观泄漏
