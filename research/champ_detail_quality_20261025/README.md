# CHAMP pristine_v2 脸与手部清晰度审查

采集时间：2026-10-25 17:19（Asia/Shanghai）。源目录为 `sg:/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024/cleaned_training_images/train_rgb_pristine_v2`。

**这批图里仍有运动模糊、细节像素不足、脸部效果与局部遮挡。继续提高单个 Laplacian 阈值不能保证变成高清细节 GT。** 建议保留原件，通过清单分别管理身体姿态、Face ID、Face 细节与左右手监督资格。本次只读审查，没有新建正式清洗集，也没有删除、移动或改写任何原图。

## 里面有什么、怎样查看

- [analysis.ipynb](analysis.ipynb)：4 个实际执行的只读 Python 单元，可从本目录或项目根目录复算计数、配对覆盖与哈希。
- [snapshot.json](snapshot.json)：46 张图的源路径、原尺寸输入、PNG SHA、原始 DWPose 数值和元数据哈希。
- `source/`：既有质量记录、ROI 布局/几何门控代码、原始清洗文档及两版训练清单；不是重新执行旧清洗脚本。
- [inspection/records.json](inspection/records.json)：逐帧已有分数、几何可用性、原生 ROI 与审查提示。新算的框内/凸包内 Laplacian 仅作诊断，没有新增自动准入阈值。
- [inspection/proposed_frame_policies.jsonl](inspection/proposed_frame_policies.jsonl)：46 条按用途的**待确认建议**，没有替换训练清单。`review`、`human_confirmation_required` 不表示训练已获准。
- `input/train_rgb_pristine_v2/`：46 张逐字节一致的原图副本；`inspection/native_crops/`：123 张可得的原生像素脸/手裁剪，没有锐化、超分或 AI 修复。

按文件名核对这 8 页；预览只改变显示尺寸，判断细节时可打开对应原生裁剪。`geom=True` 只表示既有位置/可见点门通过，**不表示纹理清晰**。手窗口 A/B 沿用原文件索引，不直接猜作画面左/右手。

[第1页](inspection/sheets/review_01.png) · [第2页](inspection/sheets/review_02.png) · [第3页](inspection/sheets/review_03.png) · [第4页](inspection/sheets/review_04.png) · [第5页](inspection/sheets/review_05.png) · [第6页](inspection/sheets/review_06.png) · [第7页](inspection/sheets/review_07.png) · [第8页](inspection/sheets/review_08.png)

## 已核对的事实

1. 当前目录有 **46 张、4 段视频**，全部 1080×1920。46 张与原 `frames/{seq}/rgb/{stem}.png` 的 SHA256 一致，导出步骤没有进一步改变像素。PNG 无损不能恢复原视频本来没有的细节。
2. 新版 `CHAMP_v1/manifests/pairs/train_pairs_pristine_v2.jsonl` 是 **158 对、46 张去重图**，帧集合与目录完全一致。旧版 862 对仅作历史对照，不据此指认新版训练仍读旧清单；本轮未核对当前训练进程的实际 CLI。
3. 6 张远景图的既有人脸检测宽度只有 **94～106 像素**。这是细节预算事实，不是新增的 128 像素自动淘汰规则。
4. 92 个手窗口中，旧质量记录有 **15 个支持点少于 12**；缺失、遮挡或出画不能算成 15 个“模糊手”。既有几何门通过 63 个窗口，也不能算作 63 个“清晰手”。
5. 已看完整概览、四张原图及覆盖全部 46 帧的 8 张局部页。判断来自模型辅助视觉审查，**不是经过多人校验的人工质量标签**；逐帧最终细节资格仍保留待确认。

| 序列 | 帧数 / 配对数 | 可见问题 | 建议用途与限制 |
|---|---:|---|---|
| 000000 | 6 / 12 | 远景脸小、手部遮挡/运动拖影；局部高分仍有糊 | 保留为身体姿态候选；Face ID 与高频 Face 细节分别检查，手也按可见区域判断 |
| 000111 | 10 / 33 | 动物耳朵/鼻子效果，挥手拖影 | 自然人脸身份/皮肤细节训练建议排除；可以保留作为效果/鲁棒性样本，不能冒充自然高清脸 |
| 000444 | 19 / 77 | 脸整体偏软、眼镜反光、挥手拖影 | 逐帧审核；服装格纹和背景边缘不能代替眼鼻口、手指质量 |
| 000666 | 11 / 36 | 烟雾或白色覆盖，部分手部窗口重叠或几何门失败 | 分区域处理；未被覆盖的脸可能有用，覆盖部分不能当干净服装/人体纹理 GT |

### 两个具体反例

- [000111_00088 原图](input/train_rgb_pristine_v2/000111_00088.png)的既有手部 Laplacian 为 **272.83**，但[手局部](inspection/native_crops/000111_00088_hand_a.png)仍有明显拖影。说明“分数高”不能直接等同“手指清晰”。
- [000000_00144 原图](input/train_rgb_pristine_v2/000000_00144.png)两只手的既有分数为 **34.25 / 80.35**，原图仍有运动模糊；其脸宽仅 98 像素。

原清洗文档也记录过全人体 Laplacian、相对手/躯干比、重影特征等的漏检。部分历史标签来自模型，不能当独立人工真值。本轮没有定位到最后一次 `pristine_v2` 导出对应的完整过滤脚本/阈值记录，因此不把推测的规则说成已确认实现。

注意旧 `detail_roi/report.json` 的手部 FAIL 诊断用过错误的身体锚点；后续 `hand_gate` 代码已按实际手腕和前臂校验。本次采用后者的既有几何判定，未因旧报告再推断所有 ROI 都坏了。

## 建议怎样处理

**先分资格，再筛帧，不再把整张图一概标成 pristine。** 一张图可以是好的身体姿态样本，但不适合高频 Face/Hand 细节监督。

1. 建四类资格：身体姿态、自然 Face ID、Face 细节、每只手细节。动物脸效果从自然 Face 监督候选中排除；手看不见、握拳、双手交叠与运动模糊分开，不能要求每帧都看见五指。新手部检查要先复用几何/可见性门，再看纹理。
2. 对高频 GT，逐帧看眼鼻口边缘、指缝与指节是否实际可辨、是否有重影，以及区域是否被覆盖。Laplacian/频谱分数用于排序候选，最终阈值要在独立、分类型的小批标签上校准；按视频取前若干名不能强迫每段都保留“高清帧”。
3. 优先从原视频同一动作附近找更清楚的帧。整段始终偏软或带效果时换素材，不在同一段中反复筛。替换帧必须作为新样本记录，并重算/校验对应 DWPose、SMPL-X、条件图、Face/Hand 缓存和配对，不能把新 RGB 套旧几何。
4. 训练时，A 参考图和 B 监督目标分别审核。想保留复杂姿态而不学习糊细节，可以保留可信几何，降低/排除不可信局部监督；**如果全图 flow/MSE 仍在模糊 ROI 上计算，仅关掉 Face/Hand 辅助损失并没有移除模糊监督**。这条是方案建议，不是本轮对当前训练实现的结论。
5. 不把 AI 修脸、补手、锐化或超分结果直接混入真实高清 GT。它们可能添加原图没有的细节；若另做伪标签实验，应独立记录来源与身份/手指一致性。本轮未修复任何图片。
6. 当前有效来源只有 4 段视频。对这些来源反复组合成 158 对不会增加身份/场景多样性；扩充无效果、细节真实清晰的来源，比在这 46 张上反复增加清洗轮数更有价值。

## 使用边界与复核

本次建议清单只供审查，没有生成新的正式 v3 配对、没有变更训练入口。可先在 `proposed_frame_policies.jsonl` 基础上补人工确认结果，再在实验目录生成新白名单/版本化清单；原图保持不动。不要把 `review` 当作通过。

只读快照脚本为 [collect_readonly.py](collect_readonly.py)，局部审查构建脚本为 [build_review.py](build_review.py)。前者在独立 `/data/shangguanrz/pic-edit-audits/` 目录执行并只向标准输出写 JSON；没有在 datasets/models 下新增产物。后者只处理本地副本，已有 inspection 目录会拒绝重建，复算计数优先用 notebook。

本地审查目录：`D:/codeplus/pictureedit/research/champ_detail_quality_20261025`。服务器脚本路径记录在 [remote_audit_path.txt](remote_audit_path.txt)。如再次采集，使用新的时间戳快照，保留这一轮证据。
