# J256 独立图像审阅

**旧协议归档（2026-10-08）**：本记录的 iPER 12 对使用 resize，现已被“全部数据使用项目 letterbox”的明确要求取代。下文旧 iPER 数值、观察与混合 36 对结论只保留作溯源，不用于当前最终比较；TikTok 24 对原本为 letterbox，仍可参考。后续最终对照必须使用同一固定 iPER 12 对的新 letterbox 输出。

已逐张检查固定 36 对：TikTok 24 对、iPER 12 对的整图和同一 B 目标脸框裁片，对照初始模型、A 身份与离线 B 姿态/表情。A/B 图像哈希与目标脸框在两次评估中全部相同。
相对初始输出，未见新增整人崩坏、严重身份/肢体破坏；记录 12 对局部中等退化、14 对局部改善、8 对稳定、2 对混合变化。中等退化并不意味着整图不可用；改善也不意味着完整满足 B。
iPER 身份与严格姿态平均改善；TikTok 平均身份持平、严格姿态略降。头朝向与表情仍不可靠：侧脸或抬头目标常变为正脸；部分闭嘴目标出现张嘴或喊叫。该记录只提供观察，不改变 J_mid 门槛。

| 数据集（配对分母） | ArcFace 对 A：初始 → J256（差值） | PCK 2%：初始 → J256（差值） | PCK 5%：初始 → J256（差值） |
| --- | --- | --- | --- |
| TikTok 24/24 | 0.4305 → 0.4298 (-0.0007) | 0.5422 → 0.5141 (-0.0281) | 0.8875 → 0.8990 (+0.0115) |
| iPER 12/12 | 0.3766 → 0.4511 (+0.0745) | 0.8811 → 0.9259 (+0.0447) | 0.9948 → 0.9948 (+0.0000) |

两次评估均 36/36 生成、评分成功；A 与输出均检测到唯一脸，姿态均检出，身份与 PCK 配对有效分母各 36/36，无身份填补、无检测失败。Face 条件 35/36 有效：TikTok index9（val_0000666）在两次运行中来源 Face 几何均无效、A 脸部裁片缺失；整张 A 与固定 B 裁片已审阅，该样本不能支持有效 Face 控制结论。

| 原视频组 / iPER 人物前缀 | 固定对数 | ArcFace 差值 | PCK 2% 差值 | PCK 5% 差值 |
| --- | ---: | ---: | ---: | ---: |
| tiktok_v3/seq_00125_00139 | 8 | +0.0280 | -0.0131 | +0.0191 |
| tiktok_v3/seq_00231_00240 | 8 | -0.0629 | -0.0146 | -0.0096 |
| tiktok_v3/seq_00282_00288 | 8 | +0.0329 | -0.0567 | +0.0250 |
| iper/001 | 2 | +0.1216 | -0.0294 | +0.0000 |
| iper/006 | 1 | +0.0466 | +0.0000 | +0.0000 |
| iper/009 | 1 | +0.0813 | -0.0588 | +0.0000 |
| iper/012 | 2 | +0.0292 | +0.0625 | +0.0000 |
| iper/015 | 2 | +0.0769 | +0.0882 | +0.0000 |
| iper/019 | 2 | +0.1485 | +0.0882 | +0.0000 |
| iper/024 | 2 | +0.0069 | +0.0882 | +0.0000 |

TikTok 各组各 8 对；iPER 按人物前缀归组，组均值与样本均值分开保存在 JSON。视频帧相关，只有 3 个 TikTok 原视频组，不据此报告置信区间、显著性或优胜。

明确退化：TikTok 6（侧脸转正脸）、9（脸部更糊，条件无效）、10（近脸手更糊/偏移）、14（视线不对称）、15（喊叫表情）、18（新增张嘴）、19（大张嘴）、20（新增露齿笑）；iPER 0（头向侧下偏移）、1（丢失目标笑容）、10（头更低/更正）、11（闭眼大张嘴）。完整逐例说明和指标在 JSON。
明确改善例：TikTok 0 嘴形更近 B，3 去掉初始不自然大笑，11 眼部更连贯，13 闭嘴更近 B，21 前伸拳透视更近 B，22 嘴形更近 B；iPER 7/8/9 的脸部或眼镜更近 A，9 去掉初始夸张嘴唇与眼镜畸变。稳定例包括 TikTok 1/7/12/16/23 和 iPER 2/3/4。
仍然存在的较大控制偏差：TikTok 3 与 18 都未达到 B 明显抬头/仰头；部分前伸拳和手指透视依旧错误。iPER 人脸原生分辨率小，细小表情判断有限。ArcFace 改善和较高 PCK 不能证明表情或精确头朝向已经成功。

J 正式训练记录 256 步，Face 激活 159 步，辅助项请求 64 次、实际使用 20 次，有效辅助脸 20；未计私有预检更新。
最初 iPER 缺缓存的基础设施尝试已排除，使用同一固定 12 对的修正后初始运行作基线；[归档原因](../diagnostic_failures/init_initial_iper_missing_cache/rerun_reason.json)与[原始失败记录](../diagnostic_failures/init_initial_iper_missing_cache/per_sample.jsonl)仍可追溯。

逐例检查链接如下：整图面板含 A、离线 B 与输出，脸部裁片固定使用 B 目标框。完整观察、检测状态、有效分母、分组数值和文件哈希见 [JSON](J_final_observations.json)。

| 数据集 / index / pair_id | 视觉变化 | 初始整图 | J 整图 | J 固定脸框 |
| --- | --- | --- | --- | --- |
| tiktok_v3/00/val_0000032 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/000_00127_0115_0127_c38174d0e9_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/000_00127_0115_0127_c38174d0e9_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/000_00127_0115_0127_c38174d0e9_output.png) |
| tiktok_v3/01/val_0000397 | 稳定 | [查看](../evaluation/init_initial_tiktok/panels/001_00138_0038_0058_35379dc46a_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/001_00138_0038_0058_35379dc46a_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/001_00138_0038_0058_35379dc46a_output.png) |
| tiktok_v3/02/val_0000116 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/002_00130_0032_0194_a1866d927b_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/002_00130_0032_0194_a1866d927b_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/002_00130_0032_0194_a1866d927b_output.png) |
| tiktok_v3/03/val_0000389 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/003_00137_0040_0382_3c4f030bdf_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/003_00137_0040_0382_3c4f030bdf_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/003_00137_0040_0382_3c4f030bdf_output.png) |
| tiktok_v3/04/val_0000192 | 混合变化 | [查看](../evaluation/init_initial_tiktok/panels/004_00132_0074_0037_d718830a37_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/004_00132_0074_0037_d718830a37_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/004_00132_0074_0037_d718830a37_output.png) |
| tiktok_v3/05/val_0000012 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/005_00126_0021_0090_3010f91535_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/005_00126_0021_0090_3010f91535_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/005_00126_0021_0090_3010f91535_output.png) |
| tiktok_v3/06/val_0000037 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/006_00128_0225_0018_0ac8a34d80_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/006_00128_0225_0018_0ac8a34d80_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/006_00128_0225_0018_0ac8a34d80_output.png) |
| tiktok_v3/07/val_0000175 | 稳定 | [查看](../evaluation/init_initial_tiktok/panels/007_00131_0207_0343_b51e5d65bd_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/007_00131_0207_0343_b51e5d65bd_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/007_00131_0207_0343_b51e5d65bd_output.png) |
| tiktok_v3/08/val_0000440 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/008_00231_0276_0189_72a1e5c263_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/008_00231_0276_0189_72a1e5c263_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/008_00231_0276_0189_72a1e5c263_output.png) |
| tiktok_v3/09/val_0000666 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/009_00238_0236_0042_7ce93ccf87_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/009_00238_0236_0042_7ce93ccf87_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/009_00238_0236_0042_7ce93ccf87_output.png) |
| tiktok_v3/10/val_0000594 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/010_00235_0026_0128_11b2ad44b6_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/010_00235_0026_0128_11b2ad44b6_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/010_00235_0026_0128_11b2ad44b6_output.png) |
| tiktok_v3/11/val_0000531 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/011_00233_0381_0129_f85dc59d64_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/011_00233_0381_0129_f85dc59d64_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/011_00233_0381_0129_f85dc59d64_output.png) |
| tiktok_v3/12/val_0000632 | 稳定 | [查看](../evaluation/init_initial_tiktok/panels/012_00236_0211_0141_ca6f32c25f_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/012_00236_0211_0141_ca6f32c25f_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/012_00236_0211_0141_ca6f32c25f_output.png) |
| tiktok_v3/13/val_0000816 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/013_00240_0110_0185_86877d9082_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/013_00240_0110_0185_86877d9082_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/013_00240_0110_0185_86877d9082_output.png) |
| tiktok_v3/14/val_0000761 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/014_00239_0365_0319_8b619e6a30_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/014_00239_0365_0319_8b619e6a30_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/014_00239_0365_0319_8b619e6a30_output.png) |
| tiktok_v3/15/val_0000499 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/015_00232_0165_0223_52a4dc955a_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/015_00232_0165_0223_52a4dc955a_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/015_00232_0165_0223_52a4dc955a_output.png) |
| tiktok_v3/16/val_0000896 | 稳定 | [查看](../evaluation/init_initial_tiktok/panels/016_00283_0187_0014_ef70627d4e_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/016_00283_0187_0014_ef70627d4e_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/016_00283_0187_0014_ef70627d4e_output.png) |
| tiktok_v3/17/val_0001022 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/017_00286_0281_0055_30ac51553d_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/017_00286_0281_0055_30ac51553d_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/017_00286_0281_0055_30ac51553d_output.png) |
| tiktok_v3/18/val_0000868 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/018_00282_0002_0166_f5055f4ce0_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/018_00282_0002_0166_f5055f4ce0_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/018_00282_0002_0166_f5055f4ce0_output.png) |
| tiktok_v3/19/val_0000974 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/019_00285_0270_0105_f149120786_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/019_00285_0270_0105_f149120786_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/019_00285_0270_0105_f149120786_output.png) |
| tiktok_v3/20/val_0001112 | 中等退化 | [查看](../evaluation/init_initial_tiktok/panels/020_00287_0130_0275_52a5a2b810_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/020_00287_0130_0275_52a5a2b810_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/020_00287_0130_0275_52a5a2b810_output.png) |
| tiktok_v3/21/val_0000941 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/021_00284_0239_0017_b4f8b981c3_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/021_00284_0239_0017_b4f8b981c3_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/021_00284_0239_0017_b4f8b981c3_output.png) |
| tiktok_v3/22/val_0000929 | 局部改善 | [查看](../evaluation/init_initial_tiktok/panels/022_00283_0187_0251_90713646df_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/022_00283_0187_0251_90713646df_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/022_00283_0187_0251_90713646df_output.png) |
| tiktok_v3/23/val_0000990 | 稳定 | [查看](../evaluation/init_initial_tiktok/panels/023_00285_0301_0221_9184769dad_whole.jpg) | [查看](../evaluation/J_common_tiktok/panels/023_00285_0301_0221_9184769dad_whole.jpg) | [查看](../evaluation/J_common_tiktok/faces/023_00285_0301_0221_9184769dad_output.png) |
| iper/00/val_000 | 中等退化 | [查看](../evaluation/init_initial_iper/panels/000_001_2_source_standard_f000681_target_f001344_5236e955b1_whole.jpg) | [查看](../evaluation/J_common_iper/panels/000_001_2_source_standard_f000681_target_f001344_5236e955b1_whole.jpg) | [查看](../evaluation/J_common_iper/faces/000_001_2_source_standard_f000681_target_f001344_5236e955b1_output.png) |
| iper/01/val_001 | 中等退化 | [查看](../evaluation/init_initial_iper/panels/001_001_5_source_motion_f000815_target_f000135_85d9ea30d8_whole.jpg) | [查看](../evaluation/J_common_iper/panels/001_001_5_source_motion_f000815_target_f000135_85d9ea30d8_whole.jpg) | [查看](../evaluation/J_common_iper/faces/001_001_5_source_motion_f000815_target_f000135_85d9ea30d8_output.png) |
| iper/02/val_002 | 稳定 | [查看](../evaluation/init_initial_iper/panels/002_006_1_source_motion_f001625_target_f001189_d146897ead_whole.jpg) | [查看](../evaluation/J_common_iper/panels/002_006_1_source_motion_f001625_target_f001189_d146897ead_whole.jpg) | [查看](../evaluation/J_common_iper/faces/002_006_1_source_motion_f001625_target_f001189_d146897ead_output.png) |
| iper/03/val_003 | 稳定 | [查看](../evaluation/init_initial_iper/panels/003_009_3_source_motion_f000168_target_f000420_6ad164b82a_whole.jpg) | [查看](../evaluation/J_common_iper/panels/003_009_3_source_motion_f000168_target_f000420_6ad164b82a_whole.jpg) | [查看](../evaluation/J_common_iper/faces/003_009_3_source_motion_f000168_target_f000420_6ad164b82a_output.png) |
| iper/04/val_004 | 稳定 | [查看](../evaluation/init_initial_iper/panels/004_012_4_source_motion_f000491_target_f000579_8b1151c115_whole.jpg) | [查看](../evaluation/J_common_iper/panels/004_012_4_source_motion_f000491_target_f000579_8b1151c115_whole.jpg) | [查看](../evaluation/J_common_iper/faces/004_012_4_source_motion_f000491_target_f000579_8b1151c115_output.png) |
| iper/05/val_005 | 混合变化 | [查看](../evaluation/init_initial_iper/panels/005_012_4_source_motion_f000491_target_f001406_e712de22a6_whole.jpg) | [查看](../evaluation/J_common_iper/panels/005_012_4_source_motion_f000491_target_f001406_e712de22a6_whole.jpg) | [查看](../evaluation/J_common_iper/faces/005_012_4_source_motion_f000491_target_f001406_e712de22a6_output.png) |
| iper/06/val_006 | 局部改善 | [查看](../evaluation/init_initial_iper/panels/006_015_1_source_motion_f000445_target_f000504_7df8fadc47_whole.jpg) | [查看](../evaluation/J_common_iper/panels/006_015_1_source_motion_f000445_target_f000504_7df8fadc47_whole.jpg) | [查看](../evaluation/J_common_iper/faces/006_015_1_source_motion_f000445_target_f000504_7df8fadc47_output.png) |
| iper/07/val_007 | 局部改善 | [查看](../evaluation/init_initial_iper/panels/007_015_1_source_motion_f000747_target_f001553_6c9edd3985_whole.jpg) | [查看](../evaluation/J_common_iper/panels/007_015_1_source_motion_f000747_target_f001553_6c9edd3985_whole.jpg) | [查看](../evaluation/J_common_iper/faces/007_015_1_source_motion_f000747_target_f001553_6c9edd3985_output.png) |
| iper/08/val_008 | 局部改善 | [查看](../evaluation/init_initial_iper/panels/008_019_3_source_motion_f001183_target_f001253_a3c19f067a_whole.jpg) | [查看](../evaluation/J_common_iper/panels/008_019_3_source_motion_f001183_target_f001253_a3c19f067a_whole.jpg) | [查看](../evaluation/J_common_iper/faces/008_019_3_source_motion_f001183_target_f001253_a3c19f067a_output.png) |
| iper/09/val_009 | 局部改善 | [查看](../evaluation/init_initial_iper/panels/009_019_3_source_motion_f001183_target_f000311_29049b864f_whole.jpg) | [查看](../evaluation/J_common_iper/panels/009_019_3_source_motion_f001183_target_f000311_29049b864f_whole.jpg) | [查看](../evaluation/J_common_iper/faces/009_019_3_source_motion_f001183_target_f000311_29049b864f_output.png) |
| iper/10/val_010 | 中等退化 | [查看](../evaluation/init_initial_iper/panels/010_024_2_source_motion_f000454_target_f000714_168c3a39aa_whole.jpg) | [查看](../evaluation/J_common_iper/panels/010_024_2_source_motion_f000454_target_f000714_168c3a39aa_whole.jpg) | [查看](../evaluation/J_common_iper/faces/010_024_2_source_motion_f000454_target_f000714_168c3a39aa_output.png) |
| iper/11/val_011 | 中等退化 | [查看](../evaluation/init_initial_iper/panels/011_024_2_source_motion_f000454_target_f001410_29e8395318_whole.jpg) | [查看](../evaluation/J_common_iper/panels/011_024_2_source_motion_f000454_target_f001410_29e8395318_whole.jpg) | [查看](../evaluation/J_common_iper/faces/011_024_2_source_motion_f000454_target_f001410_29e8395318_output.png) |

## 同 4 对 Body-only / Face-only 诊断

额外检查 TikTok index 0、7、14、22 的 8 张分支诊断输出，并与相同 4 对 J Body+Face 输出逐对比较。使用同一 J256 底座、seed42、30 步、512 分辨率；不把 4 对均值与全量 36 对均值当作公平性能对比。

| 同一 4 对（身份/PCK 有效分母各 4/4） | ArcFace 对 A | PCK 2% | PCK 5% |
| --- | ---: | ---: | ---: |
| J | 0.4032 | 0.4468 | 0.7830 |
| JBodyOnly | 0.3875 | 0.4468 | 0.7830 |
| JFaceOnly | 0.9043 | 0.2307 | 0.3560 |

Face-only 四对都明显保留 A 的姿态/构图：0 保留侧脸和俯身位置，7 保留双臂举在头顶，14 保留低头/闭眼与向侧边摆臂，22 保留前伸手臂及 A 脸位置。22 的固定 B 脸框裁片甚至只剩天空，完整输出仍检出唯一脸。高 ID 主要伴随来源保留，不能解释为成功重定向。
Body+Face 与 Body-only 的主体姿态几乎一致，四对 PCK 逐例相同；同四对 ID 均值仅增加 0.0157。这只是当前四例诊断，不能推出 Face 无用或推广到所有样本。Face-only 关闭 Body 几何，A 的保留也可能来自底层图像编辑管线，不能单独归因于 Face。
辅助像素核对：Face-only 对 A 的有效画幅平均绝对 RGB 差（0—1）分别为 0.0079、0.0103、0.0305、0.0104，对离线 B 分别为 0.0612、0.1491、0.1771、0.0599。它只辅助验证来源位置保留，不作为身份/感知质量指标。

| index | Body-only 整图 | Face-only 整图 | Face-only 固定 B 脸框 |
| --- | --- | --- | --- |
| 0 | [查看](../evaluation/JBodyOnly_common_tiktok/panels/000_00127_0115_0127_c38174d0e9_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/panels/000_00127_0115_0127_c38174d0e9_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/faces/000_00127_0115_0127_c38174d0e9_output.png) |
| 7 | [查看](../evaluation/JBodyOnly_common_tiktok/panels/007_00131_0207_0343_b51e5d65bd_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/panels/007_00131_0207_0343_b51e5d65bd_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/faces/007_00131_0207_0343_b51e5d65bd_output.png) |
| 14 | [查看](../evaluation/JBodyOnly_common_tiktok/panels/014_00239_0365_0319_8b619e6a30_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/panels/014_00239_0365_0319_8b619e6a30_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/faces/014_00239_0365_0319_8b619e6a30_output.png) |
| 22 | [查看](../evaluation/JBodyOnly_common_tiktok/panels/022_00283_0187_0251_90713646df_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/panels/022_00283_0187_0251_90713646df_whole.jpg) | [查看](../evaluation/JFaceOnly_common_tiktok/faces/022_00283_0187_0251_90713646df_output.png) |
