# TikTok Body + Face 联合训练与 BFS 小规模对照方案

状态：历史 512 方案，已获用户同意并部分执行，现已停止。用户最终要求保留原始像素宽高，后续实现与实际进度以 [实验交接](../../experiments/2026-10-08-bfs-body-face-joint-handoff.md) 为准。本文件保留旧设计，不作为当前启动方案。

## 判断与目标

Body 与 Face 同时参与生成，因此让它们共同适配有合理动机：几何分支改变人物位置、头部投影和 DiT 隐状态，Face 分支需要适应这些变化。完整架构支持组合输入和残差输出，但这不能保证联合优化胜过分阶段训练。联合更新也可能使身份、身体姿态和局部纹理目标产生冲突。

这次只回答短预算下的两个问题：已有 Body/Face 在原 Qwen 上共同微调是否有可见改善；在同一联合检查点后，固定 BFS Body 或固定 BFS Head 是否改善相同任务。对“联合优于冻结 Body”的因果结论，需要额外的匹配冻结对照，不能只用新模型对旧图片得出。

## 已核对的复用资产

- Face 现有训练清单：`/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/data/train_pairs.jsonl`，1,000 对、203 个分段序列、20 个原视频组。
- 现有 Face 验证清单：同目录 `val24_pairs.jsonl`，24 对。
- Face 训练的 Body 初始化来自 `tiktok_body_fullpass_20261004/training/checkpoint_step_006000.pt`，因此 Face 检查点中已经包含这一 Body 状态。
- 共同起点候选固定为 `/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_003000.pt`。执行前核对权重继承和当前兼容性，并重新计算文件哈希；旧 Face 候选的质量结果不能视为已经通过正式验收。
- BFS 本地已具备 Body V1.0、Head V1.1、Head V1.1 alternative 和 Head V1，均有 Qwen 2.1 版本。首轮只选 Body V1.0 和 Head V1.1，不遍历所有版本。
- 证据：`D:/codeplus/pictureedit/research/bfs_body_face_joint_design_20261008/context.json`，记录当前清单计数、哈希、代码指纹和初始化路径。

## 推荐预算与分组

从已有 1,000 对训练池固定抽取 512 对，seed 1708。按 20 个原视频组尽量均衡，先保证不同目标帧，减少同一片段相邻帧重复；不足的组在其他组中均衡补齐。筛选只读取输入几何、源身份特征及监督有效性，不读取生成结果。通过检查后锁定清单和顺序。

所有候选使用相同的 512 对及顺序。共同阶段使用前 256 对，分叉阶段使用后 256 对；一次遍历，不反复刷少量图。训练分辨率 512，沿用 letterbox。

| 阶段/组 | 固定底座 | Adapter 更新 | 追加步数 |
| --- | --- | --- | ---: |
| 共同阶段 J | 原 Qwen 2.1 | Body + Face | 256 |
| A | 原 Qwen 2.1 | 从 J 的同一状态更新 Body + Face | 256 |
| B | Qwen 2.1 + BFS Body V1.0 | 从 J 的同一状态更新 Body + Face | 256 |
| C | Qwen 2.1 + BFS Head V1.1 | 从 J 的同一状态更新 Body + Face | 256 |

主试验新增 1,024 次训练更新。每个最终候选经历共同 256 步及自身 256 步，合计新增 512 步。J 需要保存为一个独立、可恢复的检查点。

A/B/C 分叉时均新建相同配置的 optimizer 和学习率调度，保留相同 Adapter 参数状态、样本顺序及 sigma/噪声序列。不能让 A 继承优化器而 B/C 重置。B 与 C 分别从 J 开始，不能串行 Body-LoRA 训练后再加载 Head-LoRA。

所有 Qwen、BFS 参数固定；只更新 Adapter 的 Body 与 Face 参数并集，去除重复参数。Hand、共享 Detail 和 Interaction 冻结并关闭。BFS 只改变固定底座，不能因名称叫 Head 就只训练 Face，或叫 Body 就只训练 Body。

LoRA strength 首轮固定 1.0，两组记录实际加载、转换、缩放与权重指纹。混合加载 Body+Head、alternative、不同强度及更高分辨率均留给后续实验。

## 联合训练的具体含义

当前 `train_iper_qwen21_full.py` 的 `--branch` 只接受 `body/face`；`parameter_groups()`、`set_training_phase()` 与 `run_loss()` 也按两个独立分支执行。因此现有 CLI 不能直接冒充联合训练。

联合路径必须同时提供：

- Body：normal、Body25 heatmaps、part、mask、global26，以及原配置启用时的 depth。
- Face：B 的 jaw3、expression10、72 点面部几何；A 的 ArcFace/DINO 身份参考。
- 同一生成前向里的 Body 和 Face 残差。只合并 optimizer、仍使用 `with_face=False` 是无效接入。

Global26 保持原定义，细节通过已有独立 Face 接口传递。B RGB 仅用于同人样本的监督与离线评测，不能作为生成参考图或目标身份条件。跨人物 A/B 不作为重建训练样本。

首轮使用一套新的、明确标记的联合目标：有效区 Body flow loss，加 Face ROI 单独归一化的 flow loss，以及可靠低噪声阶段的身份/感知辅助项。建议初值为 `L_body + 0.5 * L_face_roi + 0.10 * L_id + 0.05 * L_perc`，所有候选一致；这是工程初值，需检查分项梯度与尺度，不是已证实最优的配方。

旧 Face 目标含 `.02 * outside`，参照是关闭 Face 的 Body 输出。首轮联合目标不沿用这一冻结教师约束；有效背景仍由现有有效区重建目标监督。若后续保留动态参照的越界约束，应独立记录定义，不能默默复用旧 Face 报表。

保留当前 Face 去噪时间调度和完整 Body sigma 范围，不将所有 Body 微调限制到 Face 的低噪声区。Face 调度关闭的步数仍更新 Body；记录 Face 激活次数。身份/感知辅助项只在 Face 激活、监督有效且 sigma 不超过 0.35 时，按既有四步间隔使用。不能强求每个高噪声步都出现 Face 梯度。

继承已训练输出桥接，不重新置零。建议 Body 使用旧学习率的 0.25 倍，Face 使用旧学习率的 0.5 倍；两者各保留原参数组比例，前 20 步做学习率线性 warmup。具体学习率随 manifest 锁定，A/B/C 不独立调参。

## 验证与查看效果

先做 4 对低噪声/Face 激活的梯度检查，要求 Body 和 Face 的关键模块均有有限、非零梯度；Qwen/BFS/Hand 保持冻结。另检查高噪声时 Body 仍正常学习。做一次短恢复一致性检查，确保分支、底座、LoRA 指纹、条件坐标和 optimizer 状态可追溯。

主比较复用原 24 对 TikTok；12 对 iPER 用于跨数据集诊断。当前 Face 缓存与监督资格需在生成前检查，不因检测失败更换样本。封存最终 test 不参与这轮筛选。

- 开始前评估共同旧检查点；共同训练后评估 J，展示 Body/Face 开关的少量诊断图。
- 各分叉保存追加 0/128/256 步。0 步含 BFS 的热加载诊断；128 步先查看固定 12 对，256 步完整查看 36 对。
- 所有组使用同一 A/B、prompt、采样 seed42、30 步、CFG1、512 分辨率。底座切换后清空 DiT 条件缓存，不共享跨权重 KV。
- 报告全图和脸部裁片，分别比较对 A 的身份、对 B 的身体姿态/头部方向、表情、衣服、背景、无脸/多脸和严重失败。脸部 ROI loss 下降不构成图像质量通过。
- 人物/视频组是统计聚类单位，重复帧不能当作独立身份。当前同人 TikTok 的提升不等于跨人物身份迁移得到验证。

B/C 最终必须对照相同追加步数的 A；同时对照自身 0 步，区分热加载与 Adapter 适配。小规模负结果只表示本协议的短适配没有显示收益，不能证明 BFS 原生能力无效或联合训练永远不如分阶段。

四对诊断出现错误来源、非有限值、冻结失效或梯度中断时先修复接口。若固定 12 对监测中相对起点新增至少 3 对明确严重失败，则停止后续更新并审查。首轮不会自动扩展超过上述预算。

## 想更严格验证联合训练时

可加一组 D：从共同阶段之前的同一旧检查点开始，在前 256 对上只更新 Face，Body 冻结，使用与 J 相同的联合条件、目标和 sigma/噪声序列。将 D 与 J 在相同数据暴露量下对照，隔离是否允许 Body 更新的影响。Face 调度关闭且无可训练路径时跳过更新并记录，不能改变 sigma 来制造梯度。

这增加最多 256 个训练迭代和一轮评测。它比较的是新的共同目标下的冻结消融；与历史 Face-only 配方不同。若不加 D，J 相对旧模型的改善仅能说明共同微调有效，不能单独证明联合优于继续 Face-only。首轮推荐先完成主试验，再根据结果决定是否需要这一归因组。

## BFS 的适用边界

[BFS Body 官方说明](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1-body.md)使用场景 RGB 与人物 RGB 两图，并强调参考格式和近似姿态匹配。[Head 官方说明](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md)也使用目标/场景图与身份参考图。

本轮仍是 A RGB + B 几何，验证固定 BFS 先验在现有输入协议下是否有帮助。它不会直接复现 BFS 原生两图工作流。Head V1.1 对应 Qwen 2.1，不能使用名称相近的 Edit 2509/2511 权重替代。

已有 30 对 Head 二次编辑实验中，BFS 更倾向保持中间图头部方向，但对 A 的身份相似度未胜过同条件原 Qwen 二次编辑。这支持继续验证取舍，不支持预设 BFS 全面更好；旧实验也没有测试本轮完整联合 Adapter。

## 怎样使用这个方案

先确认 512 对、共同 256 + 三组各 256 的预算与共同起点。实现联合路径及冻结 BFS 加载后，执行梯度/恢复检查，再按 J → A/B/C 的顺序运行，输出独立目录，不覆盖旧训练。

推荐实验目录：`/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/`。若该目录已有不同配置，使用新后缀。保存锁定清单、代码/权重指纹、共同初始化、逐模块训练梯度、分项 loss、全部比较图和失败记录。

当前没有可以直接运行的联合 CLI；本文件是具体实验提案，不代表已经完成接入或训练。执行入口必须在实现与检查通过后提供，不能使用虚构的 `--branch joint` 命令。
