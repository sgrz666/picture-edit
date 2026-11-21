# Native V7 Source 身份与背景保持

用户已根据300步实验授权改进。继续原生4Ref、冻结Qwen基座，在main完成；不另建worktree，不覆盖旧权重或实验。

## 依据与选择

训练／观察PCK@0.2达到95.83%，训练检测对齐身份却从[.711,.873]降至[.566,.770]，背景PSNR也下降。180步控制显示退步已在Joint前存在。现有身份正样本来自Target GT；相对GT的ID／contrastive下降不保证Source相似度提高。

首选改进训练目标，保留已经可学的PoseLink／MaskPose／Contact架构。单独调小Joint学习率不能解释Joint前的身份退步；立即增加主干身份模块会同时引入容量与优化变量，先不采用。

## 配置与兼容

`TrainingConfigV7`增加可选`preservation`配置和`preservation_index`引用。默认None，`record()`省略这两个关闭字段，旧配置签名与checkpoint继续有效。启用时必须显式绑定index；Source身份要求同一SHA固定的identity evaluator。旧的GT身份项保留，不静默替换。

`PreservationConfig`字段：`source_identity`与`background`，有限非负数，至少一项非零。实验先使用0.2／0.1；Source ID复用route现有sigma／update gate，新配置显式改为`auxiliary_every=1, auxiliary_sigma_max=.5`，增加有效身份更新曝光。数据不足显式拒绝，不静默关闭loss。

## 离线Source资产

每条样本独立保存schema `native_v7_preservation_sample_v1`：

- `source_features`：FP32 `[2,512]`，同一frozen TS从真实Source检测五点对齐提取，再按`source_indices`映射到target人物槽位。
- `source_valid`：bool `[2]`，对应可用Source人脸；缺失项为零且标记不可用，不能伪造。
- `background_support`：FP32 `[1,N]`，Source人物Mask与Target projected人物支持的并集膨胀16px后取补集，并与valid相交，再以16px stride平均池化。支持已含padding覆盖率，不重复乘coverage。
- metadata：source RGB SHA、source／target Canvas ID、target person IDs、source_indices、driver record SHA、identity evaluator SHA、target token grid、background margin与生成实现证据。

index按`source_id + NUL + pair_id`引用带SHA的记录，含schema与evaluator引用。DataPlan必须核对当前样本绑定；index／记录闭包进入data identity，不能将其他图或其他embedding空间的特征当成Source。

背景首轮只支持单场景Source、Source/Target Canvas完全相同的4Ref；不自动resize，不对独立多Source选择任意背景。

## 损失与梯度

Source身份项：对已有辅助解码的生成脸特征计算`mean(1-cos(generated, detached_source))`。生成脸仍使用已资格的GT五点定位，因此无需增加不可微检测到训练图。按有效Source+GT槽位及整个accumulation window的人脸数归一化；复用同一次VAE decode及同一frozen evaluator，不能再解码一次。Source reference不参与梯度。

实现通过现有`identity_regularizer(generated, gt_reference, slots)`回调扩展。GT contrastive与Source绝对ID同时返回独立parts／eligible计数，V7 backend分别按自己的window分母加权。Source-only也必须触发decode与evaluator加载。

背景项：`x0 = noisy - sigma * prediction`，在Source背景支持上计算`mean((x0-source_latent)^2)`，按有效图像window归一化，reference／support都detach。不能直接把`noise-source_latent`当成velocity监督，因为当前noisy来自Target；不除以小sigma，不增加VAE decode。

Source latent必须来自同一冻结条件缓存，Source SHA、Canvas、token grid和shape均匹配。缺少有效背景时显式记录0支持；请求背景但全数据无背景则拒绝。

## 验证与边界

先红后绿验证：Source吸引／GT与Source分歧、person mapping、Source-only gate、无额外decode、window归一化、错误evaluator／SHA／Canvas拒绝、参考无梯度、BG排除人物／padding、x0公式和旧record签名。

使用原实验数据与180步checkpoint，在新实验目录准备新资产和显式配置；旧基线不改。先CPU回归与实际data/grad预检，再做有界小步验证。保持原生尺寸与旧的seed42出图、seed24042姿态评价；身份同时看Target对齐和真实检测对齐，不将位置收益冒充身份收益。真实GPU不足则保留预检证据和可运行配置，不停止其他用户进程。

首轮不会改变主干、强制Source像素合成输出或启动大规模训练。新目标有效性须由新生成与Source身份／背景指标证明，不能仅凭loss下降宣称改善。
