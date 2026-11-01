# Hi4D 3D grounded v5 实施计划

> For agentic workers: 使用 subagent-driven-development，事实生成、下游接入、独立验证分开负责。

**目标**：修复复审确认的 Hi4D 提示词动作错误，生成完整新库，使新下游输入按正确人物/肢体读取，并在历史错例及全库上验证。保留所有旧库与训练产物，不重跑长训练。

**架构**：人物编号由源帧 source-left 身份固定；解剖左右由官方 SMPL-24 joint ID 固定。官方 joints_3d 与原相机分别提供物理动作和投影位置，禁止将 camera_Z 当 confidence。采用确定性逐人句子生成，无法证明的接触/离地/面部朝向省略。新 v5 bank 和新日期 pilot 根绑定完整 SHA，不复用旧 V1 训练缓存。

## 1. 3D 事实及全量库（fact worker）

- [x] 新增实验私有 `scripts/hi4d_grounded_v5.py` 与 `tests/test_hi4d_grounded_v5.py`，先验证投影近直线而 3D 肘角 118.7°的回归。
- [x] 输入验证官方 3D、camera R/t/K_512、SMPL24 npy=[u,v,camZ,1]、A/B身份与来源 SHA。按原 1,500 目标/3,000 人缓存输出有来源的 3D 事实。
- [x] 生成稳定 anatomical left/right 臂腿句子、3D 躯干倾角、手相对肩腰高度和骨架侧别；不写无法验证的接触/抬脚/头部转动/人体表面远近。屏幕布局按投影描述，不能当物理左右/接触。
- [x] 先生成历史 bend09/fight01、yoga92/yoga38、dance2、taichi72/73 试点；最终新目录 `v5_grounded_3d_20261030_r3` 全6,558行（5,358/600/600）。r1/r2及v4保留。

## 2. 下游接入（consumer worker）

- [x] 阅读并快照当前 dirty 的 prepare/assembler/person_order/conditioning，最小范围更新 active bank/schema 到v5 r3（只改prepare，其他共享读取器未改）。
- [x] 新 pilot 根显式落 `/data/shangguanrz`，具备选择记录、真实四图 condition、person_order、prompt/图片 SHA；覆盖 source-left=A/B、crossed 和 3D 反例。
- [x] 对7条新condition及2条真实typed样本调用读取器验证，配置明确无旧frozen、无预算、training_ready=false，不自动训练。
- [x] 保持 TikTok 与显式旧版本兼容；5种SHA/顺序/换图负例全部拒绝。

## 3. 独立验收（validator worker + root）

- [x] validator 独立从官方3D/相机重算每个人的事实、全部句子和完整正文；r3全6,558行0错误。
- [x] 全库检查人数、split、source-left A/B、稳定 anatomical side、来源/正文SHA及target缓存；错误与不支持的断言均不发布。
- [x] root逐图核对12张最终对照，涵盖历史错例、3D反例、crossed、近等高腕、target中心重叠、验证集和真实3D跨身体；r2表面远近措辞因语义代理问题删除后重验r3。
- [x] 74 passed/1 skipped，远端py_compile、真实consumer读回、5负例、全库6558条tokenizer与9组四图processor均验证；中文README与复现命令已完成。

完成标准：新库全量生成并通过独立 3D/绑定验收；新日期 pilot 真读取通过；默认下游版本已对齐；旧数据保留；文档明确训练/冻结缓存状态。不是只完成少量 pilot 或只更换默认路径。
