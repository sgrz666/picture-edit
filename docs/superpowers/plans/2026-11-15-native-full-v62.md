# Native Full V6.2 implementation plan

已批准：只实现人物路由＋分支空间 Gate，CPU 验证；现有输入不变，不要求新 Mask 图。不做动态预算、Sigma 调度、几何-only K/V、Loss 改造、真实迁移、7B、训练或生成。直接在 main 工作，最后同步 GitHub／服务器，保留旧工作区与运行快照。

## 决策与接口

- 独立 `native_v62`；`PoseLinkConfigV62` 继承 V6.1，新增 person_routing/body_spatial_gate/interaction_spatial_gate，默认 true，owner_threshold 默认 0.75；512维／8 heads／六层／每图256预算不变。Face／Hand／Detail 使用已有 owner ROI 权重与原模态开关，不再乘一层软 Mask。
- 明确可见 owner 优先，归一化人物权重最大值达到0.75才硬绑定；无可见 owner 时依次参考投影、空间支持、已校验 repair 归属；混合／不确定处保留软约束。
- Combined／target_pose 的全局预算在实际人物间固定均分，先按各自支持池化再标记人物；Joint／Local 保持 V6.1 固定预算与人物归属。所有 Key 仍进入同一次 Attention，不新增 Qwen Tokens。
- `RoutingConditionV62`：query_person_ids[B,N] int64（-1=不确定）、part_tokens[B,2,N,14] FP32、contact_positive/contact_forbidden[B,14,14] bool、person_valid、target_token_hw、canvas_ids、稳定人物 IDs 与 metadata。所有 Tensor 转换／索引／stack 保留 FP32、bool、int64 和梯度。
- `routing_permissions(query_ids,query_parts,key_person_ids,key_parts,positive,forbidden,key_valid,epsilon)` 返回[B,Q,K] bool。同人物可读；明确 owner 的跨人读取需对应部位正接触且无禁止；未知／无法映射 Contact 不放宽；不确定 Query 保留软约束。所有 Key 被屏蔽时连已训练 Head bias 也输出零。
- `BranchSpatialGatesV62`：body/interaction[B,N] FP32 条件 Gate、网格／Canvas／metadata。body 有显式合法 repair（包括显式空 Mask）才用 V6.1 edit_gate；否则全1并记录 missing_repair_legacy_coverage。interaction 用双人人体支持且仍遵循 relation_valid，不仅限接触点。已有局部权重直接复用。
- 新配置关闭时走 V6.1 完全相同路径。版本化 Runtime／Prepared／训练／推理／初始化工具，V6.1 header 与源码不改；不继承 optimizer／RNG 冒充续训。
- 可选诊断只保留标量：硬／软路由、接触开放、无有效 Key、回退、残差幅度和相对当前 Hidden 的幅度；默认关闭，不改变 RNG 或数学结果，不持有 Tensor。

## 执行清单

- [x] 运行旧135项 CPU回归，核对 main 与远端。
- [x] TDD：版本配置、路由条件、Gate 构建与 permission helper。
- [x] TDD：Combined 全局拆分，支持在池化／插值前应用；部位采样和分块路由 Attention。
- [x] TDD：独立 Runtime／Prepared／integration，Body／Interaction Gate、局部不重复权重、同原 Hidden、前缀与重算。
- [x] 版本化数据／训练／推理／迁移与可选诊断；旧 Loss 与固定 Token 预算复用。
- [x] 旧135＋新增完整 CPU 套件；初始化／严格回读／冻结／数值和梯度零依赖／关系许可等验收。
- [ ] 中文 README，独立规格／质量审查；py_compile／安全 prepare；main 提交推送；服务器时间戳快照、定向同步、索引核对。

## 验证界限

只证明 Adapter 内直接 Key 读取权限和残差空间契约。Qwen 原生 Attention 与继承 Reasoner 仍可混合上下文，不据此宣称身份绝不串位或生成质量改善。所有测试用合成 CPU 张量和随机小 Qwen；不安装依赖或运行真实资产。

本地完整验收：192 passed（原135＋新增57）。规格／质量审查发现并修复双人部位图子记录 Driver SHA 未核对、新功能关闭时额外绑定旧局部身份、contact消融仍可能开放路由等边界；新增 static 同 Hidden full/base 对照保留局部贡献。诊断接入配置／CLI，Windows旧编码下 prepare 输出UTF-8、迁移帮助可用。所有新Python已编译核对；远端交付前快照为 `/data/shangguanrz/code-snapshots/native-full-v62-20261115T064701Z/`。
