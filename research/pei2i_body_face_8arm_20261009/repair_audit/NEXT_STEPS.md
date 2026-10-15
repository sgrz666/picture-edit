# 实验继续运行记录（2026-10-11）

本实验未完成。用户已确认阶段 Word 报告并授权正式八组训练；不要把进程启动、PE 重写或阶段网页当成全部实验完成。当前线程没有另行创建 goal。

## 当前真实状态

- sg 工作目录：`/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009`。
- Python：`/home/shangguanrz/nihaochen/envs/qwen21/bin/python`。
- QA v1：24/24 完成，因删除源球杆、臆测隐藏腿部与表情等问题未批准。证据在 `prompts/development_archive/instruction_v1/`。
- QA v2：主动终止（exit -15），保留 PG 4 条。它明确冻结 A 表情/头角，与 B Face 几何矛盾。证据在 `prompts/development_archive/instruction_v2/`，停止原因在 repair_audit。
- QA v3：24/24 已完成（exit 0）；全部真实 raw/compiled/40 个唯一输入图片已下载并核对哈希，逐条审查完成。PG 仍可能误读头向、手势或腿姿，PB/PBG 也有配件、服装名称和视线误读，没有宣称 PE 完美。
- `qa8_v3_visual_review.json` 保存逐条发现；`prompts/qa8.json` 已签名放行，review SHA `f4ead9d7e5f23ee80764a7b64d0f296ca79bf52b6a0489bb4153e35a668afe6f`。没有手改个别提示词或替换样本。
- 旧控制器遗漏了最终 test 的 rewrite/cache，已在零 Adapter 更新时停止本实验独立进程组；旧日志/状态保存于 `controller_v1_before_test_fix.json`，生产训练/模型/提示词协议未改。
- 当前 v2 控制器 PID 3682236。顺序为 rewrite/cache/selfcheck/preflight/initial/train/**rewrite_test/cache_test**/finaltest/report/export。测试阶段仍受八组训练完成门禁。不要重复启动。
- 最后 2026-10-11 17:53 北京核对：status=running，phase=rewrite；P0 train 512、PG train 8，正式 Adapter 更新 0，正式八组推理 0。PE 后续仍在逐样本生成；实时进度用 `check_formal_pipeline.py` 刷新，不依赖本文静态数字。
- Face 数据和锁准备已完成。

## 修复与验证

- 40 个限定代码文件已同步 sg 和本地 D 项目，并保存原文件；未创建 Git 提交，未覆盖其他用户改动。
- 原代码清单 `code_snapshot_repaired.json`，随后报告条件补丁 `code_report_conditions_patch.json`，随后 PE v3 补丁 `code_pe_instruction_v3_patch.json`；合并这三个清单得到当前 SHA。
- 最后 sg 完整回归：326 passed（16.06 秒），当前代码 SHA 核对通过；`verification_current_remote.json/.log` 已下载本地。
- 本地完整回归早于最后小补丁：324 passed、2 skipped。之后报告相关 42 passed/1 skipped，PE 41 passed，单报告 1 passed。
- 数据锁在正式产物出现前扩展，备份在 `repair_audit/pretraining_lock_migration`。新锁 SHA `4e851090805188941cbe2bd6c78aa95be20a4b6f5412cf7bd8e00279034bde83`，566 对、7924 项文件引用的哈希。
- 损失模型 2 个与评分 ONNX 4 个、评分代码/库版本已锁定。评分 signature `61940e2d04761292950fbbdb307770764ed4e763177e492aca8892dcd351886b`。
- 真实 CPU 评分/损失模型冒烟检查通过；8 张 A 检测唯一脸，8 张 B 检出姿态，两个 TorchScript 损失模型可加载。该检查不算正式生成次数。
- 评分程序错误保留图片，不能计入完成/指标；合法检测失败记 None 和原因。

## 下一步

1. 用 `python research/pei2i_body_face_8arm_20261009/repair_audit/check_formal_pipeline.py` 核对远端当前 PID/阶段/成功提示词数/已记录训练更新；它会保存本地 `formal_pipeline_status.json`。remote_exec 默认使用 qwen21 环境。
2. 当前本地 QA 已是 instruction_v3 全部 24 条；第一版审查网页独立留档，Word 阶段报告保留原快照，不能覆盖成虚假的正式结果。
3. 本轮按已确认方案训练 Body＋Face Adapter，PE/Qwen/BFS Head 均冻结。每组从同一个 J256 新增 1024 更新；Head 从本轮 step0 加载，不是训练后热挂载。
4. 已有签名 QA，无需重新放行或重复启动。控制器会串行执行所有阶段；PE 准备需几十小时，真实 Adapter 更新开始前不要声称参数已训练。
5. 观察 `full_pipeline_state.json` 和每阶段 `logs/full_pipeline_v2_*.log`，失败时根据真实日志修复，不旁路门禁。GPU 预检的 10 次私有更新不计入正式预算。
6. 完成 remote 全流程后检查 8192 正式更新、1698 PE 记录、3696 成功推理/评分，最后测试仅推理。实际审核生成图片和成对指标，写中文实验结论与限制。
7. 下载完整 `export/results.tar.gz` 与 `files.json`，检查每个文件 SHA 和网页相对链接，保留本地原尺寸图片；据真实结果制作最终 Word。远端结束后状态为 remote_complete_waiting_local_copy，并不等于整项交付完成。

## 本地阶段报告

`D:/codeplus/pictureedit/research/pei2i_body_face_8arm_20261009/report/index.html` 当前为“进行中”，81 张历史图，237 个阶段报告文件已校验，两个网页相对链接缺失为 0。历史图不属于正式八组结果。

第一版预检：`report/development_qa8_v1/index.html`。当前报告保存 `conditions.json`，已修正 PB 的含义、学习率与加载方式；计划条件和实际配置分开。

第三版全部 24 条预检：`report/development_qa8_v3/index.html`，含原始输入、最终文本、逐条偏差及 raw 回复；67 个文件哈希和所有链接已核对。正式训练说明：实验根目录 `正式训练说明_20261011.md`。

阶段导出不是最终导出：`export_current_report_remote.py` 会保留网页引用的其他目录文件，`import_current_report.py` 与 `validate_current_local_report.py` 校验字节和链接。最终使用生产 `package_export`。


## 2026-10-11 Word stage report delivered

Local `report/PEI2I_BodyFace_实验报告_阶段版_20261011.docx` contains 26 pages, all 81 historical native generation PNGs, 48 A/B comparison images and 40 PE QA input images. All historical image hashes exist in DOCX media. Every page visually reviewed; final pages 1–25 are pixel-identical to reviewed rendering, final page 26 individually reviewed after removing the orphan last page. `word_report_manifest.json` and `word_visual_review.json` bind evidence. The report snapshot is 2026-10-11 16:59 Beijing time: formal train updates 0, formal eval jobs 0, QA v3 PG6/PB0/PBG0. This is a stage report, not final eight-arm outcomes. Preserve it; create a final Word report only after actual runs complete and final export is verified. QA v3 sign-off has since completed; the v2 formal controller is running.
