# 竖图原生画布验收（GPU 段）—— 方案 §8 里 CPU 测不到的那半

## 目的

`src/data/canvas_contract.py` 的等比性、矩形 token 网格、相机一致性都已经由
`tests/test_canvas_contract.py`（CPU）钉住。但方案 §8 明确要求「不应只停留在 CPU 单元测试，
还需要通过真实 Qwen VAE、Transformer、训练反向传播和完整采样检查」。本目录补的就是这一段，
并且顺手回答方案 §5 留下的那个未决问题：**padding token 到底有没有参与 Qwen Attention**。

## 里面有什么

- `run_native_portrait_acceptance.py` —— 单跑脚本，qwen21 环境。四段，逐段打印 JSON：
  - **A `vae_pair`**：真实 TikTok 帧（604×1080）作 Source、750×1333 作 Target，走生产路径
    `Qwen21IPERCache(resolution='native').pair()`，断言 VAE 出来的 token 网格是
    `(68,38)=2584` 与 `(84,48)=4032`，而不是 `(64,64)=4096`。源与目标**宽高比不同**，
    同时验 §8 的「源/目标比例不同不得复用同一画布变换」。
  - **B `vae_decode`**：latent 反解回画布尺寸（32 像素 unpack 那一层）。失败不中断，只记录。
  - **C `dit_backward`**：真实 DiT 在矩形网格上前向 + `loss.backward()`，验梯度有限、形状对得上。
  - **D `padding_participation`**：按 16×16 像素块算每个 target token 的有效面积，分桶
    （完全无效 / 部分 / 完全有效），比较三桶的梯度范数。**这就是那个未决问题的量化答案。**
  - **E `sample`**：`pipe(prompt, image=[真实帧], width=768, height=1344, steps=4)` 出一张竖图，
    存 `output/portrait_768x1344.png` 与去掉 padding 的 `portrait_750x1333_unpadded.png`。
- `output/results.json` —— 上面各步的汇总，含 `total_seconds`。
- `check_canvas_locks.py` —— 方案 §4「删除危险默认值」的门禁验收：8 条必须拒 + 5 条必须放行，
  全过返回 exit 0。它是 CPU 检查，不加载 7B 主干。
  输出留 `/data/shangguanrz/canvas_audit/canvas_locks.json`。
- `probe_hi4d_real_cache_guard.py` —— 只读探测真实 Hi4D 控制缓存的 key/shape 词表，
  量新画布门会不会误拒磁盘上已有的文件（下面 Hi4D 一节的数字全部来自它）。

## 关键数字

真实 Qwen-Image-2.1（`models/Qwen-Image-2.1`，bf16，管道加载 10.8 s）：

- **VAE 编码（`vae_pair`）**：Source = 真实 TikTok 帧 604×1080 → 画布 1088×608 → token 网格 **(68,38) = 2584**；
  Target = 750×1333 → 画布 1344×768 → token 网格 **(84,48) = 4032**，单 token 仍 64 维。
  `rectangular=true`、`not_square_64=true` —— 方案 §8「768×1344 得到 84×48 而非 64×64」在真 VAE 上成立，
  且源与目标**宽高比不同**也走通了同一条 `Qwen21IPERCache(resolution='native')` 生产路径。
- **VAE 反解（`vae_decode`）**：`_unpack_latents(latents, 1344, 768, vae_scale_factor)` → `(1,64,1,84,48)`
  → decode → **`(1,4,1,1344,768)`**，与画布逐维相等（`decoded_matches_canvas=true`）。
  diffusers 的 2×2 packing 不会把竖图掰成方形。
- `dit_forward`：4032 token 大网格的独立前向这次只在我的**报表行**上抛 `TypeError`（`img_shapes` 被
  变换器改写后再迭代），前向本身没报错；同一个 4032 token 网格已经由 `sample` 段完整跑过两步去噪。
- **`dit_backward`（真实 20B 变换器 + 反向传播）**：小竖图 225×400 → 画布 416×256 → 网格 **(26,16)=416**，
  loss 0.9807，梯度形状 `[1,416,64]`、全部有限 ⇒ 矩形网格在真实 DiT 上能前向也能训练反向。
- **`padding_participation`（回答方案 §5 的未决问题）**：41 个完全无效 token / 25 个半覆盖 / 350 个有效；
  梯度范数 完全无效 **0.0276** vs 有效 **0.0207**，比值 **1.332**。
  ⇒ **padding token 确实参与 Qwen 自注意力**（甚至比有效 token 平均还热一点，因为它是块常值区域）。
  所以「完全无效 token 梯度为 0」在 DiT 层面目前**不成立**；要成立必须做 attention 级屏蔽，
  而不是只在残差与 loss 上乘 mask。损失层面由 `tests/test_qwen21_canvas_coverage.py` 逐元素证到排除，
  注入层面证到无效 token 残差精确为 0，两条与这里的注意力层面结论是三件不同的事，别混。
- **`sample`（出图）**：2 步、7.1 s，管道按请求返回 **768×1344**；
  `CanvasSpec.unpad()` 得到 **750×1333**，`aspect_kept=true`（比例误差 < 1e-6）。
  注意：这一段只验「画幅与比例不被破坏」，**没有**验身体比例/姿态/ID/服装，那需要接 Adapter 与真实姿态重定向。

## P0 源资产审计（方案 §1/§9 的那一半，实测结论）

脚本：`audit_asset_aspects.py`（只看 PNG/JPEG 头，全量 13,812 对）、
`audit_part_roundtrip.py`（part 前景与 normal 前景在帧画布上的 IoU，抽样 150 对）。

1. **全量尺寸清点**（train 11042 / val 1364 / test 1406）：帧一律 604×1080；
   `source_rgb`、`target_rgb`、`target_normal`、`target_depth_vis`、`target_dwpose` 也都是 604×1080，
   与帧**零**纵横比偏差。**但 `part_512` 全部存成 512×512**（13812/13812）。
   另：val 里 510/1364 张 `target_person_mask` 存成 278×500（纵横比与帧差 0.6%，<1% 容差内）。
2. **`part_512` 会不会是方形 padding/crop？** 不会。以 native 尺寸的 normal 为参考，
   把 part 按 `_native_item` 的实际做法 NEAREST 拉到 604×1080 后比对前景：
   **IoU 均值 0.9963、最小 0.9936、p05 0.9941**；左上角 bbox 偏移归一化后
   **dy ≤0.0009、dx ≤0.0017**。⇒ part_512 是**把帧非等比压成方形后渲染**的资产，
   再拉回帧尺寸是它的**逆变换**，几何上自洽。
3. **letterbox 路径同结论**（推演而非猜测）：因为资产是帧归一化空间的纯 rescale，
   `Letterbox.image()` 把 512² 直接缩到 286×512 再贴到 left=113，与先把帧拉回 604×1080
   再 letterbox 的仿射复合**逐点相同**（`x_p = x_f·512/604`，`113 + x_p·286/512 = 113 + x_f·286/604`）。
   所以三条路径（native / letterbox / square）都不存在 part 与 RGB/normal 错位。
4. **真正的代价是分辨率不是几何**：512² 存 604×1080 的内容，纵向等于 1 像素顶 2.1 像素，
   part 边界比 normal 糊约 2 倍。新管线应当按原生尺寸直接渲染 part，而不是再走一次方形往返。

**这一条推翻了"竖图隐患来自 TikTok 资产"的猜测**：TikTok 侧的资产是自洽的，
隐患确实只在**加载器的方形 Resize 默认值**上（已在 `src/data/canvas_contract.py` +
`geometry_pair_dataset_v2.py` 处理）。iPER 侧不同：`normal_512/dwpose_512/part_512` 与
被拉伸的 RGB 同源，Dataset 端无法还原，仍按 legacy 自述处理。

## padding 是否参与 Qwen Attention（方案 §5 的未决问题，已用正确方法测出）

**先撤回一个错判**：早先我用「对未 mask 的 loss 求梯度」得到 padding/valid 梯度比 1.33，
并据此说"padding 参与注意力"。那个推理站不住 —— loss 没 mask，padding 当然有梯度，
它测的是我自己写的 loss，不是注意力。正确的探针在 `probe_padding_attention.py`：
**只扰动完全无效的 token，看有效 token 的输出动不动**。

画布 416×256（帧 400×225，边缘 16px 全是 padding）→ 网格 26×16=416，其中 **41 个完全无效、375 个有效**。

- **空白对照**：同一输入跑两次，有效 token 输出位移 `max 0.000000`（前向确定）⇒ 下面的数不是噪声。
- **扰动 padding 潜变量 10×**：有效 token 输出位移 `max 4.037 / mean 0.170`，
  相对有效信号 RMS(0.805) 为 **501%**；padding 自身位移 7.568（说明扰动确实进去了）。
- **只改 padding 的填充内容**（黑 → 边缘复制，**图像像素完全相同**）：有效 token 输出位移
  `max 3.250 / mean 0.241`，相对信号 **404%**。

⇒ **padding 不是惰性的**：DiT 的联合注意力让 margin 的取值以与信号同量级的幅度改变有效 token 的输出。
"最小化 padding"救不了这件事，因为问题不在 padding 有多少，而在它参与不参与。

**顺带查出一条真实不一致**：`_native_item` / `_budget_item` 里 RGB 用 `fill=(128,128,128)` 补边
（`rgb_transform`），而 normal/mask/part/depth 用 `fill=0`（`raster`）。
也就是说**同一块 margin 在目标潜变量里是灰的、在几何条件里是黑的**。
padding 惰性尚且不成立的前提下，这种跨模态不一致会把偏差进一步放大，属于方案 §2
"RGB、Pose、Depth、Normal、Part、Mask 和相机必须经过同一套可追溯几何变换"里没被满足的一条。

两条可选路径（都会改变已训练权重的语义，需定方向）：
1. 统一 margin 语义（所有模态同一个 pad 值），成本小、先做；
2. attention 级屏蔽 padding token，才能真正让 §8 的"完全无效 Token 梯度为 0"成立。

## 统一训练入口在真实竖图上的验收（GPU，2026-10-22 实测）

`scripts/train/train_qwen21_native_full.py` 用真实 TikTok `train_pairs.jsonl`（11042 对）
/ `val_pairs.jsonl`（1364 对）跑通，两种画布各一次：

| 跑法 | 原图 | 画布 | 真实 token 网格 | 方形猜测会给 | 结果 |
|---|---|---|---|---|---|
| `--canvas-policy native-pad --preflight-only` | 1080×604 | 1088×608 | **(68,38)=2584** | (68,68)=4624 | 前向契约通过，反向在此网格 OOM |
| `--canvas-policy aspect-preserving --max-pixels 65536 --steps 4 --warmup-steps 1 --adapter-init scratch` | 1080×604 | 352×192（缩放后 342×191） | **(22,12)=264** | (22,22)=484 | 4 步全部 `optimizer.step()`，峰值 54.2 GiB |

小网格那一次逐步的 `grad_norm` = **0.0 / 3.27 / 55.7 / 65.2**：第 1 步在 warmup 内（trainable 相位为 0，
梯度必然为 0），第 2 步起非零 —— 这正是之前那条「全绿但其实没在学」假阳性被修掉的位置。
`residual_rms` 从第 3 步起是 0.054 / 0.068…（前两步零初始化 head 输出为 0），说明几何残差真的进了 transformer。
loss 0.226 → 0.484 → 4.05 → 1.99 **不能当学习曲线读**：每步 `sigma` 不同（0.61/0.12/0.64/0.99），
4 步只证明「前向—反向—更新—落盘」这条管脚是通的。
产物：`/data/shangguanrz/canvas_audit/native_full_train_ap_v1/`（`checkpoint_last.pt` 3.7 GB、
`cache/` 902 MB、`train_metrics.jsonl`、`run_manifest.json`、`canvas_preflight.json`）。

`run_manifest.json` 与 checkpoint 都带画布自述，这是方案 §7 要的绑定：
`canvas_policy=aspect-preserving`、`canvas_schema_version=canvas_v2`、`max_pixels=65536`、
`pad_multiple=32`，另有 `train_pairs_sha256`/`val_pairs_sha256`/`initialization_provenance`。

**这一跑的边界要说清**：`adapter initialised from the DeepGen config WITHOUT pretrained V6 weights` ——
它验的是画布管脚，不是姿态迁移质量。真做竖图迁移要先定 warm-start 方向（见最后一节）。

## 危险默认值删除的门（方案 §4，`check_canvas_locks.py`，13 项全过，exit 0）

8 条**必须拒**的路（原样抄它打印的话）：

- 新入口 `--canvas-policy resize-square` → `refused by this entry: it either stretches the person to a square or crops the field of view`
- 新入口根本**没有** `--legacy-policy-ok` 这个后路 → argparse `unrecognized arguments`（退出码 2）
- `--steps 4 --warmup-steps 4` → `every step would sit inside warmup ... A run like that looks green while learning nothing`
- TikTok loader 不给 `align_mode` → `align_mode must be given explicitly for TikTok`
- `align_mode='resize'` 无 opt-in → `CanvasPolicyError: ... stretches the person to a square`
- 把 legacy 记录喂进训练 → `non-uniform resize (1080, 604) -> (512, 512) changes body proportions by x1.788`
  （**这张真实竖图被压成方形的各向异性形变是 78.8%**，就是这次隐患的量化 headline）
- 画布缺失 → `missing CanvasSpec: ... may not infer a canvas from an integer resolution`（不静默当方形）
- 无画布记录的 checkpoint → `may only inherit it as an explicit warm start (warm_start_ok=True)`

5 条**必须放行**（否则历史 run 不可复现，门就成了自锁）：真画布过 `require_canonical`、
`resize + legacy_policy_ok=True` 构造成功、`canvas_v2` checkpoint 接受、
legacy checkpoint 带 `warm_start_ok=True` 接受、
`PYTHONPATH=. python scripts/train/train_iper_qwen21_full.py --help` 正常。
最后一条顺带确认：旧入口**没有** repo-root `sys.path` 兜底（HEAD 里也没有，不是本次改动引入），
按路径直接执行会 `ModuleNotFoundError: No module named 'src'`，复现历史 run 要带 `PYTHONPATH=.`；
新入口自带兜底，按路径直接跑就行。

## Hi4D 双人线的画布记录与缓存绑定（方案 §7 落在 Hi4D 上的那部分）

先给结论：**Hi4D 不在方形拉伸隐患里**。`letterbox_array` 只有一个各向同性 scale
`min(512/w, 512/h)`；128×96（H×W）竖图落到 384×512 的内容框，宽高比 0.75 保持不变
（`tests/test_hi4d_canvas_record.py::test_letterbox_keeps_the_source_aspect_instead_of_stretching_it`）。
画布是方的，内容不是 —— 和 iPER/TikTok 那种把整张人像压成方形是两回事。

但它有两个真缺口：(a) 控件完全不落画布记录，下游没法验证它经过的几何变换；
(b) 缓存 key 只 hash 文件路径 + `VERSION`，不含 letterbox 数学，所以改 `SIZE` 或改 letterbox
会**静默复用**磁盘上那 1389 个旧缓存。

处理方式上我否掉了两个看起来更"干净"的做法：不换缓存 key（强制重算 1389 个），
也不把像素改走 `CanvasSpec`（cv2 与 PIL 重采样不同，等于静默改掉已在用的控制图）。
改成**声明 + 读取时校验**：`convert()` 写 `canvas_hw=[SIZE,SIZE]` 与 `native_hw=[h,w]`，
`__getitem__` 弹出并校验，记录被消费掉、不向下游传。

没有记录的旧文件用张量形状兜底：任何边长 >16 的方形张量必须等于 (512, 512)。
这条启发式只有在真实词表上才可信，所以做了全量实测（`probe_hi4d_real_cache_guard.py`）：

| 测什么 | 实测 |
|---|---|
| 磁盘上的控制缓存 | 1389 个，全部探测 |
| 会被新门拒绝的 | **0** |
| 带 `canvas_hw` 记录的 | 0（全部是整改前写的，正好压住兜底这条路） |
| 不同 key/shape 组合 | 24 |
| 其中边长 >16 的方形张量 | 16668 个（12 × 1389），**全部** 512×512 |
| 反向对照 | `SIZE` 临时改 768 后读同一份真实缓存 → `cached Hi4D control 'normal_a' sits on (512, 512) but this build renders onto (768, 768); re-run precompute rather than training on a mixed canvas`；改回 512 照常打开 |
| 端到端 | `main_pairs_val.jsonl`（822 对）第 0/1/2/821 条经新 `__getitem__` 全部加载，样本 29 个 key，`canvas_hw` 不在其中 |

证据落 `/data/shangguanrz/canvas_audit/hi4d_real_cache_guard_all.json`。
顺带确认 `/home/shangguanrz/nihaochen/qwen21_hi4d_controls` 是指向
`/data/shangguanrz/pic-edit-caches/qwen21_hi4d_controls` 的软链接 —— 同一份 1389 个文件，
不存在需要分别保护的两批缓存。

**已知残留（不粉饰）**：形状兜底是启发式而非证明。它成立的前提是">16 的方形张量只有画布栅格"，
今天的 24-key 词表上实测为真；将来若加进一个方形但非空间的张量（例如 24×24 的关系矩阵），
会被误判成画布。新写入的文件带 `canvas_hw`，不走这条路。

回归：`test_canvas_contract / test_geometry_pair_dataset_v2 / test_qwen21_canvas_coverage /
test_qwen21_padding_attention / test_hi4d_canvas_record / test_hi4d_geometry_registration /
test_hi4d_native_binding / test_tiktok_data_contract` = 85 passed, 1 failed。
失败的是 `test_smoke_test_runner_dry_run` → `No module named 'run_tiktok_sample_00001_smoke'`，
入口层迁到 `scripts/` 时留下的旧 import，与本次改动无关（同一类还有 `eval_iper_four_arm`）。

## §8「完全无效 Token 的梯度为 0」——native 路径已做到，Full 路径没有

之前只做了残差阻尼：`valid_mask` 乘在 token 残差上，padding token 自己写不进东西，
但它的**卷积输入还在**。实测记在 `tests/test_qwen21_canvas_coverage.py` 里：
不屏蔽时 `padding-token gradient / valid-token gradient = 0.310`，即 §8 这一条不成立。

现在补了**卷积入口门**：`Qwen21NativeBody.prepare_geometry(features, conv_input_valid=...)`
在 projection **之后**、局部卷积之前把无效 token 的行乘成硬 0。
顺序很关键 —— 先 masked 原始特征再归一化是没用的，norm/projection 的 bias 会把全 0 行变成非 0 行。
`bind_native_body(..., conv_input_gate=True)` 打开；**默认 False**，因为这是改已训权重看到的语义。

新加的 4 条测试（该文件 11 passed 里含它们）：

| 测试 | 钉住什么 |
|---|---|
| `..._makes_the_padding_gradient_exactly_zero` | `count_nonzero(grad[padding row]) == 0`，同时 3 个有效 token 每行仍有非零梯度（不是整体停了学习） |
| `..._removes_the_leak_that_the_unmasked_path_measured` | 门一开，有效邻居的输出**确实变了**（说明漏的是真的），而 padding token 自身输出仍与 hidden 逐位相等 |
| `test_an_all_valid_gate_is_bit_identical_to_no_gate_at_all` | 全 1 门 = 空操作，逐位相同：证明这根管子本身不扰动数值 |
| `..._shape_and_finateness_are_checked_before_it_is_used` | 形状不对 / NaN / 超出 0..1 都在用之前拒 |

原有的 `test_invalid_tokens_get_no_residual_but_still_leak_gradient_through_the_local_conv`
仍然通过 —— 那就是**默认关闭**的证据，两边行为都有测试压着，不是只测了想让我通过的那一侧。

**范围要说清，不能算成 §8 全过**：这道门装在 native 路径（`native_body.py` +
`native_integration.py`，方案 §1 指定的正式架构入口）。
而 `train_qwen21_native_full.py` 现在走的是 **Full control context**
（`execution.py:642` → `qwen_adapter.py:435`），那里 mask 同样只乘在 token 残差上；
它的卷积跑在画布像素的条件图上，padding 像素喂进卷积后**照样因为 conv bias 出非 0 特征**。
要在那条路上补齐，得在 controller 的卷积输出上按画布有效区再乘一次 ——
`qwen_adapter.py` 不在 47 个 SHA pin 文件里（实测 `pinned=False`），技术上可以改，
但它是 V6 全线的共享语义，属于"另一个实验变量"，方案 §9 也把它排在 P2 之后。
**这一条我没动，留作明确的下一步，不是已经做完。**

## 还没定的三件事（代码已经到位，方向要人拍板）

1. **竖图训练从哪套权重起**。可用的旧权重实测在盘上：
   `nihaochen/qwen21_full/body_512_100/checkpoint_last.pt`（4.0 GB）、
   `face_512_100/checkpoint_last.pt`（2.8 GB）、`iper1024/body_3k_final/checkpoint_best.pt`（772 MB）、
   `qwen21_face/face_train_3k/checkpoint_best_loss.pt`（216 MB）。它们全部 `canvas_policy=None`
   （方形拉伸时代训的），所以 `require_canvas_compatible` 默认拒绝、`--warm-start-ok` 才放行 ——
   两个方向都在真实文件上验过。`scripts/train/README.md` 提到的
   `experiments/iper_v6_native_20k_v1/checkpoint_last.pt` 盘上没有，且三批
   `/data/shangguanrz/_trash_ckpt_2026101{3,6,7}` 清单里都不含 `iper_v6_native` 条目
   （不是被这次清理掉的）。要真正的竖图迁移效果，得选：接受旧权重的方形失真当 warm start，
   还是 `--adapter-init scratch` 从头训。
2. **padding 要不要进 attention 屏蔽**。门已装好可开（`padding_attention.install_padding_key_mask`），
   但这是重训决定，前面 padding 一节量到的是「padding 不惰性」。
3. **margin 语义统一**。RGB 补 128 灰、条件补 0 黑，同一块 padding 在两个模态里颜色不同；
   改任何一边都会改变已训权重的输入分布。

另外两笔与画布无关但挡路的实测缺口：2584 token 的反向需要 gradient checkpointing（现在装不下）；
iPER 侧要真竖图得回到原始 SMPL-X 参数与相机重渲染资产（`normal_512/dwpose_512/part_512` 已是方形）。

## 与其他目录的关系

- 上游：`src/data/canvas_contract.py`（画布协议）与 `tests/test_canvas_contract.py`（CPU 验收）。
- 平行：`tests/test_qwen21_canvas_coverage.py` 已经量到 **native body 局部卷积**那条路上
  padding token 的梯度渗漏比值为 0.310；本脚本量的是 **Qwen Attention** 那条路，两者分开看。
- 不取代任何训练实验：这里只做单样本前向/反向与 4 步采样，不产 checkpoint、不写实验缓存。

## 注意与坑

- **sg 上有另一个会话在同时改这棵树**（实测 2026-10-22 17:5x UTC）：本机 scratch 副本里
  `native_body.py`/`native_integration.py`/`face_coordinates.py`/`tests/README.md` 等 10 个文件
  都落后于 sg，sg 还多出整套 Native Depth 线（`native_depth.py`、`train_qwen21_native_depth.py`、
  `NATIVE_DEPTH.md`，README 里 2026-10-23 那节是对方加的）。
  **sg 的时钟比 UTC 快约 8 小时**，`20261023` 这种目录名不代表"明天的实验"。
  任何部署前先比 sha1、再 `cp` 回本地对齐，否则会把对方的改动覆盖成旧文件。
  本轮做法：把 sg 的 50 个改动文件 `tar` 回本地做基线，改完只上传自己动的那 3 个。
- GPU 与人共用；加载 20B bf16 管道要占满显存几十 GB，跑之前先 `torch.cuda.mem_get_info()`。
- `nvidia-smi` 在这台机器上报 `Driver/library version mismatch`，**不代表 CUDA 不可用**；
  以 `torch.cuda.is_available()` 和 `mem_get_info()` 为准。
  实测根因：内核模块 `580.173.02` 与用户态 `libnvidia-ml.so.580.178.04` 版本不一致。
  小算子前向/反向都正常，但 PyTorch 缓存分配器在大计算图上会走到 NVML 分支并
  `INTERNAL ASSERT`。**绕过办法**：`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`
  （换分配器后端，绕开那条 NVML 路径）。这是主机层问题，要根治得对齐驱动或重启，不该由脚本改。
- 20B 变换器在 4032 token 上做**未 checkpoint 的 fp32 反向**装不进 83 GiB（实测 OOM）。
  因此大网格只验前向，反向与 padding 梯度用一个小竖图（225×400 → 画布 256×416 → 网格 26×16）承担，
  并把 text encoder 临时挪到 CPU 腾显存（采样前要挪回 `cuda`）。
- Source 用 `real_spec`、Target 用 `odd_spec`，两者画布不同，因此 `img_shapes` 是
  `[(1,68,38), (1,84,48)]`；若哪天又变成两个相同正方形，就是方形假设回来了。
