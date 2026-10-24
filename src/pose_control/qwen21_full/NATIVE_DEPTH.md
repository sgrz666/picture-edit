# Native Depth：CHAMP式编码与Qwen适配

本页是Depth V1复现说明。新建Depth默认V2，架构及未验收状态见 [NATIVE_DEPTH_V2.md](NATIVE_DEPTH_V2.md)。旧训练命令需显式 `--depth-version v1`。

当前默认 Body21 Geometry v2，详见 [GLOBAL_GEOMETRY_V2.md](GLOBAL_GEOMETRY_V2.md)。下列第三步历史复现显式选 legacy；Depth 编码与 512 维接口保持不变。旧 iPER 512 Depth 图可在 manifest 中声明 `asset_to_original` 后读取，未声明尺寸差异仍拒绝。

## 实现与结构

原Full的Pose25 / Normal3 / Part14 frontend继续输出512维geometry。新Depth独立编码 `[relative_depth, depth_valid]`，投影到同一个1/16 target网格，与原geometry做残差相加，再交给已有Native Body的4096→512→4096实时反馈。

`G = G_full + depth_strength * depth_residual`。

`Qwen21NativeDepth`共2,254,656个参数。CNN采用CHAMP的16→32→96→256通道节奏；空间Transformer保留GN、self-attention、LayerNorm、GEGLU与残差，使用PyTorch SDPA。默认`attention_scale=16`；显式设8保留官方attention位置，之后再降到1/16。移动attention是我们的适配，不声称与原CHAMP整网等价。

新512投影零初始化、融合系数默认1。禁用、strength=0或整幅无有效Depth时返回原features；interaction/person_valid不变。旧Full的384通道fusion与默认入口不变。

## 输入与来源

`DepthCondition`明确区分float相对深度、bool depth valid、bool canvas valid、available和来源。统一近1远0、背景0；有效远端也可以为0，不能按灰度>0推valid。

- `camera_depth_condition`：有效camera Z按场景共享范围归一化；支持同相机/尺度的人体按最近Z合成，保留可见表面camera Z与原场景范围。平面为0.5，全无效明确不可用。
- `relative_depth_condition`：输入必须声明`near_one`或`near_zero`。PNG不标metric；只有camera Z、metre和calibrated同时成立才标metric。
- `load_depth_record`：显式NPY或PNG、独立valid资产，不做fallback；按target_canvas做mask-aware缩放与padding，不猜源尺寸、不拉伸成方形。
- `NativeDepthRecords`：JSONL按pair_id及target_stem绑定，检查RGB/Depth画布一致，并锁定首次读取的资产SHA256、camera、尺度等。运行中漂移报错。

manifest每行示例（路径可相对于manifest）：

```json
{"pair_id":"train_0000001","target_stem":"0004","kind":"relative_png","polarity":"near_one","depth_path":"/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native/00001/depth_vis/0004.png","valid_path":"/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native/00001/smplx_mask/0004.png","camera_id":"tiktok_smplestx:00001:0004","units":"unknown","scale_status":"unknown"}
```

camera_z记录改为NPY路径、删除polarity。有来源证据后才改units/scale_status。camera_id是调用者声明，接口不会校准独立重建的尺度。

## CHAMP权重

`load_champ_state_dict`只迁入CNN和`attentions.5`有效路径：33个源张量、1,533,136个源元素。RGB首层核求和变为Depth核、valid核置0后，实际迁入1,532,992个元素。原SD专用320输出head舍弃；新降采样与512投影独立初始化。

未知key、缺key、shape冲突或非有限数会在写入前报错；已知未使用模块也检查shape/finite。CPU已验证移植后的空间attention与官方原类最大差0。来源和MIT许可证在`third_party/champ_guidance/`，权重不随Git分发。

## 训练和测试怎么用

在服务器`/home/shangguanrz/project/pic-edit`使用现有Qwen环境。新入口`train_qwen21_native_depth.py`实际构造Native Geometry/Body/Depth，B=1、KV关闭、gradient checkpointing。默认冻结继承geometry与Qwen，仅训练新Body/Depth；loss为canvas coverage加权velocity MSE，保留部分覆盖token。Face/Hand本步使用原生参考图外观路径，未迁入旧专用分支。

入口依赖服务器现有canonical canvas工作：`src/data/canvas_contract.py`、`geometry_pair_dataset_v2.py`、支持aspect_preserving的loader、`Qwen21IPERCache(canvas_policy=...)`。本步保留这些并行修改。仅检出10月22日Native分支时，运行训练入口还需集成canvas工作；核心Depth模块、checkpoint重建与单测不依赖该loader。实验的source快照保存了实际核验时的依赖。

先做CPU预检，output须使用新目录：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/train/train_qwen21_native_depth.py \
  --pairs /data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_step3_20261023_v1/data/pairs.jsonl \
  --geometry-version legacy_v1 --depth-version v1 \
  --depth-mode relative_png \
  --depth-manifest /data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_step3_20261023_v1/data/depth.jsonl \
  --full-checkpoint /home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt \
  --output /data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_preflight_manual_v1 \
  --preflight-only
```

移除`--preflight-only`可训练；用`--steps`和`--max-pixels`明确预算，`--champ-checkpoint`可选初始化Depth。`--depth-mode off`且不传depth manifest/CHAMP checkpoint可建noDepth Native对照。两组应固定pairs、canvas、loss、seed、预算与geometry冻结策略。

继承已训练noDepth Native时，用`--warm-start-native`及`--warm-start-components geometry-body`；不恢复optimizer/RNG。`all`要求Depth配置一致。非空output会拒绝启动，checkpoint保存为新schema。

恢复新checkpoint无需原Full文件：

```python
import torch
from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
saved = torch.load("/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_step3_20261023_v1/training_smoke_v1/checkpoint_last.pt",
                   map_location="cpu", weights_only=True, mmap=True)
system = NativeDepthRuntime.from_native_checkpoint(saved, legacy=True).to("cuda").eval()
```

推理时`system.encode`只读目标几何/Depth，通过已有`install_native_body` / `bind_native_body`接入Qwen原pipeline；源RGB走原生参考图路径。训练用target RGB不能作为生成条件，KV cache仍关闭。

当前3次真实更新只是工程验收：第1步Depth head梯度0、第2步head开始学习、第3步CNN有梯度，符合串联零head的延迟。该checkpoint不是质量基线。数字及复查命令见实验`report/STEP3_VERIFICATION.md`。
