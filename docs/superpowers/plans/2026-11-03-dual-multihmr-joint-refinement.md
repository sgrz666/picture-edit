# MultiHMR + Sapiens2 双人联合精修 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Hi4D 冻结的 127 帧上，实现并验证“MultiHMR 共享相机初值 + Sapiens2 可见二维监督 + 双人联合优化 + 有符号网格穿透约束”，用四臂对照证明穿透下降且二维、三维和双人相对位置不退化。

**Architecture:** 新实验放在 `experiments/mete-test/dual_mesh_refine_20261103_v1/`，不修改生产拟合代码和既有 benchmark 产物。纯 NumPy 模块负责输入契约、候选选择、可见性和统计；`smplx` 负责可微人体重建；Warp 1.18.0 在 kernel 内做双向有符号点到网格查询；Torch 以半隐式最近面约束优化两人。拟合 worker 只接收 RGB、真实 K 和模型预测，GT 仅由独立 evaluator 读取。大产物写 `/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/`。

**Tech Stack:** Python 3.10、NumPy、PyTorch、SMPL-X、NVIDIA Warp 1.18.0、pytest、Matplotlib；远端统一使用 `/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python`，不改动 `qwen21` 环境。

---

## 0. 执行约束与文件地图

### 不可放宽的实验约束

1. 输入固定为 `/data/shangguanrz/models/two-man/benchmark_results/M2_MultiHMR_672/predictions.pkl`、原 127 帧 `request.json` 和 `/data/shangguanrz/models/two-man/benchmark_results/M3_Sapiens2_detr/`。实测 pkl 是 `{sample_id: {sample_id, category, image_path, orig_size, K_orig, num_detected, predictions}}`；每个候选含 `score/bbox_orig/transl_pelvis/v3d/j3d/j2d_orig/rotvec/shape`，没有 `expression`，重建时用 10 维零向量并验证原始网格。
2. 拟合侧记录白名单只有 `id/image/image_sha256/size/K/split/pair/action/category/camera_id/frame_id`；`gt_*`、实例 mask、原始 SMPL、接触标注不得进入 selector、refiner 或接受规则。`pair` 只作为离线 bootstrap 分组，不进入每帧拟合输入。
3. 所有 127 帧均有状态；不足两个可靠不同候选时输出 `insufficient_observation`，不复制人体凑数。四臂共享一次冻结的选人结果。
4. 主穿透量是双向全部 10475 顶点的平均正穿透深度；无符号最近距离和顶点对应距离都不是替代指标。
5. 四臂为 `selected-init/reproj-only/collision-only/joint-full`。每个优化臂都使用同一可行集接受规则；没有合格迭代时回退所选初值并记录 `no_improving_iterate`。
6. 默认配置只用于 train pilot；train 筛选至多两个配置，val 选择唯一配置并写冻结哈希，test 只运行一次。任何 test 后代码或配置变更都必须把该 test 标为已接触，不能继续声称冻结测试。
7. GPU 作业串行。输入、缓存或 SMPL-X 资产缺失时启动失败；每帧异常保留分母。产物独占创建，默认不覆盖；远端脚本用 `scp` 投递并执行 `py_compile`。
8. 下列提交步骤供获批执行实施计划时使用；计划编写本身不创建 commit。

### 执行环境补充（2026-11-03 实测）

远端 `/data/shangguanrz/models/two-man/M4_MultiHMR2/env` 没有 `pytest`，但有标准库 `unittest`、Warp、Torch 和 SMPL-X；本实验所有远端单元与探针运行改用 `python -m unittest discover -s tests -p 'test_*.py' -v`，不向该环境安装 pytest。本地可用 pytest，两套测试文件均为标准 `unittest.TestCase`。2026-11-03 已实测远端完整套件 `54 tests OK`；六项探针由独立 `unittest` 子进程退出码生成，全部为 true。真实自动选人为 `125/127`，两帧各只有一个 MultiHMR 候选。 reconstruction cache、输入精度校准、selected-init 与 joint-full 单帧 smoke 已完成；这些实测结果优先于计划正文中早先给出的示例测试数。

### 文件职责

```text
experiments/mete-test/dual_mesh_refine_20261103_v1/
├── README.md
├── data/
│   ├── request.json              # 绝对输入/输出路径与默认优化配置
│   ├── fit_records.json          # 独立冻结器生成的白名单记录，不含 GT
│   └── selected.json             # 一次生成、四臂只读共享的自动选人结果
├── scripts/
│   ├── common.py                 # body-12 常量、严格 JSON、SHA256、GT 防火墙、投影
│   ├── freeze_inputs.py          # 唯一读取原始 request 的输入边界，只导出白名单
│   ├── body.py                   # full53 解码、标准 transl、可微 SMPL-X、重建验收
│   ├── select_candidates.py      # 去重、候选—Sapiens2 一致匹配、缺人状态
│   ├── collision.py              # Warp kernel 查询、穿透指标、Torch 半隐式项
│   ├── refine.py                 # 可见性/冻结门、四臂优化、可行集接受、每帧 CLI
│   ├── evaluate.py               # 独立 GT 指标、配对 bootstrap、门禁和报告
│   ├── probes.py                 # 按真实 pytest 退出码保存六项反例证据
│   └── visualize.py              # separated/hug/severe 双色网格前后对照
└── tests/
    ├── test_common.py
    ├── test_body.py
    ├── test_selection.py
    ├── test_collision.py
    ├── test_refine.py
    └── test_evaluate.py
```

### Task 1: 建立自包含实验与 GT 防火墙

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_common.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/common.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/freeze_inputs.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/data/request.json`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/README.md`

- [ ] **Step 1: 写失败的公共契约测试**

```python
# tests/test_common.py
import importlib.util
import json
from pathlib import Path
import unittest
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('dual_common', ROOT / 'scripts/common.py')
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


class CommonTests(unittest.TestCase):
    def test_body12_mapping_is_named_and_not_coco(self):
        self.assertEqual(common.BODY_NAMES[4:6], ['left_wrist', 'right_wrist'])
        self.assertEqual(common.SMPL_IDS, [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8])

    def test_fit_view_drops_every_privileged_field(self):
        record = {key: 1 for key in common.FIT_RECORD_KEYS}
        record.update(gt_boxes=[[0, 0, 1, 1]], gt_xy='secret', gt_xyz='secret',
                      masks='secret', smpl='secret', contacts='secret')
        view = common.fit_record(record)
        self.assertEqual(set(view), set(common.FIT_RECORD_KEYS))
        encoded = json.dumps(view)
        for forbidden in ('secret', 'gt_', 'mask', 'smpl', 'contact'):
            self.assertNotIn(forbidden, encoded)

    def test_nonfinite_json_and_overwrite_are_rejected(self):
        with self.assertRaises(ValueError):
            common.strict_loads(b'{"x": NaN}')
        with self.assertRaises(FileExistsError):
            common.write_json_xb(Path(__file__), {'x': 1})

    def test_projection_rejects_nonpositive_depth(self):
        k = np.array([[100., 0., 50.], [0., 200., 60.], [0., 0., 1.]])
        np.testing.assert_allclose(common.project(np.array([[1., 2., 4.]]), k), [[75., 160.]])
        with self.assertRaises(ValueError):
            common.project(np.array([[1., 2., 0.]]), k)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_common.py -q`
Expected: FAIL，collection 阶段报 `FileNotFoundError` 或 `ModuleNotFoundError`，因为 `scripts/common.py` 尚未创建。

- [ ] **Step 3: 实现完整公共模块**

```python
# scripts/common.py
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

BODY_NAMES = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
SMPL_IDS = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
ARMS = ['selected-init', 'reproj-only', 'collision-only', 'joint-full']
FIT_RECORD_KEYS = ['id', 'image', 'image_sha256', 'size', 'K', 'split',
                   'camera_id', 'frame_id', 'category', 'action']
SCHEMA = 'dual_mesh_refine_v1'


def finite(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f'Expected shape {tuple(shape)}, got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError('Expected finite numeric values')
    return array


def fit_record(record):
    missing = [key for key in FIT_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f'Missing fit fields: {missing}')
    return {key: record[key] for key in FIT_RECORD_KEYS}


def strict_loads(raw):
    def reject(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    return json.loads(raw, parse_constant=reject)


def read_json(path):
    raw = Path(path).read_bytes()
    return strict_loads(raw), hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_xb(path, payload):
    raw = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def project(xyz, k):
    xyz = finite(xyz)
    k = finite(k, (3, 3))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Expected Nx3 camera points and a pinhole K')
    if (xyz[:, 2] <= 0).any():
        raise ValueError('Non-positive camera depth')
    return finite((xyz[:, :2] / xyz[:, 2:]) @ k[:2, :2].T + k[:2, 2])
```

- [ ] **Step 4: 写默认请求与实验 README**

`data/request.json` 完整内容：

```json
{
  "schema": "dual_mesh_refine_v1",
  "source_request": "/home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_pose_compare_20261029_v1/data/request.json",
  "predictions_pkl": "/data/shangguanrz/models/two-man/benchmark_results/M2_MultiHMR_672/predictions.pkl",
  "observations_dir": "/data/shangguanrz/models/two-man/benchmark_results/M3_Sapiens2_detr",
  "smplx_model": "/home/shangguanrz/project/pic-edit/add/tools/multi-hmr/models/smplx/SMPLX_NEUTRAL.npz",
  "out_dir": "/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1",
  "device": "cuda:0",
  "seed": 20261103,
  "selection": {
    "confidence": 0.3,
    "minimum_matched_joints": 4,
    "match_max_cost": 0.35,
    "duplicate_pelvis_m": 0.12,
    "duplicate_pose_m": 0.08,
    "duplicate_box_iomin": 0.5,
    "minimum_detection_score": 0.25
  },
  "visibility": {
    "confidence": 0.3,
    "ownership_margin": 0.02,
    "maximum_projection_cost": 0.15,
    "occlusion_depth_margin_m": 0.05,
    "occlusion_segment_distance": 0.08,
    "border_px": 2
  },
  "optimization": {
    "steps": 35,
    "learning_rate": 0.003,
    "reproj_weight": 1.0,
    "pose_reg_weight": 0.015,
    "root_reg_weight": 0.015,
    "transl_reg_weight": 0.1,
    "collision_weight": 5.0,
    "huber_delta": 0.05,
    "max_dist_m": 2.0,
    "penetration_epsilon_m": 1e-8,
    "reprojection_tolerance": 1e-6,
    "max_translation_drift_m": 0.15,
    "max_root_drift_rad": 0.35,
    "max_body_joint_drift_rad": 0.5
  }
}
```

`README.md` 完整内容：

```markdown
# MultiHMR + Sapiens2 双人联合精修

本实验在 Hi4D 冻结的 127 张双人帧上验证：仅用 RGB、真实相机内参和模型预测，能否在可见点投影与双人相对位置不退化的前提下减少真实网格穿透。GT 只进入独立评测，不参与选人、损失或迭代接受。

`scripts/common.py` 固定 body-12 与输入防火墙；`body.py` 重建 MultiHMR 的 SMPL-X；`select_candidates.py` 生成四臂共享的两人选择；`collision.py` 提供 Warp 有符号查询和 Torch 半隐式穿透项；`refine.py` 跑四臂并逐帧回退；`evaluate.py` 独立读取 GT、做演员组 bootstrap；`visualize.py` 生成双色网格前后对照。

大产物位于 `/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/`。原始 predictions、Sapiens2 缓存和旧报告均只读。所有帧以 127 为状态分母；不足两名可靠不同候选时明确失败，不复制人体。穿透指标是顶点—网格近似，不等同于三角面相交体积或服装级碰撞。

先运行 `python -m pytest tests -q`，再在远端运行 Warp/CUDA 测试。train 仅用于筛选，val 仅用于选择配置，test 冻结后只跑一次。最终是否可接生产由穿透改善、二维/三维非劣效、覆盖率和全部反向探针共同决定。
```

再创建唯一允许读取原始含 GT request 的输入冻结器；它不导入 selector/refiner，输出仅含白名单：

```python
# scripts/freeze_inputs.py
from __future__ import annotations
import argparse
from pathlib import Path
from common import fit_record, read_json, write_json_xb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request, _ = read_json(args.request)
    source, source_sha = read_json(request['source_request'])
    records = [fit_record(record) for record in source['records']]
    if len(records) != 127 or len({record['id'] for record in records}) != 127:
        raise ValueError('Expected 127 unique fit records')
    write_json_xb(args.request.parent / 'fit_records.json',
                  {'schema': 'dual_mesh_fit_records_v1', 'source_request_sha256': source_sha,
                   'records': records})


if __name__ == '__main__':
    main()
```

在远端实验根目录运行 `python scripts/freeze_inputs.py --request data/request.json`，然后检查 `fit_records.json` 不含 `gt_`、mask、SMPL 或 contact 字段。selector 和 refiner 只读这个白名单文件；原始 request 的后续读取者是独立 evaluator。

- [ ] **Step 5: 运行公共测试并提交**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_common.py -q`
Expected: `4 passed`。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1
git commit -m "test(dual-refine): establish fit input and camera contracts"
```

### Task 2: 固定 MultiHMR full53 到 SMPL-X 的重建契约

**实施校准（2026-11-03，用户已批准）：** 冻结 pkl 未保存 `expression` 且经过 autocast，原全网格/全关节 `1e-4 m / 0.25 px` 门实测不适用。新增原始 MultiHMR SMPL_Layer + autocast 输入重建器，从原始 `v3d` 反解并冻结 expression；body-12 三维继续执行 `1e-4 m` 硬门，body-12 按原模型空间投影再逆仿射的路径执行 `0.25 px` 硬门；完整网格、全部 127 关节三维和全部 127 关节投影残差全量记录。train/val 202 个候选实测 body-12 p100 `0.0009765625 m / 0.952392578 px`，校准门 `0.0011 m / 1.1 px`，test 不参与定门。float32 可微导出仍执行同一路径 round-trip 一致性检查。远端 M2 权威代码为 `/data/shangguanrz/models/two-man/M2_MultiHMR/code/blocks/smpl_layer.py`，原始生成器明确输出 `expression`，但 pkl 漏存该字段。

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_body.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/body.py`

- [ ] **Step 1: 写 full53、平移和冻结参数的失败测试**

```python
# tests/test_body.py
import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('dual_body', ROOT / 'scripts/body.py')
body = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(body)


class BodyTests(unittest.TestCase):
    def setUp(self):
        self.rotvec = np.arange(159, dtype=np.float32).reshape(53, 3) / 1000
        self.shape = np.arange(10, dtype=np.float32) / 100

    def test_full53_layout_and_zero_expression(self):
        params = body.decode_full53(self.rotvec, self.shape)
        np.testing.assert_array_equal(params['global_orient'], self.rotvec[0])
        np.testing.assert_array_equal(params['body_pose'], self.rotvec[1:22].reshape(-1))
        np.testing.assert_array_equal(params['left_hand_pose'], self.rotvec[22:37].reshape(-1))
        np.testing.assert_array_equal(params['right_hand_pose'], self.rotvec[37:52].reshape(-1))
        np.testing.assert_array_equal(params['jaw_pose'], self.rotvec[52])
        np.testing.assert_array_equal(params['expression'], np.zeros(10))
        self.assertEqual(params['betas'].shape, (10,))

    def test_translation_is_camera_pelvis_minus_untranslated_pelvis(self):
        np.testing.assert_allclose(body.standard_translation([4., 5., 6.], [1., 2., 3.]), [3., 3., 3.])
        with self.assertRaises(ValueError):
            body.standard_translation([np.nan, 0., 1.], [0., 0., 0.])

    def test_trainable_delta_cannot_change_frozen_body_dimensions(self):
        initial = np.zeros((2, 63), np.float32)
        mask = np.ones((2, 63), bool)
        mask[0, 15:21] = False
        delta = np.full((2, 63), 3., np.float32)
        effective = body.apply_frozen_delta(initial, delta, mask)
        np.testing.assert_array_equal(effective[0, 15:21], np.zeros(6))
        np.testing.assert_array_equal(effective[1], np.full(63, 3.))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_body.py -q`
Expected: FAIL，`scripts/body.py` 不存在。

- [ ] **Step 3: 实现 pure NumPy 解码和 SMPL-X 后端**

```python
# scripts/body.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from common import SMPL_IDS, finite, project

SIZES = {'global_orient': 3, 'body_pose': 63, 'left_hand_pose': 45,
         'right_hand_pose': 45, 'jaw_pose': 3, 'expression': 10,
         'betas': 10, 'transl': 3}


def decode_full53(rotvec, shape):
    rotvec = finite(rotvec, (53, 3)).astype(np.float32)
    shape = finite(shape, (10,)).astype(np.float32)
    return {'global_orient': rotvec[0].copy(),
            'body_pose': rotvec[1:22].reshape(-1).copy(),
            'left_hand_pose': rotvec[22:37].reshape(-1).copy(),
            'right_hand_pose': rotvec[37:52].reshape(-1).copy(),
            'jaw_pose': rotvec[52].copy(), 'betas': shape.copy(),
            'expression': np.zeros(10, np.float32)}


def standard_translation(pelvis_cam, rest_pelvis):
    pelvis_cam = finite(pelvis_cam, (3,))
    rest_pelvis = finite(rest_pelvis, (3,))
    return (pelvis_cam - rest_pelvis).astype(np.float32)


def apply_frozen_delta(initial, delta, trainable_mask):
    initial, delta = np.asarray(initial), np.asarray(delta)
    mask = np.asarray(trainable_mask, bool)
    if initial.shape != delta.shape or initial.shape != mask.shape:
        raise ValueError('Initial, delta and trainable mask shapes must match')
    return initial + delta * mask


class BodyModel:
    def __init__(self, model_path, device='cuda:0'):
        import torch
        import smplx
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for refinement')
        self.torch = torch
        self.device = device
        self.layer = smplx.SMPLX(str(model_path), use_pca=False, flat_hand_mean=True,
            use_face_contour=False, num_betas=10, num_expression_coeffs=10).to(device).eval()
        self.faces = np.asarray(self.layer.faces, dtype=np.int32)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError('SMPL-X triangle topology is invalid')

    def tensors(self, params):
        torch = self.torch
        kwargs = {}
        for key, size in SIZES.items():
            value = params[key]
            if torch.is_tensor(value):
                kwargs[key] = value.to(device=self.device, dtype=torch.float32).reshape(1, size)
            else:
                kwargs[key] = torch.as_tensor(np.asarray(value, np.float32).reshape(1, size), device=self.device)
        return kwargs

    def forward(self, params):
        kwargs = self.tensors(params)
        kwargs['leye_pose'] = self.torch.zeros((1, 3), device=self.device)
        kwargs['reye_pose'] = self.torch.zeros((1, 3), device=self.device)
        output = self.layer(**kwargs)
        if output.vertices.shape != (1, 10475, 3) or output.joints.shape[1] != 127:
            raise ValueError('Unexpected SMPL-X output shape')
        if not self.torch.isfinite(output.vertices).all() or not self.torch.isfinite(output.joints).all():
            raise ValueError('SMPL-X produced non-finite geometry')
        return output.vertices[0], output.joints[0]

    def initial_parameters(self, human):
        params = decode_full53(human['rotvec'], human['shape'])
        params['transl'] = np.zeros(3, np.float32)
        tensors = self.tensors(params)
        with self.torch.no_grad():
            _, joints = self.forward(tensors)
        rest_pelvis = joints[0].detach().cpu().numpy()
        params['transl'] = standard_translation(human['transl_pelvis'], rest_pelvis)
        return params

    def verify_initial(self, human, params, k):
        with self.torch.no_grad():
            vertices, joints = self.forward(params)
        vertices = vertices.detach().cpu().numpy()
        joints = joints.detach().cpu().numpy()
        mesh_error = float(np.max(np.abs(vertices - finite(human['v3d'], (10475, 3)))))
        joint_error = float(np.max(np.abs(joints - finite(human['j3d'], (127, 3)))))
        projection_error = float(np.max(np.abs(project(joints, k) - finite(human['j2d_orig'], (127, 2)))))
        if max(mesh_error, joint_error) >= 1e-4 or projection_error >= .25:
            raise ValueError(f'MultiHMR reconstruction mismatch: mesh={mesh_error}, joints={joint_error}, projection={projection_error}')
        return {'mesh_max_error_m': mesh_error, 'joint_max_error_m': joint_error,
                'projection_max_error_px': projection_error,
                'expression_source': 'zeros_because_frozen_pkl_has_no_expression'}
```

- [ ] **Step 4: 运行 CPU 测试与远端真实候选重建烟雾测试**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_body.py -q`
Expected: `3 passed`。

Run on `sg` after `scp` and `py_compile`:

```bash
cd /home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_mesh_refine_20261103_v1
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -c "import pickle,sys,numpy as np; sys.path.insert(0,'scripts'); from body import BodyModel; p=pickle.load(open('/data/shangguanrz/models/two-man/benchmark_results/M2_MultiHMR_672/predictions.pkl','rb'))['809de42848f7edd5']; m=BodyModel('/home/shangguanrz/project/pic-edit/add/tools/multi-hmr/models/smplx/SMPLX_NEUTRAL.npz'); q=m.initial_parameters(p['predictions'][0]); print(m.verify_initial(p['predictions'][0],q,np.asarray(p['K_orig'])))"
```

Expected: 退出码 0，打印的 mesh/joint 误差 `<1e-4`，projection 误差 `<0.25`；不满足则停止后续任务，不通过猜 hand mean 或 transl 继续。

- [ ] **Step 5: 提交**

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/body.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_body.py
git commit -m "feat(dual-refine): reconstruct frozen MultiHMR SMPL-X parameters"
```

### Task 3: 无 GT 的重复实例归并与双人归属

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_selection.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/select_candidates.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/data/selected.json`（远端生成，不手工写选择结果）

- [ ] **Step 1: 写重复、交换、缺人和分母反例**

```python
# tests/test_selection.py
import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('dual_selection', ROOT / 'scripts/select_candidates.py')
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
K = np.array([[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]])
CFG = {'confidence': .3, 'minimum_matched_joints': 4, 'match_max_cost': .35,
       'duplicate_pelvis_m': .12, 'duplicate_pose_m': .08, 'duplicate_box_iomin': .5,
       'minimum_detection_score': .25}


def candidate(x, score=.9):
    joints = np.zeros((127, 3)); joints[:, 2] = 1.
    joints[selection.SMPL_IDS, 0] = np.arange(12) * .03 + x
    joints[0] = [x, 0., 1.]
    return {'score': score, 'bbox_orig': np.array([x * 100 - 10, -10, x * 100 + 50, 50.]),
            'j3d': joints, 'transl_pelvis': joints[0]}


def observation(x):
    return {'bbox': [x * 100 - 10, -10, x * 100 + 50, 50.],
            'xy': (np.arange(12)[:, None] * [3., 0.] + [x * 100, 0.]).tolist(),
            'scores': [1.] * 12, 'joint_names': list(selection.BODY_NAMES)}


class SelectionTests(unittest.TestCase):
    def test_duplicate_candidates_cannot_become_two_people(self):
        result = selection.select_people([candidate(0), candidate(.01)], [], K, CFG)
        self.assertEqual(result['status'], 'insufficient_observation')
        self.assertEqual(result['candidate_indices'], [])

    def test_two_consistent_people_are_selected(self):
        result = selection.select_people([candidate(0), candidate(2)], [observation(0), observation(2)], K, CFG)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['candidate_indices'], [0, 1])
        self.assertEqual(result['observation_indices'], [0, 1])
        self.assertEqual(result['basis'], 'sapiens_consistent')

    def test_forced_cross_person_observation_swap_has_high_matching_cost(self):
        candidates = [candidate(0), candidate(2)]
        observations = [observation(0), observation(2)]
        correct = [selection.match_cost(candidates[i], observations[i], K, CFG) for i in range(2)]
        swapped = [selection.match_cost(candidates[i], observations[1 - i], K, CFG) for i in range(2)]
        self.assertLess(max(correct), CFG['match_max_cost'])
        self.assertGreater(min(swapped), CFG['match_max_cost'])
        result = selection.select_people(candidates, observations[::-1], K, CFG)
        self.assertEqual(result['observation_indices'], [1, 0])

    def test_one_observation_falls_back_without_copying_candidate(self):
        result = selection.select_people([candidate(0), candidate(2)], [observation(0)], K, CFG)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['basis'], 'multihmr_score_fallback')
        self.assertEqual(result['observation_indices'], [0, None])

    def test_low_score_or_single_candidate_is_insufficient(self):
        for candidates in ([candidate(0)], [candidate(0, .1), candidate(2, .1)]):
            result = selection.select_people(candidates, [], K, CFG)
            self.assertEqual(result['status'], 'insufficient_observation')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_selection.py -q`
Expected: FAIL，`select_candidates.py` 不存在。

- [ ] **Step 3: 实现具名匹配、非重复对选择和冻结 CLI**

```python
# scripts/select_candidates.py
from __future__ import annotations
import argparse
from itertools import combinations, permutations
from pathlib import Path
import pickle
import numpy as np
from common import BODY_NAMES, SMPL_IDS, SCHEMA, finite, fit_record, project, read_json, sha256_file, write_json_xb


def box_iomin(a, b):
    a, b = finite(a, (4,)), finite(b, (4,))
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = np.prod(a[2:] - a[:2]), np.prod(b[2:] - b[:2])
    return float(intersection / max(min(area_a, area_b), 1e-12))


def duplicate_metrics(a, b):
    pa, pb = finite(a['j3d'][0], (3,)), finite(b['j3d'][0], (3,))
    ja = finite(a['j3d'], (127, 3))[SMPL_IDS] - pa
    jb = finite(b['j3d'], (127, 3))[SMPL_IDS] - pb
    return {'pelvis_distance_m': float(np.linalg.norm(pa - pb)),
            'root_relative_pose_m': float(np.linalg.norm(ja - jb, axis=1).mean()),
            'bbox_iomin': box_iomin(a['bbox_orig'], b['bbox_orig'])}


def is_duplicate(a, b, config):
    metrics = duplicate_metrics(a, b)
    duplicate = (metrics['pelvis_distance_m'] < config['duplicate_pelvis_m'] and
                 metrics['root_relative_pose_m'] < config['duplicate_pose_m'] and
                 metrics['bbox_iomin'] > config['duplicate_box_iomin'])
    return duplicate, metrics


def observation_arrays(observation, image_size=None):
    if tuple(observation['joint_names']) != tuple(BODY_NAMES):
        raise ValueError('Sapiens2 body-12 names differ from the benchmark contract')
    xy = finite(observation['xy'], (12, 2))
    scores = finite(observation['scores'], (12,))
    box = finite(observation['bbox'], (4,))
    scale = float(np.linalg.norm(box[2:] - box[:2]))
    if scale <= 0:
        raise ValueError('Degenerate Sapiens2 box')
    return xy, scores, scale


def match_cost(candidate, observation, k, config):
    xy, scores, scale = observation_arrays(observation)
    predicted = project(finite(candidate['j3d'], (127, 3))[SMPL_IDS], k)
    valid = scores >= config['confidence']
    if valid.sum() < config['minimum_matched_joints']:
        return 4.0
    normalized = np.linalg.norm(predicted[valid] - xy[valid], axis=1) / scale
    return float(np.minimum(normalized, 2.).mean())


def fallback_pair(candidates, config):
    eligible = [i for i, c in enumerate(candidates) if float(c['score']) >= config['minimum_detection_score']]
    options = []
    for i, j in combinations(eligible, 2):
        duplicate, metrics = is_duplicate(candidates[i], candidates[j], config)
        if not duplicate:
            options.append((-(float(candidates[i]['score']) + float(candidates[j]['score'])), i, j, metrics))
    return sorted(options)[0] if options else None


def ordered_pair(i, j, candidates, k):
    depth_x = [candidates[index]['j3d'][0, 0] / candidates[index]['j3d'][0, 2] for index in (i, j)]
    return [i, j] if depth_x[0] <= depth_x[1] else [j, i]


def select_people(candidates, observations, k, config):
    k = finite(k, (3, 3))
    if len(observations) >= 2:
        options = []
        for i, j in combinations(range(len(candidates)), 2):
            duplicate, metrics = is_duplicate(candidates[i], candidates[j], config)
            if duplicate:
                continue
            for oi, oj in permutations(range(len(observations)), 2):
                ci = match_cost(candidates[i], observations[oi], k, config)
                cj = match_cost(candidates[j], observations[oj], k, config)
                if max(ci, cj) <= config['match_max_cost']:
                    options.append((ci + cj, i, j, oi, oj, metrics))
        if not options:
            return {'status': 'insufficient_observation', 'reason': 'no_consistent_two_person_match',
                    'candidate_indices': [], 'observation_indices': []}
        best = sorted(options)[0]
        cost, i, j, oi, oj, metrics = best
        order = ordered_pair(i, j, candidates, k)
        mapping = {i: oi, j: oj}
        return {'status': 'ok', 'basis': 'sapiens_consistent', 'candidate_indices': order,
                'observation_indices': [mapping[index] for index in order],
                'matching_cost': float(cost), 'pair_diagnostics': metrics}
    pair = fallback_pair(candidates, config)
    if pair is None:
        return {'status': 'insufficient_observation', 'reason': 'no_reliable_nonduplicate_pair',
                'candidate_indices': [], 'observation_indices': []}
    _, i, j, metrics = pair
    order = ordered_pair(i, j, candidates, k)
    assigned = [None, None]
    costs = []
    if observations:
        values = [match_cost(candidates[index], observations[0], k, config) for index in order]
        best_slot = int(np.argmin(values))
        if values[best_slot] <= config['match_max_cost']:
            assigned[best_slot] = 0
            costs = [values[best_slot]]
    return {'status': 'ok', 'basis': 'multihmr_score_fallback', 'candidate_indices': order,
            'observation_indices': assigned, 'matching_cost': float(sum(costs)) if costs else None,
            'pair_diagnostics': metrics}


def validate_observation(record, observation, request_sha):
    expected = {'schema': 'dual_pose_benchmark_v1', 'id': record['id'], 'mode': 'sapiens_detr070',
                'status': 'ok', 'image_sha256': record['image_sha256'], 'request_sha256': request_sha}
    if any(observation.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Stale or invalid Sapiens2 cache: {record["id"]}')
    if observation['det_count'] != len(observation['instances']):
        raise ValueError('Sapiens2 det_count does not match instances')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request, request_sha = read_json(args.request)
    fit_input, fit_records_sha = read_json(args.request.parent / 'fit_records.json')
    source_sha = fit_input['source_request_sha256']
    if fit_input.get('schema') != 'dual_mesh_fit_records_v1':
        raise ValueError('Invalid sanitized fit-records schema')
    with Path(request['predictions_pkl']).open('rb') as stream:
        predictions = pickle.load(stream)
    if request['schema'] != SCHEMA or len(fit_input['records']) != 127 or set(predictions) != {r['id'] for r in fit_input['records']}:
        raise ValueError('Expected the frozen 127-record prediction set')
    rows = []
    for record in fit_input['records']:
        observation_path = Path(request['observations_dir']) / f"{record['id']}.json"
        observation, observation_sha = read_json(observation_path)
        validate_observation(record, observation, source_sha)
        entry = predictions[record['id']]
        if entry['sample_id'] != record['id'] or entry['num_detected'] != len(entry['predictions']):
            raise ValueError(f'MultiHMR prediction mismatch: {record["id"]}')
        if sha256_file(record['image']) != record['image_sha256']:
            raise ValueError(f'Image hash mismatch: {record["id"]}')
        if not np.allclose(entry['K_orig'], record['K']) or list(entry['orig_size']) != list(record['size']):
            raise ValueError(f'Camera or image-size mismatch: {record["id"]}')
        selected = select_people(entry['predictions'], observation['instances'], np.asarray(record['K']), request['selection'])
        selected.update(id=record['id'], split=record['split'], candidate_count=len(entry['predictions']),
                        observation_count=len(observation['instances']), observation_sha256=observation_sha,
                        fit_record=fit_record(record))
        rows.append(selected)
    output = {'schema': 'dual_mesh_selection_v1', 'records': rows, 'total': 127,
              'ok': sum(row['status'] == 'ok' for row in rows),
              'source_request_sha256': source_sha, 'fit_records_sha256': fit_records_sha,
              'predictions_sha256': sha256_file(request['predictions_pkl']), 'config': request['selection']}
    write_json_xb(args.request.parent / 'selected.json', output)
    print({'total': 127, 'ok': output['ok'], 'insufficient': 127 - output['ok']})


if __name__ == '__main__':
    main()
```

注意：`FIT_RECORD_KEYS` 不含 `pair`，但 evaluator 从原始 request 读取 `pair` 做离线演员组分组；selector 输出的 `split` 只用于运行调度，不进入损失。

- [ ] **Step 4: 运行测试并生成共享选择**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_selection.py -q`
Expected: `5 passed`。

Run on `sg`:

```bash
cd /home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_mesh_refine_20261103_v1
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python scripts/select_candidates.py --request data/request.json
```

Expected: 退出码 0，`total=127`；`ok` 不预设为 125。若 `ok<125`，后续报告必须逐帧列出差距原因。

- [ ] **Step 5: 检查真实重复样本与输出不可覆盖性，再提交**

```bash
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -c "import json; p=json.load(open('data/selected.json')); r=next(x for x in p['records'] if x['id']=='243780fa6b6994fc'); print(r); assert r['status']=='ok'; assert len(set(r['candidate_indices']))==2"
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python scripts/select_candidates.py --request data/request.json; test $? -ne 0
```

Expected: 第一条打印真实选择且两索引不同；第二条因 `selected.json` 已存在而失败，随后的 `test` 成功。随后提交：

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/select_candidates.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_selection.py
git commit -m "feat(dual-refine): select nonduplicate people without ground truth"
```

### Task 4: 可见性、归属与真正冻结的身体部位

**Files:**
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/refine.py`

- [ ] **Step 1: 写低置信、出界、归属歧义和遮挡冻结测试**

```python
# tests/test_refine.py
import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('dual_refine', ROOT / 'scripts/refine.py')
refine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refine)
K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
VIS = {'confidence': .3, 'ownership_margin': .02, 'maximum_projection_cost': .15,
       'occlusion_depth_margin_m': .05, 'occlusion_segment_distance': .08, 'border_px': 2}


def candidate(x, z):
    joints = np.zeros((127, 3)); joints[:, 2] = z; joints[:, 0] = x
    joints[0] = [x, 0., z]
    return {'j3d': joints, 'bbox_orig': [x * 100, 0, x * 100 + 100, 200]}


def observation(x, score=1.):
    return {'xy': np.tile([x * 100 + 50., 50.], (12, 1)).tolist(),
            'scores': [score] * 12, 'bbox': [x * 100, 0, x * 100 + 100, 200],
            'joint_names': list(refine.BODY_NAMES)}


class VisibilityTests(unittest.TestCase):
    def test_low_confidence_and_out_of_frame_points_are_not_supervision(self):
        obs = observation(0); obs['scores'][0] = .1; obs['xy'][1] = [-1, 50]; obs['xy'][2] = [350, 50]
        candidates = [candidate(0, 4), candidate(2, 4)]
        candidates[0]['j3d'][refine.SMPL_IDS[2], 0] = 12.
        gates = refine.build_gates(candidates, [obs, None], K, [400, 200], VIS)
        self.assertFalse(gates['valid'][0, 0]); self.assertFalse(gates['valid'][0, 1])
        self.assertTrue(gates['valid'][0, 2])
        self.assertEqual(int(gates['valid'].sum()), 10)

    def test_arm_occlusion_freezes_gradient_dimensions_not_only_learning_rate(self):
        candidates = [candidate(0, 4), candidate(0, 3)]
        gates = refine.build_gates(candidates, [observation(0), observation(0)], K, [200, 400], VIS)
        self.assertTrue(gates['freeze_body'][0, [15, 16, 17, 18, 19, 20]].any())
        initial = np.zeros((2, 63), np.float32)
        effective = refine.apply_frozen_delta(initial, np.ones((2, 63), np.float32), ~gates['freeze_body'])
        np.testing.assert_array_equal(effective[gates['freeze_body']], 0.)

    def test_missing_observation_has_no_loss_denominator(self):
        gates = refine.build_gates([candidate(0, 4), candidate(2, 4)], [None, None], K, [200, 400], VIS)
        self.assertEqual(gates['valid'].sum(), 0)
        self.assertEqual(gates['counts']['valid_joints'], 0)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py -q`
Expected: FAIL，`refine.py` 不存在。

- [ ] **Step 3: 实现观测门；遮挡明确标为预测几何代理**

在 `refine.py` 写入以下完整首版内容；后续 Task 7 在同一文件追加优化函数：

```python
# scripts/refine.py
from __future__ import annotations
import numpy as np
from common import BODY_NAMES, SMPL_IDS, finite, project
from body import apply_frozen_delta

ARM_BODY_POSE_INDICES = [15, 16, 17, 18, 19, 20]
ARM_BODY12_INDICES = [0, 1, 2, 3, 4, 5]


def segment_distance(point, a, b):
    direction = b - a
    denominator = float(direction @ direction)
    t = 0. if denominator == 0 else np.clip(float((point - a) @ direction) / denominator, 0., 1.)
    return float(np.linalg.norm(point - (a + t * direction)))


def predicted_occlusion(person, other, k, person_scale, config):
    own = project(finite(person['j3d'])[SMPL_IDS], k)
    other_points = project(finite(other['j3d'])[SMPL_IDS], k)
    if person['j3d'][0, 2] <= other['j3d'][0, 2] + config['occlusion_depth_margin_m']:
        return np.zeros(12, bool)
    segments = [(0, 2), (1, 3), (2, 4), (3, 5), (6, 8), (7, 9), (8, 10), (9, 11), (6, 7)]
    result = np.zeros(12, bool)
    for joint in range(12):
        distance = min(segment_distance(own[joint], other_points[a], other_points[b]) for a, b in segments)
        result[joint] = distance < config['occlusion_segment_distance'] * person_scale
    return result


def build_gates(candidates, observations, k, image_size, config):
    if len(candidates) != 2 or len(observations) != 2:
        raise ValueError('Visibility gates require two selected people and two aligned observation slots')
    width, height = map(int, image_size)
    xy = np.zeros((2, 12, 2)); scores = np.zeros((2, 12)); valid = np.zeros((2, 12), bool)
    scales = np.ones(2); reasons = {}
    for person in range(2):
        observation = observations[person]
        if observation is None:
            reasons[str(person)] = ['missing_observation']
            continue
        if tuple(observation['joint_names']) != tuple(BODY_NAMES):
            raise ValueError('Observation body-12 order mismatch')
        xy[person] = finite(observation['xy'], (12, 2))
        scores[person] = finite(observation['scores'], (12,))
        box = finite(observation['bbox'], (4,)); scales[person] = np.linalg.norm(box[2:] - box[:2])
        predicted = project(finite(candidates[person]['j3d'])[SMPL_IDS], k)
        own_error = np.linalg.norm(predicted - xy[person], axis=1) / scales[person]
        inside = ((xy[person, :, 0] >= config['border_px']) & (xy[person, :, 0] < width - config['border_px']) &
                  (xy[person, :, 1] >= config['border_px']) & (xy[person, :, 1] < height - config['border_px']))
        valid[person] = (scores[person] >= config['confidence']) & inside & (own_error <= config['maximum_projection_cost'])
        other = observations[1 - person]
        if other is not None:
            other_xy = finite(other['xy'], (12, 2))
            other_error = np.linalg.norm(predicted - other_xy, axis=1) / scales[person]
            ambiguous = other_error + config['ownership_margin'] < own_error
            valid[person, ambiguous] = False
        occluded = predicted_occlusion(candidates[person], candidates[1 - person], k, scales[person], config)
        valid[person, occluded] = False
        reasons[str(person)] = {'low_confidence': np.flatnonzero(scores[person] < config['confidence']).tolist(),
                                'outside_or_border': np.flatnonzero(~inside).tolist(),
                                'projection_or_ownership': np.flatnonzero(own_error > config['maximum_projection_cost']).tolist(),
                                'predicted_other_mesh_occlusion_proxy': np.flatnonzero(occluded).tolist()}
    freeze_body = np.zeros((2, 63), bool)
    for person in range(2):
        if observations[person] is None or not valid[person, ARM_BODY12_INDICES].all():
            freeze_body[person, ARM_BODY_POSE_INDICES] = True
    return {'xy': xy, 'scores': scores, 'valid': valid, 'scales': scales,
            'freeze_body': freeze_body, 'reasons': reasons,
            'counts': {'valid_joints': int(valid.sum()), 'possible_joints': 24},
            'visibility_semantics': 'Sapiens confidence and predicted-geometry proxy, not anatomical GT visibility'}
```

- [ ] **Step 4: 运行可见性测试并提交**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py -q`
Expected: `3 passed`。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/refine.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py
git commit -m "feat(dual-refine): gate observations and freeze unreliable arms"
```

### Task 5: 先锁定 Warp kernel 查询，再建立球体解析量尺

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_collision.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/collision.py`

- [ ] **Step 1: 写 Warp 调用、分离/相切/重叠和绕向反例测试**

```python
# tests/test_collision.py
import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('dual_collision', ROOT / 'scripts/collision.py')
collision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collision)


def has_warp_cuda():
    if importlib.util.find_spec('warp') is None or importlib.util.find_spec('torch') is None:
        return False
    import torch
    return torch.cuda.is_available()


def sphere(center=(0., 0., 0.), radius=1., segments=64, rings=32):
    vertices = []
    for r in range(rings + 1):
        theta = np.pi * r / rings
        for s in range(segments):
            phi = 2 * np.pi * s / segments
            vertices.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    faces = []
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s; b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + s; d = (r + 1) * segments + (s + 1) % segments
            faces.extend([(a, c, b), (b, c, d)])
    return np.asarray(vertices, np.float32) * radius + center, np.asarray(faces, np.int32)


class CollisionTests(unittest.TestCase):
    @unittest.skipUnless(has_warp_cuda(), 'requires Warp and CUDA')
    def test_sphere_signed_depth_has_analytic_scale(self):
        vertices, faces = sphere()
        points = np.array([[0, 0, 0], [.5, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
        query = collision.query_signed(points, vertices, faces, 2.)
        np.testing.assert_allclose(query['penetration_depth'][:2], [1., .5], atol=.02)
        self.assertLessEqual(query['penetration_depth'][2], .02)
        self.assertEqual(query['penetration_depth'][3], 0.)

    @unittest.skipUnless(has_warp_cuda(), 'requires Warp and CUDA')
    def test_separated_tangent_shallow_and_deep_overlap_are_ordered(self):
        values = []
        for centre in (3., 2., 1.5, .5):
            source, faces = sphere(center=(centre, 0, 0)); target, _ = sphere()
            query = collision.query_signed(source, target, faces, 2.)
            values.append(collision.penetration_metrics([query])['mean_positive_depth_m'])
        self.assertEqual(values[0], 0.)
        self.assertLessEqual(values[1], 1e-6)
        self.assertGreater(values[2], values[1]); self.assertGreater(values[3], values[2])

    @unittest.skipUnless(has_warp_cuda(), 'requires Warp and CUDA')
    def test_reversed_winding_fails_orientation_probe(self):
        vertices, faces = sphere()
        self.assertGreater(collision.signed_volume(vertices, faces), 0.)
        self.assertLess(collision.signed_volume(vertices, faces[:, ::-1]), 0.)
        with self.assertRaises(ValueError):
            collision.query_signed(np.zeros((1, 3), np.float32), vertices, faces[:, ::-1], 2.)

    def test_both_directions_and_all_vertices_are_in_denominator(self):
        query = {'hit': np.array([1, 1, 1, 0]), 'sign': np.array([-1., -1., 1., 0.]),
                 'penetration_depth': np.array([.002, .01, 0., 0.])}
        metrics = collision.penetration_metrics([query, query])
        self.assertEqual(metrics['query_vertex_count'], 8)
        self.assertAlmostEqual(metrics['mean_positive_depth_m'], .003)
        self.assertAlmostEqual(metrics['penetrating_vertex_rate_2mm'], .5)
        self.assertAlmostEqual(metrics['penetrating_p95_m'], .01)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 在远端 M4 环境运行并确认先失败**

Run on `sg`: `/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m pytest tests/test_collision.py -q`
Expected: FAIL，`collision.py` 不存在。不得改用不存在的 `wp.mesh_query_point` Python 属性或 `wp.Mesh.query_point`。

- [ ] **Step 3: 实现 Warp 1.18.0 kernel 查询与指标**

```python
# scripts/collision.py
from __future__ import annotations
import numpy as np


def signed_volume(vertices, faces):
    vertices = np.asarray(vertices, np.float64); faces = np.asarray(faces, np.int64)
    triangles = vertices[faces]
    return float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.)


def query_signed(source_points, target_vertices, target_faces, max_dist_m=2., device='cuda:0'):
    import warp as wp
    wp.init()
    source_points = np.ascontiguousarray(source_points, dtype=np.float32).reshape(-1, 3)
    target_vertices = np.ascontiguousarray(target_vertices, dtype=np.float32).reshape(-1, 3)
    target_faces = np.ascontiguousarray(target_faces, dtype=np.int32).reshape(-1, 3)
    if not np.isfinite(source_points).all() or not np.isfinite(target_vertices).all():
        raise ValueError('Collision geometry must be finite')
    if signed_volume(target_vertices, target_faces) <= 0:
        raise ValueError('Target mesh must have outward winding and positive signed volume')

    @wp.kernel
    def closest_signed(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), max_dist: wp.float32,
                       hit: wp.array(dtype=wp.int32), sign: wp.array(dtype=wp.float32),
                       face_out: wp.array(dtype=wp.int32), u_out: wp.array(dtype=wp.float32),
                       v_out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        inside = wp.float32(0.)
        face = wp.int32(0)
        u = wp.float32(0.)
        v = wp.float32(0.)
        found = wp.mesh_query_point(mesh_id, points[tid], max_dist, inside, face, u, v)
        hit[tid] = 1 if found else 0
        sign[tid] = inside
        face_out[tid] = face
        u_out[tid] = u
        v_out[tid] = v

    with wp.ScopedDevice(device):
        vertices_wp = wp.array(target_vertices, dtype=wp.vec3)
        indices_wp = wp.array(target_faces.reshape(-1), dtype=wp.int32)
        points_wp = wp.array(source_points, dtype=wp.vec3)
        mesh = wp.Mesh(points=vertices_wp, indices=indices_wp, support_winding_number=False)
        outputs = [wp.zeros(len(source_points), dtype=dtype) for dtype in
                   (wp.int32, wp.float32, wp.int32, wp.float32, wp.float32)]
        wp.launch(closest_signed, dim=len(source_points),
                  inputs=[mesh.id, points_wp, float(max_dist_m)] + outputs)
        hit, sign, face, u, v = [value.numpy().copy() for value in outputs]
    hit = hit.astype(bool); face = np.clip(face, 0, len(target_faces) - 1)
    triangles = target_vertices[target_faces[face]]
    closest = ((1. - u[:, None] - v[:, None]) * triangles[:, 0] +
               u[:, None] * triangles[:, 1] + v[:, None] * triangles[:, 2])
    distance = np.linalg.norm(source_points - closest, axis=1)
    depth = np.where(hit & (sign < 0), distance, 0.)
    return {'hit': hit, 'sign': sign, 'face': face, 'u': u, 'v': v,
            'closest': closest, 'distance': distance, 'penetration_depth': depth}


def penetration_metrics(directional_queries):
    depth = np.concatenate([np.asarray(q['penetration_depth'], float).reshape(-1) for q in directional_queries])
    penetrating = depth[depth >= .002]
    positive = depth[depth > 0]
    return {'query_vertex_count': int(len(depth)),
            'mean_positive_depth_m': float(depth.mean()),
            'penetrating_vertex_rate_2mm': float(np.mean(depth >= .002)),
            'penetrating_p95_m': float(np.percentile(positive, 95)) if len(positive) else 0.,
            'max_penetration_m': float(depth.max()) if len(depth) else 0.,
            'max_penetration_over_5mm': bool(len(depth) and depth.max() > .005)}
```

`query_signed` 每次构造目标 BVH；refiner 每次都使用当前两人网格，不能复用上一轮 target points。若 Warp 编译或符号量尺失败，Task 6 之后全部 GPU 精修停止。

- [ ] **Step 4: 运行远端量尺并确认反向探针有效**

Run on `sg`: `/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m pytest tests/test_collision.py -q`
Expected: `4 passed`。特别检查 reversed winding 不是“通过但数值无变化”，而是正体积断言由正转负且查询明确抛出 `ValueError`。

- [ ] **Step 5: 提交**

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/collision.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_collision.py
git commit -m "feat(dual-refine): add signed Warp penetration oracle"
```

### Task 6: Torch 半隐式穿透项及梯度反例

**Files:**
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_collision.py`
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/collision.py`

- [ ] **Step 1: 添加目标面冻结、查询点可微和加深重叠测试**

```python
# 追加到 tests/test_collision.py 的 CollisionTests
    def test_semi_implicit_loss_moves_query_point_but_not_target(self):
        import torch
        source = torch.tensor([[0., 0., 0.]], requires_grad=True)
        target = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]], requires_grad=True)
        query = {'hit': np.array([True]), 'sign': np.array([-1.]), 'face': np.array([0]),
                 'u': np.array([0.]), 'v': np.array([0.])}
        loss = collision.semi_implicit_penetration(source, target, np.array([[0, 1, 2]]), query)
        self.assertAlmostEqual(float(loss), 1.)
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertGreater(float(source.grad.norm()), 0.)
        self.assertIsNone(target.grad)

    def test_artificially_deeper_query_has_larger_torch_loss(self):
        import torch
        target = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        query = {'hit': np.array([True]), 'sign': np.array([-1.]), 'face': np.array([0]),
                 'u': np.array([0.]), 'v': np.array([0.])}
        shallow = collision.semi_implicit_penetration(torch.tensor([[0., 0., .8]]), target, np.array([[0, 1, 2]]), query)
        deep = collision.semi_implicit_penetration(torch.tensor([[0., 0., 0.]]), target, np.array([[0, 1, 2]]), query)
        self.assertGreater(float(deep), float(shallow))
```

- [ ] **Step 2: 运行并确认函数缺失**

Run on `sg`: `/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m pytest tests/test_collision.py -q`
Expected: 原 4 项通过，新 2 项因 `semi_implicit_penetration` 不存在而 FAIL。

- [ ] **Step 3: 实现 Torch 重组与双向损失**

```python
# 追加到 scripts/collision.py

def semi_implicit_penetration(source_vertices, target_vertices, faces, query):
    import torch
    face = torch.as_tensor(query['face'], dtype=torch.long, device=source_vertices.device)
    u = torch.as_tensor(query['u'], dtype=source_vertices.dtype, device=source_vertices.device)
    v = torch.as_tensor(query['v'], dtype=source_vertices.dtype, device=source_vertices.device)
    hit = torch.as_tensor(query['hit'], dtype=torch.bool, device=source_vertices.device)
    inside = torch.as_tensor(query['sign'] < 0, dtype=torch.bool, device=source_vertices.device)
    triangles = target_vertices.detach()[torch.as_tensor(faces, dtype=torch.long, device=source_vertices.device)[face]]
    closest = ((1. - u[:, None] - v[:, None]) * triangles[:, 0] +
               u[:, None] * triangles[:, 1] + v[:, None] * triangles[:, 2])
    distance = torch.linalg.vector_norm(source_vertices - closest, dim=1)
    active = hit & inside
    return (distance * active.to(distance.dtype)).mean()


def bidirectional_collision(vertices, faces, max_dist_m=2.):
    queries = [query_signed(vertices[0], vertices[1], faces, max_dist_m),
               query_signed(vertices[1], vertices[0], faces, max_dist_m)]
    return queries, penetration_metrics(queries)
```

`semi_implicit_penetration` 的分母始终是当前方向全部查询顶点；只有 `hit & sign<0` 的点有非零值，target vertices 明确 `detach()`，查询 face/bary/sign 是本轮刷新的常数。

- [ ] **Step 4: 重跑全部碰撞测试并提交**

Run on `sg`: `/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m pytest tests/test_collision.py -q`
Expected: `6 passed`。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/collision.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_collision.py
git commit -m "feat(dual-refine): differentiate query points against detached target faces"
```

### Task 7: 四臂联合优化与可行集接受/回退

**Files:**
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py`
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/refine.py`

- [ ] **Step 1: 写接受规则、非劣效、冻结和回退测试**

```python
# 追加到 tests/test_refine.py
class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'penetration_epsilon_m': 1e-8, 'reprojection_tolerance': 1e-6,
                    'max_translation_drift_m': .15, 'max_root_drift_rad': .35,
                    'max_body_joint_drift_rad': .5}

    def row(self, depth=.01, reproj=.1, finite=True, depth_ok=True, frozen=True,
            translation=.0, root=.0, body=.0):
        return {'mean_positive_depth_m': depth, 'visible_reprojection': reproj,
                'finite': finite, 'positive_depth': depth_ok, 'frozen_unchanged': frozen,
                'translation_drift_m': translation, 'root_drift_rad': root,
                'body_joint_drift_rad_max': body}

    def test_acceptance_requires_real_penetration_decrease(self):
        initial = self.row(depth=.02)
        self.assertTrue(refine.acceptable(self.row(depth=.01), initial, self.cfg))
        self.assertFalse(refine.acceptable(self.row(depth=.02), initial, self.cfg))
        self.assertFalse(refine.acceptable(self.row(depth=.03), initial, self.cfg))

    def test_visible_reprojection_and_geometry_drift_are_noninferiority_gates(self):
        initial = self.row(depth=.02, reproj=.1)
        self.assertFalse(refine.acceptable(self.row(reproj=.10001), initial, self.cfg))
        self.assertFalse(refine.acceptable(self.row(translation=.2), initial, self.cfg))
        self.assertFalse(refine.acceptable(self.row(root=.4), initial, self.cfg))
        self.assertFalse(refine.acceptable(self.row(body=.6), initial, self.cfg))

    def test_nan_missing_depth_or_frozen_change_never_passes(self):
        initial = self.row(depth=.02)
        for changed in ({'finite': False}, {'depth_ok': False}, {'frozen_unchanged': False}):
            self.assertFalse(refine.acceptable(self.row(**changed), initial, self.cfg))

    def test_fallback_is_explicit_and_keeps_initial(self):
        result = refine.choose_export([self.row(depth=.03), self.row(depth=.02)],
                                      self.row(depth=.02), self.cfg, iteration_rows=[])
        self.assertEqual(result['export_reason'], 'no_improving_iterate')
        self.assertEqual(result['selected_iteration'], -1)

    def test_lowest_penetration_feasible_iterate_wins_not_last_iterate(self):
        rows = [self.row(depth=.015), self.row(depth=.008), self.row(depth=.012)]
        result = refine.choose_export(rows, self.row(depth=.02), self.cfg, iteration_rows=[0, 1, 2])
        self.assertEqual(result['selected_iteration'], 1)
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py -q`
Expected: 原 3 项通过，新测试因 `acceptable/choose_export` 不存在而 FAIL。

- [ ] **Step 3: 实现接受函数和四臂优化循环**

```python
# 追加到 scripts/refine.py

def acceptable(row, initial, config):
    if not all(row.get(key, False) for key in ('finite', 'positive_depth', 'frozen_unchanged')):
        return False
    if row['mean_positive_depth_m'] >= initial['mean_positive_depth_m'] - config['penetration_epsilon_m']:
        return False
    if initial['visible_reprojection'] is not None:
        if row['visible_reprojection'] is None or row['visible_reprojection'] > initial['visible_reprojection'] + config['reprojection_tolerance']:
            return False
    return (row['translation_drift_m'] <= config['max_translation_drift_m'] and
            row['root_drift_rad'] <= config['max_root_drift_rad'] and
            row['body_joint_drift_rad_max'] <= config['max_body_joint_drift_rad'])


def choose_export(rows, initial, config, iteration_rows):
    feasible = [(row['mean_positive_depth_m'], iteration, index)
                for index, (row, iteration) in enumerate(zip(rows, iteration_rows))
                if acceptable(row, initial, config)]
    if not feasible:
        return {'selected_iteration': -1, 'selected_row_index': -1,
                'export_reason': 'no_improving_iterate', 'feasible_iterate_count': 0}
    _, iteration, index = min(feasible)
    return {'selected_iteration': iteration, 'selected_row_index': index,
            'export_reason': 'best_feasible_penetration', 'feasible_iterate_count': len(feasible)}


def visible_reprojection(joints, gates, k):
    import torch
    values = []
    for person in range(2):
        mask = torch.as_tensor(gates['valid'][person], device=joints.device)
        if not mask.any():
            continue
        scale = torch.as_tensor(gates['scales'][person], dtype=joints.dtype, device=joints.device)
        target = torch.as_tensor(gates['xy'][person], dtype=joints.dtype, device=joints.device)
        xyz = joints[person][:, SMPL_IDS]
        uv = (xyz[:, :2] / xyz[:, 2:]) @ k[:2, :2].T + k[:2, 2]
        errors = torch.linalg.vector_norm(uv - target, dim=1) / scale
        confidence = torch.as_tensor(gates['scores'][person], dtype=joints.dtype, device=joints.device)
        values.append((errors * confidence)[mask])
    return torch.cat(values) if values else None


def refine_frame(model, candidates, gates, k_native, arm, config):
    import torch
    from collision import bidirectional_collision, semi_implicit_penetration
    if arm not in ('reproj-only', 'collision-only', 'joint-full'):
        raise ValueError('refine_frame expects an optimizing arm')
    initial_params = [model.initial_parameters(person) for person in candidates]
    k = torch.as_tensor(k_native, dtype=torch.float32, device=model.device)
    root0 = torch.stack([torch.as_tensor(p['global_orient'], device=model.device) for p in initial_params])
    body0 = torch.stack([torch.as_tensor(p['body_pose'], device=model.device) for p in initial_params])
    transl0 = torch.stack([torch.as_tensor(p['transl'], device=model.device) for p in initial_params])
    root_delta = torch.zeros_like(root0, requires_grad=True)
    body_delta = torch.zeros_like(body0, requires_grad=True)
    transl_delta = torch.zeros_like(transl0, requires_grad=True)
    optimizer = torch.optim.Adam([root_delta, body_delta, transl_delta], lr=config['learning_rate'])
    freeze = torch.as_tensor(gates['freeze_body'], device=model.device)
    history, snapshots = [], []

    def current():
        params = []
        for person, initial in enumerate(initial_params):
            item = dict(initial)
            item['global_orient'] = root0[person] + root_delta[person]
            item['body_pose'] = body0[person] + body_delta[person] * (~freeze[person]).to(body0.dtype)
            item['transl'] = transl0[person] + transl_delta[person]
            params.append(item)
        outputs = [model.forward(item) for item in params]
        return params, torch.stack([x[0] for x in outputs]), torch.stack([x[1] for x in outputs])

    def diagnostic(params, vertices, joints):
        numpy_vertices = vertices.detach().cpu().numpy()
        queries, penetration = bidirectional_collision(numpy_vertices, model.faces, config['max_dist_m'])
        visible = visible_reprojection(joints, gates, k)
        with torch.no_grad():
            root_drift = torch.linalg.vector_norm(root_delta, dim=1).max()
            body_drift = torch.linalg.vector_norm(body_delta * (~freeze).to(body0.dtype), dim=2).max()
            transl_drift = torch.linalg.vector_norm(transl_delta, dim=1).max()
        return {**penetration,
                'visible_reprojection': float(visible.mean()) if visible is not None else None,
                'finite': bool(torch.isfinite(vertices).all() and torch.isfinite(joints).all()),
                'positive_depth': bool((vertices[..., 2] > 0).all()),
                'frozen_unchanged': bool(torch.equal((body_delta * freeze.to(body0.dtype)), torch.zeros_like(body_delta))),
                'translation_drift_m': float(transl_drift), 'root_drift_rad': float(root_drift),
                'body_joint_drift_rad_max': float(body_drift)}, queries

    with torch.no_grad():
        params, vertices, joints = current()
        initial, _ = diagnostic(params, vertices, joints)
    for iteration in range(config['steps']):
        optimizer.zero_grad(set_to_none=True)
        params, vertices, joints = current()
        visible = visible_reprojection(joints, gates, k)
        delta = config['huber_delta']
        if visible is None:
            reproj = vertices.sum() * 0.
        else:
            reproj = (delta ** 2 * (torch.sqrt(1. + (visible / delta) ** 2) - 1.)).mean()
        numpy_vertices = vertices.detach().cpu().numpy()
        queries, penetration = bidirectional_collision(numpy_vertices, model.faces, config['max_dist_m'])
        collision_terms = [semi_implicit_penetration(vertices[0], vertices[1], model.faces, queries[0]),
                           semi_implicit_penetration(vertices[1], vertices[0], model.faces, queries[1])]
        collision_loss = .5 * sum(collision_terms)
        pose_reg = (body_delta ** 2).mean()
        root_reg = (root_delta ** 2).mean()
        transl_reg = (transl_delta ** 2).mean()
        use_reproj = arm in ('reproj-only', 'joint-full')
        use_collision = arm in ('collision-only', 'joint-full')
        loss = (config['reproj_weight'] * reproj * use_reproj +
                config['pose_reg_weight'] * pose_reg + config['root_reg_weight'] * root_reg +
                config['transl_reg_weight'] * transl_reg +
                config['collision_weight'] * collision_loss * use_collision)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at iteration {iteration}')
        loss.backward()
        with torch.no_grad():
            body_delta.grad.mul_((~freeze).to(body0.dtype))
        optimizer.step()
        with torch.no_grad():
            params, vertices, joints = current()
            row, _ = diagnostic(params, vertices, joints)
        row.update(iteration=iteration, total_loss=float(loss.detach()),
                   reprojection_loss=float(reproj.detach()), collision_loss=float(collision_loss.detach()),
                   pose_regularization=float(pose_reg.detach()), root_regularization=float(root_reg.detach()),
                   translation_regularization=float(transl_reg.detach()))
        history.append(row)
        snapshots.append([{key: value.detach().cpu().numpy().copy() for key, value in item.items()}
                          for item in params])
    rows = [{key: row[key] for key in ('mean_positive_depth_m', 'visible_reprojection', 'finite',
            'positive_depth', 'frozen_unchanged', 'translation_drift_m', 'root_drift_rad',
            'body_joint_drift_rad_max')} for row in history]
    export = choose_export(rows, initial, config, list(range(len(history))))
    if export['selected_row_index'] < 0:
        chosen = [{key: np.asarray(value).copy() for key, value in item.items()} for item in initial_params]
        final = dict(initial)
    else:
        index = export['selected_row_index']
        chosen = list(snapshots[index]); final = history[index]
    return {'initial': initial, 'history': history, 'params': chosen, 'final': final, **export}
```

- [ ] **Step 4: 运行接受与回退测试**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py -q`
Expected: `8 passed`。`choose_export` 的反例必须证明选择第二迭代而非最后迭代，`snapshots` 必须是每次迭代复制后的具体列表。

- [ ] **Step 5: 做 GPU 单步反向探针并提交**

在 1 个 train 帧上分别跑 `reproj-only` 与 `joint-full`，并对真实 SMPL-X 网格做三项量尺：双向 `query_vertex_count` 必须等于 `2×10475=20950`；把两人沿相机 X 方向分开到网格包围盒无交集后，双向平均正穿透必须为 0；把其中一人向另一人平移 5 cm 制造更深重叠后，`mean_positive_depth_m` 与 `collision_loss` 必须高于原初值。任一项不满足都禁止扩大实验。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/refine.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py
git commit -m "feat(dual-refine): optimize four-arm objectives with feasible rollback"
```

### Task 8: 每帧 CLI、完整状态、哈希和默认不覆盖

**Files:**
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py`
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/refine.py`

- [ ] **Step 1: 添加失败输出也保留分母和指纹的测试**

```python
# 追加到 tests/test_refine.py
class ResultContractTests(unittest.TestCase):
    def test_every_arm_has_127_slots_even_when_selection_failed(self):
        records = [{'id': f'{i:016x}', 'status': 'insufficient_observation'} for i in range(127)]
        rows = refine.status_rows(records, 'joint-full')
        self.assertEqual(len(rows), 127)
        self.assertTrue(all(row['arm'] == 'joint-full' for row in rows))
        self.assertTrue(all(row['status'] == 'insufficient_observation' for row in rows))

    def test_cache_requires_same_config_and_input_hash(self):
        expected = {'id': 'a' * 16, 'arm': 'joint-full', 'selection_sha256': 'b' * 64,
                    'config_sha256': 'c' * 64, 'status': 'ok'}
        self.assertFalse(refine.cache_reusable(dict(expected, status='error'), expected))
        self.assertTrue(refine.cache_reusable(dict(expected), expected))
        with self.assertRaises(FileExistsError):
            refine.cache_reusable(dict(expected, config_sha256='d' * 64), expected)
```

- [ ] **Step 2: 运行并确认新函数缺失**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py -q`
Expected: 新 2 项 FAIL。

- [ ] **Step 3: 实现 CLI 状态与输出契约**

```python
# 追加到 scripts/refine.py

def status_rows(selection_records, arm):
    return [{'id': row['id'], 'arm': arm, 'status': row['status'],
             'error': row.get('reason')} for row in selection_records]


def cache_reusable(saved, expected):
    if not saved:
        return False
    if any(saved.get(key) != value for key, value in expected.items() if key != 'status'):
        raise FileExistsError('Existing output has a different fingerprint')
    return saved.get('status') == 'ok' and expected.get('status') == 'ok'


def save_frame_output(directory, result, model):
    from common import sha256_file, write_json_xb
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for person, params in enumerate(result['params']):
        for key, value in params.items():
            arrays[f'person{person}_{key}'] = np.asarray(value, np.float32)
        with model.torch.no_grad():
            vertices, joints = model.forward(result['params'][person])
        arrays[f'person{person}_vertices'] = vertices.detach().cpu().numpy()
        arrays[f'person{person}_joints'] = joints.detach().cpu().numpy()
    arrays['faces'] = model.faces
    geometry_path = directory / 'geometry.npz'
    metrics_path = directory / 'metrics.json'
    with geometry_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    payload = {key: value for key, value in result.items() if key != 'params'}
    write_json_xb(metrics_path, payload)
    return {'geometry_sha256': sha256_file(geometry_path), 'metrics_sha256': sha256_file(metrics_path)}


def run_frames(request, selection, arm, split, limit, ids, out_root):
    import hashlib
    import json
    import pickle
    import torch
    from body import BodyModel
    from common import read_json, sha256_file
    if arm not in ('selected-init', 'reproj-only', 'collision-only', 'joint-full'):
        raise ValueError('Unknown arm')
    config = {key: request[key] for key in ('selection', 'visibility', 'optimization')}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    code_sha = {name: sha256_file(Path(__file__).with_name(name))
                for name in ('common.py', 'body.py', 'collision.py', 'refine.py')}
    with open(request['predictions_pkl'], 'rb') as stream:
        predictions = pickle.load(stream)
    model = None
    selected_rows = [row for row in selection['records'] if row['split'] == split]
    if ids:
        unknown = set(ids) - {row['id'] for row in selected_rows}
        if unknown:
            raise ValueError(f'Unknown ids for {split}: {sorted(unknown)}')
        selected_rows = [row for row in selected_rows if row['id'] in ids]
    if limit is not None:
        selected_rows = selected_rows[:limit]
    summaries = []
    for row in selected_rows:
        destination = out_root / arm / split / row['id']
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}')
        if row['status'] != 'ok':
            summaries.append({'id': row['id'], 'arm': arm, 'status': row['status'], 'error': row.get('reason')})
            continue
        try:
            if model is None:
                model = BodyModel(request['smplx_model'], request['device'])
            candidates_all = predictions[row['id']]['predictions']
            candidates = [candidates_all[index] for index in row['candidate_indices']]
            observation_path = Path(request['observations_dir']) / f"{row['id']}.json"
            cached, observation_sha = read_json(observation_path)
            if observation_sha != row['observation_sha256']:
                raise ValueError('Sapiens2 cache changed after selection freeze')
            observations = [None if index is None else cached['instances'][index]
                            for index in row['observation_indices']]
            k = np.asarray(row['fit_record']['K'], np.float64)
            initial_params = [model.initial_parameters(person) for person in candidates]
            reconstruction_checks = [model.verify_initial(person, params, k)
                                     for person, params in zip(candidates, initial_params)]
            gates = build_gates(candidates, observations, k, row['fit_record']['size'], request['visibility'])
            if arm == 'selected-init':
                params = initial_params
                with torch.no_grad():
                    outputs = [model.forward(person) for person in params]
                    vertices = np.stack([x[0].cpu().numpy() for x in outputs])
                    joints = np.stack([x[1].cpu().numpy() for x in outputs])
                from collision import bidirectional_collision
                _, initial = bidirectional_collision(vertices, model.faces, request['optimization']['max_dist_m'])
                visible = visible_reprojection(torch.as_tensor(joints, device=model.device), gates,
                                               torch.as_tensor(k, dtype=torch.float32, device=model.device))
                initial['visible_reprojection'] = float(visible.mean()) if visible is not None else None
                result = {'params': params, 'history': [], 'initial': initial, 'final': initial,
                          'selected_iteration': -1, 'selected_row_index': -1,
                          'export_reason': 'selected_initialization', 'feasible_iterate_count': 0}
            else:
                result = refine_frame(model, candidates, gates, k, arm, request['optimization'])
            result.update(schema='dual_mesh_frame_v1', id=row['id'], arm=arm, split=split,
                          status='ok', selection_sha256=selection['selection_sha256'],
                          observation_sha256=observation_sha, config_sha256=config_sha, code_sha256=code_sha,
                          candidate_indices=row['candidate_indices'], observation_indices=row['observation_indices'],
                          reconstruction_checks=reconstruction_checks,
                          visibility_counts=gates['counts'], visibility_reasons=gates['reasons'])
            artifacts = save_frame_output(destination, result, model)
            summaries.append({'id': row['id'], 'arm': arm, 'status': 'ok',
                              'export_reason': result['export_reason'], **artifacts})
        except Exception as exc:
            summaries.append({'id': row['id'], 'arm': arm, 'status': 'error',
                              'error': f'{type(exc).__name__}: {exc}'})
    return summaries


def main():
    import argparse
    import json
    from common import SCHEMA, read_json, write_json_xb
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--arm', choices=['selected-init', 'reproj-only', 'collision-only', 'joint-full'], required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    request, request_sha = read_json(args.request)
    if request.get('schema') != SCHEMA or (args.limit is not None and args.limit < 0):
        raise ValueError('Invalid request schema or negative limit')
    selection, selection_sha = read_json(args.request.parent / 'selected.json')
    selection['selection_sha256'] = selection_sha
    out_root = args.out_root.resolve()
    if not out_root.is_relative_to(Path(request['out_dir']).resolve()):
        raise ValueError('out-root must be inside the experiment output directory')
    rows = run_frames(request, selection, args.arm, args.split, args.limit, args.ids, out_root)
    summary = {'schema': 'dual_mesh_run_summary_v1', 'arm': args.arm, 'split': args.split,
               'request_sha256': request_sha, 'selection_sha256': selection_sha,
               'requested_frame_count': len(rows), 'completed_frame_count': len(rows),
               'valid_two_person_frames': sum(row['status'] == 'ok' for row in rows),
               'error_frames': sum(row['status'] == 'error' for row in rows), 'records': rows}
    write_json_xb(out_root / args.arm / args.split / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'records'}))
    return 1 if summary['error_frames'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试、语法编译和独占创建反例**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py -q`
Expected: 全部通过，至少包含前述 10 项测试。

Run on `sg`：

```bash
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m py_compile scripts/*.py
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python scripts/refine.py --request data/request.json --arm selected-init --split train --limit 1 --out-root /data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/pilot
```

Expected: 退出码 0，输出一个帧目录或一个明确 selection failure 状态；同命令第二次必须因已有输出失败，不覆盖。

- [ ] **Step 5: 提交**

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/refine.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_refine.py
git commit -m "feat(dual-refine): export hashed frame outputs without overwrites"
```

### Task 9: 独立 GT 评测、配对 cluster bootstrap 和硬门

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/evaluate.py`

- [ ] **Step 1: 写漏人分母、配对差、bootstrap 和反转门测试**

```python
# tests/test_evaluate.py
import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('dual_evaluate', ROOT / 'scripts/evaluate.py')
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class EvaluateTests(unittest.TestCase):
    def test_missing_person_keeps_gt_joint_denominator(self):
        row = {'gt_joint_count': 24, 'pck10_hits': 12, 'status': 'ok'}
        self.assertEqual(evaluate.coverage([row])['e2e_pck10'], .5)
        self.assertEqual(evaluate.coverage([row])['frames'], 1)

    def test_delta_direction_is_joint_full_minus_selected_init(self):
        a = [{'id': 'x', 'value': .1, 'pair': 'p1'}]
        b = [{'id': 'x', 'value': .2, 'pair': 'p1'}]
        result = evaluate.paired_bootstrap(a, b, 'value', 'pair', seed=1, repetitions=100)
        self.assertAlmostEqual(result['mean_delta'], -.1)
        self.assertEqual(result['n'], 1)

    def test_cluster_bootstrap_resamples_actors_not_individual_frames(self):
        a = [{'id': str(i), 'value': 1., 'pair': 'p' + str(i % 2)} for i in range(8)]
        b = [{'id': str(i), 'value': 0., 'pair': 'p' + str(i % 2)} for i in range(8)]
        result = evaluate.paired_bootstrap(a, b, 'value', 'pair', seed=2, repetitions=100)
        self.assertEqual(result['actor_clusters'], 2)
        self.assertEqual(result['n'], 8)

    def test_effect_gate_must_fail_when_direction_is_reversed(self):
        intervals = {'mean_positive_depth_m': [-1., -.1], 'root_mpjpe_mm': [0., 1.],
                     'absolute_mpjpe_mm': [0., 1.], 'pelvis_vector_error_mm': [0., 1.],
                     'e2e_pck10': [0., .1]}
        self.assertTrue(evaluate.metric_gates(intervals)['mean_positive_depth_m'])
        intervals['mean_positive_depth_m'] = [-.1, .1]
        self.assertFalse(evaluate.metric_gates(intervals)['mean_positive_depth_m'])
        intervals['mean_positive_depth_m'] = [-1., -.1]
        intervals['root_mpjpe_mm'] = [0., 5.1]
        self.assertFalse(evaluate.metric_gates(intervals)['root_mpjpe_mm'])
        intervals['e2e_pck10'] = [-.01, 0.]
        self.assertFalse(evaluate.metric_gates(intervals)['e2e_pck10'])

    def test_coverage_gate_cannot_pass_with_partial_denominator(self):
        self.assertFalse(evaluate.coverage_gate(
            {'joint-full': 100, 'selected-init': 100},
            {'joint-full': 100, 'selected-init': 100}, 127))
        self.assertTrue(evaluate.coverage_gate(
            {'joint-full': 125, 'selected-init': 125},
            {'joint-full': 127, 'selected-init': 127}, 127))
        self.assertFalse(evaluate.coverage_gate(
            {'joint-full': 124, 'selected-init': 125},
            {'joint-full': 127, 'selected-init': 127}, 127))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py -q`
Expected: FAIL，`evaluate.py` 不存在。

- [ ] **Step 3: 实现统计与门禁核心**

```python
# scripts/evaluate.py
from __future__ import annotations
import numpy as np

METRICS = ['mean_positive_depth_m', 'root_mpjpe_mm', 'absolute_mpjpe_mm',
           'pelvis_vector_error_mm', 'e2e_pck10']


def coverage(rows):
    gt = sum(row['gt_joint_count'] for row in rows)
    hits05 = sum(row.get('pck05_hits', 0) for row in rows)
    hits10 = sum(row['pck10_hits'] for row in rows)
    result = {'frames': len(rows), 'gt_joint_count': gt, 'pck05_hits': hits05, 'pck10_hits': hits10,
              'e2e_pck05': hits05 / gt if gt else None, 'e2e_pck10': hits10 / gt if gt else None}
    for key in ('root_mpjpe_mm', 'absolute_mpjpe_mm', 'pelvis_vector_error_mm',
                'mean_positive_depth_m', 'penetrating_vertex_rate_2mm', 'penetrating_p95_m'):
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = {'n': len(values), 'mean': float(np.mean(values)) if values else None,
                       'p95': float(np.percentile(values, 95)) if values else None}
    exceeding = [bool(row['max_penetration_over_5mm']) for row in rows
                 if row.get('max_penetration_over_5mm') is not None]
    result['max_penetration_over_5mm_frame_rate'] = float(np.mean(exceeding)) if exceeding else None
    depths = np.asarray([value for row in rows for value in row.get('_penetrating_depths', [])], float)
    result['penetrating_p95_m_pooled_vertices'] = float(np.percentile(depths, 95)) if len(depths) else None
    return result


def paired_bootstrap(full_rows, init_rows, metric, group_key, seed=20261103, repetitions=10000):
    full = {row['id']: row for row in full_rows}
    initial = {row['id']: row for row in init_rows}
    ids = sorted(set(full) & set(initial))
    pairs = [(full[i], initial[i]) for i in ids
             if full[i].get(metric) is not None and initial[i].get(metric) is not None]
    if not pairs:
        return {'n': 0, 'actor_clusters': 0, 'mean_delta': None, 'ci95': None}
    deltas = np.asarray([a[metric] - b[metric] for a, b in pairs], float)
    groups = {}
    for (a, _), delta in zip(pairs, deltas):
        groups.setdefault(a[group_key], []).append(float(delta))
    names = sorted(groups)
    if len(names) < 2:
        return {'n': len(deltas), 'actor_clusters': len(names), 'mean_delta': float(deltas.mean()),
                'ci95': None, 'direction': 'joint-full minus selected-init',
                'estimator': 'insufficient actor clusters for a non-degenerate bootstrap interval'}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repetitions):
        sampled = rng.choice(names, len(names), replace=True)
        values = np.concatenate([groups[name] for name in sampled])
        boot.append(float(values.mean()))
    return {'n': len(deltas), 'actor_clusters': len(names), 'mean_delta': float(deltas.mean()),
            'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'direction': 'joint-full minus selected-init',
            'estimator': 'per-frame paired delta; bootstrap resamples actor clusters and retains all frames in each sampled cluster'}


def metric_gates(intervals):
    return {'mean_positive_depth_m': intervals['mean_positive_depth_m'][1] < 0.,
            'root_mpjpe_mm': intervals['root_mpjpe_mm'][1] <= 5.,
            'absolute_mpjpe_mm': intervals['absolute_mpjpe_mm'][1] <= 5.,
            'pelvis_vector_error_mm': intervals['pelvis_vector_error_mm'][1] <= 5.,
            'e2e_pck10': intervals['e2e_pck10'][0] >= -.005}


def coverage_gate(valid_by_arm, state_rows_by_arm, total):
    required = {'selected-init', 'reproj-only', 'collision-only', 'joint-full'}
    return (total == 127 and required <= set(valid_by_arm) and required <= set(state_rows_by_arm) and
            all(state_rows_by_arm[arm] == total for arm in required) and
            valid_by_arm['selected-init'] >= 125 and
            valid_by_arm['joint-full'] >= valid_by_arm['selected-init'])
```

- [ ] **Step 4: 实现独立帧评分与全量汇总**

在 `evaluate.py` 追加以下函数。它只接受原始 GT record、输出 `geometry.npz/metrics.json` 和共享 selection，不导入 `refine.py`：

```python

def score_gt_frame(record, geometry, status):
    from common import SMPL_IDS, project
    gt_xy = np.asarray(record['gt_xy'], float)
    gt_valid = np.asarray(record['gt_valid'], bool)
    gt_boxes = np.asarray(record['gt_boxes'], float)
    scales = np.linalg.norm(gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1)
    result = {'id': record['id'], 'split': record['split'], 'pair': record['pair'],
              'category': record['category'], 'status': status,
              'gt_joint_count': int(gt_valid.sum()), 'pck05_hits': 0, 'pck10_hits': 0,
              'e2e_pck05': 0.0, 'e2e_pck10': 0.0,
              'root_mpjpe_mm': None, 'absolute_mpjpe_mm': None, 'pelvis_vector_error_mm': None,
              'mean_positive_depth_m': None, 'penetrating_vertex_rate_2mm': None,
              'penetrating_p95_m': None, 'max_penetration_over_5mm': None}
    if status != 'ok':
        return result
    joints = np.stack([geometry[f'person{i}_joints'] for i in range(2)])
    vertices = np.stack([geometry[f'person{i}_vertices'] for i in range(2)])
    predicted = np.stack([project(person[SMPL_IDS], np.asarray(record['K'])) for person in joints])
    candidates = []
    from itertools import combinations, permutations
    for gt_order in combinations(range(2), min(2, len(predicted))):
        for pred_order in permutations(range(len(predicted)), len(gt_order)):
            cost = 0.
            for g, p in zip(gt_order, pred_order):
                errors = np.linalg.norm(predicted[p] - gt_xy[g], axis=1) / scales[g]
                cost += float(errors[gt_valid[g]].mean())
            candidates.append((cost, list(zip(gt_order, pred_order))))
    matches = min(candidates)[1] if candidates else []
    normalized = []
    root_errors, absolute_errors = [], []
    assignment = [None, None]
    for g, p in matches:
        assignment[g] = p
        errors = np.linalg.norm(predicted[p] - gt_xy[g], axis=1) / scales[g]
        normalized.extend(errors[gt_valid[g]].tolist())
        pred_xyz = joints[p, SMPL_IDS]; gt_xyz = np.asarray(record['gt_xyz'][g], float)
        root_errors.extend((1000 * np.linalg.norm(pred_xyz - joints[p, 0] - gt_xyz + np.asarray(record['gt_pelvis'][g]), axis=1)).tolist())
        absolute_errors.extend((1000 * np.linalg.norm(pred_xyz - gt_xyz, axis=1)).tolist())
    normalized = np.asarray(normalized)
    result['pck05_hits'] = int((normalized < .05).sum())
    result['pck10_hits'] = int((normalized < .1).sum())
    result['e2e_pck05'] = result['pck05_hits'] / result['gt_joint_count'] if result['gt_joint_count'] else None
    result['e2e_pck10'] = result['pck10_hits'] / result['gt_joint_count'] if result['gt_joint_count'] else None
    result['root_mpjpe_mm'] = float(np.mean(root_errors)) if root_errors else None
    result['absolute_mpjpe_mm'] = float(np.mean(absolute_errors)) if absolute_errors else None
    if len(matches) == 2:
        pa, pb = joints[assignment[0], 0], joints[assignment[1], 0]
        reference = np.asarray(record['gt_pelvis'], float)
        result['pelvis_vector_error_mm'] = float(1000 * np.linalg.norm((pb - pa) - (reference[1] - reference[0])))
    from collision import bidirectional_collision
    queries, penetration = bidirectional_collision(vertices, np.asarray(geometry['faces']), 2.)
    result.update(penetration)
    depths = np.concatenate([np.asarray(query['penetration_depth']) for query in queries])
    result['_penetrating_depths'] = depths[depths > 0].tolist()
    return result


def build_summary(frame_rows, total=127):
    summaries = {}
    for arm in ('selected-init', 'reproj-only', 'collision-only', 'joint-full'):
        rows = [row for row in frame_rows if row['arm'] == arm]
        summaries[arm] = {'overall': coverage(rows), 'valid_two_person_frames': sum(row['status'] == 'ok' for row in rows),
                          'state_rows': len(rows)}
        for field in ('split', 'category'):
            summaries[arm][field] = {value: coverage([row for row in rows if row[field] == value])
                                     for value in sorted({row[field] for row in rows})}
    full = [row for row in frame_rows if row['arm'] == 'joint-full']
    initial = [row for row in frame_rows if row['arm'] == 'selected-init']
    paired = {metric: paired_bootstrap(full, initial, metric, 'pair') for metric in METRICS}
    intervals = {metric: paired[metric]['ci95'] for metric in METRICS}
    gates = metric_gates(intervals) if all(value is not None for value in intervals.values()) else {}
    valid = {arm: summaries[arm]['valid_two_person_frames'] for arm in summaries}
    state_rows = {arm: summaries[arm]['state_rows'] for arm in summaries}
    gates['coverage'] = coverage_gate(valid, state_rows, total)
    complete = all(gates.values()) and len(gates) == 6
    return {'schema': 'dual_mesh_evaluation_v1', 'total_denominator': total,
            'summaries': summaries, 'paired': paired, 'gates': gates,
            'promotion_pass': complete, 'rows': frame_rows}
```

本任务先完成可单测的统计核心；Task 10 在报告函数之后给出完整 CLI。CLI 必须为四臂各构造 127 个状态槽，缺文件记为 `missing`，写 `results.json/metrics.csv/summary.json/report/FINAL.md`，并在 incomplete、区间缺失或任一门失败时返回非零。

- [ ] **Step 5: 运行统计测试并提交**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py -q`
Expected: `5 passed`。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/evaluate.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py
git commit -m "test(dual-refine): enforce paired effects and full coverage gates"
```

### Task 10: 自动生成可审计报告和反向探针清单

**Files:**
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/evaluate.py`
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py`

- [ ] **Step 1: 添加“功效不足不能写成胜出”和 125/127 差距解释测试**

```python
# 追加到 tests/test_evaluate.py
class ReportTests(unittest.TestCase):
    def test_missing_ci_is_insufficient_power_not_improvement(self):
        text = evaluate.render_conclusion({'mean_positive_depth_m': None}, {'mean_positive_depth_m': -.01})
        self.assertIn('指标功效不足', text)
        self.assertNotIn('统计上可分辨的穿透下降', text)

    def test_selection_gap_lists_every_frame(self):
        records = [{'id': 'a', 'status': 'ok'}, {'id': 'b', 'status': 'insufficient_observation'},
                   {'id': 'c', 'status': 'insufficient_observation'}]
        self.assertEqual(evaluate.selection_gap(records), ['b', 'c'])
```

- [ ] **Step 2: 运行并确认新函数缺失**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py -q`
Expected: 新 2 项 FAIL。

- [ ] **Step 3: 实现结论与 Markdown 报告生成**

```python
# 追加到 scripts/evaluate.py

def selection_gap(selection_records):
    return [row['id'] for row in selection_records if row['status'] != 'ok']


def render_conclusion(intervals, deltas):
    interval = intervals.get('mean_positive_depth_m')
    if interval is None:
        return '指标功效不足：平均正穿透深度缺少可用的演员组 bootstrap 区间，点估计不能解释为确定胜出。'
    if interval[1] < 0:
        return f'完整臂取得统计上可分辨的穿透下降：配对差 {deltas["mean_positive_depth_m"]:.9g} m，95% CI {interval}。'
    return f'尚未证明穿透下降：配对差 {deltas["mean_positive_depth_m"]:.9g} m，95% CI {interval}。'


def render_report(summary, selection_records, probe_status):
    deltas = {key: value['mean_delta'] for key, value in summary['paired'].items()}
    intervals = {key: value['ci95'] for key, value in summary['paired'].items()}
    lines = ['# MultiHMR 双人联合精修最终报告', '', render_conclusion(intervals, deltas), '',
             f'- 状态分母：{summary["total_denominator"]}',
             f'- 自动选人不足两可靠候选的帧：{selection_gap(selection_records)}',
             f'- 量尺、交换、缺失候选和重复实例反向探针：{probe_status}',
             f'- 四臂与覆盖率门：{summary["gates"]}',
             f'- 接生产结论：{summary["promotion_pass"]}', '',
             '所有差值均为 joint-full − selected-init；完整四臂、分场景结果和逐帧残差见 results.json 与 metrics.csv。',
             '穿透是双向顶点—网格近似，不是三角面相交体积；样本区间功效不足时只保留实验结论。']
    return '\n'.join(lines) + '\n'


def analysis_summary(frame_rows, total):
    arms = ['selected-init', 'reproj-only', 'collision-only', 'joint-full']
    present = [arm for arm in arms if any(row['arm'] == arm for row in frame_rows)]
    summaries = {arm: {'overall': coverage([row for row in frame_rows if row['arm'] == arm]),
                       'valid_two_person_frames': sum(row['status'] == 'ok' for row in frame_rows if row['arm'] == arm),
                       'state_rows': sum(row['arm'] == arm for row in frame_rows)} for arm in present}
    full = [row for row in frame_rows if row['arm'] == 'joint-full']
    initial = [row for row in frame_rows if row['arm'] == 'selected-init']
    paired = {metric: paired_bootstrap(full, initial, metric, 'pair') for metric in METRICS}
    return {'schema': 'dual_mesh_subset_analysis_v1', 'total_denominator': total,
            'summaries': summaries, 'paired': paired, 'gates': {}, 'promotion_pass': False}


def public_row(row):
    return {key: value for key, value in row.items() if not key.startswith('_')}


def main():
    import argparse
    import csv
    import json
    from pathlib import Path
    from common import ARMS, read_json, write_json_xb
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--source-request', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    if args.split and not args.analysis_only:
        raise ValueError('Subset evaluation must use --analysis-only')
    source, source_sha = read_json(args.source_request)
    records = [row for row in source['records'] if args.split is None or row['split'] == args.split]
    selection, selection_sha = read_json(args.request.parent / 'selected.json')
    selected_by_id = {row['id']: row for row in selection['records']}
    rows = []
    for arm in args.arms:
        for record in records:
            saved = None
            status = selected_by_id[record['id']]['status']
            frame = args.run_root / arm / record['split'] / record['id']
            try:
                if (frame / 'metrics.json').is_file():
                    saved, _ = read_json(frame / 'metrics.json')
                    status = saved.get('status', 'error')
                elif status == 'ok':
                    status = 'missing'
                if status == 'ok':
                    with np.load(frame / 'geometry.npz', allow_pickle=False) as geometry:
                        row = score_gt_frame(record, geometry, status)
                else:
                    row = score_gt_frame(record, None, status)
            except Exception as exc:
                row = score_gt_frame(record, None, 'error')
                row['error'] = f'{type(exc).__name__}: {exc}'
            row['arm'] = arm
            if saved is not None:
                row['config_sha256'] = saved.get('config_sha256')
            rows.append(row)
    summary = (analysis_summary(rows, len(records)) if args.analysis_only else build_summary(rows, total=127))
    required_probes = ['sphere_signed_scale', 'reversed_winding', 'artificial_deeper_overlap',
                       'swapped_observations', 'missing_candidate', 'duplicate_candidate']
    probe_status = {name: False for name in required_probes}
    if args.probes.is_file():
        saved_probes, _ = read_json(args.probes)
        probe_status.update({name: saved_probes.get(name) is True for name in required_probes})
    incomplete = any(row['status'] in ('missing', 'error') for row in rows)
    summary['probe_status'] = probe_status
    summary['source_request_sha256'] = source_sha
    summary['selection_sha256'] = selection_sha
    if args.analysis_only:
        summary['status'] = 'incomplete' if incomplete else 'analysis_complete'
        success = not incomplete
    else:
        summary['promotion_pass'] = bool(summary['promotion_pass'] and all(probe_status.values()) and not incomplete)
        summary['status'] = 'incomplete' if incomplete else ('ok' if summary['promotion_pass'] else 'gate_failure')
        success = summary['promotion_pass']
    public = [public_row(row) for row in rows]
    result = dict(summary, rows=public)
    write_json_xb(args.output / 'results.json', result)
    write_json_xb(args.output / 'summary.json', {key: value for key, value in result.items() if key != 'rows'})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'metrics.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in public for key in row}))
        writer.writeheader(); writer.writerows(public)
    report_path = args.output / 'report/FINAL.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(render_report(summary, selection['records'], probe_status))
    print(json.dumps({'status': summary['status'], 'promotion_pass': summary['promotion_pass'],
                      'gates': summary['gates'], 'probes': probe_status}))
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
```

`probe_status` 必须来自实际测试命令保存的 JSON，而不是常量；缺探针文件时六项均为 false 且 `promotion_pass=false`。

- [ ] **Step 4: 运行报告测试并提交**

Run: `python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py -q`
Expected: `7 passed`。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/evaluate.py experiments/mete-test/dual_mesh_refine_20261103_v1/tests/test_evaluate.py
git commit -m "feat(dual-refine): report uncertainty and selection coverage gaps"
```

### Task 11: 双色网格前后对照与视觉反例

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/visualize.py`

- [ ] **Step 1: 实现固定坐标尺的 before/after 渲染和真实结果选帧**

```python
# scripts/visualize.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common import read_json


def add_mesh(axis, vertices, faces, color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    axis.add_collection3d(Poly3DCollection(vertices[faces], facecolor=color, edgecolor='none', alpha=.42))


def render_pair(before_path, after_path, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    before, after = np.load(before_path, allow_pickle=False), np.load(after_path, allow_pickle=False)
    try:
        figure = plt.figure(figsize=(12, 6))
        all_vertices = np.concatenate([geometry[f'person{person}_vertices']
                                       for geometry in (before, after) for person in range(2)])
        lower, upper = all_vertices.min(0), all_vertices.max(0)
        centre = (lower + upper) / 2
        radius = max(float((upper - lower).max()) / 2, .1)
        for column, geometry in enumerate((before, after)):
            axis = figure.add_subplot(1, 2, column + 1, projection='3d')
            for person, color in ((0, '#0072B2'), (1, '#D55E00')):
                add_mesh(axis, geometry[f'person{person}_vertices'], geometry['faces'], color)
            axis.set_xlim(centre[0] - radius, centre[0] + radius)
            axis.set_ylim(centre[1] - radius, centre[1] + radius)
            axis.set_zlim(centre[2] - radius, centre[2] + radius)
            axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=12, azim=-65)
            axis.set_title('selected-init' if column == 0 else 'joint-full')
            axis.set_xlabel('camera X (m)'); axis.set_ylabel('camera Y (m)'); axis.set_zlabel('depth Z (m)')
        figure.suptitle(title); figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=180)
        plt.close(figure)
    finally:
        before.close(); after.close()


def choose_panels(rows):
    full_ok = {row['id'] for row in rows if row['arm'] == 'joint-full' and row['status'] == 'ok'}
    chosen = []
    for category in ('separated', 'hug', 'severe'):
        candidates = [row for row in rows if row['arm'] == 'selected-init' and row['status'] == 'ok'
                      and row['category'] == category and row['id'] in full_ok
                      and row.get('mean_positive_depth_m') is not None]
        if not candidates:
            raise ValueError(f'No jointly valid visualization frame for {category}')
        chosen.append(max(candidates, key=lambda row: row['mean_positive_depth_m']))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    results, _ = read_json(args.results)
    chosen = choose_panels(results['rows'])
    destinations = [args.output_dir / f"{row['category']}_{row['id']}.png" for row in chosen]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Refusing to overwrite an existing visualization panel')
    for row, destination in zip(chosen, destinations):
        relative = Path(row['arm']) / row['split'] / row['id'] / 'geometry.npz'
        before = args.run_root / relative
        after = args.run_root / 'joint-full' / row['split'] / row['id'] / 'geometry.npz'
        render_pair(before, after, destination, f"{row['category']} {row['id']}")
        print(destination)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 在最终评测后生成 separated/hug/severe 面板**

`choose_panels` 只从两臂共同有效帧中选取 selected-init 穿透最大的样本，选择仅用于展示，不改变统计。运行：

```bash
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python scripts/visualize.py --results /data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/final/evaluation/results.json --run-root /data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/final --output-dir /data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/final/evaluation/report/panels
```

Expected: 恰好生成 separated、hug、severe 三张 PNG；缺任一类共同有效帧时明确失败，不拿其他类别替代。

- [ ] **Step 3: 打开 PNG 检查并提交脚本**

用 Read 查看三张 PNG：两人颜色不同、前后坐标范围一致、无全黑/空图、严重遮挡仍能辨认。若视觉退化，保留面板并在报告中注明，不能仅因数值通过而删除反例。PNG 留在远端产物目录，不进入 Git。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/visualize.py
git commit -m "docs(dual-refine): add dual-colour mesh comparison renderer"
```

### Task 12: 全量静态/单元验证与小样 pilot

**Files:**
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/probes.py`
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/README.md`
- Create on remote only: `/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/pilot/probes.json`

- [ ] **Step 1: 在本地运行全部 CPU 可执行测试并检查变更**

```bash
python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests -q
python -m compileall -q experiments/mete-test/dual_mesh_refine_20261103_v1/scripts
git diff --check
```

Expected: 除明确标记 `warp`/`cuda` 且本地无依赖的测试外全部通过；不得通过删除 Warp 测试获得绿灯。若本地 collection 因 Warp 缺失，分开运行 CPU 文件，并到远端执行完整套件。

- [ ] **Step 2: 核对已逐文件部署的远端副本并跑完整测试**

前面每个任务只通过 `scp` 投递当任务新增的具体文件；部署前先比较远端同名文件 SHA256，存在分歧就停止，不做整目录覆盖。然后运行：

```bash
ssh sg "test -d /home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_mesh_refine_20261103_v1 && /data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m py_compile /home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_mesh_refine_20261103_v1/scripts/*.py"
ssh sg "cd /home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_mesh_refine_20261103_v1 && /data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python -m pytest tests -q"
```

Expected: 全部通过；远端目录缺失或本地/远端 SHA256 不一致都先停止核对，不用整目录 `scp -r` 覆盖别人实时修改。

- [ ] **Step 3: 串行运行 6 个 train 帧的四臂 pilot**

对四臂依次执行，不并行：

```bash
PY=/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python
ROOT=/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/pilot
$PY scripts/refine.py --request data/request.json --arm selected-init --split train --limit 6 --out-root "$ROOT"
$PY scripts/refine.py --request data/request.json --arm reproj-only --split train --limit 6 --out-root "$ROOT"
$PY scripts/refine.py --request data/request.json --arm collision-only --split train --limit 6 --out-root "$ROOT"
$PY scripts/refine.py --request data/request.json --arm joint-full --split train --limit 6 --out-root "$ROOT"
```

Expected: 每臂 6 个请求状态全覆盖；所有有效输出 SHA256 不同于其他 arm 的 metrics，但 selected-init 的 candidate/observation 索引与其他三臂完全相同。记录显存、耗时和回退数，不用 pilot 宣称总体有效。

- [ ] **Step 4: 用真实 pytest 退出码保存六项探针并做反向验证**

```python
# scripts/probes.py
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from common import write_json_xb

TESTS = {
    'sphere_signed_scale': 'tests/test_collision.py::CollisionTests::test_sphere_signed_depth_has_analytic_scale',
    'reversed_winding': 'tests/test_collision.py::CollisionTests::test_reversed_winding_fails_orientation_probe',
    'artificial_deeper_overlap': 'tests/test_collision.py::CollisionTests::test_artificially_deeper_query_has_larger_torch_loss',
    'swapped_observations': 'tests/test_selection.py::SelectionTests::test_forced_cross_person_observation_swap_has_high_matching_cost',
    'missing_candidate': 'tests/test_selection.py::SelectionTests::test_low_score_or_single_candidate_is_insufficient',
    'duplicate_candidate': 'tests/test_selection.py::SelectionTests::test_duplicate_candidates_cannot_become_two_people',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = {}
    for name, nodeid in TESTS.items():
        completed = subprocess.run([args.python, '-m', 'pytest', nodeid, '-q'], cwd=root,
                                   text=True, capture_output=True, check=False)
        details[name] = {'nodeid': nodeid, 'returncode': completed.returncode,
                         'stdout': completed.stdout, 'stderr': completed.stderr}
    payload = {name: detail['returncode'] == 0 for name, detail in details.items()}
    payload['details'] = details
    write_json_xb(args.output, payload)
    print({name: payload[name] for name in TESTS})
    return 0 if all(payload[name] for name in TESTS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
```

运行：

```bash
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python scripts/probes.py --output /data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/pilot/probes.json
```

六项均由独立 pytest 进程的实际退出码决定。反向验证使用 `test_artificially_deeper_query_has_larger_torch_loss` 和 `test_effect_gate_must_fail_when_direction_is_reversed`：前者加深重叠必须增大损失，后者把区间方向反转必须让门失败；任何一项恒真或恒假都停止全量实验。

- [ ] **Step 5: 提交 pilot 审计说明**

README 增加一段实际审计：pilot 请求数、成功数、回退数、Warp 版本、测试命令和六项探针结果。数字直接从 `summary.json/probes.json` 生成，不手写未运行的结果。

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/README.md experiments/mete-test/dual_mesh_refine_20261103_v1/scripts/probes.py
git commit -m "test(dual-refine): document pilot and collision probe evidence"
```

### Task 13: train 筛选、val 选择、冻结 test 四臂终验

**Files:**
- Create on remote only: `data/request_lr*_cw*.json`
- Create on remote only: `screen_configs.json`
- Create on remote only: `data/frozen_request.json` 与 `frozen_config.json`
- Create on remote only: final `results.json/summary.json/metrics.csv/report/FINAL.md`

**实际执行记录（2026-11-03）：** 六配置 train screening 全部完成；候选 `lr0.001_cw1`、`lr0.003_cw5` 进入 val，`lr0.003_cw1` 因非劣效点估计排除。val 四臂分析后唯一胜出 `lr0.003_cw5`，已冻结 `data/frozen_request.json`，SHA256 `1d17f985a44b5a912b8be6e2b7cd7d81da2d71e8796f353c02e9dbb3852fbf77`；`frozen_config.json` 初始 `test_has_run=false`，test 四臂完成后才写 true。

- [ ] **Step 1: 生成六个请求并仅在 train 上筛选**

固定其余默认参数，仅搜索 `learning_rate ∈ {0.001, 0.003}` 与 `collision_weight ∈ {1.0, 5.0, 20.0}` 的笛卡尔积。在实验根目录执行：

```bash
/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python - <<'PY'
import copy, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from common import read_json, write_json_xb
base, _ = read_json('data/request.json')
manifest = []
for learning_rate in (0.001, 0.003):
    for collision_weight in (1.0, 5.0, 20.0):
        name = f'lr{learning_rate:g}_cw{collision_weight:g}'
        request = copy.deepcopy(base)
        request['optimization'].update(learning_rate=learning_rate, collision_weight=collision_weight)
        path = Path('data') / f'request_{name}.json'
        write_json_xb(path, request)
        manifest.append({'config_id': name, 'request': str(path.resolve())})
write_json_xb('screen_configs.json', {'configurations': manifest, 'candidate_ids': []})
PY
```

每个配置串行运行 `selected-init` 与 `joint-full`，再只做 train 子集分析；`--analysis-only` 不产生接生产绿灯：

```bash
set -e
PY=/data/shangguanrz/models/two-man/M4_MultiHMR2/env/bin/python
OUT=/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1
SOURCE=/home/shangguanrz/project/pic-edit/experiments/9.hi4d/mete-test/dual_pose_compare_20261029_v1/data/request.json
for lr in 0.001 0.003; do
  for cw in 1 5 20; do
    id="lr${lr}_cw${cw}"
    for arm in selected-init joint-full; do
      "$PY" scripts/refine.py --request "data/request_${id}.json" --arm "$arm" --split train --out-root "$OUT/screen/$id"
    done
    "$PY" scripts/evaluate.py --request "data/request_${id}.json" --source-request "$SOURCE" --run-root "$OUT/screen/$id" --probes "$OUT/pilot/probes.json" --output "$OUT/screen/$id/evaluation" --split train --arms selected-init joint-full --analysis-only
  done
done
```

筛选顺序：排除任何非有限、量尺失败、incomplete 或 joint-full 有效覆盖率低于 selected-init 的配置；排除 train 上 root/absolute MPJPE 或髋距误差点估计恶化超过 5 mm、PCK@0.1 下降超过 .005 的配置；余下按 train 平均正穿透深度升序取前两名，把实际 ID 写入 `screen_configs.json:candidate_ids`。不足两名时保留实际合格数并记录原因，不读取 val/test 来扩充网格。

- [ ] **Step 2: 在 val 上对候选配置跑完整四臂并冻结唯一配置**

```bash
set -e
for id in $("$PY" -c "import json; print(' '.join(json.load(open('screen_configs.json'))['candidate_ids']))"); do
  for arm in selected-init reproj-only collision-only joint-full; do
    "$PY" scripts/refine.py --request "data/request_${id}.json" --arm "$arm" --split val --out-root "$OUT/select/$id"
  done
  "$PY" scripts/evaluate.py --request "data/request_${id}.json" --source-request "$SOURCE" --run-root "$OUT/select/$id" --probes "$OUT/pilot/probes.json" --output "$OUT/select/$id/evaluation" --split val --analysis-only
done
```

实际目录为 `select/${id}/${arm}/val`。选择顺序：四臂均有 26/26 状态槽且非 incomplete；joint-full 有效覆盖率不低于 selected-init；val 非劣效点估计满足同一边界；最后取 joint-full 平均正穿透深度较低者。将胜出的完整请求复制为 `data/frozen_request.json`，并写 `frozen_config.json`：`selected_config_id`、配置 JSON、request/selection/代码/val 结果 SHA256、UTC 时间和 `test_has_run=false`。此后不得根据 test 改配置。

- [ ] **Step 3: 用冻结配置跑最终 train/val 四臂，并仅跑一次 test 四臂**

```bash
set -e
"$PY" -c "import json; assert json.load(open('frozen_config.json'))['test_has_run'] is False"
for split in train val test; do
  for arm in selected-init reproj-only collision-only joint-full; do
    "$PY" scripts/refine.py --request data/frozen_request.json --arm "$arm" --split "$split" --out-root "$OUT/final"
  done
done
"$PY" -c "import json,os; from pathlib import Path; p=Path('frozen_config.json'); d=json.loads(p.read_text()); d['test_has_run']=True; t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2)+'\n'); os.replace(t,p)"
```

实际目录为 `final/${arm}/${split}`。若中途失败，保留所有状态与错误；修复后的 test 只能标为“已接触后的调试复跑”，不能当冻结终验。

**实际冻结终验（2026-11-04）：** `lr0.003_cw5` request SHA256 `1d17f985a44b5a912b8be6e2b7cd7d81da2d71e8796f353c02e9dbb3852fbf77`。四臂状态均127、有效双人均125。joint-full−selected-init 平均正穿透 `-0.325363 mm`，10演员组 CI `[-0.439288,-0.219526] mm`；root/absolute/hip CI上界 `2.250/2.924/4.851 mm`，PCK@0.1 CI下界 `-0.003667`，六探针全真，`promotion_pass=true`。

- [ ] **Step 4: 独立评测 127 帧并计算全部门**

```bash
"$PY" scripts/evaluate.py --request data/frozen_request.json --source-request "$SOURCE" --run-root "$OUT/final" --probes "$OUT/pilot/probes.json" --output "$OUT/final/evaluation"
```

Expected: evaluator 为每臂生成 127 个状态槽；分别给 overall、train/val/test、separated/light/hug/severe 结果；配对指标仅用两臂共同有效帧且 n 明确；覆盖率仍以 127 为分母。自动选人少于 125/127 时，`selection_gap` 逐帧列出所有缺口，覆盖率门失败。

- [ ] **Step 5: 根据实际门结论收尾，不以点估计替代 CI**

只有以下全部成立才允许 `promotion_pass=true`：穿透差 CI 上界 `<0`；root/absolute MPJPE 和髋距向量误差差 CI 上界 `≤+5 mm`；PCK@0.1 差 CI 下界 `≥−.005`；joint-full 有效双人覆盖率不低于 selected-init；四臂均公开；六项反向探针通过。否则报告保留实验结论并写具体失败门。样本或演员组不足时明确写“指标功效不足”。

**完成审计（2026-11-04）：** 本地59 passed / 6 skipped，远端65 passed。`audit_final_gates.py` 证明反转 joint-full 穿透会使效果门失败、删除一个 joint-full 有效输出会使覆盖率门失败且状态分母仍为127，原始 results SHA256不变。predictions SHA `82102d931cc0654a47df770c2d04ae761ef5cda2dc80665e720269116c452971`、source request SHA `0a20bb25a301a28d7023913e706a64dd9859737446f4468720257ccaadb81e64` 与选择记录一致，127个 Sapiens2 缓存逐文件 SHA 全部一致。四臂各125份有效输出、evaluation三件套、三张面板和 `test_has_run=true` 均存在。

### Task 14: 完成审计与交付

**Files:**
- Modify: `experiments/mete-test/dual_mesh_refine_20261103_v1/README.md`
- Create: `experiments/mete-test/dual_mesh_refine_20261103_v1/report/FINAL.md`（从远端 evaluator 产物逐字节复制后审查，不手改数字）

- [ ] **Step 1: 对照设计做提示词—产物审计**

逐项核对真实文件和命令输出：127 冻结输入哈希；GT 白名单；full53/真实 K 重建；自动去重和交换反例；可见性与两处冻结；Warp kernel 与球体符号量尺；双向全顶点主指标；Torch 半隐式梯度；四臂同选人；逐帧接受/回退；train/val/test 隔离；演员组 bootstrap；五类效果/非劣效门；127 覆盖率门；三类双色面板；默认不覆盖和错误分母。任何缺文件、缺状态、恒等式探针或测试未覆盖分支都记为未完成。

- [ ] **Step 2: 做一次“反转被测变量，门必须报警”的最终验证**

在内存中的结果深拷贝上把 joint-full 的穿透值改大到高于 selected-init，重跑 `build_summary`，`mean_positive_depth_m` 门必须失败；再把副本中一个 joint-full 行的 `status` 改为 `missing` 并重算，覆盖率门必须失败且状态分母仍为 127。原始结果文件和拟合产物不参与写操作，结束后核对其 SHA256 未改变；不删除或移动任何图像、视频、网格。

- [ ] **Step 3: 重跑完整验证并检查 Git/产物边界**

```bash
python -m pytest experiments/mete-test/dual_mesh_refine_20261103_v1/tests -q
git diff --check
git status --short
```

远端再跑完整 Warp/CUDA 测试并检查 final 汇总。确认 `.pt/.safetensors/pkl/npz/png` 未进入 Git，原始 `predictions.pkl`、Sapiens2 缓存和旧报告 SHA256 与 manifest 一致。

- [ ] **Step 4: 更新 README 的最终实测结论与复现入口**

README 只从最终 `summary.json` 提取：各 split 有效帧数、四臂穿透均值、配对 CI、非劣效门、覆盖率、回退数和最终决策。不得把 pilot 数字写成 127 帧结果，也不得把“未测量”写成 0。随后逐字节带回生成的报告：

```bash
ls experiments/mete-test/dual_mesh_refine_20261103_v1
mkdir -p experiments/mete-test/dual_mesh_refine_20261103_v1/report
test ! -e experiments/mete-test/dual_mesh_refine_20261103_v1/report/FINAL.md
scp sg:/data/shangguanrz/pic-edit-experiments/mete-test/dual_mesh_refine_20261103_v1/final/evaluation/report/FINAL.md experiments/mete-test/dual_mesh_refine_20261103_v1/report/FINAL.md
```

- [ ] **Step 5: 提交最终实验并报告限制**

```bash
git add experiments/mete-test/dual_mesh_refine_20261103_v1/README.md experiments/mete-test/dual_mesh_refine_20261103_v1/scripts experiments/mete-test/dual_mesh_refine_20261103_v1/tests experiments/mete-test/dual_mesh_refine_20261103_v1/report/FINAL.md
git commit -m "docs(dual-refine): record four-arm evaluation and promotion decision"
```

最终回复必须给出实验目录、最终报告、实际运行的测试、四臂主要数字、是否通过接生产门，以及未能验证的限制。若未完成全量运行或任一门失败，明确说实验代码完成但效果/覆盖验收未通过，不能宣称方案已验证有效。
