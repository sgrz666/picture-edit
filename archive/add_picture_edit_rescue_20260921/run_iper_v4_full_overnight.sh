#!/usr/bin/env bash
set -euo pipefail

BASE=/home/shangguanrz/project/pic-edit
ADD="$BASE/add"
SMPL="$ADD/tools/SMPLest-X"
V4="$ADD/data/iper_sampled_v4_6src64tgt"

export PATH="$ADD/envs/smplestx_bw/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "============================================================"
echo "iPER V4 FULL OVERNIGHT PIPELINE"
echo "start: $(date)"
echo "host : $(hostname)"
echo "============================================================"


###############################################################################
# A. 先对已经存在的 001_1 做完整 end-to-end smoke
###############################################################################

echo
echo "============================================================"
echo "E2E SMOKE: 001_1"
echo "============================================================"

python \
  "$SMPL/scripts/run_iper_v4_smplestx.py" \
  --appearance 001_1

PYOPENGL_PLATFORM=osmesa \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python \
  "$SMPL/scripts/build_iper_v4_geometry.py" \
  --appearance 001_1

test -f \
  "$V4/001_1/smplestx/_SMPLX_DONE"

test -f \
  "$V4/001_1/geometry/source_beta/_GEOMETRY_DONE"

N=$(find \
  "$V4/001_1/geometry/source_beta/target/normal_512_ndr_exact" \
  -type f -name "*.png" \
  | wc -l)

test "$N" -eq 64

echo
echo "001_1 END-TO-END: PASS"


###############################################################################
# B. Stage 1：103 appearance 全量 V4 sampling
#    已完成的前三组自动跳过
###############################################################################

echo
echo "============================================================"
echo "STAGE 1: FULL V4 SAMPLING"
echo "============================================================"

python \
  "$SMPL/scripts/build_iper_sampled_v4_6src64tgt.py" \
  --raw-root "$ADD/data/iper_raw_1024" \
  --out-root "$V4" \
  --source-candidates 64 \
  --motion-candidates 256 \
  --standard-sources 3 \
  --arbitrary-sources 3 \
  --targets 64


###############################################################################
# C. 严格检查 Stage 1
###############################################################################

python - <<'PY'
from pathlib import Path
import json

root = Path(
    "/home/shangguanrz/project/pic-edit/"
    "add/data/iper_sampled_v4_6src64tgt"
)

s = json.loads(
    (
        root
        / "summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

expected = {
    "appearance_count": 103,
    "source_frames": 618,
    "target_frames": 6592,
    "unique_frames": 7210,
    "source_target_pairs": 39552,
}

for k, v in expected.items():
    assert s.get(k) == v, (
        k,
        s.get(k),
        v,
    )

assert not s.get(
    "failed_in_this_run",
    []
)

done = len(
    list(
        root.glob(
            "*/_STAGE1_DONE"
        )
    )
)

assert done == 103, done

(
    root
    / "FULL_STAGE1_PASS"
).write_text(
    "PASS\n",
    encoding="utf-8",
)

print(
    "FULL STAGE1: PASS"
)
PY


###############################################################################
# D. 所有 103 appearance -> SMPLest-X params
###############################################################################

echo
echo "============================================================"
echo "STAGE 2A: SMPLest-X"
echo "============================================================"

python \
  "$SMPL/scripts/run_iper_v4_smplestx.py" \
  --all


###############################################################################
# E. 只有 SMPLest-X 103/103 才进入 geometry
###############################################################################

python - <<'PY'
from pathlib import Path
import json

root = Path(
    "/home/shangguanrz/project/pic-edit/"
    "add/data/iper_sampled_v4_6src64tgt"
)

p = (
    root
    / "smplestx_stage2_summary.json"
)

s = json.loads(
    p.read_text(
        encoding="utf-8"
    )
)

assert (
    s["smplx_done"]
    == 103
), s

assert not s[
    "failures"
], s

print(
    "SMPLest-X 103/103: PASS"
)
PY


###############################################################################
# F. source beta + target theta
#    -> depth / mask / exact normal / 3D joints
###############################################################################

echo
echo "============================================================"
echo "STAGE 2B: DENSE GEOMETRY"
echo "============================================================"

PYOPENGL_PLATFORM=osmesa \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python \
  "$SMPL/scripts/build_iper_v4_geometry.py" \
  --all


###############################################################################
# G. FINAL AUDIT
###############################################################################

echo
echo "============================================================"
echo "FINAL AUDIT"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import json
import sys

root = Path(
    "/home/shangguanrz/project/pic-edit/"
    "add/data/iper_sampled_v4_6src64tgt"
)

def count(pattern):
    return len(
        list(
            root.glob(pattern)
        )
    )

checks = {
    "stage1_done":
        count(
            "*/_STAGE1_DONE"
        ),

    "smplx_done":
        count(
            "*/smplestx/"
            "_SMPLX_DONE"
        ),

    "geometry_done":
        count(
            "*/geometry/source_beta/"
            "_GEOMETRY_DONE"
        ),

    "stable_beta":
        count(
            "*/geometry/source_beta/"
            "stable_beta_standard3_median.npy"
        ),

    "target_vertices":
        count(
            "*/geometry/source_beta/"
            "target_vertices_sourcebeta.npy"
        ),

    "target_joints3d":
        count(
            "*/geometry/source_beta/"
            "target_joints3d_sourcebeta.npy"
        ),

    "depth_1024":
        count(
            "*/geometry/source_beta/"
            "target/depth_1024/*.npy"
        ),

    "depth_512":
        count(
            "*/geometry/source_beta/"
            "target/depth_512/*.npy"
        ),

    "mask_1024":
        count(
            "*/geometry/source_beta/"
            "target/mask_1024/*.png"
        ),

    "mask_512":
        count(
            "*/geometry/source_beta/"
            "target/mask_512/*.png"
        ),

    "normal_1024":
        count(
            "*/geometry/source_beta/"
            "target/"
            "normal_1024_ndr_exact/*.png"
        ),

    "normal_512":
        count(
            "*/geometry/source_beta/"
            "target/"
            "normal_512_ndr_exact/*.png"
        ),
}

print(
    json.dumps(
        checks,
        indent=2,
    )
)

expected103 = [
    "stage1_done",
    "smplx_done",
    "geometry_done",
    "stable_beta",
    "target_vertices",
    "target_joints3d",
]

expected6592 = [
    "depth_1024",
    "depth_512",
    "mask_1024",
    "mask_512",
    "normal_1024",
    "normal_512",
]

errors = []

for k in expected103:
    if checks[k] != 103:
        errors.append(
            f"{k}: "
            f"{checks[k]} != 103"
        )

for k in expected6592:
    if checks[k] != 6592:
        errors.append(
            f"{k}: "
            f"{checks[k]} != 6592"
        )

geo_summary = json.loads(
    (
        root
        / "geometry_stage2_summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

if (
    geo_summary[
        "geometry_done"
    ]
    != 103
):
    errors.append(
        "geometry_done "
        "summary != 103"
    )

if geo_summary[
    "failures"
]:
    errors.append(
        str(
            geo_summary[
                "failures"
            ]
        )
    )

if errors:
    print()
    print(
        "FINAL PIPELINE: FAIL"
    )

    for e in errors:
        print(
            "FAIL:",
            e,
        )

    sys.exit(1)

(
    root
    / "FULL_PIPELINE_PASS"
).write_text(
    "PASS\n",
    encoding="utf-8",
)

print()
print(
    "====================================="
)
print(
    "FULL PIPELINE: PASS"
)
print(
    "103 appearances"
)
print(
    "618 source RGB"
)
print(
    "6592 target RGB / DWPose"
)
print(
    "7210 selected unique frames"
)
print(
    "6592 source-beta / target-theta "
    "dense geometry assets"
)
print(
    "39552 potential training pairs"
)
print(
    "====================================="
)
PY

echo
du -sh "$V4"
echo
echo "finish: $(date)"
