#!/usr/bin/env bash
set -euo pipefail

BASE=/home/shangguanrz/project/pic-edit
ADD="$BASE/add"
SMPL="$ADD/tools/SMPLest-X"
V4="$ADD/data/iper_sampled_v4_6src64tgt"

ENV="$ADD/envs/smplestx_bw"
PY="$ENV/bin/python"

export PATH="$ENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1

echo "============================================================"
echo "iPER V4 MASTER DATASET PIPELINE"
echo "start  : $(date)"
echo "host   : $(hostname)"
echo "python : $PY"
echo "============================================================"

test -x "$PY" || {
    echo "FAIL: Python missing: $PY"
    exit 1
}

###############################################################################
# STAGE 1
# 103 appearance
# 3 standard source + 3 arbitrary source + 64 target
# DWPose / keypoints / RGB / manifest
###############################################################################

echo
echo "============================================================"
echo "STAGE 1: RGB SELECTION + DWPOSE"
echo "============================================================"

"$PY" \
  "$SMPL/scripts/build_iper_sampled_v4_6src64tgt.py" \
  --raw-root "$ADD/data/iper_raw_1024" \
  --out-root "$V4" \
  --source-candidates 64 \
  --motion-candidates 256 \
  --standard-sources 3 \
  --arbitrary-sources 3 \
  --targets 64


###############################################################################
# STAGE 1 AUDIT
###############################################################################

"$PY" - <<'PY'
from pathlib import Path
import json

root = Path(
    "/home/shangguanrz/project/pic-edit/"
    "add/data/iper_sampled_v4_6src64tgt"
)

summary_path = root / "summary.json"

assert summary_path.exists(), (
    f"missing {summary_path}"
)

s = json.loads(
    summary_path.read_text(
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

for k, expected_value in expected.items():
    actual = s.get(k)

    assert actual == expected_value, (
        f"{k}: "
        f"{actual} != "
        f"{expected_value}"
    )

assert not s.get(
    "failed_in_this_run",
    []
), s

done = len(
    list(
        root.glob(
            "*/_STAGE1_DONE"
        )
    )
)

assert done == 103, (
    f"_STAGE1_DONE: "
    f"{done}/103"
)

print(
    "STAGE 1: 103/103 PASS"
)
PY


###############################################################################
# STAGE 2
# SMPLest-X ONLY
#
# 注意：
# 这里只恢复参数。
# 不生成 Depth / Mask / Normal。
###############################################################################

echo
echo "============================================================"
echo "STAGE 2: SMPLest-X MASTER PARAMETERS"
echo "============================================================"

"$PY" \
  "$SMPL/scripts/run_iper_v4_smplestx.py" \
  --all


###############################################################################
# SMPLest-X AUDIT
###############################################################################

"$PY" - <<'PY'
from pathlib import Path
import json

root = Path(
    "/home/shangguanrz/project/pic-edit/"
    "add/data/iper_sampled_v4_6src64tgt"
)

summary_path = (
    root
    / "smplestx_stage2_summary.json"
)

assert summary_path.exists(), (
    f"missing {summary_path}"
)

s = json.loads(
    summary_path.read_text(
        encoding="utf-8"
    )
)

assert (
    s.get("appearance_total")
    == 103
), s

assert (
    s.get("smplx_done")
    == 103
), s

assert not s.get(
    "failures",
    []
), s

print(
    "SMPLest-X: 103/103 PASS"
)
PY


###############################################################################
# FINAL MASTER AUDIT
#
# 不检查任何 Depth / Mask / Normal。
###############################################################################

echo
echo "============================================================"
echo "FINAL MASTER AUDIT"
echo "============================================================"

"$PY" - <<'PY'
from pathlib import Path
import json
import os

root = Path(
    "/home/shangguanrz/project/pic-edit/"
    "add/data/iper_sampled_v4_6src64tgt"
)

IMG_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def nimg(path):
    if not path.is_dir():
        return -1

    return sum(
        p.is_file()
        and p.suffix.lower()
        in IMG_EXT
        for p in path.iterdir()
    )


apps = sorted(
    p
    for p in root.iterdir()
    if (
        p.is_dir()
        and (
            p
            / "manifest.json"
        ).exists()
    )
)

assert len(apps) == 103, (
    f"appearance count: "
    f"{len(apps)} != 103"
)

records = []
errors = []

for app in apps:

    rgb = (
        app
        / "rgb_1024"
    )

    dwp = (
        app
        / "dwpose_512"
    )

    smpl = (
        app
        / "smplestx"
    )

    checks = {
        "rgb_standard":
            nimg(
                rgb
                / "source_standard"
            ),

        "rgb_motion":
            nimg(
                rgb
                / "source_motion"
            ),

        "rgb_target":
            nimg(
                rgb
                / "target"
            ),

        "dwpose_standard":
            nimg(
                dwp
                / "source_standard"
            ),

        "dwpose_motion":
            nimg(
                dwp
                / "source_motion"
            ),

        "dwpose_target":
            nimg(
                dwp
                / "target"
            ),
    }

    expected = {
        "rgb_standard": 3,
        "rgb_motion": 3,
        "rgb_target": 64,
        "dwpose_standard": 3,
        "dwpose_motion": 3,
        "dwpose_target": 64,
    }

    for k, exp in expected.items():
        if checks[k] != exp:
            errors.append(
                f"{app.name}: "
                f"{k}="
                f"{checks[k]}, "
                f"expected {exp}"
            )

    params = (
        smpl
        / "params.pt"
    )

    frame_map = (
        smpl
        / "frame_map.json"
    )

    smpl_done = (
        smpl
        / "_SMPLX_DONE"
    )

    if (
        not params.exists()
        or params.stat().st_size == 0
    ):
        errors.append(
            f"{app.name}: "
            "missing smplestx/params.pt"
        )

    if not frame_map.exists():
        errors.append(
            f"{app.name}: "
            "missing frame_map.json"
        )

    else:
        fm = json.loads(
            frame_map.read_text(
                encoding="utf-8"
            )
        )

        frames = fm.get(
            "frames",
            []
        )

        if len(frames) != 70:
            errors.append(
                f"{app.name}: "
                f"frame_map={len(frames)} "
                "expected 70"
            )

    if not smpl_done.exists():
        errors.append(
            f"{app.name}: "
            "missing _SMPLX_DONE"
        )

    records.append({
        "appearance":
            app.name,

        **checks,

        "smplestx_params":
            str(
                params.relative_to(
                    root
                )
            ),

        "frame_map":
            str(
                frame_map.relative_to(
                    root
                )
            ),
    })


summary = {
    "version":
        "iper-v4-master-v1",

    "description":
        (
            "Resolution-independent "
            "Master dataset. "
            "Dense geometry is generated "
            "later as experiment cache."
        ),

    "appearance_count":
        103,

    "source_count":
        618,

    "target_count":
        6592,

    "selected_unique_frames":
        7210,

    "potential_pairs":
        39552,

    "rgb_master_resolution":
        1024,

    "dwpose_render_resolution":
        512,

    "master_contains": [
        "RGB 1024",
        "DWPose keypoints / render",
        "sampling manifest",
        "candidate cache",
        "SMPLest-X per-frame parameters",
        "camera parameters",
        "frame mapping",
    ],

    "dense_geometry_policy":
        (
            "Depth / Mask / Exact Normal "
            "are NOT permanent Master assets. "
            "They are pre-rendered later "
            "for a requested experiment "
            "resolution."
        ),

    "records":
        records,

    "errors":
        errors,
}

(
    root
    / "MASTER_INDEX.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

if errors:

    print(
        "\nMASTER AUDIT FAILED"
    )

    for e in errors[:100]:
        print(
            "FAIL:",
            e
        )

    raise SystemExit(1)


(
    root
    / "FULL_MASTER_PASS"
).write_text(
    "PASS\n",
    encoding="utf-8",
)

print()
print(
    "===================================="
)
print(
    "IPER V4 MASTER: PASS"
)
print(
    "103 appearances"
)
print(
    "618 source RGB @1024"
)
print(
    "6592 target RGB @1024"
)
print(
    "7210 selected frames"
)
print(
    "103 SMPLest-X parameter sequences"
)
print(
    "NO dense geometry generated"
)
print(
    "===================================="
)
PY


echo
echo "============================================================"
echo "MASTER PIPELINE FINISHED"
echo "finish: $(date)"
echo "============================================================"

du -sh "$V4"
