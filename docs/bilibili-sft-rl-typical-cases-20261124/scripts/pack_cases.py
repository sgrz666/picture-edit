import csv, json, os, tarfile

R = "/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1"
rows = list(csv.DictReader(open(os.path.join(R, "per_pair_metrics.csv"), encoding="utf-8-sig")))
by_no = {r["case_no"]: r for r in rows}

CASES = ["5", "15", "19", "20", "31", "36", "37"]

def f(x):
    try:
        return float(x)
    except Exception:
        return None

meta = []
out = "/tmp/k5_cases.tar.gz"
os.makedirs("/tmp/k5_cases", exist_ok=True)
with tarfile.open(out, "w:gz") as tf:
    for no in CASES:
        r = by_no[no]
        pid = r["pair_id"]
        rels = [
            f"context/{pid}/A.png",
            f"context/{pid}/pose.png",
            f"context/{pid}/GT.png",
            f"images/0/{pid}.png",
            f"images/200/{pid}.png",
            f"images/300/{pid}.png",
        ]
        for rel in rels:
            src = os.path.join(R, rel)
            if not os.path.exists(src):
                print("MISSING", rel)
                continue
            tf.add(src, arcname=f"k5_cases/{rel}")
        meta.append({
            "case_no": int(no),
            "clip_id": r["clip_id"],
            "pair_id": pid,
            "category": r["category"],
            "id_applicable": r["id_applicable"],
            "pose_sft": f(r["pose_sft"]),
            "pose_rl200": f(r["pose_rl200"]),
            "delta_pose_200": f(r["delta_pose_200"]),
            "pose_rl300": f(r["pose_rl300"]),
            "delta_pose_300": f(r["delta_pose_300"]),
            "pck002_sft": f(r["pck002_sft"]),
            "pck002_rl200": f(r["pck002_rl200"]),
            "delta_pck002_200": f(r["delta_pck002_200"]),
            "pck005_sft": f(r["pck005_sft"]),
            "pck005_rl200": f(r["pck005_rl200"]),
            "glintr_sft": f(r["glintr_sft"]),
            "glintr_rl200": f(r["glintr_rl200"]),
            "delta_glintr_200": f(r["delta_glintr_200"]),
            "buffalo_sft": f(r["buffalo_sft"]),
            "buffalo_rl200": f(r["buffalo_rl200"]),
            "nme_sft": f(r["nme_sft"]),
            "nme_rl200": f(r["nme_rl200"]),
        })

with open("/tmp/k5_cases/meta.json", "w", encoding="utf8") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=1)

print("packed", out, os.path.getsize(out))
for m in meta:
    print(m["case_no"], m["clip_id"], m["category"], "dPose", m["delta_pose_200"], "dGlintr", m["delta_glintr_200"])
