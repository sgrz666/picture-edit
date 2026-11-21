import json
import os
import glob

def read_json(p):
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return f"Error: {e}"
    return None

def print_section(title, data):
    print("=" * 60)
    print(f"=== {title} ===")
    if isinstance(data, dict):
        for k, v in list(data.items())[:15]:
            print(f"  {k}: {str(v)[:160]}")
    elif isinstance(data, list):
        print(f"  List of {len(data)} items:")
        for item in data[:5]:
            print(f"    {str(item)[:160]}")
    else:
        print(f"  {data}")

# 1. qwen21_iper_body
d_four = read_json("/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body/four_arm_eval_50_s40/summary.json")
print("=" * 60)
print("=== qwen21_iper_body four_arm_eval_50_s40 keys & summary ===")
if isinstance(d_four, dict):
    if "metrics_mean" in d_four:
        print("metrics_mean:", json.dumps(d_four["metrics_mean"], indent=2))
    elif "mean" in d_four:
        print("mean:", json.dumps(d_four["mean"], indent=2))
    for k in ["summary", "averages", "arms"]:
        if k in d_four:
            print(k, d_four[k])

d_hug = read_json("/home/shangguanrz/project/pic-edit/experiments/qwen21_hi4d_dual/single_hug_eval/summary.json")
print("=" * 60)
print("=== qwen21_hi4d_dual single_hug_eval mean ===")
if isinstance(d_hug, dict):
    if "mean" in d_hug:
        print("mean:", json.dumps(d_hug["mean"], indent=2))


d_overfit = read_json("/home/shangguanrz/project/pic-edit/experiments/tiktok_overfit_sanity_full/summary.json")
print("=" * 60)
print("=== tiktok_overfit_sanity_full summary ===")
if isinstance(d_overfit, dict):
    for k, v in d_overfit.items():
        if k != "per_sample":
            print(f"  {k}: {v}")


# 2. tiktok_overfit_sanity_full
print_section("tiktok_overfit_sanity_full summary", 
              read_json("/home/shangguanrz/project/pic-edit/experiments/tiktok_overfit_sanity_full/summary.json"))

# 3. tiktok_00001_condition_repair
print_section("tiktok_00001_condition_repair train_pose_part_1500", 
              read_json("/home/shangguanrz/project/pic-edit/experiments/tiktok_00001_condition_repair/train_pose_part_1500/metrics.json"))

# 4. iper_v6_native_20k_v1
print_section("iper_v6_native_20k_v1 metrics_summary", 
              read_json("/home/shangguanrz/project/pic-edit/experiments/iper_v6_native_20k_v1/three_row_showcase/metrics_summary.json"))

# 5. qwen21_hi4d_dual
print_section("qwen21_hi4d_dual single_hug_eval", 
              read_json("/home/shangguanrz/project/pic-edit/experiments/qwen21_hi4d_dual/single_hug_eval/summary.json"))

# 6. 16-test
print("=" * 60)
print("=== 16-test detailed files ===")
p_16 = "/home/shangguanrz/project/pic-edit/experiments/16-test"
if os.path.exists(p_16):
    for root, dirs, files in os.walk(p_16):
        js = [f for f in files if f.endswith(".json")]
        if js:
            for j in js:
                jp = os.path.join(root, j)
                print(f"  [{os.path.relpath(jp, p_16)}]: {str(read_json(jp))[:180]}")

# 7. iper_v65_face_overfit_024_6
print_section("iper_v65_face_overfit_024_6 metrics", 
              read_json("/home/shangguanrz/project/pic-edit/experiments/iper_v65_face_overfit_024_6/metrics.json"))

# 8. tiktok_objective_eval
print_section("tiktok_objective_eval", 
              read_json("/home/shangguanrz/project/pic-edit/experiments/tiktok_objective_eval/objective_eval.json"))
