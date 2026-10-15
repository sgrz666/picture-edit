import os,subprocess
R="/home/shangguanrz/project/pic-edit/experiments"
def sh(c):
    return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
for d in ["qwen21_tiktok_v3_smoke","qwen21_hi4d_dual","hi4d_overfit_archive","qwen-2.1-test","qwen_edit_results","16-test","qwen21_dual_v8"]:
    print("\n===",d,"===")
    print("size:",sh(f"du -sh --si {R}/{d} 2>/dev/null"))
    print("subtree files:",sh(f"find {R}/{d} -type f | wc -l"))
    print("png in subtree:",sh(f"find {R}/{d} -type f -iname '*.png' | wc -l"))
# smoke subdirs body/face detail
T=R+"/qwen21_tiktok_v3_smoke"
print("\nface ckpt sizes:",sh(f"ls -la {T}/face/*.pt"))
print("body metrics tail:",sh(f"tail -n1 {T}/body/train_metrics.jsonl"))
print("face metrics tail:",sh(f"tail -n1 {T}/face/train_metrics.jsonl"))
print("face_features count:",sh(f"ls {T}/face_features/*.pt | wc -l"))
print("qwen21_cache count:",sh(f"ls {T}/qwen21_cache/*.pt | wc -l"))
# 16-test sizes
print("\n16-test source/conditions/stage2_cache:")
for s in ["source","conditions_ab","stage2_cache"]:
    print(" ",s,sh(f"ls {R}/16-test/{s}"))
print("\nqwen_edit_results:",sh(f"ls -la {R}/qwen_edit_results"))
print("hi4d_overfit_archive cond cache:",sh(f"ls {R}/hi4d_overfit_archive/hug_1000steps/condition_cache"))
