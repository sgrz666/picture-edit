import re, os
BASE = "/home/shangguanrz/project/pic-edit/scripts"
targets = [
 "inspect_deep_metrics.py","inspect_experiments.py","parse_audit.py",
 "audit/audit_tiktok_v3_blur_sample.py","audit/audit_tiktok_v3_final_pairs.py",
 "audit/audit_tiktok_v3_pose_redundancy.py","audit/audit_tiktok_v3_strict_final.py",
 "audit/audit_tiktok_v3_visual_quality_v2.py","audit/audit_tiktok_v3_visual_quality_v3.py",
 "data/build_tiktok_v3_pairs.py","data/build_tiktok_v3_pairs_strict.py",
 "data/build_tiktok_v3_strict_visual_pool.py","data/build_tiktok_v3_usable_pool.py",
 "data/create_debug_samples.py","data/fill_tiktok_v3_missing_geometry.py",
 "data/finalize_tiktok_v3_visual_quality_v1.py","data/inspect_hand_inputs.py",
 "data/run_tiktok_v3_dwpose.py","data/sample_tiktok_v3_dwpose_qc.py",
 "data/sample_tiktok_v3_geometry_rejects.py","data/split_tiktok_v3_groupwise.py",
 "data/test_control_signals.py","run/run_demo_pose_edit.py",
 "reports/visualize_3dpw_20_pairs_assets_diverse.py","reports/visualize_tiktok_v3_final_global_blur.py",
 "train/train_adapter.py",
]
for t in targets:
    p = os.path.join(BASE, t)
    src = open(p, encoding="utf-8", errors="replace").read().splitlines()
    print("=====", t)
    head = [l for l in src[:24] if l.strip()]
    print(" | ".join(head[:8])[:400])
    hits = re.findall(r'["\']((?:[\w/.-]*?)(?:experiments|results|TikTok_v3|reportdoc|output|TikTok_3d|iper_)[^"\']{0,90})["\']', "\n".join(src))
    uniq = []
    for h in hits:
        if h not in uniq: uniq.append(h)
    print("PATHS:", uniq[:10])
