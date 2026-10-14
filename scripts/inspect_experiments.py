import os
import glob
import json
from pathlib import Path

exp_root = "/home/shangguanrz/project/pic-edit/experiments"
entries = sorted(os.listdir(exp_root))

results = []

for entry in entries:
    full_path = os.path.join(exp_root, entry)
    if not os.path.isdir(full_path) and not os.path.islink(full_path):
        continue
    
    is_symlink = os.path.islink(full_path)
    real_path = os.path.realpath(full_path) if is_symlink else full_path
    
    info = {
        "name": entry,
        "is_symlink": is_symlink,
        "real_path": real_path,
        "subdirs": [],
        "checkpoints": [],
        "configs": [],
        "logs": [],
        "metrics": [],
        "image_count": 0,
        "first_log_sample": "",
        "config_sample": {},
        "metrics_sample": {}
    }
    
    try:
        sub_items = os.listdir(real_path)
    except Exception as e:
        info["error"] = str(e)
        results.append(info)
        continue
        
    for item in sub_items:
        p = os.path.join(real_path, item)
        if os.path.isdir(p):
            info["subdirs"].append(item)
        elif item.endswith(('.pt', '.pth', '.safetensors', '.bin', '.ckpt')):
            info["checkpoints"].append(item)
        elif item.endswith(('.json', '.yaml', '.yml', '.toml')):
            if any(k in item.lower() for k in ['metric', 'eval', 'result', 'score', 'summary', 'report', 'acceptance']):
                info["metrics"].append(item)
            else:
                info["configs"].append(item)
        elif item.endswith(('.log', '.txt')) or 'log' in item.lower():
            info["logs"].append(item)
        elif item.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            info["image_count"] += 1

    # Look deeper into subdirs
    for subdir in info["subdirs"]:
        sub_p = os.path.join(real_path, subdir)
        try:
            for root, dirs, files in os.walk(sub_p):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), real_path)
                    if f.endswith(('.pt', '.pth', '.safetensors', '.bin', '.ckpt')):
                        info["checkpoints"].append(rel)
                    elif f.endswith('.json') and any(k in f.lower() for k in ['metric', 'eval', 'result', 'score', 'summary', 'report', 'acceptance']):
                        info["metrics"].append(rel)
                    elif f.endswith(('.json', '.yaml', '.yml')) and any(k in f.lower() for k in ['config', 'args', 'param']):
                        info["configs"].append(rel)
                    elif f.endswith('.log'):
                        info["logs"].append(rel)
                    elif f.endswith(('.png', '.jpg', '.jpeg')):
                        info["image_count"] += 1
                if len(info["checkpoints"]) > 10:
                    break
        except Exception:
            pass

    # Read config if available
    for cf in info["configs"][:3]:
        cf_path = os.path.join(real_path, cf)
        if os.path.exists(cf_path) and cf_path.endswith('.json'):
            try:
                with open(cf_path, 'r', encoding='utf-8') as f:
                    cdata = json.load(f)
                    info["config_sample"][cf] = cdata
            except Exception:
                pass

    # Read metrics if available
    for mf in info["metrics"][:4]:
        mf_path = os.path.join(real_path, mf)
        if os.path.exists(mf_path) and mf_path.endswith('.json'):
            try:
                with open(mf_path, 'r', encoding='utf-8') as f:
                    mdata = json.load(f)
                    info["metrics_sample"][mf] = mdata
            except Exception:
                pass

    # Read log snippet
    for lf in info["logs"][:3]:
        lf_path = os.path.join(real_path, lf)
        if os.path.exists(lf_path) and os.path.isfile(lf_path):
            try:
                with open(lf_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    head = lines[:10]
                    tail = lines[-15:] if len(lines) > 15 else []
                    info["first_log_sample"] = f"[{lf} HEAD ({len(lines)} lines)]:\n" + "".join(head) + "\n[TAIL]:\n" + "".join(tail)
            except Exception:
                pass
            break

    results.append(info)

out_file = "/home/shangguanrz/project/pic-edit/experiments_audit.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Audit completed. Total experiments audited: {len(results)}. Output saved to {out_file}")
