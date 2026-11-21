import json
import os

with open("/home/shangguanrz/project/pic-edit/experiments_audit.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    name = item["name"]
    is_link = item["is_symlink"]
    print(f"==================================================")
    print(f"EXP: {name} (is_symlink: {is_link})")
    print(f"Path: {item['real_path']}")
    print(f"Subdirs ({len(item['subdirs'])}): {item['subdirs'][:8]}")
    print(f"Checkpoints ({len(item['checkpoints'])}): {item['checkpoints'][:8]}")
    print(f"Configs ({len(item['configs'])}): {item['configs'][:6]}")
    print(f"Metrics ({len(item['metrics'])}): {item['metrics'][:6]}")
    print(f"Logs ({len(item['logs'])}): {item['logs'][:4]}")
    print(f"Image count: {item['image_count']}")
    
    if item["config_sample"]:
        print("  --- Config Samples ---")
        for k, v in item["config_sample"].items():
            print(f"    [{k}]: {str(v)[:250]}")

    if item["metrics_sample"]:
        print("  --- Metric Samples ---")
        for k, v in item["metrics_sample"].items():
            print(f"    [{k}]: {str(v)[:250]}")

    if item["first_log_sample"]:
        print("  --- Log Snippet ---")
        lines = item["first_log_sample"].splitlines()
        for l in lines[:10]:
            print("   ", l[:120])
        if len(lines) > 10:
            print("    ...")
            for l in lines[-5:]:
                print("   ", l[:120])
    print()
