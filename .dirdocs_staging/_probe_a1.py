import json, os, glob, csv, statistics
V='/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2'
print('=== EXPERIMENT_CONCLUSION.md ===')
print(open(os.path.join(V,'EXPERIMENT_CONCLUSION.md'),errors='replace').read()[:6000])
