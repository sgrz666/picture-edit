#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-check load_case() on one tiktok run, then run the 4B arm on a single case."""
import os
import sys

REPO = "/home/shangguanrz/project/pic-edit"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "workflow"))
import lowlatency_probe as L  # noqa: E402

runs = sorted(d for d in os.listdir(L.TIKTOK) if d.startswith("tt_"))
print("tiktok runs found:", len(runs))
for name in runs[:3]:
    case = L.load_case(os.path.join(L.TIKTOK, name))
    print("--- %s" % name)
    print("    images=%d  hf_tokens=%s  hf_seconds=%s  hf_seed=%s" % (
        len(case["images"]), case["hf_tokens"], case["hf_seconds"], case["hf_seed"]))
    for p in case["images"]:
        print("      %s  (%d B)" % (p, os.path.getsize(p)))
    print("    instruction chars=%d  system chars=%d" % (
        len(case["instruction"]), len(case["system"])))
