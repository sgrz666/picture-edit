#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-latency PE probe: run PE-style prompt rewriting through llama.cpp on the tiktok runs.

Two arms, same eight workflow runs, same three input images each:

  --mode pe        official PE-I2I contract: the model's own system prompt + instruction,
                   JSON answer validated by compile_prompt().  This is the quality arm.

  --mode describe  the 4B arm.  The model is NOT PE-trained, so it is not asked to emit the
                   PE contract.  It gets a short system prompt and a short ask, and is
                   expected to return a brief action description.  The identity / clothing /
                   background preservation clauses are added by this script, mirroring
                   compile_prompt()'s preamble and suffix.

Both arms record the HF baseline tokens/seconds from each run's prompt/pe_result.json so the
three-way comparison (HF PE / GGUF PE / GGUF 4B) is on identical inputs.

Writes only under the experiment dir.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = "/home/shangguanrz/project/pic-edit"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "workflow"))

from src.pose_edit.prompt_contract import PE_SAMPLING  # noqa: E402
from pose_transfer.prompts import compile_prompt  # noqa: E402

TIKTOK = os.path.join(REPO, "workflow/runs/tiktok")

DESCRIBE_SYSTEM = (
    "You describe human body poses for an image-editing system. You are given three images: "
    "the source person, the target photograph, and a coloured skeleton guide of the target pose. "
    "Describe ONLY what the target body is doing. Be brief, concrete and literal. "
    "Never describe identity, face, hair, clothing, accessories or background - those come from "
    "the source image and are handled separately. Never mention the skeleton lines or their colours."
)

DESCRIBE_ASK = (
    "In under 90 words, describe the target's: leg stance and weight distribution; both arm "
    "positions using viewer-left / viewer-right; head orientation and gaze direction; facial "
    "expression; and anything held in the hands. Plain prose, one paragraph, no lists, no JSON."
)

# Same contract shape compile_prompt() builds for the PE arm.
PREAMBLE = (
    "Edit the single person in <image1> into the desired pose. "
    "<image1> supplies all identity, hairstyle, face, outfit, accessories and scene appearance. "
    "<image2> is only a colored skeletal pose guide; use its joint positions, never its line "
    "colors as appearance. Preserve objects already in the source. "
)
SUFFIX = (
    " Preserve the source person and their original clothing and setting. "
    "Render a natural photograph without skeleton lines, diagrams or pose-guide colors."
)


def image_part(path):
    with open(path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return {"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (mime, b64)}}


def reroot(path, run_dir):
    """A run moved under runs/tiktok keeps absolute paths recorded at its old location.

    Re-root the recorded path under the run directory instead of failing.
    """
    if os.path.exists(path):
        return path
    name = os.path.basename(run_dir.rstrip("/"))
    marker = os.sep + name + os.sep
    idx = path.find(marker)
    if idx >= 0:
        candidate = os.path.join(run_dir, path[idx + len(marker):])
        if os.path.exists(candidate):
            return candidate
    raise SystemExit("missing input image %s (run dir churned?)" % path)


def load_case(run_dir):
    ev = json.load(open(os.path.join(run_dir, "prompt", "pe_result.json"), encoding="utf-8"))
    images = [reroot(it["path"], run_dir) for it in ev["input_roles"]]
    system = open(os.path.join(run_dir, "prompt", "effective_system_prompt.txt"), encoding="utf-8").read()
    return {"name": os.path.basename(run_dir.rstrip("/")), "run_dir": run_dir,
            "images": images, "system": system, "instruction": ev["instruction"],
            "hf_tokens": ev.get("output_tokens"),
            "hf_seconds": round(ev.get("seconds", 0), 1) if ev.get("seconds") else None,
            "hf_seed": ev.get("seed")}


def wait_ready(port, proc, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(3)
    return False


def ask(port, system, instruction, images, seed, max_tokens):
    payload = {"messages": [{"role": "system", "content": system},
                            {"role": "user", "content": [{"type": "text", "text": instruction},
                                                         *[image_part(p) for p in images]]}],
               "max_tokens": max_tokens, "temperature": PE_SAMPLING["temperature"],
               "top_p": PE_SAMPLING["top_p"], "seed": seed}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        body = json.loads(resp.read().decode())
    return body, time.perf_counter() - t0


def check_description(text):
    """Cheap content check for the 4B arm: does it actually say what we need?"""
    low = text.lower()
    keys = {
        "arm": any(w in low for w in ("arm", "elbow", "forearm", "hand")),
        "head": any(w in low for w in ("head", "face", "gaze", "look", "chin")),
        "legs": any(w in low for w in ("leg", "stand", "feet", "foot", "stance", "knee")),
        "left_right": ("left" in low and "right" in low),
    }
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pe", "describe"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dirs; default = all tt_* under workflow/runs/tiktok")
    ap.add_argument("--port", type=int, default=18777)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: PE_SAMPLING for pe mode, 512 for describe mode")
    ap.add_argument("--attempts", type=int, default=1,
                    help="attempts per case; pe mode retries with a corrective suffix like pe_worker")
    ap.add_argument("--tag", default=None, help="label used in the record")
    args = ap.parse_args()

    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = PE_SAMPLING["max_new_tokens"] if args.mode == "pe" else 512

    runs = args.runs or sorted(
        os.path.join(TIKTOK, d) for d in os.listdir(TIKTOK) if d.startswith("tt_"))
    cases = [load_case(r) for r in runs]

    out_root = os.path.abspath(args.out)
    for sub in ("records", "raw", "prompts"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    log_path = os.path.join(out_root, "server.log")

    cmd = [args.server, "-m", args.model, "--mmproj", args.mmproj, "--no-mmproj-offload",
           "-ngl", "99", "-c", "32768", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.port),
           "--temp", str(PE_SAMPLING["temperature"]), "--top-p", str(PE_SAMPLING["top_p"]),
           "--top-k", str(PE_SAMPLING["top_k"]), "--no-context-shift"]
    if args.mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    print("launching server (load takes ~1 min): %s" % " ".join(cmd[-6:]), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        env = dict(os.environ, LLAMA_LLVEC_LIBRARY="")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if not wait_ready(args.port, proc):
        print(open(log_path, encoding="utf-8", errors="replace").read()[-2500:])
        raise SystemExit("server not healthy")
    print("server ready; mode=%s cases=%d max_tokens=%d" % (args.mode, len(cases), max_tokens), flush=True)

    records = []
    try:
        for case in cases:
            attempts, chosen, wall = [], None, 0.0
            for attempt in range(max(1, args.attempts)):
                seed = 42 + attempt
                if args.mode == "pe":
                    system, instruction = case["system"], case["instruction"]
                else:
                    system, instruction = DESCRIBE_SYSTEM, DESCRIBE_ASK
                body, secs = ask(args.port, system, instruction, case["images"], seed, max_tokens)
                wall += secs
                content = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                raw_path = os.path.join(out_root, "raw", "%s_attempt%d.txt" % (case["name"], attempt + 1))
                open(raw_path, "w", encoding="utf-8").write(content)
                ev = {"attempt": attempt + 1, "seed": seed, "seconds": round(secs, 1),
                      "completion_tokens": usage.get("completion_tokens"),
                      "prompt_tokens": usage.get("prompt_tokens"),
                      "raw_path": raw_path,
                      "tok_per_s": round((usage.get("completion_tokens") or 0) / secs, 2) if secs else None}
                if args.mode == "pe":
                    try:
                        parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
                        compiled = compile_prompt(parsed["rewritten_prompt"], "source_only")
                        ev.update(status="accepted", ratio_follow=parsed.get("ratio_follow"))
                        p_path = os.path.join(out_root, "prompts", "%s.txt" % case["name"])
                        open(p_path, "w", encoding="utf-8").write(compiled + "\n")
                        ev.update(compiled_path=p_path, compiled_words=len(compiled.split()),
                                  compiled_chars=len(compiled))
                        chosen = ev
                    except Exception as exc:
                        ev.update(status="rejected", error=str(exc))
                        print("  %s attempt%d REJECTED: %s" % (case["name"], attempt + 1, exc), flush=True)
                else:
                    desc = " ".join(content.split())
                    compiled = PREAMBLE + "Apply this pose: " + desc + SUFFIX
                    p_path = os.path.join(out_root, "prompts", "%s.txt" % case["name"])
                    open(p_path, "w", encoding="utf-8").write(compiled + "\n")
                    ev.update(status="accepted", description_chars=len(desc),
                              description_words=len(desc.split()),
                              compiled_chars=len(compiled),
                              content_checks=check_description(desc),
                              compiled_path=p_path)
                    chosen = ev
                attempts.append(ev)
                if chosen:
                    break

            rec = {"case": case["name"], "run_dir": case["run_dir"], "mode": args.mode,
                   "model": os.path.basename(args.model), "mtp": args.mtp,
                   "tag": args.tag or os.path.basename(args.model),
                   "status": "complete" if chosen else "failed",
                   "seconds_total": round(wall, 1), "attempts": attempts,
                   "hf_tokens": case["hf_tokens"], "hf_seconds": case["hf_seconds"],
                   "hf_seed": case["hf_seed"]}
            if chosen:
                rec.update(tokens=chosen.get("completion_tokens"),
                           prompt_tokens=chosen.get("prompt_tokens"),
                           tok_per_s=chosen.get("tok_per_s"),
                           compiled_chars=chosen.get("compiled_chars"))
            records.append(rec)
            json.dump(rec, open(os.path.join(out_root, "records", "%s.json" % case["name"]), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
            print("  %-22s %s tokens=%s %.1fs tok/s=%s" % (
                case["name"], rec["status"], rec.get("tokens"), wall,
                rec.get("tok_per_s")), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except Exception:
            proc.kill()

    ok = [r for r in records if r["status"] == "complete"]
    summary = {"schema": "lowlatency_pe_probe_v1", "mode": args.mode,
               "model": os.path.basename(args.model), "mmproj": os.path.basename(args.mmproj),
               "mtp": args.mtp, "max_tokens": max_tokens,
               "cases": len(records), "complete": len(ok), "failed": len(records) - len(ok)}
    if ok:
        secs = [r["seconds_total"] for r in ok]
        toks = [r["tokens"] for r in ok if r.get("tokens")]
        tps = [r["tok_per_s"] for r in ok if r.get("tok_per_s")]
        hf = [r["hf_seconds"] for r in ok if r.get("hf_seconds")]
        summary.update(seconds_mean=round(sum(secs) / len(secs), 1),
                       tokens_mean=round(sum(toks) / len(toks)) if toks else None,
                       tok_per_s_mean=round(sum(tps) / len(tps), 2) if tps else None,
                       hf_seconds_mean=round(sum(hf) / len(hf), 1) if hf else None,
                       speedup_vs_hf=round((sum(hf) / len(hf)) / (sum(secs) / len(secs)), 2) if hf else None)
        if args.mode == "describe":
            keys = ("arm", "head", "legs", "left_right")
            summary["content_check_pass"] = {
                k: sum(1 for r in ok for a in r["attempts"]
                       if (a.get("content_checks") or {}).get(k)) for k in keys}
    json.dump(summary, open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
