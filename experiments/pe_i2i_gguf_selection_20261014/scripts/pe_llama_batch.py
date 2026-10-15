#!/usr/bin/env python3
"""Gate 1: run PE-I2I through llama.cpp on real workflow cases, with retry parity.

Cases are read out of existing run directories, so the three images, the effective system
prompt and the instruction are byte-identical to what the HF stage used. The two-attempt
retry and the corrective suffix mirror workflow/pe_worker.py, so "contract pass rate" means
the same thing on both arms.

Writes only under the experiment dir -- nothing here touches workflow/runs.
"""
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.pose_edit.prompt_contract import PE_SAMPLING, parse_pe_answer  # noqa: E402
from pose_transfer.prompts import compile_prompt  # noqa: E402

RETRY_SUFFIX = ("\nYour previous answer was rejected: {error}. Write the action in explicit words "
                "without any reference to image2. Keep the official JSON format and source ratio contract.")


def image_part(path):
    with open(path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    return {"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % ("jpeg" if ext in ("jpg", "jpeg") else ext, b64)}}


def load_case(run_dir):
    ev = json.load(open(os.path.join(run_dir, "prompt", "pe_result.json"), encoding="utf-8"))
    images = [it["path"] for it in ev["input_roles"]]
    for p in images:
        if not os.path.exists(p):
            raise SystemExit("missing input image %s (run dir churned?)" % p)
    system = open(os.path.join(run_dir, "prompt", "effective_system_prompt.txt"), encoding="utf-8").read()
    return {"name": os.path.basename(run_dir.rstrip("/")), "images": images, "system": system,
            "instruction": ev["instruction"], "hf_tokens": ev.get("output_tokens"),
            "hf_seconds": round(ev.get("seconds", 0), 1), "hf_prompt_sha": ev.get("input_roles", [{}])[0].get("sha256")}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="workflow run dirs to take cases from")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--libdir", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18777)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=PE_SAMPLING["max_new_tokens"])
    args = ap.parse_args()

    cases = [load_case(r) for r in args.runs]
    out_root = os.path.abspath(args.out)
    for sub in ("records", "raw", "prompts"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    log_path = os.path.join(out_root, "server.log")

    env = dict(os.environ)
    if args.libdir:
        env["LD_LIBRARY_PATH"] = ":".join(args.libdir) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [args.server, "-m", args.model, "--mmproj", args.mmproj, "--no-mmproj-offload",
           "-ngl", "99", "-c", "32768", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.port),
           "--temp", str(PE_SAMPLING["temperature"]), "--top-p", str(PE_SAMPLING["top_p"]),
           "--top-k", str(PE_SAMPLING["top_k"]), "--no-context-shift"]
    if args.mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    print("launching server (load takes ~1 min)", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if not wait_ready(args.port, proc):
        print(open(log_path, encoding="utf-8", errors="replace").read()[-2500:])
        raise SystemExit("server not healthy")
    print("server ready", flush=True)

    records = []
    try:
        for case in cases:
            attempts, status, chosen = [], None, None
            wall = 0.0
            for attempt in range(2):
                seed = 42 + attempt
                instruction = case["instruction"] if not attempts else (
                    case["instruction"] + RETRY_SUFFIX.format(error=attempts[-1]["error"]))
                body, secs = ask(args.port, case["system"], instruction, case["images"], seed, args.max_tokens)
                wall += secs
                content = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                raw_path = os.path.join(out_root, "raw", "%s_attempt%d.txt" % (case["name"], attempt + 1))
                open(raw_path, "w", encoding="utf-8").write(content)
                ev = {"attempt": attempt + 1, "seed": seed, "seconds": round(secs, 1),
                      "completion_tokens": usage.get("completion_tokens"),
                      "prompt_tokens": usage.get("prompt_tokens"),
                      "raw_path": raw_path, "raw_sha256": None}
                try:
                    parsed = parse_pe_answer(content)
                    compiled = compile_prompt(parsed["rewritten_prompt"], "source_only")
                    ev["status"] = "accepted"
                    p_path = os.path.join(out_root, "prompts", "%s.txt" % case["name"])
                    open(p_path, "w", encoding="utf-8").write(compiled + "\n")
                    ev["compiled_path"] = p_path
                    ev["compiled_words"] = len(compiled.split())
                    ev["ratio_follow"] = parsed.get("ratio_follow")
                    status, chosen = parsed, ev
                except ValueError as exc:
                    ev["status"] = "rejected"
                    ev["error"] = str(exc)
                    print("  %s attempt%d REJECTED: %s" % (case["name"], attempt + 1, exc), flush=True)
                attempts.append(ev)
                if status:
                    break
            rec = {"schema": "pe_llama_gate1_v1", "case": case["name"], "mtp": args.mtp,
                   "model": os.path.basename(args.model), "status": "complete" if status else "failed",
                   "seconds_total": round(wall, 1), "attempts": attempts,
                   "tokens": chosen["completion_tokens"] if chosen else None,
                   "prompt_tokens": chosen["prompt_tokens"] if chosen else attempts[-1].get("prompt_tokens"),
                   "hf_tokens": case["hf_tokens"], "hf_seconds": case["hf_seconds"],
                   "compiled_words": chosen.get("compiled_words") if chosen else None,
                   "hf_prompt_sha": case["hf_prompt_sha"]}
            if chosen:
                rec["tok_per_s"] = round(chosen["completion_tokens"] / chosen["seconds"], 2)
            records.append(rec)
            json.dump(rec, open(os.path.join(out_root, "records", "%s.json" % case["name"]), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
            print(json.dumps({k: rec[k] for k in ("case", "status", "tokens", "prompt_tokens",
                                                  "tok_per_s", "compiled_words")}, ensure_ascii=False), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except Exception:
            proc.kill()

    ok = [r for r in records if r["status"] == "complete"]
    secs = [r["seconds_total"] for r in records]
    stats = {"out": out_root, "records": len(records), "complete": len(ok),
             "failed": len(records) - len(ok), "mtp": args.mtp,
             "model": os.path.basename(args.model),
             "seconds_mean": round(float(np.mean(secs)), 1) if secs else None,
             "tokens_mean": int(np.mean([r["tokens"] for r in ok])) if ok else None,
             "hf_tokens_mean": int(np.mean([r["hf_tokens"] for r in ok])) if ok else None,
             "hf_seconds_mean": round(float(np.mean([r["hf_seconds"] for r in ok])), 1) if ok else None,
             "tok_per_s_mean": round(float(np.mean([r.get("tok_per_s", 0) for r in ok])), 2) if ok else None,
             "retried_pairs": sum(1 for r in records if len(r["attempts"]) > 1),
             "hours_per_1000_pairs": round(float(np.mean(secs)) * 1000 / 3600, 1) if secs else None}
    json.dump(stats, open(os.path.join(out_root, "stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
