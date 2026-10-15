#!/usr/bin/env python3
"""Gate 0: does llama.cpp run PE-I2I on this box, at what speed, with what contract rate?

Starts llama-server with the MTP GGUF + mmproj, sends ONE real workflow PE request
(A_reference + B_rgb + B_3d_dwpose + the run's own effective system prompt + instruction),
then reports: GPU/offload status from the server log, timing, tokens, tok/s,
and the verdict from the repo's existing contract validators.

Nothing here writes into workflow/runs; outputs land under the experiment dir.
"""
import argparse
import base64
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = "/home/shangguanrz/project/pic-edit"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "workflow"))
from src.pose_edit.prompt_contract import (PE_SAMPLING, parse_pe_answer,  # noqa: E402
                                           validate_generator_prompt)
from pose_transfer.prompts import compile_prompt  # noqa: E402

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def data_uri(path):
    with open(path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode()
    return "data:%s;base64,%s" % (MIME.get(os.path.splitext(path)[1], "image/png"), b64)


def find_case(run_dir):
    """Pull the three images + system prompt + instruction recorded by a real PE stage."""
    ev_path = os.path.join(run_dir, "prompt", "pe_result.json")
    ev = json.load(open(ev_path, encoding="utf-8"))
    images = [it["path"] for it in ev["input_roles"]]
    missing = [p for p in images if not os.path.exists(p)]
    if missing:
        raise SystemExit("case images missing on disk (runs/ is shared and gets edited): %s" % missing)
    system = open(os.path.join(run_dir, "prompt", "effective_system_prompt.txt"), encoding="utf-8").read()
    return images, system, ev["instruction"], ev


def free_gib():
    out = subprocess.run([sys.executable, "-c",
                          "import torch;p=torch.cuda.mem_get_info();print(p[0]/2**30, p[1]/2**30)"],
                         capture_output=True, text=True)
    parts = out.stdout.split()
    return (float(parts[0]), float(parts[1])) if len(parts) == 2 else (None, None)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--run", default=os.path.join(REPO, "workflow/runs/wp1_accept_hard02"))
    ap.add_argument("--server", required=True, help="path to llama-server binary")
    ap.add_argument("--port", type=int, default=18777)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--mtp", action="store_true", help="enable --spec-type draft-mtp")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)) + "/../output")
    ap.add_argument("--dry-run", action="store_true", help="resolve the case and exit; launches nothing")
    args = ap.parse_args()

    images, system, instruction, ev = find_case(args.run)
    print("case:", args.run)
    print("  images:", [os.path.relpath(p, REPO) for p in images])
    print("  system_prompt chars:", len(system), " instruction chars:", len(instruction))
    print("  recorded BF16 baseline: output_tokens=%s seconds=%.1f" % (ev.get("output_tokens"), ev.get("seconds", 0)))
    free, total = free_gib()
    print("  cuda free/total GiB: %s / %s" % (free, total))
    if args.dry_run:
        return

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_path = os.path.join(out_dir, "server_%s.log" % stamp)

    cmd = [args.server, "-m", args.model, "--mmproj", args.mmproj,
           "--no-mmproj-offload", "-ngl", str(args.ngl), "-c", str(args.ctx),
           "-fa", "on", "--host", "127.0.0.1", "--port", str(args.port),
           "--temp", str(PE_SAMPLING["temperature"]), "--top-p", str(PE_SAMPLING["top_p"]),
           "--top-k", str(PE_SAMPLING["top_k"]), "--no-context-shift"]
    if args.mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    print("launch:", " ".join(cmd[:6]), "...", "(+%d flags)" % (len(cmd) - 6))

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, LLAMA_LLVEC_LIBRARY=""))
        ready = wait_ready(args.port, proc)
        log.flush()
        text = open(log_path, encoding="utf-8", errors="replace").read()
        gpu = re.findall(r"(CUDA[0-9]?|CUDA0|offloaded (\d+/\d+)|system memory|model buffer size|load time|slot print|sm_[0-9]+|compute capability[^\n]*)", text)
        print("server ready:", ready, " exit:", proc.poll())
        print("gpu lines:", gpu[:12] or "NONE (check log)")
        if not ready:
            print(text[-3000:])
            proc.terminate()
            raise SystemExit("server did not become healthy; log at %s" % log_path)

        payload = {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                *[{"type": "image_url", "image_url": {"url": data_uri(p)}} for p in images]]}],
            "max_tokens": args.max_tokens, "temperature": PE_SAMPLING["temperature"],
            "top_p": PE_SAMPLING["top_p"], "seed": 42}
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % args.port,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=7200) as resp:
            body = json.loads(resp.read().decode())
        secs = time.perf_counter() - t0

    proc.terminate()
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()

    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    tok = usage.get("completion_tokens") or 0
    raw = os.path.join(out_dir, "answer_%s.txt" % stamp)
    open(raw, "w", encoding="utf-8").write(content)
    result = {"case": args.run, "model": os.path.basename(args.model), "mtp": args.mtp,
              "seconds": round(secs, 1), "completion_tokens": tok,
              "prompt_tokens": usage.get("prompt_tokens"), "tok_per_s": round(tok / secs, 2) if secs else None,
              "answer_words": len(content.split()), "raw_path": raw}
    try:
        parsed = parse_pe_answer(content)
        compiled = compile_prompt(parsed["rewritten_prompt"], "source_only")
        result["contract"] = "accepted"
        result["ratio_follow"] = parsed.get("ratio_follow")
        result["wh_ratio"] = parsed.get("wh_ratio")
        result["compiled_words"] = len(compiled.split())
        result["forbidden_image2"] = "<image2>" in compiled
    except ValueError as exc:
        result["contract"] = "rejected: %s" % exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    manifest = os.path.join(out_dir, "gate0_%s.json" % stamp)
    json.dump(result, open(manifest, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("->", manifest)


if __name__ == "__main__":
    main()
