#!/usr/bin/env python3
"""Read GGUF headers by HTTP range request: no full download.

Reports general.* metadata (arch, chat template, thinking flags, vision hooks) and the
per-tensor dtype histogram grouped by module, which is how we see what a K-quant actually
left in high precision (attention? MLP? vision?).
"""
import collections
import io
import json
import re
import struct
import subprocess
import sys
from urllib.parse import quote

# ggml tensor dtype enum (ggml.h): F32=0 F16=1 Q4_0=2 Q4_1=3 Q5_0=6 Q5_1=7 Q8_0=8 Q8_1=9
# Q2_K=10 Q3_K=11 Q4_K=12 Q5_K=13 Q6_K=14 Q8_K=15 IQ2_XXS=16 IQ2_XS=17 IQ3_XXS=18 IQ1_S=19
# IQ4_NL=20 IQ3_S=21 IQ2_S=22 IQ4_XS=23 I8=24 I16=25 I32=26 I64=27 F64=28 IQ1_M=29 BF16=30
DTYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}
GB = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I32": 4, "I8": 1, "U8": 1, "I16": 2, "I64": 8}
# bytes per element = block_bytes / 32 for the K-quants used here
BLOCK_BYTES = {
    "Q2_K": 84 / 32, "Q3_K": 110 / 32, "Q4_0": 18 / 32, "Q4_1": 20 / 32,
    "Q4_K": 144 / 32, "Q4_K_S": 144 / 32, "Q4_K_M": 144 / 32, "IQ4_NL": 34 / 32,
    "IQ4_XS": 272 / 64, "Q5_0": 40 / 32, "Q5_1": 40 / 32, "Q5_K": 176 / 32,
    "Q5_K_S": 176 / 32, "Q5_K_M": 176 / 32, "Q6_K": 210 / 32, "Q8_0": 33 / 32,
    "IQ2_XXS": 106 / 256, "IQ2_XS": 131 / 256, "IQ3_XXS": 152 / 256,
}


def fetch_range(url, start, end, attempts=4):
    last = None
    for i in range(attempts):
        out = subprocess.run(["curl", "-sSL", "--retry", "2", "--max-time", "420",
                              "-r", f"{start}-{end}", url], capture_output=True)
        if out.returncode == 0 and out.stdout:
            return out.stdout
        last = f"rc={out.returncode} {out.stderr[:160]!r}"
        print(f"  retry {i + 1} after {last}", flush=True)
        import time
        time.sleep(8)
    raise RuntimeError(f"curl failed after {attempts}: {last}")


def rstr(buf, pos):
    (n,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    s = buf[pos:pos + n].decode("utf-8", "replace")
    return s, pos + n


def rvalue(buf, pos, t):
    if t == 8:  # string
        return rstr(buf, pos)
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
    if t in fmt:
        (v,) = struct.unpack_from(fmt[t], buf, pos)
        return v, pos + struct.calcsize(fmt[t])
    if t == 9:  # array: must walk every element, even when only a prefix is kept
        (atype,) = struct.unpack_from("<I", buf, pos)
        (cnt,) = struct.unpack_from("<Q", buf, pos + 4)
        pos += 12
        vals = []
        for i in range(cnt):
            v, pos = rvalue(buf, pos, atype)
            if i < 64:
                vals.append(v)
        if cnt > 64:
            vals.append(f"...({cnt} elements total)")
        return vals, pos
    raise ValueError(f"unknown GGUF type {t} at {pos}")


def parse_header(buf, label):
    if buf[:4] != b"GGUF":
        print(f"!! {label}: not GGUF (got {buf[:4]!r}); likely truncated or redirect page")
        return None
    (version,) = struct.unpack_from("<I", buf, 4)
    (n_tensors,) = struct.unpack_from("<Q", buf, 8)
    (n_kv,) = struct.unpack_from("<Q", buf, 16)
    print(f"\n=== {label}: GGUF v{version} tensors={n_tensors} kv={n_kv} header_buf={len(buf)/2**20:.1f} MiB")
    pos = 24
    kv = {}
    for _ in range(n_kv):
        try:
            key, pos = rstr(buf, pos)
            (t,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            val, pos = rvalue(buf, pos, t)
            kv[key] = val
        except Exception as exc:
            print(f"  KV parse stopped at {pos}: {exc}")
            break
    interesting = [k for k in kv if re.match(r"^(general\.|tokenizer\.(chat_template|.ggml_.*|"
                                             r"add_bos|bos_id|eos_id|eot_id|gpt2|pre|byte_fallback|model)|"
                                             r"vision|clip\.|att_?ention|block_count|embedding_length|"
                                             r"feed_forward_length|expert_.*|linear_.*|ssm_.*|rope|arch)", k)
                   or "think" in k.lower() or "image" in k.lower()]
    for k in sorted(interesting):
        v = kv[k]
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        if len(s) > 240:
            s = s[:240].replace("\n", "\\n") + f"...[{len(s)} chars]"
        print("  %-42s %s" % (k, s))
    # tensor info
    dt_hist = collections.Counter()
    grp_hist = collections.defaultdict(collections.Counter)
    approx_bytes = collections.Counter()
    per_tensor = {}
    got_tensors = 0
    for _ in range(n_tensors):
        try:
            name, pos = rstr(buf, pos)
            (nd,) = struct.unpack_from("<I", buf, pos)
            dims = list(struct.unpack_from("<" + "Q" * nd, buf, pos + 4))
            pos += 4 + 8 * nd
            (dt,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            (off,) = struct.unpack_from("<Q", buf, pos)
            pos += 8
        except Exception:
            break
        got_tensors += 1
        dtn = DTYPES.get(dt, f"type{dt}")
        dt_hist[dtn] += 1
        per_tensor[name] = dtn
        low = name.lower()
        # llama.cpp tensor naming: blk.N.{attn_*,ffn_*,ssm_*,mtp.*}, token_embd.*, output.*, v.* for vision
        if low.startswith("v."):
            g = "VISION(mmproj)"
        elif "mtp" in low:
            g = "MTP"
        elif "ssm_" in low:
            g = "LINEAR_ATTN(ssm_*)"
        elif "attn_gate" in low:
            g = "ATTN_GATE"
        elif "attn_" in low:
            g = "FULL_ATTN(attn_*)"
        elif "ffn_" in low:
            g = "MLP(ffn_*)"
        elif "output" in low:
            g = "LM_HEAD(output)"
        elif "token_embd" in low:
            g = "EMBED"
        else:
            g = "OTHER"
        grp_hist[g][dtn] += 1
        n_el = 1
        for d in dims:
            n_el *= d
        if dtn in GB:
            approx_bytes[g] += n_el * GB[dtn]
        elif dtn in BLOCK_BYTES:
            approx_bytes[g] += n_el * BLOCK_BYTES[dtn]
    print(f"  parsed tensors: {got_tensors}/{n_tensors}")
    ct = kv.get("tokenizer.chat_template")
    if ct:
        print("  chat_template: %d chars; mentions think=%s add_bos=%s eos_id=%s vocab=%s" % (
            len(ct), ("think" in ct), kv.get("tokenizer.ggml.add_bos_token"),
            kv.get("tokenizer.ggml.eos_token_id"), kv.get("tokenizer.ggml.model")))
    else:
        print("  chat_template: ABSENT in KV seen (parser may have stopped early)")
    sens = [n for n in per_tensor if any(s in n for s in
            ("ssm_out", "ssm_alpha", "ssm_beta", "attn_gate", "ssm_conv1d", "mtp"))]
    print("  sensitive/mtp tensor dtypes (%d shown of %d):" % (min(len(sens), 14), len(sens)))
    for n in sens[:14]:
        print("     %-56s %s" % (n, per_tensor[n]))

    print("  dtype histogram:", dict(dt_hist.most_common()))
    for g, c in sorted(grp_hist.items()):
        print("   %-12s %s  ~%.2f GiB" % (g, dict(c.most_common()), approx_bytes[g] / 2**30))
    return kv


def main():
    repo = sys.argv[1]
    files = sys.argv[2:]
    nbytes = 128 * 2**20
    for fn in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{quote(fn)}"
        buf = fetch_range(url, 0, nbytes - 1)
        parse_header(buf, fn)


if __name__ == "__main__":
    main()
