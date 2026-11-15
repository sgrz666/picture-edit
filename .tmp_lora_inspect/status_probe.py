"""Read-only status analysis for the bilibili-dwpose-edit-lora full run.

Nothing here writes into the experiment tree. It aggregates:
  1. validation loss trend (fixed 10 samples x 3 densities, independent seed)
  2. training loss trend controlled for flow-matching timestep bucket
  3. one-epoch coverage / duplicate check on pair_id
  4. resource boundaries and violation flags
  5. paired baseline-vs-LoRA per-image deltas with a sign test
"""
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict

ROOT = "/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora"
TRAIN = os.path.join(ROOT, "output/train_full_20261111_v1")
REPORT = os.path.join(ROOT, "report/full_report_20261111_v1")


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def jl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


print("=" * 78)
print("1. VALIDATION LOSS TREND")
print("=" * 78)
vfiles = sorted(
    f for f in os.listdir(TRAIN)
    if f.startswith("validation_") and f.endswith(".json")
)
vrows = []
for f in vfiles:
    d = load(os.path.join(TRAIN, f))
    recs = d["records"]
    by_den = defaultdict(list)
    for r in recs:
        den = r.get("density", r.get("timestep_density"))
        by_den[den].append(r["loss"])
    vrows.append((d["step"], d["mean_loss"], d.get("training_rng_preserved"),
                  {k: st.mean(v) for k, v in sorted(by_den.items(), key=lambda x: str(x[0]))},
                  len(recs)))
print("step   mean_loss   n   rng_preserved   per-density means")
for step, mean, rng, dens, n in vrows:
    ds = "  ".join("%s:%.5f" % (k, v) for k, v in dens.items())
    print("%5d  %.6f  %3d  %-14s  %s" % (step, mean, n, rng, ds))

if len(vrows) >= 2:
    first, last = vrows[0], vrows[-1]
    print("\nfirst->last mean_loss: %.6f -> %.6f  (delta %+.6f, %+.2f%%)"
          % (first[1], last[1], last[1] - first[1],
             100.0 * (last[1] - first[1]) / first[1]))
    best = min(vrows, key=lambda r: r[1])
    print("lowest validation mean_loss: step %d = %.6f" % (best[0], best[1]))
    # per-density first vs last, since pooling densities hides direction
    print("\nper-density first vs last:")
    for k in first[3]:
        if k in last[3]:
            print("  density %s: %.6f -> %.6f (%+.6f)"
                  % (k, first[3][k], last[3][k], last[3][k] - first[3][k]))
    # per-sample paired first vs last
    d0 = load(os.path.join(TRAIN, vfiles[0]))["records"]
    d1 = load(os.path.join(TRAIN, vfiles[-1]))["records"]

    def key(r):
        return (r.get("pair_id", r.get("id")), r.get("density", r.get("timestep_density")))

    m0 = {key(r): r["loss"] for r in d0}
    m1 = {key(r): r["loss"] for r in d1}
    common = sorted(set(m0) & set(m1), key=str)
    deltas = [m1[k] - m0[k] for k in common]
    down = sum(1 for x in deltas if x < 0)
    print("\nper-(sample,density) paired: n=%d, improved=%d, worsened=%d, mean delta %+.6f"
          % (len(deltas), down, len(deltas) - down, st.mean(deltas) if deltas else float("nan")))
    print("record keys sample:", sorted(d0[0].keys()))

print()
print("=" * 78)
print("2. TRAINING LOSS, CONTROLLED FOR TIMESTEP")
print("=" * 78)
ups = jl(os.path.join(TRAIN, "updates_000000_009520.jsonl"))
print("logged update records: %d (log cadence every 10 steps)" % len(ups))
print("record keys:", sorted(ups[0].keys()))
steps = [u["step"] for u in ups]
print("step range: %d .. %d ; monotonic=%s ; duplicates=%d"
      % (min(steps), max(steps), steps == sorted(steps), len(steps) - len(set(steps))))

# bucket by timestep decile so sigma sampling noise does not masquerade as learning
def bucket(u):
    return min(9, int(u["timestep"] // 100))

early = [u for u in ups if u["step"] <= 1000]
late = [u for u in ups if u["step"] > 8520]
print("\nwindow A = steps 1..1000 (n=%d), window B = steps 8521..9520 (n=%d)"
      % (len(early), len(late)))
print("timestep-bucket   meanA     meanB     delta      nA   nB")
tot_a, tot_b, wsum = [], [], []
for b in range(10):
    a = [u["loss"] for u in early if bucket(u) == b]
    bb = [u["loss"] for u in late if bucket(u) == b]
    if a and bb:
        ma, mb = st.mean(a), st.mean(bb)
        print("  %3d-%3d        %.5f   %.5f   %+.5f   %4d %4d"
              % (b * 100, b * 100 + 99, ma, mb, mb - ma, len(a), len(bb)))
        w = min(len(a), len(bb))
        tot_a.append(ma * w)
        tot_b.append(mb * w)
        wsum.append(w)
    else:
        print("  %3d-%3d        nA=%d nB=%d (insufficient)" % (b * 100, b * 100 + 99, len(a), len(bb)))
if wsum:
    sa, sb = sum(tot_a) / sum(wsum), sum(tot_b) / sum(wsum)
    print("\nbucket-weighted (equal-weight per shared bucket) A=%.5f B=%.5f delta=%+.5f (%+.2f%%)"
          % (sa, sb, sb - sa, 100.0 * (sb - sa) / sa))
print("unweighted pooled mean A=%.5f B=%.5f  <-- confounded by sigma mix"
      % (st.mean([u["loss"] for u in early]), st.mean([u["loss"] for u in late])))
print("mean timestep A=%.1f B=%.1f (shows the confound)"
      % (st.mean([u["timestep"] for u in early]), st.mean([u["timestep"] for u in late])))

# 1000-step block means, both raw and bucket-adjusted
print("\n1000-step blocks: raw mean / bucket-adjusted mean (fixed reference bucket weights)")
ref_w = Counter(bucket(u) for u in ups)
tot = sum(ref_w.values())
ref_w = {b: ref_w[b] / tot for b in ref_w}
for lo in range(0, 9520, 1000):
    hi = min(lo + 1000, 9520)
    blk = [u for u in ups if lo < u["step"] <= hi]
    if not blk:
        continue
    per = defaultdict(list)
    for u in blk:
        per[bucket(u)].append(u["loss"])
    adj, wacc = 0.0, 0.0
    for b, w in ref_w.items():
        if per.get(b):
            adj += w * st.mean(per[b])
            wacc += w
    print("  %5d-%5d  raw=%.5f  adj=%.5f  (bucket coverage %.2f, n=%d)"
          % (lo + 1, hi, st.mean([u["loss"] for u in blk]),
             adj / wacc if wacc else float("nan"), wacc, len(blk)))

print("\ngradient_norm: min=%.5f median=%.5f max=%.5f ; zero-grad steps=%d ; non-finite=%d"
      % (min(u["gradient_norm"] for u in ups),
         st.median([u["gradient_norm"] for u in ups]),
         max(u["gradient_norm"] for u in ups),
         sum(1 for u in ups if u["gradient_norm"] == 0),
         sum(1 for u in ups if not math.isfinite(u["loss"]) or not math.isfinite(u["gradient_norm"]))))
print("lr values seen:", sorted(set(u["lr"] for u in ups)))
print("step_seconds: median=%.3f mean=%.3f max=%.3f"
      % (st.median([u["step_seconds"] for u in ups]),
         st.mean([u["step_seconds"] for u in ups]),
         max(u["step_seconds"] for u in ups)))
tok = Counter((u["condition_tokens"], u["target_tokens"], u["image_slots"]) for u in ups)
print("token shapes (condition,target,image_slots):", dict(tok))

print()
print("=" * 78)
print("3. ONE-EPOCH COVERAGE")
print("=" * 78)
pid = [u["pair_id"] for u in ups]
c = Counter(pid)
print("distinct pair_id in logged records: %d of %d logged" % (len(c), len(pid)))
print("max repeats of a single pair_id: %d" % max(c.values()))
mani = os.path.join(ROOT, "data/data_20261111_v1/train.jsonl")
if os.path.exists(mani):
    m = jl(mani)
    print("train.jsonl entries: %d" % len(m))
    print("manifest keys:", sorted(m[0].keys()))
    mk = set(x.get("pair_id") for x in m)
    print("manifest distinct pair_id: %d" % len(mk))
    print("logged pair_ids all in manifest: %s" % set(pid).issubset(mk))
    print("manifest pairs never seen in log: %d (log is 1/10 sampled, so ~90%% expected unseen)"
          % len(mk - set(pid)))
else:
    print("train.jsonl NOT FOUND at", mani)

print()
print("=" * 78)
print("4. RESOURCES")
print("=" * 78)
rp = os.path.join(TRAIN, "resources_000000_009520.json")
r = load(rp)
print("top-level keys:", sorted(r.keys()) if isinstance(r, dict) else type(r))
recs = r["records"] if isinstance(r, dict) else r
if isinstance(r, dict):
    for k in ("pid", "samples", "violation", "error"):
        if k in r:
            print("  %s = %s" % (k, r[k]))
print("resource samples: %d" % len(recs))
for field, agg in (("gpu_free_mib", min), ("ram_available_mib", min),
                   ("gpu_used_mib", max), ("rss_mib", max),
                   ("gpu_utilization", st.mean)):
    vals = [x[field] for x in recs if field in x and x[field] is not None]
    if vals:
        print("  %-20s %s = %.1f" % (field, agg.__name__, agg(vals)))
print("cuda_allocated_mib in update log: min=%.1f max=%.1f"
      % (min(u["cuda_allocated_mib"] for u in ups), max(u["cuda_allocated_mib"] for u in ups)))
zero_util = sum(1 for x in recs if x.get("gpu_utilization") == 0)
print("samples with gpu_utilization==0: %d/%d" % (zero_util, len(recs)))

print()
print("=" * 78)
print("5. PAIRED BASELINE vs LORA (per image)")
print("=" * 78)
pi = jl(os.path.join(REPORT, "per_image.jsonl"))
print("per_image records: %d ; keys: %s" % (len(pi), sorted(pi[0].keys())))
idx = {}
for r in pi:
    k = (r.get("split"), r.get("intervention"), r.get("name", r.get("id", r.get("case"))))
    idx.setdefault(k, {})[r.get("method")] = r
print("distinct case keys: %d" % len(idx))
sample = pi[0]
print("example record:", json.dumps(sample, ensure_ascii=False)[:600])

metrics = [("pck_002_diagonal", "PCK@.02", +1), ("pck_005_diagonal", "PCK@.05", +1),
           ("pck_010_diagonal", "PCK@.10", +1), ("nme_diagonal", "NME", -1),
           ("identity_a_cosine", "FaceID", +1)]
for split, interv in (("val", "normal"), ("train", "normal")):
    cases = [v for k, v in idx.items() if k[0] == split and k[1] == interv
             and "baseline" in v and "lora" in v]
    if not cases:
        continue
    print("\n--- split=%s intervention=%s  paired n=%d ---" % (split, interv, len(cases)))
    for field, label, sign in metrics:
        ds = []
        for v in cases:
            a, b = v["baseline"].get(field), v["lora"].get(field)
            if a is not None and b is not None:
                ds.append(b - a)
        if not ds:
            print("  %-8s no paired values (field absent)" % label)
            continue
        win = sum(1 for x in ds if x * sign > 0)
        loss_ = sum(1 for x in ds if x * sign < 0)
        tie = len(ds) - win - loss_
        # exact two-sided sign test on non-tie pairs
        n = win + loss_
        if n:
            k = min(win, loss_)
            tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
            p = min(1.0, 2 * tail)
        else:
            p = float("nan")
        print("  %-8s base=%.5f lora=%.5f  mean delta %+.5f | better=%d worse=%d tie=%d | sign-test p=%.3f"
              % (label,
                 st.mean([v["baseline"][field] for v in cases]),
                 st.mean([v["lora"][field] for v in cases]),
                 st.mean(ds), win, loss_, tie, p))

print("\n--- diagnostics (n=1 each, NOT powered) ---")
for interv in ("swap_pose", "blank_pose"):
    for meth in ("baseline", "lora"):
        rr = [r for r in pi if r.get("intervention") == interv and r.get("method") == meth]
        for r in rr:
            print("  %-10s %-8s PCK@.02=%s PCK@.05=%s NME=%s FaceID=%s"
                  % (interv, meth,
                     r.get("pck_002_diagonal"), r.get("pck_005_diagonal"),
                     r.get("nme_diagonal"), r.get("identity_a_cosine")))
print("\nDONE")