#!/usr/bin/env python3
"""Analyze merge conflicts between local main (de606d0b) and sg line (2ebd026).

Parses `git merge-tree --write-tree` output, classifies each conflicted path,
and computes per-side last-touch timestamps to drive the resolution policy:
  - content/add-add conflicts -> newer side wins
  - modify/delete -> keep the modified side
"""
import subprocess
import sys
from collections import defaultdict

OURS = "de606d0b019b718cbe65cd52b5558b978816db88"   # local main / GitHub main
THEIRS = "2ebd026b8568737d2f0e13c4ebea6a0634c54fff"  # sg main + today sync


def run(args):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def unquote(path):
    path = path.strip()
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        inner = path[1:-1]
        # git C-style quoting: decode common escapes
        out = []
        i = 0
        while i < len(inner):
            c = inner[i]
            if c == "\\" and i + 1 < len(inner):
                nxt = inner[i + 1]
                if nxt == "n":
                    out.append("\n"); i += 2; continue
                if nxt == "t":
                    out.append("\t"); i += 2; continue
                if nxt == "\\":
                    out.append("\\"); i += 2; continue
                if nxt == '"':
                    out.append('"'); i += 2; continue
                # octal escapes like \346\234\252 (utf-8 bytes)
                if nxt.isdigit():
                    oct_digits = inner[i + 1:i + 4]
                    try:
                        out.append(bytes(int(oct_digits[j:j + 3], 8) for j in range(0, len(oct_digits), 3) if oct_digits[j:j + 3]).decode("utf-8", errors="replace") if False else chr(int(oct_digits, 8)))
                        i += 4
                        continue
                    except ValueError:
                        pass
                out.append(nxt); i += 2; continue
            out.append(c); i += 1
        return "".join(out)
    return path


def last_touch_map(commit):
    """path -> (timestamp, short_hash) of the newest commit on `commit` touching path."""
    out = run(["git", "log", "--format=%x01%H %ct", "--name-only", commit])
    entries = out.split("\x01")
    result = {}
    for e in entries:
        lines = [l for l in e.split("\n")]
        if not lines or not lines[0].strip():
            continue
        parts = lines[0].split()
        if len(parts) < 2:
            continue
        h, ts = parts[0], int(parts[1])
        for l in lines[1:]:
            p = l.strip()
            if p and p not in result:  # first seen = newest
                result[p] = (ts, h[:7])
    return result


def main():
    mt = run(["git", "merge-tree", "--write-tree", OURS, THEIRS]).splitlines()
    stages = defaultdict(dict)  # path -> {stage: object}
    for line in mt:
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        mode, obj, stage = parts[0], parts[1], int(parts[2])
        stages[unquote(path)][stage] = obj

    ours_map = last_touch_map(OURS)
    theirs_map = last_touch_map(THEIRS)

    import datetime

    def fmt(ts):
        return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "never"

    stats = defaultdict(int)
    rows = []
    for path in sorted(stages):
        st = stages[path]
        has1, has2, has3 = 1 in st, 2 in st, 3 in st
        if has2 and has3:
            kind = "add/add" if not has1 else "content"
            o_ts, o_h = ours_map.get(path, (0, "-"))
            t_ts, t_h = theirs_map.get(path, (0, "-"))
            if o_ts >= t_ts:
                winner, w_ts, w_h = "OURS(local)", o_ts, o_h
            else:
                winner, w_ts, w_h = "THEIRS(sg)", t_ts, t_h
            rows.append((kind, path, f"ours {fmt(o_ts)} {o_h} vs theirs {fmt(t_ts)} {t_h}", winner))
        elif has3 and not has2:
            kind = "mod/del(ours deleted)"
            t_ts, t_h = theirs_map.get(path, (0, "-"))
            rows.append((kind, path, f"theirs {fmt(t_ts)} {t_h}", "THEIRS(sg) keep modified"))
        elif has2 and not has3:
            kind = "mod/del(theirs deleted)"
            o_ts, o_h = ours_map.get(path, (0, "-"))
            rows.append((kind, path, f"ours {fmt(o_ts)} {o_h}", "OURS(local) keep modified"))
        else:
            kind = f"stages{sorted(st.keys())}"
            rows.append((kind, path, "-", "NEEDS MANUAL"))
        stats[kind if not (has2 and has3) else kind] += 1

    print("=== CONFLICT TYPE STATS ===")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"{v:5d}  {k}")
    print()
    side_count = defaultdict(int)
    for kind, path, detail, winner in rows:
        side_count[winner] += 1
    print("=== WINNER SIDE DISTRIBUTION ===")
    for k, v in sorted(side_count.items(), key=lambda x: -x[1]):
        print(f"{v:5d}  {k}")
    print()
    print("=== FULL RESOLUTION TABLE ===")
    for kind, path, detail, winner in rows:
        print(f"{winner:28s} | {kind:24s} | {path}")
        print(f"{'':28s} | {'':24s} |   {detail}")


if __name__ == "__main__":
    main()
