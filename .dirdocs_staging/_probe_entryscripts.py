import ast, os, re, json
BASE = "/home/shangguanrz/project/pic-edit/scripts"
dirs = ["", "audit", "data", "eval", "reports", "run", "train", "weights"]
out = []
for d in dirs:
    p = os.path.join(BASE, d)
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        if os.path.islink(fp):
            continue
        if not os.path.isfile(fp):
            continue
        if f in ("README.md",) or f.endswith("目录说明.md") or f.endswith(".pyc"):
            continue
        rel = os.path.join(d, f) if d else f
        if f.endswith(".py"):
            try:
                src = open(fp, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
                doc = ast.get_docstring(tree) or ""
                doc = " ".join(doc.split())[:240]
                # argparse defaults & add_argument dests
                args = re.findall(r'add_argument\(\s*[\'"]--?([\w-]+)', src)
                paths = re.findall(r'default\s*=\s*[\'"]([^\'"]{4,120})[\'"]', src)
                paths = [x for x in paths if "/" in x or x.endswith((".json", ".sh", ".npz", ".pt"))]
                consts = re.findall(r'^(?:ROOT|OUTPUT|OUT|DEFAULT_[A-Z_]+)\s*=\s*[\'"]([^\'"]{4,140})[\'"]', src, re.M)
                out.append({"file": rel, "doc": doc, "args": sorted(set(args))[:40], "defaults": (paths[:8] + consts[:8])})
            except Exception as e:
                out.append({"file": rel, "error": str(e)})
        elif f.endswith((".sh", ".ps1")):
            src = open(fp, encoding="utf-8", errors="replace").read()
            lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#!")]
            out.append({"file": rel, "head": " | ".join(lines[:12])[:600]})
print(json.dumps(out, ensure_ascii=False, indent=1))
