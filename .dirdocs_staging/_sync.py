"""把 .dirdocs_staging 内的人工手写 README 同步到远端，并强制不覆盖人工/上游自带稿。

用法：
    python .dirdocs_staging/_sync.py            # 只列计划，不动远端（dry-run）
    python .dirdocs_staging/_sync.py --apply    # 实际同步
"""
import io
import os
import subprocess
import sys

STAGE = os.path.dirname(os.path.abspath(__file__))
REPO_STAGE = os.path.join(STAGE, 'repo')
DATA_STAGE = os.path.join(STAGE, 'data')
REMOTE_REPO = '/home/shangguanrz/project/pic-edit'
REMOTE_DATA = '/data/shangguanrz'
RAW_PROTECT = os.path.join(STAGE, '_protect_raw.txt')


def load_protect():
    repo, data = set(), set()
    if os.path.exists(RAW_PROTECT):
        lines = io.open(RAW_PROTECT, encoding='utf-8', errors='replace').read().split('\n')
        sep = lines.index('==SPLIT==')
        repo = {l.strip().replace('\\', '/') for l in lines[:sep] if l.strip()}
        data = {l.strip().replace('\\', '/') for l in lines[sep + 1:] if l.strip()}
    extra = os.path.join(STAGE, '_data_protect.txt')
    if os.path.exists(extra):
        for line in io.open(extra, encoding='utf-8', errors='replace').read().split('\n'):
            line = line.strip()
            if line and not line.startswith('data_protect=') and line.endswith('.md'):
                data.add(line)
    return repo, data


def collect(root):
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith('.md'):
                continue
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, root).replace('\\', '/')
            out.append((rel, ap, os.path.getsize(ap)))
    return sorted(out)


def main():
    apply_ = '--apply' in sys.argv
    only = None
    for a in sys.argv:
        if a.startswith('--only='):
            only = a.split('=', 1)[1]
    pro_repo, pro_data = load_protect()
    plans = []
    for label, stage_root, remote, protect in (
        ('repo', REPO_STAGE, REMOTE_REPO, pro_repo),
        ('data', DATA_STAGE, REMOTE_DATA, pro_data),
    ):
        if only and label != only:
            continue
        items = collect(stage_root)
        keep, blocked = [], []
        for rel, ap, size in items:
            (keep if rel not in protect else blocked).append((rel, ap, size))
        plans.append((label, stage_root, remote, keep, blocked))
        print('[%s] 待同步 %d 份，因禁覆盖清单拦截 %d 份' % (label, len(keep), len(blocked)))
        for rel, _ap, _s in blocked:
            print('    BLOCKED(人工稿已存在): %s' % rel)
        small = [(rel, s) for rel, _ap, s in keep if s < 1200]
        for rel, s in small:
            print('    WARN 篇幅偏短(%dB): %s' % (s, rel))

    if not apply_:
        print('\ndry-run，未写入。加 --apply 执行。')
        return

    for label, stage_root, remote, keep, _blocked in plans:
        if not keep:
            print('[%s] 无内容，跳过' % label)
            continue
        listfile = os.path.join(STAGE, '_tarlist_%s.txt' % label)
        with io.open(listfile, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(rel for rel, _a, _s in keep) + '\n')
        cmd = 'tar -cf - -C "%s" -T "%s" | ssh sg "tar -xf - -C %s"' % (
            stage_root.replace('\\', '/'), listfile.replace('\\', '/'), remote)
        r = subprocess.run(['bash', '-c', cmd])
        print('[%s] tar|ssh 退出码=%d，共 %d 份 → %s' % (label, r.returncode, len(keep), remote))


if __name__ == '__main__':
    main()
