"""staging 人工稿质检（只读分析，不生成文档、不碰远端）。

检查项：
  1) 疑似照抄旧生成稿：正文出现生成器的固定句式；
  2) 结构缺件：四要素小节（职责 / 层级地图 / 实证 / 注意事项）是否齐；
  3) 数字实证：正文是否含实测数字（无数字的纯套话稿可疑）；
  4) 篇幅异常：过短(<500B)或疑似占位；
  5) 覆盖差集：目标清单里仍未写 README 的目录（剔除编号/叶子项）。
"""
import io
import os
import re
import sys

STAGE = os.path.dirname(os.path.abspath(__file__))
GEN_PHRASES = ('依据目录实际内容自动生成', '本文件由 `scripts/gen_folder_docs.py`', '构成画像', '子树规模：文件', '直属类型分布')
SECTION_HINTS = (r'在干什么|干什么|目录定位|职责|是什么|留档', r'层级地图|层级|子目录|兄弟|变体|位置',
                 r'实证|关键文件|内容|结果|布局', r'注意事项|调用路径|复现|关系|口径')
NUM_RE = re.compile(r'\d')


def read(p):
    return io.open(p, encoding='utf-8', errors='replace').read()


def main():
    bad = []
    thin = []
    nodigit = []
    total = 0
    for base, prefix in (('repo', 'repo/'), ('data', 'data/')):
        root = os.path.join(STAGE, base)
        if not os.path.isdir(root):
            continue
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if not fn.lower().endswith('.md'):
                    continue
                ap = os.path.join(dp, fn)
                rel = prefix + os.path.relpath(ap, root).replace('\\', '/')
                txt = read(ap)
                total += 1
                hits = [g for g in GEN_PHRASES if g in txt]
                if len(hits) >= 2:
                    bad.append((rel, hits))
                miss = [i for i, pat in enumerate(SECTION_HINTS) if not re.search(pat, txt)]
                if len(miss) >= 2:
                    bad.append((rel, '缺小节 %s' % miss))
                body = txt.split('\n', 3)[-1]
                if len(NUM_RE.findall(body)) < 8:
                    nodigit.append(rel)
                if os.path.getsize(ap) < 500:
                    thin.append(rel)

    done = set()
    for line in io.open(os.path.join(STAGE, '_done_cache.txt'), encoding='utf-8', errors='replace').read().split('\n') if os.path.exists(os.path.join(STAGE, '_done_cache.txt')) else []:
        done.add(line.strip())
    leaf = re.compile(r'(/[0-9]{2,8}|/seq_[0-9]+|/[0-9]{5}_[0-9]+|/motion-[0-9]+|/pair[0-9]+|/[0-9]{6,}|/(train|val|validation|test|source_standard|source_motion|target))$')
    uncovered = []
    for name in ('_target_repo.txt', '_target_data.txt'):
        p = os.path.join(STAGE, name)
        if not os.path.exists(p):
            continue
        for rel in read(p).split('\n'):
            rel = rel.strip()
            if not rel:
                continue
            cand = rel if name.endswith('repo.txt') else 'DATA:' + rel
            if os.path.exists(os.path.join(STAGE, 'repo', rel, 'README.md')) or os.path.exists(os.path.join(STAGE, 'data', rel, 'README.md')):
                continue
            if leaf.search(rel):
                continue
            depth = rel.count('/')
            if name.endswith('repo.txt') and depth >= 3 and rel.split('/')[0] in ('add', 'results', 'datasets', 'experiments') and depth >= 3:
                continue
            uncovered.append(cand)
    print('staged=%d  可疑(照抄/缺件)=%d  过短=%d  少数字=%d  未覆盖(已剔叶子/深层)=%d' % (
        total, len(bad), len(thin), len(nodigit), len(uncovered)))
    for rel, why in bad[:25]:
        print('  SUSPECT %s :: %s' % (rel, why))
    for rel in thin[:15]:
        print('  THIN %s' % rel)
    for rel in nodigit[:15]:
        print('  NO-DIGIT %s' % rel)
    from collections import Counter
    c = Counter(u.split('/')[0] for u in uncovered)
    print('  未覆盖分组:', dict(c.most_common(12)))


if __name__ == '__main__':
    prefix_repo = ''
    main()
