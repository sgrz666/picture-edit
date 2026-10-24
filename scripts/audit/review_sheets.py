"""Build per-sample review sheets so a human (me) can look at the actual pixels the model gets
and judge whether the prompt describes them correctly.

Panels are the exact files each rewrite pipeline feeds PE / the generator:
  tiktok : source_rgb | target_dwpose (skeleton) | target_rgb (GT)
  hi4d   : source_rgb | target_pose_combined | target_pose_A | target_pose_B | target_rgb

Cohort priority: first every row the cleaner reverted to v1 or flagged needs_review
(those are the ones most likely wrong), then a deterministic stratified walk of the rest.

Writes PNGs + a JSON manifest with the prompt text next to each sheet. Read-only on the banks.
"""
import argparse, json, math
from pathlib import Path
from PIL import Image, ImageDraw

REPO = Path('/home/shangguanrz/project/pic-edit')
OUT = Path('/data/shangguanrz/manual_review')
BANKS = {
    'tiktok': {
        'v3': {s: REPO / f'experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014/prompts_v3/{s}.jsonl'
               for s in ('train', 'val', 'test')},
        'field': 'compiled_prompt',
        'man': {s: REPO / f'datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/{s}_pairs.jsonl'
                for s in ('train', 'val', 'test')},
        'panels': ('source_rgb', 'target_dwpose', 'target_rgb'),
        'labels': ('A source', 'skeleton', 'B target GT'),
    },
    'hi4d': {
        'v3': {s: REPO / f'experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/prompts_v3/hi4d_{s}_prompts_v3.jsonl'
               for s in ('train', 'val', 'test')},
        'field': 'compiled_prompt_training',
        'man': {s: REPO / f'datasets/Hi4D_pilot_v1/final_manifests/pairs_1500_{s}.jsonl'
                for s in ('train', 'val', 'test')},
        'panels': ('source_rgb', 'target_pose_combined', 'target_pose_A', 'target_pose_B', 'target_rgb'),
        'labels': ('A source', 'combined skel', 'person A skel', 'person B skel', 'B target GT'),
    },
}
H = 400            # panel height; 5 panels x 400 stays readable in one sheet


def rows(p):
    return [json.loads(l) for l in open(p, encoding='utf-8').read().splitlines() if l.strip()]


def manifest_index(bank, split):
    """join v3 rows back to the image paths in the dataset manifest."""
    cfg = BANKS[bank]
    out = {}
    for r in rows(cfg['man'][split]):
        pid = r.get('pair_id')
        if pid:
            out[pid] = r
            continue
        key = f"{r['pair']}_{r['action']}_cam{r['camera_id']}_{int(r['source_frame_id'])}_{int(r['target_frame_id'])}"
        out[key] = r
    return out


def norm_key(pair_id):
    import re
    pid = re.sub(r'^(train|val|test)_', '', pair_id)
    m = re.match(r'(pair\d+)_(\w+)_cam(\d+)_(\d+)_(\d+)', pid)
    if not m:
        return pid
    p, a, c, s, t = m.groups()
    return f'{p}_{a}_cam{c}_{int(s)}_{int(t)}'


def make_sheet(bank, row, man_row, path):
    cfg = BANKS[bank]
    panels = []
    for key, label in zip(cfg['panels'], cfg['labels']):
        fp = man_row.get(key)
        if not fp or not Path(fp).is_file():
            im = Image.new('RGB', (200, H), (30, 30, 30))
            d = ImageDraw.Draw(im)
            d.text((6, 6), f'MISSING\n{key}', fill=(255, 80, 80))
        else:
            im = Image.open(fp).convert('RGB')
            w = max(1, int(im.width * (H / im.height)))
            im = im.resize((min(w, 900), H))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width - 1, 16], fill=(0, 0, 0))
        d.text((3, 2), label, fill=(255, 255, 0))
        panels.append(im)
    total_w = sum(p.width for p in panels)
    sheet = Image.new('RGB', (total_w, H + 46), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width
    d = ImageDraw.Draw(sheet)
    txt = (row.get(cfg['field']) or '')[:2400]
    # wrap the prompt into the caption band so the sheet is self-contained
    lines, cur = [], ''
    for word in txt.split():
        if len(cur) + len(word) > 150:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + ' ' + word).strip()
        if len(lines) >= 2:
            break
    lines.append(cur)
    d.multiline_text((3, H + 2), '\n'.join(lines), fill=(0, 0, 0))
    sheet.save(path)
    return sheet.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', choices=('tiktok', 'hi4d', 'both'), default='both')
    ap.add_argument('--want', type=int, default=20, help='sheets per bank')
    ap.add_argument('--offset', type=int, default=0, help='skip N stratified rows (for the next batch)')
    ap.add_argument('--ids', default='', help='comma separated pair_ids to force into this batch')
    ap.add_argument('--tag', default='batch1')
    args = ap.parse_args()

    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    keys = ('tiktok', 'hi4d') if args.bank == 'both' else (args.bank,)
    manifest = []
    forced = [x for x in args.ids.split(',') if x]
    for bank in keys:
        cfg = BANKS[bank]
        field = cfg['field']
        cand, flagged = [], []
        for split in ('train', 'val', 'test'):
            v3 = rows(cfg['v3'][split])
            mi = manifest_index(bank, split)
            for r in v3:
                pid = r['pair_id']
                key = norm_key(pid)
                m = mi.get(key) or mi.get(pid)
                if m is None:
                    continue
                item = (bank, split, pid, r, m)
                if forced:
                    if pid in forced:
                        cand.append(item)
                    continue
                if r.get('cleaning_reverted_fields') or r.get('cleaning_needs_review'):
                    flagged.append(item)
                else:
                    cand.append(item)
        if not forced:
            step = max(1, len(cand) // max(1, args.want))
            picked = flagged[:args.want] + cand[args.offset::step]
            picked = picked[:args.want]
            kind = f'flagged_first({len(flagged)} flagged rows)'
        else:
            picked, kind = cand[:args.want], 'forced ids'
        print(f'{bank}: {len(picked)} sheets ({kind}, want={args.want}, offset={args.offset})')
        for bank_, split, pid, r, m in picked:
            name = f'{args.tag}_{bank}_{pid}'.replace('/', '_')
            path = OUT / 'sheets' / f'{name}.png'
            size = make_sheet(bank, r, m, path)
            manifest.append({'bank': bank, 'split': split, 'pair_id': pid, 'sheet': str(path),
                             'sheet_size': list(size), 'field': field,
                             'prompt': r.get(field), 'prompt_inference': r.get('compiled_prompt_inference'),
                             'action': r.get('action'), 'sequence_id': r.get('sequence_id'),
                             'reverted': r.get('cleaning_reverted_fields'),
                             'needs_review': r.get('cleaning_needs_review'),
                             'notes': r.get('cleaning_review_notes'),
                             'images': {k: m.get(k) for k in BANKS[bank]['panels']}})
    mp = OUT / f'manifest_{args.tag}.json'
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print('WROTE', mp, len(manifest), 'sheets')
    for e in manifest:
        print('\n==== ' + e['bank'] + '/' + e['pair_id'],
              '| reverted=', e['reverted'], '| notes=', (e['notes'] or [])[:2])
        print((e['prompt'] or '')[:1100])


if __name__ == '__main__':
    main()

