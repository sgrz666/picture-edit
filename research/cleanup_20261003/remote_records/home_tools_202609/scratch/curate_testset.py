#!/usr/bin/env python3
"""Curate the downloaded candidate pool into a cross-body-type retargeting test set.

- moves unsafe / off-task candidates out of the pool
- selects appearance sources (A, heavy) and pose references (B, slim)
- writes frozen 1024px test inputs + manifest with provenance
"""
import json
import os
import shutil
import urllib.parse
import urllib.request

from PIL import Image

BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/data'
CAND = os.path.join(BASE, 'candidates')
REJ = os.path.join(CAND, '_rejected')
SET = os.path.join(BASE, 'testset')
TARGET_LONG_SIDE = 1280
API = 'https://commons.wikimedia.org/w/api.php'

# ---- 不安全 / 与任务无关，必须移出候选池 ----
REJECT = [
    'Buried_penis.jpg',                                  # 医学，裸露
    'Diagnostyka_bezdechu_śródsennego.jpg',              # 医学，病患
    'A_system_of_practical_medicine__1897___14778802172_.jpg',   # 医学插画
    '_Tom-Ton_-_Fat_Boy__Wellcome_L0040521.jpg',         # 历史猎奇
    'Annex_-_Howerton__Clarence__Major_Mite__02.jpg',    # 历史
    'De_dikke_18-jarige_Leonard_West__400_pond__uit_Los_Angeles_is_een_prima_zwemmer_en_duiker__SFA002001631.jpg',
    'Dikkerds__ongemakken__SFA022815318.jpg',            # 历史
    '2021-11-05_WK_7_Springboard_3m_Boys_Youth_A_at_Ulla-Klinger-Cup_2021__Martin_Rulsch__421.jpg',  # 涉未成年人
    '9430Poblacion__Baliuag__Bulacan_05.jpg',            # 非人像（弯腰）
    'Bookreader_at_ekushe_boimela_2023_6.jpg',           # 非人像
    'AG_Dicke_Piraten.jpg',                              # 背影，非人像
    'Toccara_Jones_e_Ashanti.jpg',                       # 双人
    '2-promotional-models-fenasucro.JPG',                # 双人
    '2BuckJumpersUptownSuperSunday09.JPG',               # 双人
    'AIDS_Auditions_London_5_with_Boikanyo_Trust_Phenyo.jpg',   # 半身
    'Aerbalonoj_en_Goodwin.jpg',                         # 主体过小
    'Barbara_Palvin_2025__facecrop_.jpg',                # 面部特写
    'Cover_of_Fashion_Street_Magazine_SS_2018.jpg',      # 杂志封面拼版
    'MKr382321_Barbara_Palvin__Cannes_2025_.jpg',        # 半身
    'MKr382323_Barbara_Palvin__Cannes_2025_.jpg',        # 半身
    'Actress_Laila_Alina_Reischer_New_York_Film_Festival_Best_supporting_actress_2011.JPG',
    '2008-10-04_Security_guard_at_Beer_Fest.jpg',        # 姿态不清晰
    '2011-09-17_Roma_Manifestante_Vaticano_paga_tu.jpg',  # 多人
    '2015-08-04-1021_0___20254253326_.jpg',              # 背景杂乱
    'ADAM_Fall-Winter_2010_170.jpg',                     # 双人走秀
    'Jill_Stuart_Fall-Winter_2010_230.jpg',              # 双人走秀
    '250824_FC_서울_vs_울산_고광민_하프타임_은퇴식_2.jpg',   # 多人
    '250824_FC_서울_vs_울산_고광민_하프타임_은퇴식_3.jpg',   # 多人
    '220528_FC_서울_vs_김천_상무_FC__김정훈_.jpg',          # 3/4 裁切
    '220622_FC서울_vs_울산_현대_FC__Leonardo_.jpg',         # 3/4 裁切
]

# ---- A：外貌来源（体型显著偏大） ----
A_HEAVY = [
    'BigGirlsRock2FashionShow22016_011__25986507981_.jpg',
    'BigGirlsRock2FashionShow22016_014__25779998480_.jpg',
    'BigGirlsRock2FashionShow22016_026__26052853385_.jpg',
    'BigGirlsRock2FashionShow22016_029__25779993480_.jpg',
    'BigGirlsRock2FashionShow22016_039__25450287583_.jpg',
    'BigGirlsRock2FashionShow22016_047__26052846665_.jpg',
    'BigGirlsRock2FashionShow22016_062__25448057424_.jpg',
    'BigGirlsRock2FashionShow22016_066__25448036534_.jpg',
    'BigGirlsRock2FashionShow22016_067__26052813995_.jpg',
    'BigGirlsRock2FashionShow22016_069__25450250993_.jpg',
    'BigGirlsRock2FashionShow22016_081__25448032524_.jpg',
    'BigGirlsRock2FashionShow22016_087__25986461641_.jpg',
    'BigGirlsRock2FashionShow22016_092__26052844055_.jpg',
    'BBW_Christmas_Model.jpg',
    'Bench_Press_-_Flickr_-_colros.jpg',
    'Toccara_Jones.jpg',
    'Toccara_Jones_2.jpg',
    'Toccara_Jones_3.jpg',
    'Toccara_Jones_cropped.jpg',
]

# ---- B：姿态参考（体型偏瘦，全身，姿态清晰） ----
B_SLIM = [
    'Miguel_Ramirez_Headshot.jpg',
    'Lucas_Oliveira_Miranda.jpg',
    'Vuslat_DS.jpg',
    'OIKN_Deputi_202411749__1_.jpg',
    'DR.Andi_Afdal_MBA__AAK-017.jpg',
    'Keonte__in_2024.jpg',
    'Mercy_Namukose.jpg',
    'Ama_Ampofo.jpg',
    'Prelude__35036862301_.jpg',
    'My_Coloured_Rectangle.jpg',
    'Siuzanna-77.jpg',
    'Lynda_Carter_at_The_Heart_Truth_2009.jpg',
    'Ringberg2021.jpg',
    'KurtMaier_FotoSylviaAsmus.jpg',
    'IOW_Oliver_Zielinski_Pressefoto_IMG_1715_KBeck.jpg',
    'Aardra_Padmakumar.jpg',
    'African_Street_Style_Festival__Shoreditch__London__24_July_2016_15.jpg',
    'African_Street_Style_Festival__Shoreditch__London__24_July_2016_33.jpg',
    '-37_Eiffel_Janell_Rosalita__Catanduanes__52266606299_.jpg',
    '230731_서울_이랜드_FC_vs_충남_아산_FC__문현호_.jpg',
    '230731_서울_이랜드_FC_vs_충남_아산_FC__이동률_.jpg',
]

PAIRS = [
    ('hard_01', 'BigGirlsRock2FashionShow22016_011__25986507981_.jpg', 'Miguel_Ramirez_Headshot.jpg'),
    ('hard_02', 'BigGirlsRock2FashionShow22016_014__25779998480_.jpg', 'Lucas_Oliveira_Miranda.jpg'),
    ('hard_03', 'BigGirlsRock2FashionShow22016_062__25448057424_.jpg', 'Vuslat_DS.jpg'),
    ('hard_04', 'BigGirlsRock2FashionShow22016_087__25986461641_.jpg', 'Ama_Ampofo.jpg'),
    ('hard_05', 'BBW_Christmas_Model.jpg', 'Prelude__35036862301_.jpg'),
    ('medium_01', 'Bench_Press_-_Flickr_-_colros.jpg', 'Ringberg2021.jpg'),
    ('medium_02', 'Toccara_Jones.jpg', 'Mercy_Namukose.jpg'),
]


def move_rejects():
    os.makedirs(REJ, exist_ok=True)
    moved = 0
    for name in REJECT:
        for bucket in ('heavy', 'slim'):
            src = os.path.join(CAND, bucket, name)
            if os.path.isfile(src):
                shutil.move(src, os.path.join(REJ, f'{bucket}__{name}'))
                moved += 1
    print(f'rejected moved: {moved}')


def freeze(src_name, bucket, dest_path, long_side=TARGET_LONG_SIDE):
    src = os.path.join(CAND, bucket, src_name)
    if not os.path.isfile(src):
        return None
    im = Image.open(src).convert('RGB')
    w, h = im.size
    scale = long_side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    im.save(dest_path, quality=95, subsampling=0)
    return im.size


def licenses(titles):
    """One batched API call for attribution metadata."""
    out = {}
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        params = {'action': 'query', 'prop': 'imageinfo',
                  'iiprop': 'extmetadata|url', 'titles': '|'.join(chunk),
                  'format': 'json', 'formatversion': '2'}
        url = API + '?' + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'pic-edit-research/1.0'})
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            for page in data.get('query', {}).get('pages', []):
                em = ((page.get('imageinfo') or [{}])[0].get('extmetadata') or {})
                out[page['title']] = {
                    'license': (em.get('LicenseShortName') or {}).get('value', ''),
                    'artist': (em.get('Artist') or {}).get('value', '')[:200],
                    'credit': (em.get('Credit') or {}).get('value', '')[:200],
                    'descriptionurl': (em.get('ImageDescription') or {}).get('value', '')[:200],
                }
        except Exception as exc:
            print(f'license lookup failed: {exc}')
    return out


def build():
    move_rejects()
    os.makedirs(os.path.join(SET, 'A_heavy'), exist_ok=True)
    os.makedirs(os.path.join(SET, 'B_slim'), exist_ok=True)

    meta = {'A': {}, 'B': {}}
    for bucket, names, key in (('heavy', A_HEAVY, 'A'), ('slim', B_SLIM, 'B')):
        dest_dir = os.path.join(SET, 'A_heavy' if key == 'A' else 'B_slim')
        for n in names:
            out = os.path.join(dest_dir, os.path.splitext(n)[0] + '.png')
            size = freeze(n, bucket, out)
            if size:
                meta[key][n] = {'file': os.path.basename(out), 'frozen_size': size}
            else:
                print(f'  MISSING {bucket}/{n}')

    # 成对目录
    pairs_dir = os.path.join(SET, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)
    pair_records = []
    for pid, a_name, b_name in PAIRS:
        d = os.path.join(pairs_dir, pid)
        os.makedirs(d, exist_ok=True)
        a_out = os.path.join(d, 'A_source.png')
        b_out = os.path.join(d, 'B_target.png')
        sa = freeze(a_name, 'heavy', a_out)
        sb = freeze(b_name, 'slim', b_out)
        if not (sa and sb):
            print(f'  PAIR {pid} incomplete (A={sa} B={sb})')
            continue
        pair_records.append({'pair': pid, 'A_file': a_name, 'A_size': sa,
                             'A_path': a_out, 'B_file': b_name, 'B_size': sb,
                             'B_path': b_out})
        print(f'  pair {pid}: A {sa}  B {sb}')

    titles = ['File:' + n for n in set(A_HEAVY + B_SLIM)]
    lic = licenses(titles)
    for key, names in (('A', A_HEAVY), ('B', B_SLIM)):
        for n in names:
            if n in meta[key]:
                meta[key][n]['attribution'] = lic.get('File:' + n, {})

    manifest = {
        'task': 'single-person cross-body-type pose retargeting capability test',
        'backbone': 'Qwen-Image-2.1 (sha 790c92633540aa0cb11d9abf19eb46d861714758)',
        'source': 'Wikimedia Commons (freely licensed)',
        'convention': 'A_source = appearance source (large build); B_target = pose reference (slim)',
        'frozen_long_side': TARGET_LONG_SIDE,
        'counts': {'A_heavy': len(meta['A']), 'B_slim': len(meta['B']), 'pairs': len(pair_records)},
        'pairs': pair_records,
        'assets': meta,
        'known_limitations': [
            'A 侧多为 3/4 身裁切（Wikimedia 上现代、全身、体型显著偏大的自由版权照片稀缺）',
            '社媒来源（小红书/抖音）需浏览器 CDP，未纳入本轮',
            '背景与光照未做统一，评测时需在指标之外人工核验',
        ],
    }
    with open(os.path.join(SET, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print('counts:', manifest['counts'])


if __name__ == '__main__':
    build()
    print('DONE')
