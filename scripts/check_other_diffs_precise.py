import json

def load_prompts(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                item = json.loads(line)
                key = item.get('pair_name') or item.get('pair_id') or item.get('id') or idx
                data[key] = item
    return data

r3_dir = '/home/shangguanrz/project/pic-edit/experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/v5_grounded_3d_20261030_r3/prompts'
r4_dir = '/home/shangguanrz/project/pic-edit/experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/v5_grounded_3d_20261030_r4/prompts'

for split in ['test', 'val', 'train']:
    p3 = load_prompts(f'{r3_dir}/hi4d_{split}_prompts_v2.jsonl')
    p4 = load_prompts(f'{r4_dir}/hi4d_{split}_prompts_v2.jsonl')
    
    diff_other = []
    diff_above_head = 0
    for k in p3:
        txt3 = p3[k]['compiled_prompt_inference']
        txt4 = p4[k]['compiled_prompt_inference']
        if txt3 != txt4:
            c3 = txt3.count('above head height')
            c4 = txt4.count('above head height')
            # Check if difference is purely removal of 'above head height'
            norm3 = txt3.replace(', above head height', '').replace('above head height', '')
            norm4 = txt4.replace(', above head height', '').replace('above head height', '')
            if norm3 == norm4:
                diff_above_head += 1
            else:
                diff_other.append((k, txt3, txt4))
                
    print(f'=== Split {split}: {len(diff_other)} true non-above-head diffs, {diff_above_head} pure above-head changes ===')
    for k, t3, t4 in diff_other:
        print(f'Key: {k}')
        print(f'  r3: {t3}')
        print(f'  r4: {t4}')
        print('-'*40)
