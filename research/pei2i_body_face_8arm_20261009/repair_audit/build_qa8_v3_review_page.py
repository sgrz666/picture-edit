"""Retain the complete reviewed QA v3 input and prompt evidence in Chinese."""
import hashlib
import json
import shutil
from html import escape
from pathlib import Path

run = Path(__file__).resolve().parents[1]
out = run / 'report/development_qa8_v3'
out.mkdir(parents=True, exist_ok=True)
review = json.loads((run / 'repair_audit/qa8_v3_visual_review.json').read_text(encoding='utf-8'))
notes = {(x['protocol'], x['pair_id']): x['review_note'] for x in review['records']}
banks = {p: [json.loads(line) for line in (run / f'prompts/qa8/{p}/train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()] for p in ('pg', 'pb', 'pbg')}
assert all(len(rows) == 8 for rows in banks.values())
roles = {'source_A': '原图 A：外观与场景', 'target_B': '目标 B：仅 PB/PBG 的 PE 看见 RGB', 'target_depth': 'B depth：PE 输入', 'target_normal': 'B normal：PE 输入', 'target_dwpose': 'B 过滤 DWPose：PE 输入'}
files = {}


def retain(relative, folder):
    src = run / relative
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    target = out / folder / f'{sha}{src.suffix}'
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(src, target)
    rel = target.relative_to(out).as_posix()
    files[rel] = sha
    return rel


parts = [
    '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PE 第三版预检审查：24 条真实输出</title>',
    '<style>body{font:16px/1.7 system-ui,sans-serif;margin:28px;max-width:1500px;color:#20232a}h1{font-size:28px}.note{background:#f1f5f9;padding:16px;border-left:4px solid #496782}.warning{background:#fff2df;padding:12px;border-left:4px solid #b86619}.inputs{display:flex;gap:14px;flex-wrap:wrap}figure{margin:8px 0;width:180px}img{max-width:180px;height:300px;object-fit:contain;background:#f2f3f5}section{border-top:1px solid #ccd0d5;margin-top:30px;padding-top:16px}summary{cursor:pointer;font-weight:600}p.prompt{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#52606d}</style>',
    '<h1>PE 第三版预检审查：输入、真实提示词与已知误读</h1>',
    '<p class="note">24 条开发预检已审查，放行用于正式八组对照。这里展示的是 PE 的真实提示词与输入，尚不是新八组模型的生成结果。放行不代表所有动作细节均正确，以下保留每条问题。</p>',
    '<p>PG：A＋depth/normal/过滤 DWPose；PB：A＋B；PBG：A＋B＋上述几何。P0 继续使用历史固定短提示词。Qwen 图像参考仅 A，PE、Qwen 和 BFS 权重冻结，Body＋Face Adapter 联合训练。B 是合法的训练监督目标，测试不反传。</p>',
    '<p>原图按原始像素保留，点击可查看原图。统一 instruction_v3 使用官方 PE 系统提示词和 thinking；没有手改个别回复，也没有按好坏替换样本。</p>',
    f'<p>审查时间（UTC）：{escape(review["reviewed_at"])}；24 条原始回答、请求与图片哈希一致，输入图片共 {review["verified_unique_input_assets"]} 个唯一文件。</p>',
    '<p><a href="../index.html">实验主网页</a> · <a href="../development_qa8_v1/index.html">第一版预检留档</a></p>',
]
for text in review['notes']:
    parts.append(f'<p class="warning">{escape(text)}</p>')
for index, row in enumerate(banks['pbg']):
    pair_id = row['pair_id']
    parts.append(f'<section><h2>{index + 1}. {escape(pair_id)}</h2><div class="inputs">')
    for item in row['inputs']:
        asset = retain(item['asset_relative_path'], 'assets')
        parts.append(f'<figure><a href="{asset}"><img src="{asset}" alt="{escape(roles[item["role"]])}"></a><figcaption>{escape(roles[item["role"]])}<br><small>{item["size"][0]}×{item["size"][1]}</small></figcaption></figure>')
    parts.append('</div>')
    for protocol in ('pg', 'pb', 'pbg'):
        record = banks[protocol][index]
        assert record['pair_id'] == pair_id
        raw = retain(record['raw_path'], 'raw')
        parts.append(f'<details><summary>{protocol.upper()} · PE {record["seconds"]:.1f} 秒 · {len(record["compiled_prompt"].split())} 个英文词</summary><p class="warning">审查：{escape(notes[protocol, pair_id])}</p><p class="prompt">{escape(record["compiled_prompt"])}</p><p><a href="{raw}">原始回答、思考和实际输入记录</a></p><small>seed={record["seed"]}；模板={escape(record["instruction_version"])}<br>最终文本 SHA256：{record["compiled_prompt_sha256"]}<br>请求 SHA256：{record["request_sha256"]}</small></details>')
    parts.append('</section>')
parts.append('</html>')
(out / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
(out / 'records.json').write_text(json.dumps(banks, indent=2, ensure_ascii=False), encoding='utf-8')
(out / 'review.json').write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding='utf-8')
for name in ('index.html', 'records.json', 'review.json'):
    files[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
(out / 'files.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
print(json.dumps({'page': str(out / 'index.html'), 'reviewed_prompts': 24, 'retained_files': len(files)}))
