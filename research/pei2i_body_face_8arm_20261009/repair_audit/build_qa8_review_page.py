import hashlib,json,shutil
from html import escape
from pathlib import Path
run=Path(__file__).resolve().parents[1]
out=run/'report/development_qa8_v1'
out.mkdir(parents=True,exist_ok=True)
rows={proto:[json.loads(l) for l in (run/f'prompts/qa8/{proto}/train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()] for proto in ('pg','pb','pbg')}
assert all(len(records)==8 for records in rows.values())
roles={'source_A':'源图 A','target_B':'目标 B（仅 PB/PBG 的 PE 可见）','target_depth':'B 深度','target_normal':'B 法线','target_dwpose':'B 过滤后的 DWPose'}
files={}
def retain(relative,folder):
 src=run/relative;digest=hashlib.sha256(src.read_bytes()).hexdigest()
 target=out/folder/f'{digest}{src.suffix}'
 target.parent.mkdir(exist_ok=True);shutil.copyfile(src,target)
 files[target.relative_to(out).as_posix()]=digest
 return target.relative_to(out).as_posix()
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PE-I2I 第一版预检审查</title>',
 '<style>body{font:16px/1.65 system-ui,sans-serif;margin:28px;max-width:1500px;color:#20232a}h1{font-size:28px}.warning{background:#fff2df;padding:16px;border-left:4px solid #b86619}.inputs{display:flex;gap:14px;flex-wrap:wrap}figure{margin:8px 0;width:180px}img{max-width:180px;height:300px;object-fit:contain;background:#f2f3f5}section{border-top:1px solid #ccd0d5;margin-top:30px;padding-top:16px}summary{cursor:pointer;font-weight:600}pre,p.prompt{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#52606d}</style>',
 '<h1>PE-I2I 第一版预检：输入、提示词与修改依据</h1>',
 '<p class="warning">开发预检已完成，尚未批准进入正式训练。以下保留全部 8 对 × 3 协议的真实输出；它们不是正式八组模型的生成结果。</p>',
 '<p>PG：PE 看 A＋几何；PB：PE 看 A＋B；PBG：PE 看 A＋B＋几何。正式 Qwen 图像参考仍只有 A。图像链接保留原始像素尺寸，网页显示仅用于查看。</p>',
 '<p>审查发现：PG 要求删除源图球杆；部分回复臆测源画面外腿部动作或从裸人体拟合脸推测表情；编译后仍反复提到生成器没有收到的几何图。统一第二版要求将保留物件、约束可见范围和表情证据，并以独立可理解的文字说明姿态。旧回复不做逐样本手工修改。</p>']
for i,pair in enumerate(rows['pbg']):
 parts.append(f'<section><h2>{i+1}. {escape(pair["pair_id"])}</h2><div class="inputs">')
 for item in pair['inputs']:
  link=retain(item['asset_relative_path'],'assets')
  parts.append(f'<figure><a href="{link}"><img src="{link}" alt="{escape(roles[item["role"]])}"></a><figcaption>{escape(roles[item["role"]])}<br><small>{item["size"][0]}×{item["size"][1]}</small></figcaption></figure>')
 parts.append('</div>')
 for proto in ('pg','pb','pbg'):
  record=rows[proto][i];assert record['pair_id']==pair['pair_id']
  raw=retain(record['raw_path'],'raw')
  parts.append(f'<details><summary>{proto.upper()} 实际提示词 · {record["tokens"]} token · {record["seconds"]:.1f} 秒</summary><p class="prompt">{escape(record["compiled_prompt"])}</p><p><a href="{raw}">原始模型回复与思考记录</a></p><small>输入角色：{escape(", ".join(record["input_roles"]))}<br>模板：{escape(record["instruction_version"])}；PE 模型：{escape(record.get("model_dir",""))}；seed：{record["seed"]}<br>提示词 SHA256：{record["compiled_prompt_sha256"]}</small></details>')
 parts.append('</section>')
parts.append('</html>')
(out/'index.html').write_text('\n'.join(parts),encoding='utf-8')
(out/'files.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
(out/'records.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'page':str(out/'index.html'),'replies':24,'retained_files':len(files)}))
