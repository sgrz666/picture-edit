"""Compact read-only paired-pilot progress, including actual Source loss parts."""
from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    read=lambda path:json.loads(path.read_text(encoding='utf8'))
    reports=args.root/'report'/args.name
    result={'reports':{name:read(reports/name) for name in ('completed.json','failure.json') if (reports/name).exists()},'segments':[]}
    ready={path.stem:read(path) for path in reports.glob('*-ready.json')}
    for folder in (args.root/'output/training/segments').glob('*'):
        if not (folder/'resolved.json').exists():continue
        cfg=read(folder/'resolved.json')['config']
        names=[name for name,value in ready.items() if value==cfg]
        if not names:continue
        latest=None
        with (folder/'metrics.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                try:row=json.loads(line)['record']
                except (json.JSONDecodeError,KeyError):continue
                if row.get('optimizer_updated'):
                    latest={key:row[key] for key in ('global_update','stage_update','gradient_norm')}
                    latest['sigmas']=[entry['sigma'] for entry in row['window']]
                    keys=('velocity','identity','source_identity','source_identity_eligible','source_background','weighted_micro_objective')
                    latest['losses']=[{key:parts[key] for key in keys if key in parts} for parts in row['losses']]
        result['segments'].append({'arm':names[0],'latest':latest})
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
