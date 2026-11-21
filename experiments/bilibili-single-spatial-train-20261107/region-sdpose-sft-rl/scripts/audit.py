"""Read-only initialization-chain and evaluation-input receipt before training."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepared=json.loads((a.root/'prepared.json').read_text());directory=Path(prepared['initialization']['path']).parent
    source='f68a2a881e57a9537ae68d69b51c660fa503a3b48cbf9f6b4ad88f29fa259e45';chain=[]
    for version in ('v6','v61','v62','v7'):
        report_path=directory/(version+'-migration.json');report=json.loads(report_path.read_text())
        if report['source_sha256']!=source or report.get('training_started') is not False or report.get('optimizer_inherited') is not False:
            raise ValueError('Initialization does not trace to the common pre-spatial production model: '+version)
        actual=digest(directory/(version+'.pt'))
        if report['output_sha256']!=actual or not report.get('strict_readback'):
            raise ValueError('Migration readback/SHA proof differs: '+version)
        chain.append({'version':version,'source_sha256':source,'output_sha256':actual,'report_sha256':digest(report_path)})
        source=actual
    if source!=prepared['initialization']['sha256']:raise ValueError('Final initializer differs from the prepared pilot')
    pairs=json.loads((a.root/'data/evaluation.json').read_text())
    train={r['group'] for r in pairs if r['split']=='train'};val={r['group'] for r in pairs if r['split']=='val'}
    if train&val:raise ValueError('Original-video leakage')
    receipt={'schema':'bilibili_region_input_audit_v1','passed':True,'initialization_chain':chain,
        'evaluation_manifest':{'path':str(a.root/'data/evaluation.json'),'sha256':digest(a.root/'data/evaluation.json')},
        'train_pairs':sum(r['split']=='train' for r in pairs),'val_pairs':sum(r['split']=='val' for r in pairs),
        'train_videos':len(train),'val_videos':len(val),'original_video_overlap':False,
        'scope':'read-only input and pre-spatial initialization proof; no models or optimizer constructed'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
