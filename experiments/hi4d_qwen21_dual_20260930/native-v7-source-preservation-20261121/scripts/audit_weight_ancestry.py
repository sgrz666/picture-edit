"""Read CPU-only checkpoint provenance; never infer inheritance from filenames."""
from pathlib import Path
import argparse
import hashlib
import json


def references(value, prefix=''):
    found=[]
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and value['path'].endswith(('.pt','.safetensors','.pth')):
            found.append({'key':prefix,'reference':{k:value[k] for k in ('path','sha256') if k in value}})
        else:
            for key,item in value.items():found.extend(references(item,prefix+'.'+key))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):found.extend(references(item,prefix+f'[{index}]'))
    elif isinstance(value,str) and value.endswith(('.pt','.pth','.safetensors')):
        found.append({'key':prefix,'value':value})
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    records=[]
    for path in args.input:
        if path.suffix=='.json':saved=json.loads(path.read_text(encoding='utf8'))
        else:
            import torch
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
        state=saved.get('training_state',{})
        metadata={k:saved[k] for k in saved if k not in ('state_dict','training_state','optimizer_state_dict','optimizer','model')}
        records.append({'file':{'path':str(path.resolve()),'sha256':sha.hexdigest()},'top_keys':sorted(saved),
            'architecture':saved.get('architecture'),'training_schema':saved.get('training_schema'),
            'metadata_keys':sorted(saved.get('metadata',{})),
            'training':{k:state.get(k) for k in ('global_update','stage_update','stage_origin','parent')},
            'initialization':state.get('config',{}).get('initialization'),
            'weight_references':references(metadata)})
        del saved,state,metadata
    result={'read_only':True,'gpu_used':False,'records':records}
    with args.report.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
