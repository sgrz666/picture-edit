"""用既有 smpl_clean 环境移除 SMPL pickle 的 Chumpy 序列化依赖。"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy import sparse


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def numeric(value):
    if hasattr(value,'r'):
        return np.asarray(value.r).copy()
    if isinstance(value,dict):
        return {k:numeric(v) for k,v in value.items()}
    if isinstance(value,list):
        return [numeric(v) for v in value]
    if isinstance(value,tuple):
        return tuple(numeric(v) for v in value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    before=sha(args.input)
    with args.input.open('rb') as stream:
        original=pickle.load(stream,encoding='latin1')
    converted=numeric(original)
    checked=[]
    for k,value in original.items():
        if hasattr(value,'r') or isinstance(value,np.ndarray) or sparse.issparse(value):
            a=value.r if hasattr(value,'r') else value
            b=converted[k]
            a=a.toarray() if sparse.issparse(a) else a
            b=b.toarray() if sparse.issparse(b) else b
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            checked.append(k)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        pickle.dump(converted,stream,protocol=4)
    assert sha(args.input)==before
    report={'source':str(args.input),'source_sha256':before,'output':str(args.output),
            'output_sha256':sha(args.output),'exact_array_checks':checked,'source_unchanged':True}
    with args.output.with_suffix('.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
