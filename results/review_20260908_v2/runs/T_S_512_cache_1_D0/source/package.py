"""Export reviewable evidence without multi-GB adapter/cache tensors."""
import argparse
from pathlib import Path
import zipfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();output=args.output.resolve()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as bundle:
        for path in root.rglob('*'):
            if not path.is_file() or path==output or path.suffix in ('.pt','.npy','.zip','.tmp'):continue
            bundle.write(path,path.relative_to(root).as_posix())
    print(output,output.stat().st_size)
if __name__=='__main__':main()
