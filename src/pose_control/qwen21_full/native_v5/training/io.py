"""Append-only, exclusive artifacts; no truncation, delete, rename or overwrite."""
from pathlib import Path
import hashlib,json,os,uuid
from .config import canonical,digest

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(ref):
    if not isinstance(ref,dict) or not ref.get('path'):raise ValueError('Explicit artifact reference required')
    p=Path(ref['path'])
    if not p.is_absolute():raise ValueError('Artifact paths must be absolute')
    actual=file_sha(p)
    if ref.get('sha256') is not None and actual!=ref['sha256']:raise ValueError('Artifact SHA changed: '+str(p))
    return {'path':str(p),'sha256':actual}

def read_json_ref(ref):
    bound=verify(ref)
    return json.loads(Path(bound['path']).read_text(encoding='utf8')),bound

def sync_directory(path):
    if os.name=='posix':
        fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:
        f.write((canonical(value)+'\n').encode());f.flush();os.fsync(f.fileno())
    sync_directory(p.parent)
    return {'path':str(p.resolve()),'sha256':file_sha(p)}

def new_segment(root,kind,parent=None):
    root=Path(root)
    if not root.is_absolute() or not str(root).startswith('/data/shangguanrz/'):raise ValueError('New run artifacts must use /data/shangguanrz/')
    p=root/'segments'/(kind+'_'+uuid.uuid4().hex);p.mkdir(parents=True,exist_ok=False)
    for directory in (p.parent,root,root.parent):sync_directory(directory)
    write_json(p/'lineage.json',{'kind':kind,'parent':parent})
    return p

class Journal:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('xb'):pass
        sync_directory(self.path.parent)
        self.sequence=0;self.chain='0'*64
    def append(self,record):
        row={'sequence':self.sequence+1,'previous':self.chain,'record':record}
        row['digest']=digest(row)
        with self.path.open('ab') as f:f.write((canonical(row)+'\n').encode());f.flush();os.fsync(f.fileno())
        self.sequence+=1;self.chain=row['digest']
    def state_dict(self):return {'path':str(self.path.resolve()),'sequence':self.sequence,'chain':self.chain}

def verify_prefix(state):
    p=Path(state['path']);raw=p.read_bytes();chain='0'*64;rows=raw.splitlines(keepends=True)
    if len(rows)<state['sequence']:raise ValueError('Committed log prefix is missing')
    for i,line in enumerate(rows[:state['sequence']],1):
        if not line.endswith(b'\n'):raise ValueError('Committed log record is truncated')
        row=json.loads(line);claimed=row.pop('digest')
        if row['sequence']!=i or row['previous']!=chain or digest(row)!=claimed:raise ValueError('Committed log prefix is corrupt')
        chain=claimed
    if chain!=state['chain']:raise ValueError('Checkpoint/log prefix differs')
    tail=b''.join(rows[state['sequence']:])
    return {'committed_records':state['sequence'],'uncommitted_tail_bytes':len(tail),
        'uncommitted_tail_sha256':hashlib.sha256(tail).hexdigest(),'tail_preserved':True,'original_log':str(p)}
