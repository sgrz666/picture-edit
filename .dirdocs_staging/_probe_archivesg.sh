set -x
R=/home/shangguanrz/project/pic-edit
echo ==DF; df -h / | tail -1
echo ==ROOT-SYMLINKS
ls -la $R | grep -- ' -> '
echo ==DEMOOUTPUT
ls -la $R/demooutput 2>/dev/null | head -5
readlink $R/demooutput
echo ==ORGANIZATION-JSON-KEYS
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/organization.json'))
print(type(d), list(d.keys())[:20] if isinstance(d,dict) else len(d))
for k,v in d.items():
    if isinstance(v,list): print(k,'list',len(v))
    elif isinstance(v,dict): print(k,'dict',list(v.keys())[:8])
    else: print(k,repr(v)[:120])
"
echo ==ASSETS-JSON
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/assets.json'))
print(type(d), list(d.keys())[:15] if isinstance(d,dict) else len(d))
if isinstance(d,dict):
    for k,v in d.items():
        print('--',k, type(v).__name__, (list(v.keys())[:12] if isinstance(v,dict) else len(v) if isinstance(v,list) else repr(v)[:100]))
"
echo ==ASSETS-ASSETLIST
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/assets.json'))
import sys
def walk(o,depth=0):
    if depth>2: return
    if isinstance(o,dict):
        for k,v in list(o.items())[:8]:
            print('  '*depth+str(k)[:60], type(v).__name__)
            walk(v,depth+1)
    elif isinstance(o,list) and o:
        print('  '*depth+f'list[{len(o)}]')
        walk(o[0],depth+1)
walk(d)
"
echo ==FINALIZATION
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/finalization.json'))
print(list(d.keys()) if isinstance(d,dict) else len(d))
print(json.dumps(d,ensure_ascii=False)[:1200])
"
echo ==SMALL-OPS
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/small_operations.json'))
print(list(d.keys()) if isinstance(d,dict) else len(d))
print(json.dumps(d,ensure_ascii=False)[:1000])
"
echo ==BEFORE-COUNT
ls $R/archive/cleanup_20261003/before | wc -l
echo ==HOLDS
ls -la $R/archive/cleanup_20261003/holds/scripts | head; ls $R/archive/cleanup_20261003/holds/scripts | wc -l
echo ==ADDBACKUPS
find $R/archive/cleanup_20261003/add_backups -type f | head -20; cat $R/archive/cleanup_20261003/add_backups/README.md
echo ==BACKUP-HOLDS
find $R/archive/cleanup_20261003/backup_holds -type f | head -20; cat $R/archive/cleanup_20261003/backup_holds/README.md
echo ==HOLDS-README
cat $R/archive/cleanup_20261003/holds/README.md
echo ==TOOLING-README
cat $R/archive/cleanup_20261003/tooling/README.md
echo ==BEFORE-DOCS
find $R/archive/cleanup_20261003/before_docs -type f | head
echo ==ENTRY-VERIF
python3 -c "
import json; d=json.load(open('$R/archive/cleanup_20261003/entry_verification.json'))
print(list(d.keys())[:15] if isinstance(d,dict) else len(d)); print(json.dumps(d,ensure_ascii=False)[:600])"
echo ==PYTEST-VERIF
head -c 600 $R/archive/cleanup_20261003/pytest_verification.json
echo ==GITINDEX
head -c 100 $R/archive/cleanup_20261003/git-index.before | od -c | head -3
