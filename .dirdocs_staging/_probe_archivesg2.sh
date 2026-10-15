set -x
R=/home/shangguanrz/project/pic-edit
echo ==ASSETS-TASKS
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/assets.json'))
for t in d['tasks']:
    print(t['source'],'->',t['target'],'| bytes',t['bytes'],'files',t['files'],'state',t['state'])
print('disk_before',d['disk_before'])
print('disk_after',d['disk_after'])
"
echo ==FINAL-RETIRED
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/finalization.json'))
print('apply',d['apply']);print('retired',len(d['retired']))
for r in d['retired']: print(' ',r)
print('small_intents',type(d['small_intents']),len(d['small_intents']))
si=d['small_intents']
import itertools
if isinstance(si,list):
    for x in si[:5]: print(json.dumps(x,ensure_ascii=False)[:200])
print('disk_before',d['disk_before'],'disk_after',d['disk_after'])
"
echo ==ORG-MOVES-SAMPLE
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261003/organization.json'))
print(json.dumps(d['moves'][0],ensure_ascii=False)[:500])
print(json.dumps(d['aliases'][0],ensure_ascii=False)[:300])
cats={}
for m in d['moves']:
    t=m.get('target','')
    top=t.replace('$R/','').split('/')[0]
    cats[top]=cats.get(top,0)+1
print(cats)
"
echo ==HOLDS-SCRIPTS
ls -la $R/archive/cleanup_20261003/holds/scripts/
find $R/archive/cleanup_20261003/holds $R/archive/cleanup_20261003/backup_holds -type f ! -name '目录说明.md' ! -name 'README.md'
echo ==DU-1003
du -sh --si $R/archive/cleanup_20261003/{before,before_docs,tooling,holds,add_backups,backup_holds}
echo ==DATA-ARCHIVE-TOP
ls -la /data/shangguanrz/pic-edit-archive/ 2>/dev/null
echo ==DATA-1003
ls /data/shangguanrz/pic-edit-archive/cleanup_20261003/ 2>/dev/null
du -sh --si /data/shangguanrz/pic-edit-archive/cleanup_20261003/* 2>/dev/null | head -20
echo ==REMOVED-LINKS-1008
python3 -c "
import json
d=json.load(open('$R/archive/cleanup_20261008/removed_links.json'))
print(type(d), list(d.keys())[:10] if isinstance(d,dict) else len(d))
import json as j
s=j.dumps(d,ensure_ascii=False)
print(s[:700])
"
echo ==SMOKE-JSON-SAMPLE
head -c 500 $R/archive/cleanup_20261008/outputs_smoke_202609/adapter_v6_smoke_single.json
echo
echo ==RESCUE-LOGS
tail -n 12 $R/archive/add_picture_edit_rescue_20260921/smoke_test_outputs/smoke_test.log
echo ---
head -n 8 $R/archive/add_picture_edit_rescue_20260921/smoke_test_outputs/edit_official.log
echo ==HOME-SYMLINKS
find /home/shangguanrz -maxdepth 1 -type l -printf '%p -> %l\n' 2>/dev/null | grep -i archive
ls -la /home/shangguanrz | grep -- ' -> ' | head -20
echo ==HOME-TOOLS-OPS
python3 -c "
import json
d=json.load(open('$R/archive/home_tools_202609/operations.json'))
print(type(d), list(d.keys())[:10] if isinstance(d,dict) else len(d))
print(json.dumps(d,ensure_ascii=False)[:600])
"
echo ==SCRATCH-COUNT
ls $R/archive/home_tools_202609/scratch | wc -l
ls $R/archive/home_tools_202609/scratch | tail -45
echo ==DU-HOME
du -sh --si $R/archive/home_tools_202609/scratch $R/archive/home_tools_202609/historical_shell
echo ==DOCS-CLEANUP
ls $R/docs | grep -i 清理
