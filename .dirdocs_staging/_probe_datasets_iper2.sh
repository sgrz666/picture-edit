D=/home/shangguanrz/project/pic-edit/datasets/iPER
echo "== summary.json"; cat $D/iper_sampled_6src64tgt/summary.json
echo "== smplestx"; cat $D/iper_sampled_6src64tgt/smplestx_stage2_summary.json
echo "== wc pairs"; wc -l $D/iper_sampled_6src64tgt/pairs.jsonl $D/iper_sampled_6src64tgt/samples.jsonl
echo "== split_summary full"; cat $D/iper_sampled_6src64tgt/splits/split_summary.json
echo "== splits wc"; wc -l $D/iper_sampled_6src64tgt/splits/*.jsonl $D/iper_sampled_6src64tgt/splits/*appearances.txt
echo "== baseline indices count"; python3 -c "import json;d=json.load(open('$D/iper_sampled_6src64tgt/splits/baseline_3000_microstep_indices.json'));print(type(d),len(d))"
echo "== appearance txt counts"; python3 -c "
import json
d=json.load(open('$D/iper_sampled_6src64tgt/024_6/manifest.json'))
print(list(d.keys()))
print(json.dumps(d,indent=1)[:1500])
"
echo "== pairs.jsonl first line keys"
python3 -c "
import json
l=open('$D/iper_sampled_6src64tgt/pairs.jsonl').readline()
d=json.loads(l)
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+'.'+k)
    elif isinstance(o,list): print(p,'list len',len(o))
    else:
        s=str(o)
        print(p,'=',s[:90])
walk(d)
"
echo "== samples.jsonl first line"
python3 -c "
import json
l=open('$D/iper_sampled_6src64tgt/samples.jsonl').readline()
d=json.loads(l)
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+'.'+k)
    elif isinstance(o,list): print(p,'list len',len(o))
    else:
        s=str(o)
        print(p,'=',s[:90])
walk(d)
"
