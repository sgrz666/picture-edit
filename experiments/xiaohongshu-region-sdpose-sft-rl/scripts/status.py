"""Read the one-shot queue and committed artifacts without loading CUDA."""
import argparse
import json
from pathlib import Path

from prepare import read
from waiting import process_snapshot,dependency_active


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--launcher',default='automatic-v1');a=p.parse_args();root=a.root
    request=read(root/'launch_request.json');pidfile=root/f'logs/{a.launcher}.pid.json'
    launcher=read(pidfile) if pidfile.exists() else None
    current=None if launcher is None else process_snapshot(launcher['pid'])
    dep=process_snapshot(request['dependency']['pid'])
    result={'run_root':str(root),'launcher':launcher,'launcher_alive':current is not None and current['state'] not in ('Z','X'),
        'dependency':request['dependency'],'dependency_still_running':dependency_active(request['dependency'],dep),
        'counts':read(root/'data/selection.json')['pilot_counts'],'budgets':request['budgets'],'followup_updates':request['followup_updates'],
        'prepared':(root/'prepared.json').is_file(),'cache_ready':(root/'cache_ready.json').is_file(),
        'test_comparison_ready':(root/'report/test-comparison-ready.json').is_file(),
        'pipeline_complete':(root/'pipeline-complete.json').is_file()}
    events=root/'logs/events.jsonl'
    result['recent_events']=list(map(json.loads,events.read_text().splitlines()[-8:])) if events.exists() else []
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
