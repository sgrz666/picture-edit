#!/bin/bash
set -u
D=/data/shangguanrz/pic-edit-datasets/TikTokDataset
R=$D/TikTok_Raw_Videos/TikTok_Raw_Videos
echo "===== E1 ffprobe 26 个源视频 ====="
for x in $R/seq_*/YouTube.mp4; do
  g=$(basename $(dirname $x))
  sz=$(stat -c '%s' $x)
  info=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,profile,width,height,r_frame_rate,nb_frames,pix_fmt -show_entries format=duration,size,bit_rate -of default=noprint_wrappers=1 $x | tr '\n' ' ')
  echo "$g | bytes=$sz | $info"
done
echo "===== E2 辅助文本内容 ====="
for x in $R/seq_00001_00009 $R/seq_00117_00124 $R/seq_00329_00340; do
  echo "--- $x ---"
  for f in $x/dance_name.txt $x/video_link.txt; do
    [ -f "$f" ] && { echo "## $(basename $f) ($(stat -c %s $f) B)"; cat "$f"; echo; }
  done
done
echo "-- dance_name.txt 是否所有组都有 --"
ls $R/seq_*/dance_name.txt | wc -l
ls $R/seq_*/video_link.txt 2>/dev/null | wc -l
echo "===== E3 每组的帧数合计 vs 视频时长 ====="
python3 - <<'PY'
import os,re,subprocess,json
R='/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_Raw_Videos/TikTok_Raw_Videos'
N='/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_dataset/TikTok_dataset'
allframes={}
for sid in os.listdir(N):
    if re.match(r'^\d{5}$',sid):
        allframes[sid]=len([f for f in os.listdir(os.path.join(N,sid,'images')) if f.endswith('.png')])
tot=0
for g in sorted(os.listdir(R)):
    p=os.path.join(R,g,'YouTube.mp4')
    if not os.path.exists(p): continue
    m=re.match(r'seq_(\d{5})_(\d{5})',g)
    a,b=int(m.group(1)),int(m.group(2))
    keys=['%05d'%i for i in range(a,b+1)]
    present=[k for k in keys if k in allframes]
    s=sum(allframes[k] for k in present)
    tot+=s
    out=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height,codec_name','-show_entries','format=duration','-of','json',p],capture_output=True,text=True).stdout
    j=json.loads(out); st=j['streams'][0]; dur=float(j['format']['duration'])
    print(g, 'range',a,b,'seqs_present',len(present),'missing',[k for k in keys if k not in allframes],
          'frames',s,'video',f"{st['width']}x{st['height']}",st['codec_name'],st.get('r_frame_rate'),
          'nb_frames',st.get('nb_frames'),'dur',round(dur,2),'fps_calc',round(s/dur,3))
print('TOTAL frames covered by these 26 videos', tot, 'sum of all frame dirs', sum(allframes.values()))
PY
