"""Lay out unmodified experiment images with run labels for visual review."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--group',default='Q0');p.add_argument('--task',required=True)
    args=p.parse_args();rows=[]
    for directory in sorted((args.root/'runs').iterdir()):
        if not (directory/'config.json').exists() or not (directory/'output.png').exists():continue
        c=json.loads((directory/'config.json').read_text())
        if c.get('group') in args.group.split(',') and c.get('task')==args.task:rows.append(directory)
    dest=args.root/'review_sheets';dest.mkdir(exist_ok=True)
    for start in range(0,len(rows),4):
        subset=rows[start:start+4];sheet=Image.new('RGB',(1024,550*((len(subset)+1)//2)),(240,240,240));draw=ImageDraw.Draw(sheet)
        for i,directory in enumerate(subset):
            x=(i%2)*512;y=(i//2)*550
            draw.text((x+8,y+10),directory.name,fill=(0,0,0))
            img=Image.open(directory/'output.png').convert('RGB');img.thumbnail((512,512))
            sheet.paste(img,(x,y+38))
        path=dest/f'{args.group.replace(",","-")}_{args.task}_{start//4}.png';sheet.save(path);print(path)
if __name__=='__main__':main()
