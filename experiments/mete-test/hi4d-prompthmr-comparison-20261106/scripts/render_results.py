"""从已有预测与 PromptHMR 新产物绘制原图对照，所有图片独占创建。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2

SMPL12 = [16,17,18,19,20,21,1,2,4,5,7,8]
ANNY12 = [50,76,52,78,54,80,2,22,4,24,6,26]
LIMBS = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
COLORS = np.array([[22,156,255],[255,135,37],[209,77,231],[38,204,157],[246,207,37]])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def font(size):
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf']:
        if Path(name).exists():
            return ImageFont.truetype(name,size)
    return ImageFont.load_default()


def save(image,path):
    with Path(path).open('xb') as stream:
        image.save(stream,format='PNG')


def skeleton(image,people,cw,ch):
    data=np.asarray(image.resize((cw,ch))).copy()
    for i,xy in enumerate(people):
        points=np.asarray(xy)*[cw/image.width,ch/image.height]
        color=tuple(map(int,COLORS[i%len(COLORS)]))
        for a,b in LIMBS:
            if np.isfinite(points[[a,b]]).all() and abs(points[[a,b]]).max()<10000:
                pa,pb=np.rint(points[[a,b]]).astype(int)
                cv2.line(data,tuple(pa),tuple(pb),color,3,cv2.LINE_AA)
        for x,y in points:
            if np.isfinite([x,y]).all() and 0<=x<cw and 0<=y<ch:
                cv2.circle(data,(round(x),round(y)),5,(255,255,255),-1,cv2.LINE_AA)
                cv2.circle(data,(round(x),round(y)),3,color,-1,cv2.LINE_AA)
    return Image.fromarray(data)


def rasterize(vertices,faces,K,width,height):
    depth=np.full((height,width),np.inf)
    ns_image=np.zeros((height,width,3),np.float32)
    owners=np.full((height,width),-1,np.int8)
    for person,verts in enumerate(vertices):
        triangles=verts[faces]
        cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals=np.zeros_like(verts)
        for j in range(3):
            np.add.at(normals,faces[:,j],cross)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        uv=verts@K.T; uv=uv[:,:2]/uv[:,2:]
        for face in faces:
            xy=uv[face]; z=verts[face,2]
            if z.min()<=0:
                continue
            low=np.maximum(np.floor(xy.min(0)).astype(int),[0,0])
            high=np.minimum(np.ceil(xy.max(0)).astype(int),[width-1,height-1])
            if (low>high).any():
                continue
            x0,y0=xy[0]; x1,y1=xy[1]; x2,y2=xy[2]
            den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            if abs(den)<1e-10:
                continue
            xx,yy=np.meshgrid(np.arange(low[0],high[0]+1)+.5,np.arange(low[1],high[1]+1)+.5)
            a=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
            b=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den; c=1-a-b
            dz=1/np.maximum(a/z[0]+b/z[1]+c/z[2],1e-12)
            region=np.s_[low[1]:high[1]+1,low[0]:high[0]+1]
            visible=(a>=-1e-6)&(b>=-1e-6)&(c>=-1e-6)&(dz<depth[region])
            if not visible.any():
                continue
            n=np.stack([a/z[0],b/z[1],c/z[2]],-1)@normals[face]
            n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
            depth[region][visible]=dz[visible]
            ns_image[region][visible]=n[visible]
            owners[region][visible]=person
    return ns_image,owners


def strip(views,labels,detail,cw,ch):
    header=96
    result=Image.new('RGB',(cw*len(views),ch+header),'#f6f8fb')
    draw=ImageDraw.Draw(result)
    for col,(view,(title,subtitle)) in enumerate(zip(views,labels)):
        x=col*cw
        draw.text((x+10,8),title,font=font(23),fill='#172333')
        draw.text((x+10,40),subtitle,font=font(16),fill='#3c4b5b')
        draw.text((x+10,67),detail,font=font(17),fill='#3c4b5b')
        result.paste(view,(x,header))
    return result


def stack(rows):
    output=Image.new('RGB',(rows[0].width,sum(row.height for row in rows)))
    y=0
    for row in rows:
        output.paste(row,(0,y)); y+=row.height
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--models-root',type=Path,default=Path('/data/shangguanrz/models/two-man/benchmark_results'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    records=read(args.run_dir/'request.json')['records']
    with (args.models_root/'M2_MultiHMR_672/predictions.pkl').open('rb') as stream:
        m2=pickle.load(stream)
    with (args.models_root/'M4_MultiHMR2/predictions.pkl').open('rb') as stream:
        m4=pickle.load(stream)
    cw,ch=376,512
    boards=[]; meshes=[]
    for record in records:
        sample=record['id']; image=Image.open(record['image']).convert('RGB')
        baseline=read(args.run_dir/'box-only'/sample/'result.json')
        p2=[np.asarray(p['j2d_orig'])[SMPL12] for p in m2[sample]['predictions']]
        s=read(args.models_root/'M3_Sapiens2_detr'/f'{sample}.json')
        p3=[p['xy'] for p in s['instances']]
        resized=ImageOps.contain(image,(768,768)); rw,rh=resized.size
        pw,ph=int(np.ceil(rw/16))*16,int(np.ceil(rh/16))*16
        offset=np.array([round((pw-rw)/2),round((ph-rh)/2)])
        assert np.allclose(m4[sample]['pred_K'][:2,2],[pw/2,ph/2],atol=1)
        p4=[(np.asarray(p)[ANNY12]-offset)/[rw/image.width,rh/image.height] for p in m4[sample]['j2d']]
        detail=f'{record["category"]} | {sample[:8]}'
        board=strip([image.resize((cw,ch)),skeleton(image,baseline['body12_xy_original'],cw,ch),
                     skeleton(image,p2,cw,ch),skeleton(image,p3,cw,ch),skeleton(image,p4,cw,ch)],
                    [('RGB input','Same original image'),('PromptHMR','Multi-HMR boxes / true K'),
                     ('Multi-HMR 672',f'{len(p2)} raw detections'),('Sapiens2 Pose 5B',f'2D / cached DETR / {len(p3)} persons'),
                     ('Multi-HMR2',f'Anny / {len(p4)} raw detections')],detail,cw,ch)
        save(board,args.output_dir/f'{record["category"]}-four-models.png'); boards.append(board)
        views=[image.resize((cw,ch))]; normals=[]
        for arm in ['box-only','interaction']:
            with np.load(args.run_dir/arm/sample/'geometry.npz') as geo:
                K=np.asarray(geo['original_K']).copy(); K[0]*=cw/image.width; K[1]*=ch/image.height
                ns,owner=rasterize(geo['vertices'].astype(np.float64),geo['faces'],K,cw,ch)
            rgb=np.asarray(image.resize((cw,ch))).copy(); valid=owner>=0
            colors=COLORS[np.maximum(owner,0)]
            rgb[valid]=(.45*rgb[valid]+.55*colors[valid]).astype(np.uint8)
            views.append(Image.fromarray(rgb))
            encoded=np.clip((ns*np.array([1,-1,-1])+1)*127.5,0,255).astype(np.uint8)
            encoded[~valid]=0; normals.append(Image.fromarray(encoded))
        mesh=strip(views+normals,[('RGB input','Same original image'),('PromptHMR box-only','Interaction = false'),
                   ('PromptHMR interaction','Interaction = true'),('Box-only normal','From fitted SMPL-X'),
                   ('Interaction normal','From fitted SMPL-X')],detail,cw,ch)
        save(mesh,args.output_dir/f'{record["category"]}-mesh-normal.png'); meshes.append(mesh)
    save(stack(boards),args.output_dir/'four-models-all.png')
    save(stack(boards[2:]),args.output_dir/'four-models-contact.png')
    save(stack(meshes),args.output_dir/'prompthmr-mesh-normal-all.png')
    save(stack(meshes[2:]),args.output_dir/'prompthmr-mesh-normal-contact.png')
    print(json.dumps({'samples':len(records),'images':len(list(args.output_dir.glob('*.png')))}),flush=True)


if __name__=='__main__':
    main()
