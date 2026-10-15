import io,os,json,zipfile,bisect,random,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
root=Path('/data/shangguanrz/pic-edit-datasets/SynBody');out=root/'inspection_20261003/visual_quality_20261003';out.mkdir(exist_ok=True)
rng=random.Random(20261003)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
records=[]
def save_sample(raw,subset,name,index):
    im=Image.open(io.BytesIO(raw));im.load();dest=out/f'{subset}_{index:02d}.jpg';dest.write_bytes(raw)
    records.append({'subset':subset,'archive_path':name,'local_copy':str(dest),'sha256':hashlib.sha256(raw).hexdigest(),'size':list(im.size)})
    return im

def sheet(ims,labels,stem):
    w,h=480,530;cols=4;rows=(len(ims)+cols-1)//cols;canvas=Image.new('RGB',(w*cols,h*rows),(245,245,245));d=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(ims,labels)):
        im=im.copy();im.thumbnail((480,480));x=(i%cols)*w;y=(i//cols)*h
        canvas.paste(im,(x+(w-im.width)//2,y+38+(480-im.height)//2));d.text((x+8,y+8),label,font=font,fill='black')
    canvas.save(out/f'{stem}.png')

hroot=root/'inspection_20261003/extracted/HumanNeRF-subset'
ids=rng.sample([f'{i:03d}' for i in range(100) if i not in [28,42]],12)
ims=[];labels=[]
for i,sid in enumerate(ids):
    n=f'{sid}/img/00/0200.jpg';ims.append(save_sample((hroot/n).read_bytes(),'HumanNeRF',n,i));labels.append(f'HumanNeRF ID {sid}, view00, frame200')
sheet(ims,labels,'HumanNeRF_contact_sheet')

class Parts(io.RawIOBase):
    def __init__(self,paths):
        self.fs=[open(p,'rb') for p in paths];self.off=[0];self.pos=0
        for p in paths:self.off.append(self.off[-1]+os.path.getsize(p))
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.off[-1]+n
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.off[-1]-self.pos,self.off[-1]-self.pos);chunks=[]
        while n>0:
            i=bisect.bisect_right(self.off,self.pos)-1;f=self.fs[i];f.seek(self.pos-self.off[i]);c=f.read(min(n,self.off[i+1]-self.pos))
            if not c:break
            chunks.append(c);self.pos+=len(c);n-=len(c)
        return b''.join(chunks)

ims=[];labels=[]
with zipfile.ZipFile(Parts(sorted((root/'SMPL-X_subset').glob('Synbody_v1_0_sampled.zip.[0-9]*')))) as z:
    names=sorted(n for n in z.namelist() if n.endswith('.jpeg'))
    for i,n in enumerate(rng.sample(names,8)):
        ims.append(save_sample(z.read(n),'SMPLX',n,i));labels.append('SMPL-X '+n.split('/')[2]+' / '+n.rsplit('/',1)[-1])
sheet(ims,labels,'SMPLX_contact_sheet')

ims=[];labels=[]
for part in range(1,5):
    with zipfile.ZipFile(root/f'SynBody-100K/S100K_rgb_part_{part}.zip') as z:
        names=sorted(n for n in z.namelist() if n.endswith('.jpeg'))
        for n in rng.sample(names,2):
            ims.append(save_sample(z.read(n),'S100K',n,len(ims)));labels.append('100K '+n.split('/')[0]+' / '+n.rsplit('/',1)[-1])
sheet(ims,labels,'S100K_contact_sheet')

with zipfile.ZipFile(root/'HumanNeRF-subset/img.zip') as z:
    n='028/img/00/0200.jpg';raw=z.read(n);copied=(hroot/n).read_bytes()
    integrity={'source_archive_path':n,'sha256_archive':hashlib.sha256(raw).hexdigest(),'sha256_extracted':hashlib.sha256(copied).hexdigest(),'byte_identical':raw==copied}
a=np.asarray(Image.open(hroot/'028/img/00/0200.jpg'),dtype=np.int16);c=np.asarray(Image.open(hroot/'028/img/00/0150.jpg'),dtype=np.int16)
ma=np.asarray(Image.open(hroot/'028/mask/00/0200.png'))>0;mc=np.asarray(Image.open(hroot/'028/mask/00/0150.png'))>0
bg=~(ma|mc);delta=np.abs(a-c).mean(axis=2)
stats={'outside_person_union_rgb_mae':float(delta[bg].mean()),'outside_person_union_fraction_rgb_delta_gt5':float((delta[bg]>5).mean()),'caveat':'Background temporal changes are descriptive only; not a physical shadow/ground-contact test.'}
(out/'sampling_manifest.json').write_text(json.dumps({'seed':20261003,'sampling':'12 other HumanNeRF subjects at view00/frame200; 8 sampled SMPLX frames; 2 random images per S100K RGB archive','samples':records,'source_integrity':integrity,'background_statistics':stats},indent=2))
print(json.dumps({'directory':str(out),'images':len(records),'ids':ids,'source_integrity':integrity,'background_statistics':stats},indent=2))
