import numpy as np
from PIL import Image, ImageDraw

H = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
smpl = np.load(f'{H}/raw/pair00/hug00/smpl/000108.npz', allow_pickle=True)
cams = np.load(f'{H}/raw/pair00/hug00/cameras/rgb_cameras.npz', allow_pickle=True)
G = np.load(f'{H}/final_assets_native_v1/pair00/hug00/cam4/geometry/000108.npz', allow_pickle=True)

J = smpl['joints_3d'].astype(np.float64)          # (2,24,3)
ids = cams['ids']; K_all = cams['intrinsics']; E = cams['extrinsics']
print('camera ids:', ids.tolist())
K_rt = G['K_runtime'].astype(np.float64)
print('K_runtime:/n', K_rt)
idx = int(np.where(ids == 4)[0][0])
Kd = K_all[idx]; Ed = E[idx]
print('dataset cam4 K:/n', Kd); print('cam4 E:/n', Ed)

W, Hh = 940, 1280
rgb = Image.open(f'{H}/raw/pair00/hug00/images/4/000108.jpg').convert('RGB')
dr = ImageDraw.Draw(rgb)

def project(X, K, E):
    Xh = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)   # (N,4)
    xc = Xh @ E.T                                                # (N,3)
    z = np.clip(xc[:, 2:3], 1e-6, None)
    uv = (xc[:, :2] / z) @ K[:2, :2].T + K[:2, 2]
    return uv, xc[:, 2]

for tag, color, K_, E_ in (('K_runtime+E', (255, 60, 60), K_rt, Ed),
                           ('Kd + E', (60, 255, 60), Kd, Ed),
                           ('K_runtime, 无外参', (255, 255, 0), K_rt, np.hstack([np.eye(3), np.zeros((3,1))]))):
    uv, z = project(J[0], K_, E_)
    ok = (uv[:,0]>0)&(uv[:,0]<W)&(uv[:,1]>0)&(uv[:,1]<Hh)
    print(f'{tag}: 落在画面内的关节 {ok.sum()}/24   z范围 [{z.min():.2f},{z.max():.2f}]')
    print('   前 5 个 uv:', np.round(uv[:5],1).tolist())
    for i in range(24):
        if ok[i]:
            x, y = uv[i]
            dr.ellipse([x-6, y-6, x+6, y+6], outline=color, width=3)
            dr.text((x+8, y-8), str(i), fill=color)

rgb.save('/tmp/proj_check.png')
print('wrote /tmp/proj_check.png')
