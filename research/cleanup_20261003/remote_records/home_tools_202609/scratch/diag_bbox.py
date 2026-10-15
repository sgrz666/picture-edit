import numpy as np, torch
SIZE = 512
for n in ('pA', 'pB'):
    d = torch.load(f'/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/demo/{n}_params.pt',
                   map_location='cpu', weights_only=False)
    v = d['mesh_cam'][0].numpy().astype(np.float64)
    W = int(d['width'][0,0]); H = int(d['height'][0,0])
    fx = float(d['render_focal'][0,0]); fy = float(d['render_focal'][0,1])
    cx, cy = (float(x) for x in d['render_princpt'][0])
    s = min(SIZE/W, SIZE/H)
    ow, oh = round(W*s), round(H*s)
    offx, offy = (SIZE-ow)//2, (SIZE-oh)//2
    u = v[:,0]/v[:,2]*fx + cx
    w2 = v[:,1]/v[:,2]*fy + cy
    u = u*s + offx; w2 = w2*s + offy
    print(f'=== {n}  image {W}x{H}  letterbox {ow}x{oh} off=({offx},{offy})')
    print(f'  顶点投影 u[{u.min():.0f},{u.max():.0f}] v[{w2.min():.0f},{w2.max():.0f}]  '
          f'-> bbox {u.max()-u.min():.0f} x {w2.max()-w2.min():.0f}')
    c = np.load(f'/home/shangguanrz/project/pic-edit/experiments/16-test/conditions_ab/{n}/conditions.npz')
    m = c['human_mask'] > 0.5
    ys, xs = np.where(m)
    print(f'  我的掩码 bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]  '
          f'-> {xs.max()-xs.min()} x {ys.max()-ys.min()}  像素={int(m.sum())}')
    inside = (u>=0)&(u<SIZE)&(w2>=0)&(w2<SIZE)
    print(f'  顶点落在画面内: {int(inside.sum())}/{len(u)}')
