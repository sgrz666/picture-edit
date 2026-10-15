import numpy as np
H = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'

for tag in ('A', 'B'):
    p = f'{H}/native_interaction_v2/dwpose_native/pair00/hug00/000108_{tag}.npz'
    d = np.load(p, allow_pickle=True)
    xy = d['xy_native']; sc = d['scores']
    print('=' * 70)
    print(f'{tag}:  dwpose 自带 width={int(d["width"])} height={int(d["height"])}  generator={d["generator"]}')
    print(f'  xy_native 范围: x[{xy[:,0].min():.1f},{xy[:,0].max():.1f}] y[{xy[:,1].min():.1f},{xy[:,1].max():.1f}]')
    print(f'  高置信(>0.3)点数: {(sc>0.3).sum()} / {len(sc)}')
    print('  前 25 个点的 (x, y, score):')
    for i in range(25):
        print(f'    {i:3d}  x={xy[i,0]:8.1f} y={xy[i,1]:8.1f} s={sc[i]:.3f}')
