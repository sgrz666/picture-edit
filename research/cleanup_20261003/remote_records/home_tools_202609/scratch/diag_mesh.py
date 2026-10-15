import numpy as np, torch
P = '/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/demo'
for n in ('pA', 'pB'):
    d = torch.load(f'{P}/{n}_params.pt', map_location='cpu', weights_only=False)
    v = d['mesh_cam'][0].numpy()
    W = int(d['width'][0,0]); H = int(d['height'][0,0])
    fx = float(d['focal_length_xy'][0,0]); fy = float(d['focal_length_xy'][0,1])
    cx0, cy0 = (float(x) for x in d['princpt'][0])
    print(f'=== {n}  image {W}x{H}  focal=({fx:.1f},{fy:.1f})  princpt=({cx0:.1f},{cy0:.1f})')
    print(f'  verts x[{v[:,0].min():.3f},{v[:,0].max():.3f}] y[{v[:,1].min():.3f},{v[:,1].max():.3f}] z[{v[:,2].min():.3f},{v[:,2].max():.3f}]')
    K = np.array([[fx,0,cx0],[0,fy,cy0],[0,0,1]], np.float32)
    pr = v @ K.T
    xy = pr[:,:2] / pr[:,2:3]
    print(f'  投影 xy x[{xy[:,0].min():.1f},{xy[:,0].max():.1f}] y[{xy[:,1].min():.1f},{xy[:,1].max():.1f}]')
    jp = d['joint_proj_model'][0].numpy()
    print(f'  joint_proj x[{jp[:,0].min():.1f},{jp[:,0].max():.1f}] y[{jp[:,1].min():.1f},{jp[:,1].max():.1f}]')
