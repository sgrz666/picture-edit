import numpy as np
H = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
G = np.load(f'{H}/final_assets_native_v1/pair00/hug00/cam4/geometry/000108.npz', allow_pickle=True)
smpl = np.load(f'{H}/raw/pair00/hug00/smpl/000108.npz', allow_pickle=True)
cams = np.load(f'{H}/raw/pair00/hug00/cameras/rgb_cameras.npz', allow_pickle=True)
ids = cams['ids']; idx = int(np.where(ids == 4)[0][0])
K = G['K_runtime'].astype(np.float64); E = cams['extrinsics'][idx].astype(np.float64)
J = smpl['joints_3d'].astype(np.float64)
W, Hh = 940, 1280
sem_A = G['semantic_A']; sem_B = G['semantic_B']; front = G['front_owner']

def project(X):
    Xh = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    xc = Xh @ E.T
    z = np.clip(xc[:, 2:3], 1e-6, None)
    return (xc[:, :2] / z) @ K[:2, :2].T + K[:2, 2]

for p in (0, 1):
    uv = project(J[p])
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, Hh - 1)
    a = int((sem_A[v, u] > 0).sum()); b = int((sem_B[v, u] > 0).sum())
    print(f'SMPL person {p}: 24 个关节中  落在 semantic_A 内 {a} 个, semantic_B 内 {b} 个')

print()
print('front_owner 唯一值:', np.unique(front).tolist(), ' 各值占比:',
      {int(k): round(float((front == k).mean()), 4) for k in np.unique(front)})
print('semantic_A 像素数:', int((sem_A > 0).sum()), ' semantic_B 像素数:', int((sem_B > 0).sum()))
# 两人是否重叠
ov = ((sem_A > 0) & (sem_B > 0)).sum()
print('A 与 B 掩码重叠像素:', int(ov))
