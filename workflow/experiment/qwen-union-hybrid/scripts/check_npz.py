import numpy as np
d = np.load("/home/shangguanrz/project/pic-edit/workflow/runs/test/hard_05_torso/geometry/B_keypoints.npz")
print("keys:", list(d.keys()))
for k in d.keys():
    print(k, d[k].shape)
