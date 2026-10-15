import torch
for n in ('pA', 'pB'):
    d = torch.load(f'/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/demo/{n}_params.pt',
                   map_location='cpu', weights_only=False)
    if 'render_focal' in d:
        f = d['render_focal'][0].tolist(); c = d['render_princpt'][0].tolist()
        b = d['render_bbox'][0].tolist()
        print(f'  {n}: render_focal={[round(x,1) for x in f]}  render_princpt={[round(x,1) for x in c]}')
        print(f'       render_bbox={[round(x,1) for x in b]}')
    else:
        print(f'  {n}: 未导出（键={[k for k in d if "render" in k]}）')
