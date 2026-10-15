import pathlib
p = pathlib.Path('/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/main/inference.py')
s = p.read_text(encoding='utf-8')
anchor = "                # transl: (1, 3)\n                all_results['transl'].append("
if anchor not in s:
    raise SystemExit('anchor not found')
add = (
    "                # [EXPORT] 渲染时实际使用的相机 —— 这才是与 mesh_cam 配套的那一组\n"
    "                all_results.setdefault('render_focal', []).append(\n"
    "                    torch.tensor([[focal[0], focal[1]]], dtype=torch.float32))\n"
    "                all_results.setdefault('render_princpt', []).append(\n"
    "                    torch.tensor([[princpt[0], princpt[1]]], dtype=torch.float32))\n"
    "                all_results.setdefault('render_bbox', []).append(\n"
    "                    torch.tensor([list(bbox)], dtype=torch.float32))\n\n"
)
s = s.replace(anchor, add + anchor, 1)
p.write_text(s, encoding='utf-8')
print('patched:', 'render_focal' in s)
