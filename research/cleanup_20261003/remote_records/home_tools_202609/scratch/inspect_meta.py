import json

d = json.load(open('/tmp/qwen21.json'))
sib = d['siblings']
print('sibling keys sample:', list(sib[0].keys()))
for s in sib:
    if s['rfilename'].endswith('vae/diffusion_pytorch_model.safetensors'):
        print(json.dumps(s, indent=2, ensure_ascii=False))
        break
print('---- top-level keys ----')
print(sorted(d.keys()))
