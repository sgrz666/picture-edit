import requests

url = "https://hf-mirror.com/deepgenteam/DeepGen-1.0-diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors"
r = requests.head(url, allow_redirects=True)
print("Final status:", r.status_code)
print("Final url prefix:", r.url[:60])
print("Content-Length:", r.headers.get("Content-Length"))
print("Accept-Ranges:", r.headers.get("Accept-Ranges"))