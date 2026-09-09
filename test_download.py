import os
import sys

os.environ["http_proxy"] = "http://127.0.0.1:5674"
os.environ["https_proxy"] = "http://127.0.0.1:5674"

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="deepgenteam/DeepGen-1.0-diffusers",
    filename="model_index.json",
    local_dir="/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers",
)
print("Successfully downloaded:", path)