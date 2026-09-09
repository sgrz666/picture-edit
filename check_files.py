from huggingface_hub import HfApi
api = HfApi()
repo = "deepgenteam/DeepGen-1.0-diffusers"
files = api.list_repo_tree(repo, recursive=True)
total_size = 0
for f in files:
    size = getattr(f, "size", 0) or 0
    total_size += size
    if size > 1024 * 1024:
        print(f"{f.path:50} {size / (1024*1024):.1f} MB")
print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")