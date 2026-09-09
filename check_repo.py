from huggingface_hub import HfApi
api = HfApi()
for repo in ['deepgenteam/DeepGen-1.0-diffusers', 'deepgenteam/DeepGen-1.0']:
    try:
        files = api.list_repo_files(repo)
        print(f"Repo: {repo}, files: {len(files)}")
        for f in files[:15]:
            print("  ", f)
    except Exception as e:
        print(f"Error for {repo}: {e}")