import os
import sys
import time
from huggingface_hub import hf_hub_download, list_repo_files

# Configure stdout/stderr encoding to utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Target directory on D: drive
dst_dir = r"D:\MiniMax-H3-FP8\transformer_ref"
os.makedirs(dst_dir, exist_ok=True)

repo_id = "abenliao/minimax-h3-ref2va-transformer-mxfp8"

print(f"=== Starting Download of transformer_ref (ref2va) from {repo_id} ===")

# List files from the repository
print("Fetching files list from Hugging Face repository...")
try:
    all_files = list_repo_files(repo_id=repo_id)
except Exception as e:
    print(f"[ERROR] Failed to list files from {repo_id}: {e}")
    sys.exit(1)

# Filter files (exclude gitattributes, README, etc. if you want, but download index and safetensors)
download_files = [f for f in all_files if f.endswith(".json") or f.endswith(".safetensors")]

if not download_files:
    print(f"[ERROR] No valid model files found in repository {repo_id}!")
    sys.exit(1)

print(f"Found {len(download_files)} model files to download. Starting download...")
t_start = time.time()

for idx, filename in enumerate(download_files):
    print(f"\n[{idx+1}/{len(download_files)}] Downloading: {filename}...")
    
    try:
        # Download file directly to D:\MiniMax-H3-FP8\transformer_ref
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dst_dir,
            local_dir_use_symlinks=False
        )
        print(f"[OK] Successfully downloaded/verified {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        sys.exit(1)

elapsed = time.time() - t_start
print(f"\n=== DOWNLOAD SUCCESS ===")
print(f"All transformer_ref (ref2va) components downloaded successfully to {dst_dir} in {elapsed:.1f}s.")
