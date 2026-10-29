import os
import sys
import shutil

# Configure stdout/stderr encoding to utf-8 if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Environment variables for max path and caching prevention
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Target directory inside models folder
TARGET_DIR = r"C:\Foundry\Griptape\video_gen_server\models\MiniMax-H3-FP8"
os.makedirs(TARGET_DIR, exist_ok=True)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed in this environment.")
    sys.exit(1)

print(f"=== MiniMax-H3 FP8 Download & Merge Script ===")
print(f"Target Directory: {TARGET_DIR}")
print("==============================================")

# Step 1: Download metadata and small folders from MiniMaxAI/MiniMax-H3
print("\n[Step 1] Downloading metadata and VAE/Schedulers from MiniMaxAI/MiniMax-H3...")
meta_patterns = [
    "model_index.json",
    "modular_model_index.json",
    "scheduler/*",
    "audio_scheduler/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*"
]
try:
    snapshot_download(
        repo_id="MiniMaxAI/MiniMax-H3",
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=meta_patterns,
        local_files_only=False
    )
    print("[OK] Metadata and VAE/Schedulers downloaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to download metadata: {e}")
    sys.exit(1)

# Step 2: Download FP8 weights from SabiQG/minimax-h3-fp8
print("\n[Step 2] Downloading FP8 weights from SabiQG/minimax-h3-fp8...")
fp8_patterns = [
    "text_encoder_fp8/*",
    "transformer_fp8/*"
]
try:
    snapshot_download(
        repo_id="SabiQG/minimax-h3-fp8",
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=fp8_patterns,
        local_files_only=False
    )
    print("[OK] FP8 weights downloaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to download FP8 weights: {e}")
    sys.exit(1)

# Step 3: Rename folders to fit standard Diffusers model structure
print("\n[Step 3] Renaming folders to standard Diffusers structure...")
rename_mappings = [
    ("text_encoder_fp8", "text_encoder"),
    ("transformer_fp8", "transformer")
]

for src_name, dest_name in rename_mappings:
    src_path = os.path.join(TARGET_DIR, src_name)
    dest_path = os.path.join(TARGET_DIR, dest_name)
    
    if os.path.exists(src_path):
        if os.path.exists(dest_path):
            print(f"Target directory {dest_name} already exists. Removing it first...")
            shutil.rmtree(dest_path)
        shutil.move(src_path, dest_path)
        print(f"[OK] Renamed {src_name} -> {dest_name}")
    else:
        if os.path.exists(dest_path):
            print(f"[OK] {dest_name} already exists.")
        else:
            print(f"[WARNING] {src_name} not found and {dest_name} does not exist.")

print("\n==============================================")
print("SUCCESS: MiniMax-H3 FP8 model is successfully assembled!")
print(f"Location: {TARGET_DIR}")
print("==============================================")
