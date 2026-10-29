import os
import sys

# 윈도우 MAX_PATH 및 캐시 충돌 방지 환경 변수 설정
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

CACHE_DIR = "C:/Users/qvrex/.cache/huggingface/hub"
REPO_ID = "MiniMaxAI/MiniMax-H3"

print(f"[MiniMax] Starting download for: {REPO_ID}")
print(f"[MiniMax] Target Cache Directory: {CACHE_DIR}")
print("[MiniMax] Size estimate: ~65GB (safetensors format only)")
print("=" * 60)

try:
    from huggingface_hub import snapshot_download
    
    local_path = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=CACHE_DIR,
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.onnx", "onnx/*", "non_diffusers/*", "*pytorch_model*"],   # 무거운 레거시 포맷 차단
        local_files_only=False,
    )
    print(f"\n✅ [SUCCESS] MiniMax-H3 model cached successfully at: {local_path}")
    
except KeyboardInterrupt:
    print("\n⚠️ [CANCEL] Interrupted by user. Progress is saved and resumable.")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ [ERROR] Download failed: {e}")
    sys.exit(1)
