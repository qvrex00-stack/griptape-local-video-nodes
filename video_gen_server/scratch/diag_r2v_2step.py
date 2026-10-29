"""2-step R2V diagnostic: reuses the cached prompt embeds, prints per-modality activation stats."""
import os, sys, glob, hashlib, shutil
sys.path.insert(0, r"C:\Foundry\Griptape\video_gen_server")
from models import minimax_handler as mh

REF = r"C:\Foundry\Griptape\inputs\videos\runAnim_shot_1.mov"
prompt = "diag"
refs = [{"type": "video", "path": REF}]
key = tuple((r["type"], r["path"]) for r in refs)
h = hashlib.md5((prompt + "ref2va" + str(key)).encode("utf-8")).hexdigest()
src = os.path.join(mh.TEMP_DIR, "prompt_cache_82bb9a94a1ee8048cc7704d4a736a7ff.pt")
dst = os.path.join(mh.TEMP_DIR, f"prompt_cache_{h}.pt")
if not os.path.exists(dst):
    shutil.copy(src, dst)

res = mh.generate(prompt=prompt, generation_mode="R2V", reference_paths=refs,
                  width=640, height=544, duration_seconds=5.0,
                  num_inference_steps=2, seed=1)
print("DONE", res)
