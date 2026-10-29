import json, os, collections
import safetensors.torch as st
base = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
for sub in ("transformer", "transformer_ref"):
    p = os.path.join(base, sub)
    idx = json.load(open(os.path.join(p, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    print("=====", sub, "keys:", len(idx))
    suffix = collections.Counter(k.rsplit(".", 1)[-1] for k in idx)
    print(suffix)
    # dtype by key category, scanning shard 1 and a block shard
    shards = sorted(set(idx.values()))
    dt = collections.Counter()
    ex = {}
    for s in shards[:2]:
        with st.safe_open(os.path.join(p, s), framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                cat = ".".join(x for x in k.split(".") if not x.isdigit())
                key = (cat, sl.get_dtype())
                dt[key] += 1
                ex.setdefault(key, (k, sl.get_shape()))
    for (cat, d), n in sorted(dt.items()):
        print(d, n, cat, ex[(cat, d)][1])
    print("non-block keys:", [k for k in idx if not k.startswith("transformer_blocks.")][:80])
print(open(os.path.join(base, "transformer_ref", "config.json")).read())
print(open(os.path.join(base, "transformer", "config.json")).read())
