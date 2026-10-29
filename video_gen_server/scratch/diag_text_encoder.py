"""Is the FP8 (scale-less) Qwen3-VL text encoder still semantically intact?"""
import torch, time, json, os
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

P = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
tok = AutoTokenizer.from_pretrained(os.path.join(P, "tokenizer"))
t0 = time.time()
m = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(P, "text_encoder"), dtype=torch.bfloat16, low_cpu_mem_usage=True)
print("loaded", round(time.time() - t0), "s", flush=True)
has_head = m.lm_head.weight.abs().sum().item() > 0
print("lm_head nonzero:", has_head, "tied:", m.config.tie_word_embeddings, flush=True)

for prompt in ["The capital of France is", "1 + 1 = 2, 2 + 2 ="]:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        out = m.generate(**ids, max_new_tokens=6, do_sample=False)
    print(repr(prompt), "->", repr(tok.decode(out[0][ids["input_ids"].shape[1]:])), flush=True)

ids = tok("The capital of France is", return_tensors="pt", add_special_tokens=False)["input_ids"]
with torch.inference_mode():
    hs = m.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
for L in (10, 30, 50, 64):
    if L < len(hs):
        h = hs[L].float()
        print(f"layer {L}: std={h.std().item():.2f} rowRMS={h.pow(2).mean(-1).sqrt().squeeze().tolist()}", flush=True)
