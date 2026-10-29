import torch

cache_path = r"C:\Foundry\Griptape\video_gen_server\temp\prompt_cache_82bb9a94a1ee8048cc7704d4a736a7ff.pt"
data = torch.load(cache_path, map_location="cpu")
embeds = data['prompt_embeds']
tags = data['text_token_tags']
print("embeds shape:", embeds.shape, "dtype:", embeds.dtype, "mean:", embeds.float().mean().item(), "std:", embeds.float().std().item())
print("tags shape:", tags.shape, "dtype:", tags.dtype, "unique tags:", torch.unique(tags))
