import os
import sys
import argparse
import torch
import traceback

def encode_prompt_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    try:
        print(f"[PromptWorker] Loading text_encoder from {args.model_path}...", flush=True)
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

        cm = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            args.model_path,
            workflow="fl2va",
            components_manager=cm,
            local_files_only=True
        )
        pipe.load_components(names=["text_encoder", "tokenizer", "processor"], dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if torch.cuda.is_available():
            pipe.text_encoder.to("cuda")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"[PromptWorker] Encoding prompt on {device}: '{args.prompt[:30]}...'...", flush=True)
        prompt_ids = pipe.tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
        text_tag = getattr(pipe, "text_tag", 0)
        token_tags = [text_tag] * len(prompt_ids)

        with torch.no_grad():
            embeds = get_qwen3vl_prompt_embeds(
                pipe.text_encoder,
                pipe.processor,
                prompt_ids,
                device=device
            )
        text_token_tags = torch.tensor(token_tags, dtype=torch.long)

        # Free text_encoder from VRAM/RAM immediately
        del pipe.text_encoder
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[PromptWorker] Saving prompt embeds (shape={embeds.shape}) and text_token_tags (shape={text_token_tags.shape}) to {args.output_path}...", flush=True)
        torch.save({"prompt_embeds": embeds.cpu(), "text_token_tags": text_token_tags.cpu()}, args.output_path)
        print("[PromptWorker] Done!", flush=True)
    except Exception as e:
        print(f"[PromptWorker ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    encode_prompt_main()
