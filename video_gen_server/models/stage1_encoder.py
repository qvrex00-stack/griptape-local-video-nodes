"""
MiniMax-H3 Stage 1 Encoder (Isolated Subprocess)
R2V / FL2VA 공용 멀티모달 인코더

실행 방법:
    python stage1_encoder.py <prompt_file> <out_file> <model_path> [ref_json_file]

ref_json_file: R2V 모드일 때 참조 파일 목록 JSON
  [{"type": "image", "path": "/path/to/img.jpg"}, {"type": "video", "path": "/path/to/vid.mp4"}]
  없거나 빈 경우 텍스트 전용 (fl2va 호환)

[2026-09-09 BugFix]
  - text_tag / video_tag 상수를 getattr(pipe, ...) 대신 diffusers 공식 상수에서 직접 임포트.
    ModularPipeline 인스턴스에는 text_tag / video_tag property가 없으므로 getattr fallback 값이
    잘못 사용되었음:
      이전 코드: text_tag = getattr(pipe, "text_tag", 0)  → 0 반환 (WRONG: TEXT_TAG=1)
                 video_tag = getattr(pipe, "video_tag", 1) → 1 반환 (WRONG: VIDEO_TAG=0)
    결과: 모든 텍스트 토큰이 VIDEO(0)로 태깅되고 vision 토큰이 TEXT(1)로 태깅되어
          token_tags 전체가 뒤집힌 상태로 transformer에 전달 → 체크무늬 아티팩트 발생.
"""
import os
import sys
import gc
import json
import torch

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
# [FIX] getattr(pipe, ...) 대신 공식 상수를 직접 임포트
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_TEXT_TAG,   # = 1
    MINIMAX_H3_VIDEO_TAG,  # = 0
)


def main():
    if len(sys.argv) < 3:
        print("Usage: stage1_encoder.py <prompt_file> <out_file> [model_path] [ref_json_file]", file=sys.stderr)
        sys.exit(1)

    prompt_file = sys.argv[1]
    out_file = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
    ref_json_file = sys.argv[4] if len(sys.argv) > 4 else None

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    # reference 목록 로드
    references_raw = []
    if ref_json_file and os.path.exists(ref_json_file):
        with open(ref_json_file, "r", encoding="utf-8") as f:
            references_raw = json.load(f)

    has_references = bool(references_raw)
    workflow = "ref2va" if has_references else "fl2va"

    print(f"[stage1_encoder] workflow={workflow}, has_references={has_references}", file=sys.stderr)
    # [FIX] 올바른 상수 값 출력으로 검증
    print(f"[stage1_encoder] Using: MINIMAX_H3_TEXT_TAG={MINIMAX_H3_TEXT_TAG}, MINIMAX_H3_VIDEO_TAG={MINIMAX_H3_VIDEO_TAG}", file=sys.stderr)

    import time
    t0 = time.time()
    def stage_log(msg):
        print(f"[stage1_encoder {time.time()-t0:.1f}s] {msg}", file=sys.stderr, flush=True)

    cm = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        model_path,
        workflow=workflow,
        components_manager=cm,
        local_files_only=True
    )
    if torch.cuda.is_available():
        free_vram_gb = int(torch.cuda.get_device_properties(0).total_memory / (1024**3)) - 3
        gpu_budget = f"{max(8, free_vram_gb)}GiB"
        stage_log(f"Loading text_encoder on GPU with gpu_budget={gpu_budget}...")
        pipe.load_components(
            names=["text_encoder"],
            device_map="auto",
            max_memory={0: gpu_budget, "cpu": "30GiB"},
            low_cpu_mem_usage=True
        )
        pipe.load_components(names=["tokenizer", "processor"])
        device = next(pipe.text_encoder.parameters()).device
    else:
        stage_log("CUDA not available. Loading text_encoder on CPU...")
        pipe.load_components(names=["text_encoder", "tokenizer", "processor"], low_cpu_mem_usage=True)
        device = torch.device("cpu")

    stage_log(f"text_encoder loaded successfully on {device} (dtype={pipe.text_encoder.dtype})")

    if not has_references:
        # ─── 텍스트 전용 (t2va / fl2va) ─────────────────────────────────────
        prompt_ids = pipe.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # [FIX] getattr(pipe, "text_tag", 0) 이었던 것을 공식 상수로 교체
        # MINIMAX_H3_TEXT_TAG = 1 (텍스트 행은 1로 태깅)
        token_tags = [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        stage_log(f"Running text-only forward pass ({len(prompt_ids)} tokens)...")
        with torch.inference_mode():
            prompt_embeds = get_qwen3vl_prompt_embeds(
                pipe.text_encoder,
                pipe.processor,
                prompt_ids,
                device=device
            ).cpu()

        text_token_tags = torch.tensor(token_tags, dtype=torch.long)

    else:
        # ─── R2V 멀티모달 인코딩 (ref2va) ──────────────────────────────────
        import numpy as np
        from PIL import Image
        import cv2
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VATextEncoderStep,
        )

        MAX_REF_DIM = 768  # Qwen3-VL multimodal reference max dimension

        ref_objects = []
        for ref in references_raw:
            t = ref.get("type", "image")
            p = ref.get("path")
            if not p or not os.path.exists(p):
                stage_log(f"WARNING: ref path not found: {p}")
                continue
            if t == "image":
                stage_log(f"Loading image reference: {p}")
                raw_img = MiniMaxH3ImageReference.from_file(p)
                img = raw_img.image
                orig_w, orig_h = img.size
                if max(orig_w, orig_h) > MAX_REF_DIM:
                    scale = min(1.0, MAX_REF_DIM / max(orig_w, orig_h))
                    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    stage_log(f"Smart-resized image from ({orig_w}, {orig_h}) to ({new_w}, {new_h}) (max_dim={MAX_REF_DIM})")
                ref_objects.append(MiniMaxH3ImageReference(image=img))
            elif t == "video":
                stage_log(f"Loading video reference: {p}")
                raw_vid = MiniMaxH3VideoReference.from_file(p)
                frames = raw_vid.frames
                f_h, f_w = frames.shape[1:3]
                if max(f_w, f_h) > MAX_REF_DIM:
                    scale = min(1.0, MAX_REF_DIM / max(f_w, f_h))
                    vid_w, vid_h = int(round(f_w * scale)), int(round(f_h * scale))
                    frames = np.stack([
                        cv2.resize(f, (vid_w, vid_h), interpolation=cv2.INTER_AREA) for f in frames
                    ])
                    stage_log(f"Smart-resized video frames from ({f_w}, {f_h}) to ({vid_w}, {vid_h}) (max_dim={MAX_REF_DIM})")
                ref_objects.append(MiniMaxH3VideoReference(
                    frames=frames,
                    fps=raw_vid.fps,
                    audio=getattr(raw_vid, "audio", None),
                    sample_rate=getattr(raw_vid, "sample_rate", None),
                ))
            elif t == "audio":
                stage_log(f"Loading audio reference: {p}")
                ref_objects.append(MiniMaxH3AudioReference.from_file(p))

        if not ref_objects:
            stage_log("ERROR: No valid references loaded")
            sys.exit(2)

        MINIMAX_H3_FPS = 24.0

        # reference normalization: video frames → 24fps, image 그대로
        normalized_refs = []
        for ref in ref_objects:
            if ref.kind == "video":
                src_fps = float(getattr(ref, "fps", MINIMAX_H3_FPS))
                frames = ref.frames  # (N, H, W, 3) uint8
                if abs(src_fps - MINIMAX_H3_FPS) > 0.1:
                    src_count = frames.shape[0]
                    duration_sec = src_count / src_fps
                    target_count = max(1, int(round(duration_sec * MINIMAX_H3_FPS)))
                    indices = np.round(np.linspace(0, src_count - 1, target_count)).astype(int)
                    frames = frames[indices]
                normalized_refs.append(MiniMaxH3VideoReference(
                    frames=frames,
                    fps=MINIMAX_H3_FPS,
                    audio=getattr(ref, "audio", None),
                    sample_rate=getattr(ref, "sample_rate", None),
                ))
            else:
                normalized_refs.append(ref)

        encoder_step = MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=2.0)

        # [FIX] getattr(pipe, "text_tag", 0) / getattr(pipe, "video_tag", 1) → 공식 상수 사용
        # 이전 코드의 fallback 값이 잘못됨:
        #   getattr(pipe, "text_tag", 0)  → 0 반환 (WRONG, TEXT_TAG=1)
        #   getattr(pipe, "video_tag", 1) → 1 반환 (WRONG, VIDEO_TAG=0)
        text_tag = MINIMAX_H3_TEXT_TAG   # 1
        video_tag = MINIMAX_H3_VIDEO_TAG  # 0

        # vision features 수집
        vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps = (
            encoder_step._gather_vision_features(
                pipe.processor,
                normalized_refs,
                MINIMAX_H3_FPS,
            )
        )

        # presentation 토큰 구성 (참조 레이블 + 텍스트 프롬프트 → input_ids + token_tags)
        token_ids, token_tags_list = encoder_step._build_presentation(
            pipe.tokenizer,
            prompt,
            normalized_refs,
            image_token_counts,
            video_block_token_counts,
            video_block_timestamps,
            text_tag=text_tag,   # 1 (올바른 값)
            video_tag=video_tag,  # 0 (올바른 값)
        )

        stage_log(f"R2V multimodal tokens: {len(token_ids)} total (img_tokens={image_token_counts}, vid_tokens={video_block_token_counts})")
        stage_log(f"vision_inputs keys: {list(vision_inputs.keys())}")
        stage_log(f"Starting Qwen3-VL forward pass ({len(token_ids)} tokens)...")

        t_fwd = time.time()
        with torch.inference_mode():
            prompt_embeds = get_qwen3vl_prompt_embeds(
                pipe.text_encoder,
                pipe.processor,
                token_ids,
                vision_inputs,
                text_encoder_layer=getattr(pipe, "text_encoder_layer", 50),
                device=device,
                dtype=pipe.text_encoder.dtype,
            ).cpu()

        stage_log(f"Qwen3-VL forward pass completed in {time.time()-t_fwd:.2f}s!")
        text_token_tags = torch.tensor(token_tags_list, dtype=torch.long)

    print(f"[stage1_encoder] prompt_embeds shape: {prompt_embeds.shape}", file=sys.stderr)
    print(f"[stage1_encoder] text_token_tags shape: {text_token_tags.shape}", file=sys.stderr)
    # [검증] text_tag 분포 출력
    unique_tags, counts = text_token_tags.unique(return_counts=True)
    for tag, cnt in zip(unique_tags.tolist(), counts.tolist()):
        tag_name = {MINIMAX_H3_TEXT_TAG: "TEXT", MINIMAX_H3_VIDEO_TAG: "VIDEO"}.get(tag, f"UNKNOWN({tag})")
        print(f"[stage1_encoder] tag {tag} ({tag_name}): {cnt} tokens", file=sys.stderr)

    torch.save({"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}, out_file)
    del pipe, cm
    gc.collect()
    print("STAGE1_SUCCESS", flush=True)


if __name__ == "__main__":
    main()
