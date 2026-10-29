import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys, io
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

"""
Griptape Video Generation Server
Wan2.1-Fun-Control + AnimateDiff local inference server

Port: 8090
Usage: python server.py

[2026-08-04] 개선:
- WanGenerateRequest에 depth_video_path / pose_video_path / depth_weight / pose_weight 분리 추가
- RTX 5090 최적화 (wan_handler 참조)
"""

import base64
from typing import Optional, List, Dict
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_gen_server")

_wan_pipeline = None       # 1.3B
_wan_last_used = 0
_wan_14b_pipeline = None   # 14B
_wan_14b_last_used = 0
_wan_inp_pipeline = None
_wan_inp_last_used = 0
_animatediff_pipeline = None
_animatediff_last_used = 0
_hunyuan_i2v_pipeline = None  # HunyuanVideo 1.5 I2V
_hunyuan_i2v_last_used = 0
_minimax_loaded = False        # MiniMax-H3 FL2VA
_minimax_last_used = 0
_cache_lock = threading.Lock()

UNLOAD_TIMEOUT = 120  # 120초 (2분)


def unload_idle_models():
    while True:
        time.sleep(15)
        now = time.time()
        with _cache_lock:
            global _wan_pipeline, _wan_last_used
            global _animatediff_pipeline, _animatediff_last_used
            global _hunyuan_i2v_pipeline, _hunyuan_i2v_last_used
            global _minimax_loaded, _minimax_last_used
            try:
                import torch
                if _wan_pipeline is not None and now - _wan_last_used > UNLOAD_TIMEOUT:
                    _wan_pipeline.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] Wan pipeline offloaded to CPU")
            except Exception:
                pass
            try:
                import torch
                if _wan_inp_pipeline is not None and now - _wan_inp_last_used > UNLOAD_TIMEOUT:
                    _wan_inp_pipeline.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] WanInp pipeline offloaded to CPU")
            except Exception:
                pass
            try:
                import torch
                if _animatediff_pipeline is not None and now - _animatediff_last_used > UNLOAD_TIMEOUT:
                    _animatediff_pipeline.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] AnimateDiff pipeline offloaded to CPU")
            except Exception:
                pass
            try:
                import torch
                if _hunyuan_i2v_pipeline is not None and now - _hunyuan_i2v_last_used > UNLOAD_TIMEOUT:
                    _hunyuan_i2v_pipeline.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] HunyuanVideo pipeline offloaded to CPU")
            except Exception:
                pass
            try:
                from models import minimax_handler
                if _minimax_loaded and now - _minimax_last_used > UNLOAD_TIMEOUT:
                    minimax_handler.unload()
                    _minimax_loaded = False
                    logger.info("[Auto-unload] MiniMax-H3 pipeline unloaded (120s timeout)")
            except Exception as e:
                logger.warning(f"[Auto-unload] MiniMax-H3 unload error: {e}")


# ── Request / Response Models ──────────────────────────────────────────────────

class WanGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted"
    # ── 기존 단일 control 경로 (하위 호환) ──
    control_video_path: str = ""      # depth/pose 합산 비디오 경로 (레거시)
    # ── 분리된 depth / pose 입력 (권장) ──
    depth_video_path: str = ""        # depth 비디오 경로
    pose_video_path: str = ""         # pose 비디오 경로
    depth_weight: float = 0.7         # depth 제어 강도 (0.0~1.0)
    pose_weight: float = 0.5          # pose 제어 강도 (0.0~1.0)
    # ── 기타 옵션 ──
    clip_image_path: str = ""         # CLIP 스타일 레퍼런스 이미지 (선택사항)
    start_image_path: str = ""        # 첫 프레임 고정 이미지 (선택사항)
    model_size: str = "1.3B"          # "1.3B" | "14B"
    num_frames: int = 16
    fps: int = 8
    width: int = 480
    height: int = 832
    guidance_scale: float = 7.5       # Wan 계열 권장값 상향 (기존 6.0)
    num_inference_steps: int = 20
    seed: int = -1


class WanInpaintRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted"
    ref_image_path: str = ""       # 스타일 이미지 경로 (FLUX로 변환된 첫 프레임)
    control_video_path: str = ""   # depth_video 경로
    num_frames: int = 16
    fps: int = 8
    width: int = 480
    height: int = 832
    guidance_scale: float = 6.0
    num_inference_steps: int = 20
    seed: int = -1


class AnimateDiffRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted, ugly, bad anatomy"
    pose_video_path: str = ""      # pose 비디오 경로
    num_frames: int = 16
    fps: int = 8
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    seed: int = -1


class HunyuanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry"
    num_frames: int = 61
    fps: int = 24
    width: int = 720
    height: int = 480
    num_inference_steps: int = 10
    seed: int = -1


class MiniMaxGenerateRequest(BaseModel):
    prompt: str
    generation_mode: str = "Auto"    # Auto, T2V, I2V, FL2VA, R2V
    first_frame_path: str = ""        # I2V용 첫 프레임 이미지 경로 (비워두면 T2V)
    last_frame_path: str = ""         # FL2VA용 마지막 프레임 이미지 경로 (선택)
    reference_paths: Optional[list[dict]] = None # R2V용 다중 레퍼런스 경로 목록 (선택)
    width: int = 960
    height: int = 544
    duration_seconds: float = 5.0     # 생성 시간 (5~15초)
    num_inference_steps: int = 30
    flow_shift: Optional[float] = None # 커스텀 shift (None이면 스텝수에 맞춰 자동 최적화)
    lora_path: Optional[str] = None   # LoRA safetensors 파일 경로
    lora_scale: Optional[float] = 1.0 # LoRA 강도 배율
    seed: int = -1


class HunyuanI2VRequest(BaseModel):
    prompt: str
    image_path: str
    negative_prompt: str = "low quality, blurry"
    num_frames: int = 61
    fps: int = 24
    width: int = 720
    height: int = 480
    num_inference_steps: int = 10
    seed: int = -1


class VideoGenResponse(BaseModel):
    video_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    info: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def frames_to_video(frames, output_path: str, fps: float):
    """PIL Image 리스트를 H264 비디오로 저장"""
    import cv2
    import numpy as np
    import subprocess
    import shutil

    if not frames:
        raise ValueError("No frames to encode")

    w, h = frames[0].size
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-crf", "23", output_path],
            capture_output=True, timeout=300
        )
        os.remove(tmp_path)
        if result.returncode != 0:
            shutil.move(tmp_path, output_path)
    except Exception as e:
        logger.warning(f"ffmpeg unavailable ({e}), using mp4v")
        if os.path.exists(tmp_path):
            shutil.move(tmp_path, output_path)


def extract_frames_from_video(video_path: str):
    """비디오에서 PIL Image 프레임 추출"""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import torch
        logger.info("=" * 50)
        logger.info("Griptape Video Generation Server Starting... (port 8090)")
        logger.info(f"torch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"Auto-unload timeout: {UNLOAD_TIMEOUT}s")
        logger.info("=" * 50)
    except Exception as e:
        logger.warning(f"torch info unavailable: {e}")
    threading.Thread(target=unload_idle_models, daemon=True).start()
    yield
    logger.info("Video gen server shutting down...")


app = FastAPI(
    title="Griptape Video Generation Server",
    description="Wan2.1-Fun-Control + AnimateDiff local video generation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    try:
        import torch
        return {
            "status": "ok",
            "port": 8090,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "wan_loaded": _wan_pipeline is not None,
            "animatediff_loaded": _animatediff_pipeline is not None,
            "hunyuan_loaded": _hunyuan_i2v_pipeline is not None,
            "minimax_loaded": _minimax_loaded,
            "unload_timeout_sec": UNLOAD_TIMEOUT,
        }
    except Exception:
        return {"status": "ok", "port": 8090, "torch_available": False}


@app.post("/cancel")
@app.post("/unload")
def cancel_and_unload():
    """즉시 백그라운드 작업을 중단하고 모든 모델 VRAM/RAM을 소각합니다."""
    global _minimax_loaded, _minimax_last_used
    try:
        try:
            from models import minimax_handler
        except ImportError:
            import minimax_handler
        minimax_handler.unload()
        _minimax_loaded = False
        _minimax_last_used = 0
    except Exception as e:
        logger.warning(f"Error unloading minimax: {e}")

    try:
        global _wan_pipeline, _wan_14b_pipeline, _wan_inp_pipeline, _animatediff_pipeline, _hunyuan_i2v_pipeline
        _wan_pipeline = None
        _wan_14b_pipeline = None
        _wan_inp_pipeline = None
        _animatediff_pipeline = None
        _hunyuan_i2v_pipeline = None
    except Exception as e:
        logger.warning(f"Error unloading other pipelines: {e}")

    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[Server] VRAM and RAM successfully purged.")
    return {"status": "ok", "message": "All background tasks cancelled and memory purged."}


@app.post("/server/stop")
def server_stop():
    """서버를 안전하게 종료하는 엔드포인트."""
    logger.info("[VideoServer] Shutdown requested via /server/stop")
    import signal
    import os
    import threading

    def _shutdown():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_shutdown, daemon=True).start()
    return {"status": "shutting_down"}


@app.get("/server/status")
def server_status():
    status_info = {
        "status": "running",
        "port": 8090,
        "minimax_loaded": False,
        "minimax_status": None
    }
    try:
        from models import minimax_handler
        status_info["minimax_loaded"] = minimax_handler.is_loaded()
        status_info["minimax_status"] = minimax_handler.get_status()
    except Exception as e:
        status_info["error"] = str(e)
    return status_info

@app.post("/wan/generate", response_model=VideoGenResponse)
def wan_generate(req: WanGenerateRequest):
    """Wan2.1-Fun-Control 영상 생성 (1.3B 또는 14B)"""
    from models.wan_handler import WanHandler
    global _wan_pipeline, _wan_last_used
    global _wan_14b_pipeline, _wan_14b_last_used

    try:
        import torch
        import random
        device = get_device()
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        model_size = req.model_size if req.model_size in ("1.3B", "14B") else "1.3B"

        with _cache_lock:
            if model_size == "14B":
                if _wan_14b_pipeline is None:
                    logger.info("Loading Wan2.1-Fun-Control 14B pipeline...")
                    _wan_14b_pipeline = WanHandler.load(device, model_size="14B")
                    logger.info("Wan 14B pipeline loaded.")
                else:
                    try:
                        _wan_14b_pipeline.to(device)
                    except Exception:
                        pass
                _wan_14b_last_used = time.time()
                pipeline = _wan_14b_pipeline
            else:
                if _wan_pipeline is None:
                    logger.info("Loading Wan2.1-Fun-Control 1.3B pipeline...")
                    _wan_pipeline = WanHandler.load(device, model_size="1.3B")
                    logger.info("Wan 1.3B pipeline loaded.")
                else:
                    try:
                        _wan_pipeline.to(device)
                    except Exception:
                        pass
                _wan_last_used = time.time()
                pipeline = _wan_pipeline

        logger.info(f"Wan [{model_size}] generate: '{req.prompt[:50]}...', frames={req.num_frames}, seed={seed}")

        output_path = WanHandler.generate(
            pipeline=pipeline,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            # 기존 단일 경로 (하위 호환)
            control_video_path=req.control_video_path,
            # 분리된 depth / pose (신규)
            depth_video_path=req.depth_video_path,
            pose_video_path=req.pose_video_path,
            depth_weight=req.depth_weight,
            pose_weight=req.pose_weight,
            clip_image_path=req.clip_image_path,
            start_image_path=req.start_image_path,
            num_frames=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            seed=seed,
            device=device,
        )

        logger.info(f"Wan [{model_size}] generation complete: {output_path}")
        return VideoGenResponse(
            video_path=output_path,
            frame_count=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            info=f"seed={seed}, steps={req.num_inference_steps}, model={model_size}"
        )
    except Exception as e:
        logger.error(f"Wan generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/animatediff/generate", response_model=VideoGenResponse)
def animatediff_generate(req: AnimateDiffRequest):
    """AnimateDiff + ControlNet 영상 생성"""
    from models.animatediff_handler import AnimateDiffHandler
    global _animatediff_pipeline, _animatediff_last_used

    try:
        import torch
        import random
        device = get_device()
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

        with _cache_lock:
            if _animatediff_pipeline is None:
                logger.info("Loading AnimateDiff pipeline...")
                _animatediff_pipeline = AnimateDiffHandler.load(device)
                logger.info("AnimateDiff pipeline loaded.")
            else:
                try:
                    _animatediff_pipeline.to(device)
                except Exception:
                    pass
            _animatediff_last_used = time.time()

        logger.info(f"AnimateDiff generate: '{req.prompt[:50]}...', frames={req.num_frames}, seed={seed}")

        output_path = AnimateDiffHandler.generate(
            pipeline=_animatediff_pipeline,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            pose_video_path=req.pose_video_path,
            num_frames=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            seed=seed,
            device=device,
        )

        logger.info(f"AnimateDiff generation complete: {output_path}")
        return VideoGenResponse(
            video_path=output_path,
            frame_count=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            info=f"seed={seed}, steps={req.num_inference_steps}"
        )
    except Exception as e:
        logger.error(f"AnimateDiff generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wan/generate_inp", response_model=VideoGenResponse)
def wan_generate_inp(req: WanInpaintRequest):
    """Wan2.1-Fun-InP: 스타일 이미지 + depth_video → 스타일 유지 영상 생성"""
    from models.wan_inpaint_handler import WanInpaintHandler
    global _wan_inp_pipeline, _wan_inp_last_used

    try:
        import torch
        import random
        device = get_device()
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

        with _cache_lock:
            if _wan_inp_pipeline is None:
                logger.info("Loading Wan2.1-Fun-InP pipeline...")
                _wan_inp_pipeline = WanInpaintHandler.load(device)
                logger.info("Wan InP pipeline loaded.")
            else:
                try:
                    _wan_inp_pipeline.to(device)
                except Exception:
                    pass
            _wan_inp_last_used = time.time()

        logger.info(f"Wan InP generate: '{req.prompt[:50]}...', frames={req.num_frames}, seed={seed}")
        logger.info(f"  ref_image={req.ref_image_path}")
        logger.info(f"  control_video={req.control_video_path}")

        output_path = WanInpaintHandler.generate(
            pipeline=_wan_inp_pipeline,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            ref_image_path=req.ref_image_path,
            control_video_path=req.control_video_path,
            num_frames=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            seed=seed,
            device=device,
        )

        logger.info(f"Wan InP generation complete: {output_path}")
        return VideoGenResponse(
            video_path=output_path,
            frame_count=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            info=f"seed={seed}, steps={req.num_inference_steps}, inp"
        )
    except Exception as e:
        logger.error(f"Wan InP generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hunyuan/i2v", response_model=VideoGenResponse)
def hunyuan_i2v(req: HunyuanI2VRequest):
    """HunyuanVideo 1.5 Image-to-Video 생성 (distilled)"""
    from models.hunyuan_handler import HunyuanHandler
    global _hunyuan_i2v_pipeline, _hunyuan_i2v_last_used

    try:
        import torch
        import random
        device = get_device()
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

        with _cache_lock:
            # 타 모델 메모리 오프로드 실행 (OOM 방지)
            _unload_all_except_hunyuan()

            if _hunyuan_i2v_pipeline is None:
                logger.info("Loading HunyuanVideo 1.5 I2V pipeline...")
                _hunyuan_i2v_pipeline = HunyuanHandler.load_i2v(device)
                logger.info("HunyuanVideo 1.5 I2V pipeline loaded.")
            else:
                try:
                    _hunyuan_i2v_pipeline.to(device)
                except Exception:
                    pass
            _hunyuan_i2v_last_used = time.time()

        output_path = HunyuanHandler.generate_i2v(
            pipeline=_hunyuan_i2v_pipeline,
            prompt=req.prompt,
            image_path=req.image_path,
            num_frames=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            seed=seed,
            device=device,
            negative_prompt=req.negative_prompt,
        )

        # 5090 용량 관리를 위해 사용 후 즉시 CPU 반환 (옵션)
        # _hunyuan_i2v_pipeline.to("cpu")
        # torch.cuda.empty_cache()

        return VideoGenResponse(
            video_path=output_path,
            frame_count=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            info=f"seed={seed}, steps={req.num_inference_steps}, hunyuan_i2v"
        )
    except Exception as e:
        logger.error(f"HunyuanVideo I2V failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hunyuan/t2v", response_model=VideoGenResponse)
def hunyuan_t2v(req: HunyuanT2VRequest):
    """HunyuanVideo 1.5 Text-to-Video 생성"""
    from models.hunyuan_handler import HunyuanHandler
    global _hunyuan_i2v_pipeline, _hunyuan_i2v_last_used

    try:
        import torch
        import random
        device = get_device()
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

        with _cache_lock:
            _unload_all_except_hunyuan()

            if _hunyuan_i2v_pipeline is None:
                logger.info("Loading HunyuanVideo 1.5 pipeline...")
                _hunyuan_i2v_pipeline = HunyuanHandler.load_i2v(device)
            else:
                try:
                    _hunyuan_i2v_pipeline.to(device)
                except Exception:
                    pass
            _hunyuan_i2v_last_used = time.time()

        output_path = HunyuanHandler.generate_t2v(
            pipeline=_hunyuan_i2v_pipeline,
            prompt=req.prompt,
            num_frames=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            seed=seed,
            device=device,
            negative_prompt=req.negative_prompt,
        )

        return VideoGenResponse(
            video_path=output_path,
            frame_count=req.num_frames,
            fps=req.fps,
            width=req.width,
            height=req.height,
            info=f"seed={seed}, steps={req.num_inference_steps}, hunyuan_t2v"
        )
    except Exception as e:
        logger.error(f"HunyuanVideo T2V failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _unload_all_except_hunyuan():
    """HunyuanVideo 메모리 확보를 위해 기존 모델들을 CPU로 오프로드"""
    global _wan_pipeline, _wan_14b_pipeline, _wan_inp_pipeline, _animatediff_pipeline
    import torch
    logger.info("[MemoryManager] Offloading other pipelines to CPU to prevent VRAM overflow")
    try:
        if _wan_pipeline is not None:
            _wan_pipeline.to("cpu")
        if _wan_14b_pipeline is not None:
            _wan_14b_pipeline.to("cpu")
        if _wan_inp_pipeline is not None:
            _wan_inp_pipeline.to("cpu")
        if _animatediff_pipeline is not None:
            _animatediff_pipeline.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Offloading failed: {e}")



# ── MiniMax-H3 Endpoints ───────────────────────────────────────────────────────

@app.post("/server/stop")
def stop_server():
    """Gracefully stop server"""
    logger.info("Stopping server process...")
    import threading
    threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
    return {"status": "stopping"}


@app.post("/minimax/generate", response_model=VideoGenResponse)
def minimax_generate(req: MiniMaxGenerateRequest):
    """MiniMax-H3 FL2VA 영상 생성 (T2V / I2V / FL2VA)"""
    import random
    from models import minimax_handler

    global _minimax_loaded, _minimax_last_used

    try:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

        # Set last used time to infinity to prevent background thread auto-unloads
        # during the long loading process or the generation process.
        _minimax_last_used = float('inf')

        with _cache_lock:
            if not minimax_handler.is_loaded():
                logger.info("Loading MiniMax-H3 FL2VA pipeline...")
                minimax_handler.load()
                _minimax_loaded = True
                logger.info("MiniMax-H3 pipeline loaded.")

        result = minimax_handler.generate(
            prompt=req.prompt,
            generation_mode=getattr(req, "generation_mode", "Auto"),
            first_frame_path=req.first_frame_path if req.first_frame_path else None,
            last_frame_path=req.last_frame_path if req.last_frame_path else None,
            reference_paths=req.reference_paths,
            width=req.width,
            height=req.height,
            duration_seconds=req.duration_seconds,
            num_inference_steps=req.num_inference_steps,
            flow_shift=getattr(req, "flow_shift", None),
            lora_path=getattr(req, "lora_path", None),
            lora_scale=getattr(req, "lora_scale", 1.0) if getattr(req, "lora_scale", None) is not None else 1.0,
            seed=seed,
        )

        # Set last used time to current time now that generation is finished
        _minimax_last_used = time.time()

        return VideoGenResponse(
            video_path=result["video_path"],
            frame_count=result["frame_count"],
            fps=result["fps"],
            width=result["width"],
            height=result["height"],
            info=result.get("info", ""),
        )

    except Exception as e:
        logger.error(f"MiniMax-H3 generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090, log_level="info")

