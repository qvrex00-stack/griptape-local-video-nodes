import os
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
MiniMax-H3 Handler (FL2VA: T2V + I2V)
diffusers PR #14355 브랜치 사용 (ModularPipeline + ComponentsManager)

[로딩 전략 - RTX 5090 24GB + RAM 64GB]
- 동적 메모리 패치 대신, Diffusers 공식 Sequential CPU Offload 기능을 사용하여 35GB 대용량 모델을 레이어 단위로 자동 CPU <-> VRAM 스왑 처리.
- 에러 발생 시(500) 혹은 예외 시 전역 파이프라인 내부 레퍼런스를 전부 None으로 소각하여 메모리 누수를 물리적으로 원천 차단.
"""

import json
import logging
import os
import time
from typing import Optional
import gc
import sys

import torch
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds


_PROMPT_EMBEDS_CACHE = {} # Cache (prompt, mode) -> (prompt_embeds, text_token_tags)
logger = logging.getLogger("video_gen_server")

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MINIMAX_LOCAL_PATH = r"C:\Foundry\Griptape\models\MiniMax-H3-FP8"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

_pipeline = None
_components_manager = None
_last_used = 0.0
_current_status = {
    "state": "idle",
    "step": 0,
    "total_steps": 0,
    "progress_pct": 0.0,
    "step_time_sec": 0.0,
    "last_step_timestamp": 0.0,
    "start_timestamp": 0.0,
    "error_message": None
}


def _get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        t = torch.cuda.get_device_properties(0).total_memory
        free = (t - r) / 1024**3
        return free
    except Exception:
        return 0.0


def _duration_to_frames(duration: float, fps: int = 24) -> int:
    requested = int(round(duration * fps))
    # MiniMax-H3 requires num_frames between 120 and 360 (17 * k + 5)
    requested = max(requested, 120)
    k = 0
    while True:
        candidate = 17 * k + 5
        if candidate >= requested:
            return candidate
        k += 1


def is_loaded() -> bool:
    return _pipeline is not None


def load():
    global _pipeline, _components_manager
    if _pipeline is not None:
        logger.info("[MiniMax-H3] Pipeline already loaded.")
        return

    logger.info("Loading MiniMax-H3 FL2VA pipeline...")
    total_vram = 0.0
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"[MiniMax-H3] VRAM Total: {total_vram:.1f}GB")

    try:
        from diffusers import ComponentsManager, ModularPipeline
    except ImportError as e:
        raise ImportError(
            f"ModularPipeline import 실패: {e}\n"
            "pip install git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head"
        )

    logger.info(f"[MiniMax-H3] Initializing ComponentsManager and ModularPipeline from {MINIMAX_LOCAL_PATH} ...")
    t0 = time.time()

    _components_manager = ComponentsManager()
    _pipeline = ModularPipeline.from_pretrained(
        MINIMAX_LOCAL_PATH, 
        workflow="fl2va",
        components_manager=_components_manager,
        local_files_only=True
    )
    
    logger.info("[MiniMax-H3] Loading vae and audio_vae on CPU...")
    _pipeline.load_components(
        names=["vae", "audio_vae"],
        dtype=torch.float32
    )
    
    logger.info("[MiniMax-H3] Loading utility components...")
    _pipeline.load_components(
        names=["tokenizer", "processor", "scheduler", "audio_scheduler"]
    )

    elapsed = time.time() - t0
    free_after = _get_free_vram_gb()
    logger.info(
        f"[MiniMax-H3] Base pipeline initialized in {elapsed:.1f}s. VRAM free: {free_after:.1f}GB"
    )



def _dequantize_transformer_weights(tf_model, model_dir, subfolder):
    """
    Dequantize all MXFP8 / Block-FP8 weights in MiniMax-H3 transformer to exact bfloat16
    and directly bind them to _cpu_weight for lossless CPU-resident streaming.
    Fixes the 1000x scale mismatch and checkerboard noise artifact permanently!
    Includes explicit per-shard memory reclamation to avoid RAM saturation.
    """
    import glob, json, gc
    import safetensors.torch as st
    from torchao.prototype.mx_formats.mx_tensor import from_blocked, get_fp_scale
    
    sub_path = os.path.join(model_dir, subfolder)
    idx_path = os.path.join(sub_path, "diffusion_pytorch_model.safetensors.index.json")
    
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        shards = glob.glob(os.path.join(sub_path, "*.safetensors"))
        weight_map = {}
        for s in shards:
            s_name = os.path.basename(s)
            m = st.load_file(s)
            for k in m.keys():
                weight_map[k] = s_name

    shard_to_keys = {}
    for k, s_name in weight_map.items():
        shard_to_keys.setdefault(s_name, []).append(k)

    scale_keys = [k for k in weight_map.keys() if "scale" in k]
    if not scale_keys:
        logger.info(f"[MiniMax-H3/Dequantizer] {subfolder} has NO scale keys (already pure bfloat16). Skipping dequantization.")
        return

    shard_items = sorted(shard_to_keys.items())
    total_shards = len(shard_items)
    dequant_count = 0

    logger.info(f"[MiniMax-H3/Dequantizer] Starting unswizzled dequantization for {subfolder} ({total_shards} shards)...")

    for shard_idx, (s_name, keys) in enumerate(shard_items):
        shard_file = os.path.join(sub_path, s_name)
        logger.info(f"[MiniMax-H3/Dequantizer] [{shard_idx+1}/{total_shards}] Loading and processing {s_name} ({len(keys)} layers)...")
        shard_data = st.load_file(shard_file)
        
        for k in keys:
            if k.endswith(".mx_scale") or k.endswith(".fp8_scale"):
                continue
            mx_scale_k = f"{k}.mx_scale"
            fp8_scale_k = f"{k}.fp8_scale"
            
            w_bf16 = None
            if mx_scale_k in shard_data:
                w_fp8 = shard_data[k]
                scale_blocked = shard_data[mx_scale_k]
                rows, cols = w_fp8.shape
                scale_unblocked = from_blocked(scale_blocked, rows, cols // 32)
                s_fp = get_fp_scale(scale_unblocked).repeat_interleave(32, dim=1)
                w_bf16 = (w_fp8.to(torch.bfloat16).float() * s_fp).to(torch.bfloat16)
                dequant_count += 1
            elif fp8_scale_k in shard_data:
                w_fp8 = shard_data[k]
                scale = shard_data[fp8_scale_k]
                if scale.ndim == 2 and w_fp8.ndim == 2:
                    block_size = w_fp8.shape[1] // scale.shape[1]
                    w_bf16 = (w_fp8.float() / scale.repeat_interleave(block_size, dim=1)).to(torch.bfloat16)
                elif scale.ndim == 1:
                    w_bf16 = (w_fp8.float() / scale.unsqueeze(0)).to(torch.bfloat16)
                else:
                    w_bf16 = (w_fp8.float() / scale).to(torch.bfloat16)
                dequant_count += 1
            elif k in shard_data and (k.endswith(".weight") or k.endswith(".bias")):
                w_bf16 = shard_data[k].to(torch.bfloat16)
                
            if w_bf16 is not None:
                parts = k.split(".")
                curr = tf_model
                for p in parts[:-1]:
                    if p.isdigit():
                        curr = curr[int(p)]
                    else:
                        curr = getattr(curr, p, None)
                    if curr is None:
                        break
                if curr is not None:
                    attr = parts[-1]
                    if hasattr(curr, attr):
                        param = getattr(curr, attr)
                        if isinstance(param, torch.nn.Parameter):
                            param.data = w_bf16
                        else:
                            setattr(curr, attr, w_bf16)
                    # Directly bind dequantized tensor to _cpu_weight / _cpu_bias
                    setattr(curr, f"_cpu_{attr}", w_bf16.to(device="cpu", dtype=torch.bfloat16))
                del w_bf16

        # Crucial: clean up shard temporary dictionary immediately to prevent RAM explosion
        del shard_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"[MiniMax-H3/Dequantizer] Successfully dequantized {dequant_count} FP8 weight layers to exact bfloat16 and bound to CPU streaming!")


def _fuse_lora_into_transformer(tf_model, lora_path: Optional[str] = None, lora_scale: float = 1.0):
    """
    Zero-VRAM, Zero-Latency In-place LoRA Weight Fusion for MiniMax-H3 Transformer.
    - Directly merges delta_W = scale * (alpha / rank) * (up @ down) into module._cpu_weight.
    - Preserves original base weights in module._orig_cpu_weight for 0.01s instant swapping or restore.
    - Supports Diffusers format (transformer.transformer_blocks...lora_A/B)
      and ComfyUI / Kohya format (lora_unet_...lora_down/up, diffusion_model...).
    """
    if tf_model is None:
        return

    # Check current fused state: (path, scale)
    current_fused = getattr(tf_model, "_current_fused_lora", None)
    target_state = (lora_path, float(lora_scale)) if (lora_path and os.path.exists(lora_path) and float(lora_scale) != 0.0) else None

    # If the requested LoRA and scale match what is already fused, do nothing!
    if current_fused == target_state:
        if target_state is not None:
            logger.info(f"[MiniMax-H3/LoRA] LoRA state unchanged ({target_state[0]}, scale={target_state[1]:.2f}). Skipping re-fusion.")
        return

    # If another LoRA was previously fused, restore original weights first!
    if current_fused is not None:
        logger.info(f"[MiniMax-H3/LoRA] Restoring original base weights before applying new LoRA (previous: {current_fused})...")
        restore_count = 0
        for mod in tf_model.modules():
            if hasattr(mod, "_orig_cpu_weight") and mod._orig_cpu_weight is not None:
                mod._cpu_weight = mod._orig_cpu_weight.clone()
                restore_count += 1
        tf_model._current_fused_lora = None
        logger.info(f"[MiniMax-H3/LoRA] Restored {restore_count} layers to pure base weights.")

    # If no new LoRA is requested, we are done
    if target_state is None:
        logger.info("[MiniMax-H3/LoRA] No LoRA requested (pure base model active).")
        return

    # Load and fuse new LoRA
    import safetensors.torch as st
    logger.info(f"[MiniMax-H3/LoRA] Loading and fusing LoRA from {lora_path} (scale={lora_scale:.2f})...")
    t0 = time.time()

    try:
        lora_data = st.load_file(lora_path)
    except Exception as e:
        logger.error(f"[MiniMax-H3/LoRA] Failed to read safetensors file {lora_path}: {e}")
        return

    # Map named modules and underscore versions
    named_mods = dict(tf_model.named_modules())
    mod_underscore_map = {k.replace(".", "_"): k for k in named_mods.keys()}

    pairs = {}
    for k in lora_data.keys():
        down_suffix = None
        up_suffix = None
        for s in [".lora_A.weight", ".lora_down.weight", ".lora.down.weight", ".down.weight"]:
            if k.endswith(s):
                down_suffix = s
                break
        for s in [".lora_B.weight", ".lora_up.weight", ".lora.up.weight", ".up.weight"]:
            if k.endswith(s):
                up_suffix = s
                break

        if down_suffix:
            prefix = k[:-len(down_suffix)]
            pairs.setdefault(prefix, {})["down"] = lora_data[k]
        elif up_suffix:
            prefix = k[:-len(up_suffix)]
            pairs.setdefault(prefix, {})["up"] = lora_data[k]
        elif k.endswith(".alpha"):
            prefix = k[:-len(".alpha")]
            try:
                pairs.setdefault(prefix, {})["alpha"] = float(lora_data[k].item())
            except Exception:
                pass

    fused_count = 0
    for prefix, pdata in pairs.items():
        if "down" not in pdata or "up" not in pdata:
            continue
        down = pdata["down"].float()
        up = pdata["up"].float()
        alpha = pdata.get("alpha", None)

        # Strip known prefixes
        clean_name = prefix
        for pfx in ["diffusion_model.", "transformer.", "lora_unet_"]:
            if clean_name.startswith(pfx):
                clean_name = clean_name[len(pfx):]

        # Try exact match or underscore match
        target_mod_name = None
        if clean_name in named_mods:
            target_mod_name = clean_name
        elif clean_name in mod_underscore_map:
            target_mod_name = mod_underscore_map[clean_name]

        if not target_mod_name:
            continue

        mod = named_mods[target_mod_name]
        if not hasattr(mod, "_cpu_weight") or mod._cpu_weight is None:
            continue

        r = down.shape[0] if down.shape[0] < down.shape[1] else down.shape[1]
        eff_alpha = alpha if alpha is not None else float(r)
        eff_scale = float(lora_scale) * (eff_alpha / float(r))

        if up.shape[1] == down.shape[0]:
            delta = up @ down
        elif up.shape[0] == down.shape[1]:
            delta = (down @ up).t()
        elif up.shape[0] == down.shape[0]:
            delta = up.t() @ down
        else:
            logger.warning(f"[MiniMax-H3/LoRA] Dimension mismatch for {target_mod_name}: up={up.shape}, down={down.shape}")
            continue

        if delta.shape != mod._cpu_weight.shape:
            # Check transpose
            if delta.t().shape == mod._cpu_weight.shape:
                delta = delta.t()
            else:
                logger.warning(f"[MiniMax-H3/LoRA] Delta shape {delta.shape} != target weight shape {mod._cpu_weight.shape} for {target_mod_name}")
                continue

        delta = delta * eff_scale

        # Backup original CPU weight if not already backed up
        if not hasattr(mod, "_orig_cpu_weight") or mod._orig_cpu_weight is None:
            mod._orig_cpu_weight = mod._cpu_weight.clone()

        # In-place merge into _cpu_weight
        mod._cpu_weight = (mod._orig_cpu_weight.float() + delta).to(dtype=torch.bfloat16)
        fused_count += 1

    del lora_data, pairs
    gc.collect()

    tf_model._current_fused_lora = target_state
    elapsed = time.time() - t0
    logger.info(
        f"[MiniMax-H3/LoRA] Successfully fused {fused_count} layers in {elapsed:.2f}s "
        f"(0MB VRAM used, 0s latency overhead)!"
    )


def generate(
    prompt: str,
    generation_mode: str = "Auto",
    first_frame_path: Optional[str] = None,
    last_frame_path: Optional[str] = None,
    reference_paths: Optional[list[dict]] = None,
    width: int = 960,
    height: int = 544,
    duration_seconds: float = 5.0,
    num_inference_steps: int = 30,
    flow_shift: Optional[float] = None,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    global _pipeline, _components_manager, _last_used
    progress_hook_handle = None

    try:
        if _pipeline is None:
            load()

        _last_used = time.time()

        # 1. generation_mode 명시적 선택 처리
        has_valid_references = False
        if reference_paths:
            for ref in reference_paths:
                p = ref.get("path")
                if p and os.path.exists(p):
                    has_valid_references = True
                    break

        g_mode = (generation_mode or "Auto").upper()

        if g_mode == "T2V":
            target_workflow = "fl2va"
            target_transformer = "transformer"
            unload_transformer = "transformer_ref"
            mode = "T2V"
            first_frame_path = None
            last_frame_path = None
            reference_paths = None
        elif g_mode == "R2V" or (g_mode == "AUTO" and has_valid_references):
            target_workflow = "ref2va"
            target_transformer = "transformer_ref"
            unload_transformer = "transformer"
            mode = "R2V"
        else:
            target_workflow = "fl2va"
            target_transformer = "transformer"
            unload_transformer = "transformer_ref"
            reference_paths = None

            if g_mode in ("I2V", "FL2VA"):
                mode = g_mode
            elif first_frame_path and last_frame_path:
                mode = "FL2VA"
            elif first_frame_path:
                mode = "I2V"
            else:
                mode = "T2V"

        from diffusers import ModularPipeline, ComponentsManager
        current_workflow = getattr(_pipeline, "workflow", "fl2va")
        if current_workflow != target_workflow or _components_manager is None:
            logger.info(f"[MiniMax-H3] Switching workflow from {current_workflow} to {target_workflow} (Fresh ComponentsManager)...")
            old_pipeline = _pipeline
            _pipeline = None
            if _components_manager is not None:
                try:
                    _components_manager.components.clear()
                    _components_manager.collections.clear()
                    _components_manager.added_time.clear()
                except Exception:
                    pass
            _components_manager = ComponentsManager()
            del old_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _pipeline = ModularPipeline.from_pretrained(
                MINIMAX_LOCAL_PATH, 
                workflow=target_workflow,
                components_manager=_components_manager,
                local_files_only=True
            )
            _pipeline.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
            _pipeline.load_components(names=["tokenizer", "processor", "scheduler", "audio_scheduler"])

        # Stage 1: text_encoder (Qwen3-VL) runs in isolated subprocess to ensure 100% OS RAM reclamation
        logger.info(f"[MiniMax-H3/Stage1] Encoding prompt in isolated process: '{prompt[:30]}...'...")
        import subprocess
        import tempfile

        import hashlib
        # R2V 모드: 참조 파일 경로도 cache_key에 포함 (텍스트 전용 캐시와 분리)
        ref_paths_key = tuple(
            (r.get("type", ""), r.get("path", ""))
            for r in (reference_paths or [])
            if r.get("path") and os.path.exists(r.get("path", ""))
        )
        cache_key = (prompt, target_workflow, ref_paths_key)
        prompt_hash = hashlib.md5(
            (prompt + target_workflow + str(ref_paths_key)).encode("utf-8")
        ).hexdigest()
        disk_cache_file = os.path.join(TEMP_DIR, f"prompt_cache_{prompt_hash}.pt")

        if cache_key in _PROMPT_EMBEDS_CACHE:
            logger.info("[MiniMax-H3/Stage1] Reusing in-memory cached prompt embeds (Instant Stage 1!).")
            prompt_embeds, text_token_tags = _PROMPT_EMBEDS_CACHE[cache_key]
        elif os.path.exists(disk_cache_file) and os.path.getsize(disk_cache_file) > 0:
            logger.info(f"[MiniMax-H3/Stage1] Reusing disk-cached prompt embeds from {disk_cache_file} (Instant Stage 1!).")
            data = torch.load(disk_cache_file, weights_only=True)
            prompt_embeds = data["prompt_embeds"]
            text_token_tags = data["text_token_tags"]
            _PROMPT_EMBEDS_CACHE[cache_key] = (prompt_embeds, text_token_tags)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as pf:
                pf.write(prompt)
                prompt_tmp = pf.name

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as ef:
                embeds_tmp = ef.name

            # R2V 모드: 참조 파일 목록을 JSON으로 임시 파일에 저장
            ref_json_tmp = None
            if target_workflow == "ref2va" and reference_paths:
                import json as _json
                valid_refs = [
                    r for r in reference_paths
                    if r.get("path") and os.path.exists(r.get("path", ""))
                    and r.get("type") in ("image", "video", "audio")
                ]
                if valid_refs:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as rjf:
                        _json.dump(valid_refs, rjf, ensure_ascii=False)
                        ref_json_tmp = rjf.name
                    logger.info(f"[MiniMax-H3/Stage1] R2V refs written to {ref_json_tmp}: {[r['path'] for r in valid_refs]}")

            try:
                # 연속 실행 시 이전 실행에서 로드된 트랜스포머(35GB)가 RAM에 남아 있으면
                # Stage 1 텍스트 인코더 실행 시 총 메모리가 64GB를 초과하므로 먼저 메모리 완전 해제
                if _pipeline is not None and hasattr(_pipeline, target_transformer) and getattr(_pipeline, target_transformer) is not None:
                    logger.info(f"[MiniMax-H3/Stage1] Unloading {target_transformer} to free ~35GB RAM before Stage 1 prompt encoding...")
                    try:
                        _pipeline.unload_components(target_transformer)
                    except Exception as e_unl:
                        logger.warning(f"[MiniMax-H3/Stage1] unload_components error: {e_unl}, applying fallback purge...")
                        setattr(_pipeline, target_transformer, None)

                    if _components_manager is not None:
                        try:
                            keys_to_del = [k for k in _components_manager.components.keys() if target_transformer in k]
                            for k in keys_to_del:
                                _components_manager.components.pop(k, None)
                        except Exception:
                            pass

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    psapi = ctypes.windll.psapi
                    handle = kernel32.GetCurrentProcess()
                    psapi.EmptyWorkingSet(handle)
                except Exception:
                    pass
                encoder_script = os.path.join(os.path.dirname(__file__), "stage1_encoder.py")
                python_exe = (
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Scripts', 'python.exe')
                    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Scripts', 'python.exe'))
                    else sys.executable
                )
                cmd = [python_exe, encoder_script, prompt_tmp, embeds_tmp, MINIMAX_LOCAL_PATH]
                if ref_json_tmp:
                    cmd.append(ref_json_tmp)
                # R2V는 비디오 프레임 디코딩 + 멀티모달 인코딩이 포함되므로 timeout을 600초로 확장
                # 외장 HDD 등 느린 디스크에서 Qwen3-VL 텍스트 인코더를 처음 읽을 때 300초를 넘길 수 있어 기본값을 1800초로 확장
                # 환경변수 MINIMAX_STAGE1_TIMEOUT 으로 조정 가능
                stage1_timeout = int(os.environ.get("MINIMAX_STAGE1_TIMEOUT", "1800"))
                logger.info(f"[MiniMax-H3/Stage1] Running encoder subprocess (timeout={stage1_timeout}s, mode={'R2V multimodal' if ref_json_tmp else 'text-only'})...")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=dict(os.environ, PYTHONWARNINGS='ignore', PYTHONIOENCODING='utf-8', HF_DEACTIVATE_ASYNC_LOAD='1')
                )
                output_lines = []
                import time as _t
                t_sub_start = _t.time()
                while True:
                    line = proc.stdout.readline()
                    if line:
                        line_str = line.strip()
                        if line_str:
                            output_lines.append(line_str)
                            logger.info(f"[MiniMax-H3/Stage1] {line_str}")
                            print(f"[Stage1] {line_str}", flush=True)
                    elif proc.poll() is not None:
                        break
                    if _t.time() - t_sub_start > stage1_timeout:
                        proc.kill()
                        raise subprocess.TimeoutExpired(cmd, stage1_timeout, output="\n".join(output_lines))
                    _t.sleep(0.01)

                proc.stdout.close()
                returncode = proc.wait()
                loaded_ok = False
                if returncode == 0 and os.path.exists(embeds_tmp) and os.path.getsize(embeds_tmp) > 100:
                    try:
                        data = torch.load(embeds_tmp, weights_only=True)
                        prompt_embeds = data["prompt_embeds"]
                        text_token_tags = data["text_token_tags"]
                        _PROMPT_EMBEDS_CACHE[cache_key] = (prompt_embeds, text_token_tags)
                        try:
                            torch.save({"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}, disk_cache_file)
                        except Exception as dce:
                            logger.warning(f"[MiniMax-H3/Stage1] Could not write disk cache: {dce}")
                        logger.info(f"[MiniMax-H3/Stage1] Prompt encoded successfully! Shape: {prompt_embeds.shape}")
                        loaded_ok = True
                    except Exception as le:
                        logger.warning(f"[MiniMax-H3/Stage1] Temporary embeds load warning: {le}")

                if not loaded_ok:
                    out_text = "\n".join(output_lines[-20:])
                    logger.error(
                        f"[MiniMax-H3/Stage1] Subprocess encoding failed (code {returncode}):\n{out_text}"
                    )
                    raise RuntimeError(
                        f"Prompt encoding failed (code {returncode}): {out_text[:500]}"
                    )

            finally:
                if os.path.exists(prompt_tmp):
                    try:
                        os.remove(prompt_tmp)
                    except Exception:
                        pass
                if os.path.exists(embeds_tmp):
                    try:
                        os.remove(embeds_tmp)
                    except Exception:
                        pass
                if ref_json_tmp and os.path.exists(ref_json_tmp):
                    try:
                        os.remove(ref_json_tmp)
                    except Exception:
                        pass

        # Stage 2: 트랜스포머 (34GB) & VAE 단독 로드 (메모리 10GB 상태에서 여유롭게 로드)
        if not hasattr(_pipeline, target_transformer) or getattr(_pipeline, target_transformer) is None:
            logger.info(f"[MiniMax-H3/Stage2] Loading {target_transformer} (dtype=bfloat16, low_cpu_mem_usage=True)...")
            _pipeline.load_components(names=[target_transformer], dtype=torch.bfloat16, low_cpu_mem_usage=True)
            tf_loaded = getattr(_pipeline, target_transformer, None)
            if tf_loaded is not None and not getattr(tf_loaded, "_is_mxfp8_dequantized", False):
                logger.info(f"[MiniMax-H3/Stage2] Dequantizing {target_transformer} MXFP8/FP8 weights to bfloat16...")
                _dequantize_transformer_weights(tf_loaded, MINIMAX_LOCAL_PATH, subfolder=target_transformer)
                tf_loaded._is_mxfp8_dequantized = True

        if target_workflow == "ref2va" and (not hasattr(_pipeline, "processor") or getattr(_pipeline, "processor") is None):
            _pipeline.load_components(names=["processor"])

        if not hasattr(_pipeline, "vae") or getattr(_pipeline, "vae") is None:
            _pipeline.load_components(names=["vae", "audio_vae", "scheduler", "audio_scheduler"], dtype=torch.float32)

        # CPU Pinned-Free Block Streaming (Zero VRAM overhead, prevents Windows kernel pool exhaustion)
        if torch.cuda.is_available():
            tf = getattr(_pipeline, target_transformer, None)
            if tf is not None and not getattr(tf, "_is_fp8_quantized", False):
                logger.info(f"[MiniMax-H3] Setting up Stable CPU-Streaming for {target_transformer}...")
                
                fp32_names = ("time_embedder", "norm_out", "proj_in", "proj_out", "audio_proj_in", "audio_proj_out", "rope")
                for name, child in tf.named_children():
                    if name not in ("blocks", "transformer_blocks"):
                        target_dtype = torch.float32 if any(fn in name for fn in fp32_names) else torch.bfloat16
                        child.to(device="cuda", dtype=target_dtype)

                blocks = getattr(tf, "blocks", None) or getattr(tf, "transformer_blocks", None) or []
                total_blocks = len(blocks)
                
                for b in blocks:
                    for mod in b.modules():
                        if isinstance(mod, torch.nn.Linear):
                            if not hasattr(mod, "_cpu_weight") or mod._cpu_weight is None or mod._cpu_weight.numel() == 0:
                                mod._cpu_weight = mod.weight.data.to(device="cpu", dtype=torch.bfloat16)
                            mod.weight.data = torch.empty(0, device="cuda", dtype=torch.bfloat16)
                            if mod.bias is not None:
                                if not hasattr(mod, "_cpu_bias") or mod._cpu_bias is None:
                                    mod._cpu_bias = mod.bias.data.to(device="cpu", dtype=torch.bfloat16)
                                mod.bias.data = torch.empty(0, device="cuda", dtype=torch.bfloat16)
                        else:
                            for p in mod.parameters(recurse=False):
                                p.data = p.data.to(device="cuda", dtype=torch.bfloat16)

                    # Move the block container and its tiny RMSNorms to CUDA ONCE (takes < 100KB VRAM)
                    b.to(device="cuda")

                    def make_pre_hook():
                        def hook(m, inp):
                            for sub in m.modules():
                                if isinstance(sub, torch.nn.Linear) and hasattr(sub, "_cpu_weight") and sub._cpu_weight is not None:
                                    sub.weight.data = sub._cpu_weight.to(device="cuda", dtype=torch.bfloat16)
                                    if hasattr(sub, "_cpu_bias") and sub._cpu_bias is not None:
                                        sub.bias.data = sub._cpu_bias.to(device="cuda", dtype=torch.bfloat16)
                        return hook

                    def make_post_hook():
                        def hook(m, inp, out):
                            for sub in m.modules():
                                if isinstance(sub, torch.nn.Linear):
                                    sub.weight.data = torch.empty(0, device="cuda", dtype=torch.bfloat16)
                                    if hasattr(sub, "bias") and sub.bias is not None and sub.bias.data.numel() > 0:
                                        sub.bias.data = torch.empty(0, device="cuda", dtype=torch.bfloat16)
                        return hook

                    b.register_forward_pre_hook(make_pre_hook())
                    b.register_forward_hook(make_post_hook())

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                tf._is_fp8_quantized = True
                logger.info(f"[MiniMax-H3] {target_transformer} CPU-Streaming setup complete ({_get_free_vram_gb():.1f}GB VRAM free)!")

            # RoPE inv_freq must be exact float32 (a bf16 round-trip corrupts positional angles). Rebuild it.
            _tf_rope = getattr(getattr(_pipeline, target_transformer, None), "rope", None)
            if _tf_rope is not None and hasattr(_tf_rope, "inv_freq"):
                _fd = _tf_rope.rope_freq_dim
                _theta = float(getattr(getattr(_pipeline, target_transformer).config, "rope_theta", 10000.0))
                _tf_rope.inv_freq = (1.0 / (_theta ** (torch.arange(0, 2 * _fd, 2, dtype=torch.float32) / (2 * _fd)))).to("cuda")
                logger.info(f"[MiniMax-H3] RoPE inv_freq rebuilt in fp32 (dtype={_tf_rope.inv_freq.dtype}).")

        # Apply LoRA Weight Fusion (Zero VRAM overhead, instant swapping)
        tf_for_lora = getattr(_pipeline, target_transformer, None)
        if tf_for_lora is not None:
            _fuse_lora_into_transformer(tf_for_lora, lora_path=lora_path, lora_scale=lora_scale)

        if hasattr(_pipeline, "vae") and getattr(_pipeline, "vae") is not None:
                vae_mod = getattr(_pipeline, "vae")
                if next(vae_mod.parameters()).device.type != "cuda":
                    logger.info(f"[MiniMax-H3] Moving VAE to CUDA (free VRAM before: {_get_free_vram_gb():.1f}GB)...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    vae_mod.to("cuda")
                    logger.info(f"[MiniMax-H3] VAE placed on CUDA (free VRAM after: {_get_free_vram_gb():.1f}GB)")
                if hasattr(_pipeline, "audio_vae") and getattr(_pipeline, "audio_vae") is not None:
                    audio_vae_mod = getattr(_pipeline, "audio_vae")
                    if next(audio_vae_mod.parameters()).device.type != "cuda":
                        audio_vae_mod.to("cuda")

        # 6. Parse references for R2V mode
        ref_objects = []
        if target_workflow == "ref2va" and reference_paths:
            from PIL import Image as _PILImage
            from diffusers.modular_pipelines.minimax_h3 import (
                MiniMaxH3AudioReference,
                MiniMaxH3ImageReference,
                MiniMaxH3VideoReference,
            )
            target_short = min(width, height)
            if hasattr(_pipeline, "config"):
                _pipeline.config.canvas_short_edge = min(768, target_short)
                _pipeline.config.canvas_max_pixels = width * height

            # Ensure before_encode step doesn't upscale image short edges to 2048px (prevents 10MP VAE explosion)
            if hasattr(_pipeline, "blocks") and hasattr(_pipeline.blocks, "sub_blocks"):
                be_block = _pipeline.blocks.sub_blocks.get("before_encode")
                if be_block is not None and hasattr(be_block, "reference_image_short_edge"):
                    be_block.reference_image_short_edge = min(768, target_short)

            for ref in reference_paths:
                t = ref.get("type", "image")
                p = ref.get("path")
                if not p or not os.path.exists(p):
                    continue
                if t == "image":
                    logger.info(f"[MiniMax-H3] Adding image reference: {p}")
                    raw_img_ref = MiniMaxH3ImageReference.from_file(p)
                    img = raw_img_ref.image
                    w, h = img.size
                    if max(w, h) > 768:
                        scale = min(1.0, 768 / max(w, h))
                        new_w, new_h = int(round(w * scale)), int(round(h * scale))
                        img = img.resize((new_w, new_h), _PILImage.Resampling.LANCZOS)
                        logger.info(f"[MiniMax-H3] Resized image reference for VAE from ({w}, {h}) to ({new_w}, {new_h})")
                    ref_objects.append(MiniMaxH3ImageReference(image=img))
                elif t == "video":
                    logger.info(f"[MiniMax-H3] Adding video reference: {p}")
                    raw_vid_ref = MiniMaxH3VideoReference.from_file(p)
                    # Video reference smart resize for VAE to prevent sequence explosion (58k -> 32k tokens)
                    frames = raw_vid_ref.frames
                    import numpy as _np
                    if isinstance(frames, _np.ndarray) and frames.ndim == 4:
                        num_f, v_h, v_w, _ = frames.shape
                        if min(v_h, v_w) > target_short or max(v_h, v_w) > 768:
                            scale = min(target_short / min(v_h, v_w), 768 / max(v_h, v_w))
                            new_w = max(32, int(round(v_w * scale / 32) * 32))
                            new_h = max(32, int(round(v_h * scale / 32) * 32))
                            import cv2
                            resized_frames = _np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames])
                            raw_vid_ref.frames = resized_frames
                            logger.info(f"[MiniMax-H3] Smart-resized video reference for VAE from ({v_w}, {v_h}) to ({new_w}, {new_h})")
                    ref_objects.append(raw_vid_ref)
                elif t == "audio":
                    logger.info(f"[MiniMax-H3] Adding audio reference: {p}")
                    ref_objects.append(MiniMaxH3AudioReference.from_file(p))

        # 7. Parse inputs and keyframes
        keyframes = []
        keyframe_anchors = []

        if first_frame_path and not reference_paths:
            from PIL import Image
            first_frame_path = str(first_frame_path).strip("'\"").replace("file:///", "").replace("file://", "")
            if os.path.exists(first_frame_path):
                logger.info(f"[MiniMax-H3] Loading first_frame: {first_frame_path}")
                img = Image.open(first_frame_path).convert("RGB")
                keyframes.append(img)
                keyframe_anchors.append("first")
            else:
                logger.warning(f"[MiniMax-H3] first_frame_path file not found: '{first_frame_path}', skipping first_frame.")

        if last_frame_path and not reference_paths:
            from PIL import Image
            last_frame_path = str(last_frame_path).strip("'\"").replace("file:///", "").replace("file://", "")
            if os.path.exists(last_frame_path):
                logger.info(f"[MiniMax-H3] Loading last_frame: {last_frame_path}")
                img = Image.open(last_frame_path).convert("RGB")
                keyframes.append(img)
                keyframe_anchors.append("last")
            else:
                logger.warning(f"[MiniMax-H3] last_frame_path file not found: '{last_frame_path}', skipping last_frame.")

        width = (width // 32) * 32
        height = (height // 32) * 32
        num_frames = _duration_to_frames(duration_seconds)

        logger.info(
            f"[MiniMax-H3] {mode}: {width}x{height}, "
            f"{num_frames}f (~{num_frames/24:.1f}s), "
            f"{num_inference_steps}steps, seed={seed}"
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"[MiniMax-H3] VRAM before inference: {_get_free_vram_gb():.1f}GB free")

        if torch.cuda.is_available():
            prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
            text_token_tags = text_token_tags.cpu()

        t0 = time.time()
        _current_status["state"] = "stage2_generating"
        _current_status["start_timestamp"] = t0
        _current_status["step"] = 0
        _current_status["total_steps"] = num_inference_steps
        _current_status["progress_pct"] = 0.0
        _current_status["last_step_timestamp"] = t0

        # Register Real-time Step Counter on Block 0
        tf = getattr(_pipeline, target_transformer, None)
        blocks = getattr(tf, "blocks", None) or getattr(tf, "transformer_blocks", None) or []
        progress_hook_handle = None
        if blocks:
            step_tracker = {"count": 0, "last_time": t0}
            def block0_progress_hook(module, inp):
                step_tracker["count"] += 1
                cur_step = step_tracker["count"]
                now = time.time()
                dt = round(now - step_tracker["last_time"], 2)
                step_tracker["last_time"] = now
                pct = round(min((cur_step / num_inference_steps) * 100.0, 100.0), 1)
                _current_status["step"] = cur_step
                _current_status["total_steps"] = num_inference_steps
                _current_status["progress_pct"] = pct
                _current_status["step_time_sec"] = dt
                _current_status["last_step_timestamp"] = now
                msg = f"[MiniMax-H3] >>> Step {cur_step}/{num_inference_steps} ({pct}%) | Step Time: {dt}s | Elapsed: {round(now - t0, 1)}s <<<"
                logger.info(msg)
                print(msg, flush=True)
            progress_hook_handle = blocks[0].register_forward_pre_hook(block0_progress_hook)

        def step_callback(pipe, step, timestep, callback_kwargs):
            global _current_status
            s = step + 1
            pct = round((s / num_inference_steps) * 100, 1)
            now = time.time()
            dt = round(now - (_current_status.get("last_step_timestamp") or now), 2)
            _current_status["step"] = s
            _current_status["progress_pct"] = pct
            _current_status["step_time_sec"] = dt
            _current_status["last_step_timestamp"] = now
            msg = f"[MiniMax-H3] Step {s}/{num_inference_steps} ({pct}%) completed in {dt}s"
            logger.info(msg)
            print(msg, flush=True)
            return callback_kwargs

        # Official MiniMax-H3 schedule: video shift = 12.0, audio shift = 3.0 (for 50 steps)
        # For lower inference steps (e.g. 15-30), shift=12.0 stalls denoising at t=0.52 leaving 90% Gaussian noise.
        # Dynamic formula: shift = (num_inference_steps - 2) / 4.0 keeps terminal sigma at 0.20 and t at 0.80 across all step counts!
        if flow_shift is not None and flow_shift > 0:
            video_shift = float(flow_shift)
        elif num_inference_steps >= 50:
            video_shift = 12.0
        else:
            video_shift = max(2.5, (num_inference_steps - 2) / 4.0)

        audio_shift = 3.0 if num_inference_steps >= 50 else max(1.0, 3.0 * (num_inference_steps / 50.0))

        if hasattr(_pipeline, "scheduler") and getattr(_pipeline, "scheduler") is not None:
            _pipeline.scheduler.set_shift(video_shift)
            logger.info(f"[MiniMax-H3] Scheduler video shift set to {video_shift:.2f} (steps={num_inference_steps}).")
        if hasattr(_pipeline, "audio_scheduler") and getattr(_pipeline, "audio_scheduler") is not None:
            _pipeline.audio_scheduler.set_shift(audio_shift)
            logger.info(f"[MiniMax-H3] Scheduler audio shift set to {audio_shift:.2f} (steps={num_inference_steps}).")

        kwargs = {
            'callback_on_step_end': step_callback,
            "prompt": prompt,
            "prompt_embeds": prompt_embeds,
            "text_token_tags": text_token_tags,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "output_type": "np",
        }
        if keyframes:
            if len(keyframes) >= 1 and keyframe_anchors[0] == "first":
                kwargs["image"] = keyframes[0]
            if len(keyframes) >= 2 and keyframe_anchors[1] == "last":
                kwargs["last_image"] = keyframes[1]
            elif len(keyframes) == 1 and keyframe_anchors[0] == "last":
                kwargs["last_image"] = keyframes[0]

        if ref_objects:
            kwargs["references"] = ref_objects

        try:
            import io, sys
            if hasattr(sys.stdout, 'encoding') and (sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8'):
                if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout, 'closed', False):
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'encoding') and (sys.stderr.encoding is None or sys.stderr.encoding.lower() != 'utf-8'):
                if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr, 'closed', False):
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

        # === Attention Optimization: PyTorch SDPA backend flags ===
        # MiniMaxH3Attention uses a custom attention class - cannot use set_attn_processor().
        # Enable Flash/MemEfficient SDPA at the PyTorch backend level instead.
        # torch is already imported at module level - do NOT re-import here.
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info("[MiniMax-H3] PyTorch Flash SDPA backend enabled.")
        except Exception as e_sdp:
            logger.warning("[MiniMax-H3] Could not enable Flash SDPA backend.")
        with torch.inference_mode():
            result = _pipeline(**kwargs)
        elapsed = time.time() - t0
        logger.info(f"[MiniMax-H3] Inference done in {elapsed:.1f}s")
        _current_status["state"] = "completed"
        _current_status["progress_pct"] = 100.0

        timestamp = int(time.time())
        seed_str = seed if seed is not None else "rand"
        output_path = os.path.join(OUTPUT_DIR, f"minimax_{timestamp}_{seed_str}.mp4")
        _save_video(result, output_path)

        logger.info(f"[MiniMax-H3] Saved: {output_path}")
        return {
            "video_path": output_path,
            "frame_count": num_frames,
            "fps": 24,
            "width": width,
            "height": height,
            "info": f"seed={seed}, steps={num_inference_steps}, model=MiniMax-H3-{mode}",
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        logger.error(f"[MiniMax-H3] Inference failed with error: {e}")
        # 에러 발생 시 메모리 누수를 완전히 방지하기 위해 로드된 모든 파이프라인 개체를 셧다운하고 삭제
        _pipeline = None
        _components_manager = None
        raise e
    finally:
        if progress_hook_handle is not None:
            try:
                progress_hook_handle.remove()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.GetCurrentProcess()
            psapi.EmptyWorkingSet(handle)
        except Exception:
            pass


def _save_video(result, output_path: str, fps: int = 24):
    import numpy as np
    from PIL import Image
    import cv2

    logger.info(f"[MiniMax-H3/SaveVideo] Saving result to {output_path}")
    video = None
    if isinstance(result, dict):
        video = result.get("videos") or result.get("video") or result.get("frames")
    elif hasattr(result, "videos"):
        video = result.videos
    elif hasattr(result, "video"):
        video = result.video
    elif hasattr(result, "frames"):
        video = result.frames
    elif isinstance(result, (list, tuple)):
        video = result[0]
    else:
        video = result

    if video is None:
        raise ValueError(f"No video found in pipeline output: {result}")

    if isinstance(video, (list, tuple)):
        if len(video) > 0 and isinstance(video[0], (list, tuple)):
            video = video[0]
        if len(video) > 0 and isinstance(video[0], Image.Image):
            video = np.stack([np.array(img) for img in video])
        elif len(video) > 0 and isinstance(video[0], np.ndarray):
            video = np.stack(video)

    if isinstance(video, torch.Tensor):
        video = video.cpu().float().numpy()

    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            video = video[0]
        if video.ndim == 4 and video.shape[1] in (1, 3, 4) and video.shape[-1] not in (1, 3, 4):
            # (T, C, H, W) -> (T, H, W, C)
            video = np.transpose(video, (0, 2, 3, 1))
        if video.dtype != np.uint8:
            if video.max() <= 1.05:
                video = (video * 255.0).clip(0, 255).astype(np.uint8)
            else:
                video = video.clip(0, 255).astype(np.uint8)

    num_frames, h, w, c = video.shape
    logger.info(f"[MiniMax-H3/SaveVideo] Final video array: {num_frames} frames, {w}x{h}, {c}ch, dtype={video.dtype}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Write video using cv2.VideoWriter
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
        for frame in video:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if c == 3 else frame
            out.write(frame_bgr)
        out.release()
        logger.info(f"[MiniMax-H3/SaveVideo] cv2.VideoWriter saved successfully: {output_path} (size: {os.path.getsize(output_path)} bytes)")
    except Exception as e:
        logger.warning(f"[MiniMax-H3/SaveVideo] cv2.VideoWriter failed: {e}, attempting ffmpeg fallback...")
        _save_video_ffmpeg(video, output_path, fps)

    # 2. Also save sample preview frames next to the output video for instant artifact inspection
    try:
        preview_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        for f_idx in [0, min(15, num_frames - 1), min(30, num_frames - 1), min(60, num_frames - 1), min(90, num_frames - 1), num_frames - 1]:
            preview_p = os.path.join(preview_dir, f"{base_name}_frame_{f_idx}.png")
            Image.fromarray(video[f_idx]).save(preview_p)
    except Exception as pe:
        logger.warning(f"[MiniMax-H3/SaveVideo] Preview frame export warning: {pe}")


def _save_video_ffmpeg(frames, output_path: str, fps: int):
    import subprocess
    import tempfile
    import shutil
    from PIL import Image

    tmpdir = tempfile.mkdtemp()
    try:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f"frame_{i:05d}.png"))

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ], check=True, capture_output=True)
        logger.info(f"[MiniMax-H3/SaveVideo] ffmpeg saved successfully to {output_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def unload():
    global _pipeline, _components_manager, _PROMPT_EMBEDS_CACHE
    logger.info("[MiniMax-H3] Unloading pipeline and purging RAM/VRAM...")
    try:
        if _components_manager is not None:
            _components_manager.components.clear()
            _components_manager.collections.clear()
            _components_manager.added_time.clear()
    except Exception:
        pass

    try:
        if _pipeline is not None:
            for comp_name in list(getattr(_pipeline, "components", {}).keys()):
                try:
                    _pipeline.unload_components(comp_name)
                except Exception:
                    setattr(_pipeline, comp_name, None)
    except Exception:
        pass

    _pipeline = None
    _components_manager = None
    _PROMPT_EMBEDS_CACHE.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    # Windows OS Working Set 즉시 회수하여 작업관리자 물리 RAM을 OS로 즉시 반환
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = kernel32.GetCurrentProcess()
        psapi.EmptyWorkingSet(handle)
    except Exception:
        pass

    logger.info("[MiniMax-H3] Unloaded successfully. Memory returned to OS.")


def get_status() -> dict:
    global _current_status
    vram_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    vram_res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    
    gpu_util = 0.0
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=2
        ).strip()
        gpu_util = float(out)
    except Exception:
        pass

    state = _current_status["state"]
    last_ts = _current_status.get("last_step_timestamp") or _current_status.get("start_timestamp") or time.time()
    elapsed_since_step = time.time() - last_ts if last_ts else 0.0
    
    is_idle_vram_held = False
    if vram_res > 5.0 and state in ("stage2_generating", "generating") and elapsed_since_step > 35.0 and gpu_util == 0.0:
        is_idle_vram_held = True

    return {
        "state": state,
        "step": _current_status["step"],
        "total_steps": _current_status["total_steps"],
        "progress_pct": _current_status["progress_pct"],
        "step_time_sec": _current_status["step_time_sec"],
        "vram_allocated_gb": round(vram_alloc, 2),
        "vram_reserved_gb": round(vram_res, 2),
        "gpu_utilization_pct": gpu_util,
        "is_idle_vram_held": is_idle_vram_held,
        "elapsed_sec": round(time.time() - (_current_status["start_timestamp"] or time.time()), 1) if state != "idle" else 0.0
    }
