from __future__ import annotations

import logging
import os
import random
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ControlParameterInput, ControlParameterOutput, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File

logger = logging.getLogger("griptape_nodes")

__all__ = ["MinimaxVideoGeneration"]

VIDEO_GEN_SERVER_URL = "http://127.0.0.1:8090"


def resolve_file_path(param_val) -> str:
    raw_path = None
    if isinstance(param_val, str):
        raw_path = param_val
    elif hasattr(param_val, "value"):
        raw_path = param_val.value
    elif isinstance(param_val, dict) and "value" in param_val:
        raw_path = param_val["value"]
    if not raw_path:
        raise ValueError("Could not extract file path.")
    try:
        resolved = File(raw_path).resolve()
        path = resolved.location if hasattr(resolved, "location") else str(resolved)
    except Exception:
        path = raw_path
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    return path


class MinimaxVideoGeneration(BaseNode):
    """Generate high-quality video & audio locally using MiniMax-H3 model."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Video"
        self.description = "Generate video and native stereo audio locally using MiniMax-H3."

        flow_in = ControlParameterInput(
            name="exec_in",
            tooltip="Input flow for looping/sequencing"
        )
        flow_in.ui_options = {"display_name": "Flow In"}
        self.add_parameter(flow_in)

        flow_out = ControlParameterOutput(
            name="exec_out",
            tooltip="Output flow for looping/sequencing"
        )
        flow_out.ui_options = {"display_name": "Flow Out"}
        self.add_parameter(flow_out)

        self.add_parameter(
            ParameterString(
                name="generation_mode",
                default_value="Auto",
                tooltip="Select generation mode: Auto (auto-detect), T2V, I2V, FL2VA, R2V",
                allow_output=False,
                ui_options={"simple_dropdown": ["Auto", "T2V", "I2V", "FL2VA", "R2V"]}
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Describe the shots, actions, and soundscape in detail.",
                multiline=True,
                placeholder_text="A warrior performing martial arts, epic space-opera orchestral score...",
                allow_output=False
            )
        )
        self.add_parameter(
            ParameterImage(
                name="first_frame",
                tooltip="First frame image reference (optional, enables I2V mode)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )
        self.add_parameter(
            ParameterImage(
                name="last_frame",
                tooltip="Last frame image reference (optional, enables FL2VA keyframe mode)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )
        self.add_parameter(
            ParameterImage(
                name="ref_image",
                tooltip="Reference image for R2V mode (optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )
        self.add_parameter(
            ParameterVideo(
                name="ref_video",
                tooltip="Reference video for R2V mode (optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )
        self.add_parameter(
            ParameterString(
                name="ref_audio_path",
                tooltip="Reference audio absolute file path for R2V mode (optional)",
                placeholder_text="C:\\path\\to\\voice.wav",
                allow_output=False
            )
        )
        self.add_parameter(
            Parameter(
                name="loras",
                type="loras",
                input_types=["loras", "dict", "list", "str"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Connect from a 'Load Lora' node or provide a LoRA file path (.safetensors).",
            )
        )
        self.add_parameter(
            ParameterFloat(
                name="lora_scale",
                default_value=1.0,
                tooltip="Multiplier for LoRA weight (default 1.0)",
                allow_output=False,
                min_val=-2.0,
                max_val=2.0,
            )
        )
        self.add_parameter(
            ParameterInt(
                name="width",
                default_value=960,
                tooltip="Output width (must be multiple of 32)",
                allow_output=False,
                min_val=256,
                max_val=1344,
                step=32
            )
        )
        self.add_parameter(
            ParameterInt(
                name="height",
                default_value=544,
                tooltip="Output height (must be multiple of 32)",
                allow_output=False,
                min_val=256,
                max_val=1344,
                step=32
            )
        )
        self.add_parameter(
            ParameterFloat(
                name="duration_seconds",
                default_value=5.0,
                tooltip="Video duration (5 to 15 seconds)",
                allow_output=False,
                min_val=5.0,
                max_val=15.0
            )
        )
        self.add_parameter(
            ParameterInt(
                name="num_inference_steps",
                default_value=30,
                tooltip="Number of inference steps (Official default: 50, Fast preview: 15-30 with dynamic shift)",
                allow_output=False,
                min_val=10,
                max_val=100
            )
        )
        self.add_parameter(
            ParameterFloat(
                name="flow_shift",
                default_value=0.0,
                tooltip="Sigma shift (0.0 = Auto-calculated for step count; Official 50-step is 12.0, 15-step is ~3.25)",
                allow_output=False,
                min_val=0.0,
                max_val=20.0
            )
        )
        self.add_parameter(
            ParameterBool(
                name="randomize_seed",
                default_value=True,
                tooltip="Randomize seed on each run",
                allow_output=False
            )
        )
        self.add_parameter(
            ParameterInt(
                name="seed",
                default_value=42,
                tooltip="Random seed",
                allow_output=False
            )
        )
        self.add_parameter(
            ParameterBool(
                name="auto_stop_server",
                default_value=False,
                tooltip="생성 완료 후 video_gen_server를 자동으로 종료하여 RAM을 완전히 반환합니다.",
                allow_output=False
            )
        )
        self.add_parameter(
            ParameterBool(
                name="start_server_if_needed",
                default_value=True,
                tooltip="서버가 꺼져 있을 때 자동으로 시작합니다.",
                allow_output=False
            )
        )
        self.add_parameter(
            ParameterVideo(
                name="output_video",
                tooltip="Generated video with synchronized stereo audio",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True
            )
        )

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output_video"] = None
        import requests

        generation_mode = self.get_parameter_value("generation_mode") or "Auto"
        prompt = self.get_parameter_value("prompt") or ""
        first_frame = self.get_parameter_value("first_frame")
        last_frame = self.get_parameter_value("last_frame")
        width = self.get_parameter_value("width") or 960
        height = self.get_parameter_value("height") or 544
        duration_seconds = self.get_parameter_value("duration_seconds") or 5.0
        num_inference_steps = self.get_parameter_value("num_inference_steps") or 30
        randomize = self.get_parameter_value("randomize_seed") or False
        seed = self.get_parameter_value("seed") or 42

        ref_image = self.get_parameter_value("ref_image")
        ref_video = self.get_parameter_value("ref_video")
        ref_audio_path = self.get_parameter_value("ref_audio_path")

        raw_loras = self.get_parameter_value("loras")
        lora_scale_mult = float(self.get_parameter_value("lora_scale") if self.get_parameter_value("lora_scale") is not None else 1.0)
        lora_path = None
        lora_weight = 1.0

        if raw_loras:
            if isinstance(raw_loras, dict):
                lora_path = raw_loras.get("path") or raw_loras.get("file_path")
                lora_weight = float(raw_loras.get("weight", 1.0))
                trigger = raw_loras.get("trigger_phrase")
                if trigger and isinstance(trigger, str) and trigger.strip() and trigger.strip() not in prompt:
                    prompt = f"{trigger.strip()}, {prompt}"
            elif isinstance(raw_loras, list) and len(raw_loras) > 0:
                first_lora = raw_loras[0]
                if isinstance(first_lora, dict):
                    lora_path = first_lora.get("path") or first_lora.get("file_path")
                    lora_weight = float(first_lora.get("weight", 1.0))
                    trigger = first_lora.get("trigger_phrase")
                    if trigger and isinstance(trigger, str) and trigger.strip() and trigger.strip() not in prompt:
                        prompt = f"{trigger.strip()}, {prompt}"
                elif isinstance(first_lora, str):
                    lora_path = first_lora
            elif isinstance(raw_loras, str):
                lora_path = raw_loras

            if lora_path:
                try:
                    lora_path = resolve_file_path(lora_path)
                except Exception as e:
                    logger.warning(f"Could not resolve LoRA file path '{lora_path}': {e}")

        if not prompt:
            raise ValueError("Prompt is required.")

        if randomize:
            seed = random.randint(0, 2**32 - 1)
            self.set_parameter_value("seed", seed)

        # 서버 실행 확인 - 안 켜져 있으면 설정에 따라 자동 시작 시도
        server_ready = False
        try:
            requests.get(f"{VIDEO_GEN_SERVER_URL}/health", timeout=3)
            server_ready = True
        except Exception:
            start_needed = self.get_parameter_value("start_server_if_needed")
            if start_needed is None:
                start_needed = True

            if start_needed:
                # 서버가 없으면 videogen_library_advanced의 시작 함수 호출 시도
                logger.warning("[MiniMax] Server not responding, attempting to start...")
                try:
                    from griptape_nodes_library.videogen_library_advanced import _ensure_server_running
                    _ensure_server_running()
                    # 서버가 올라올 때까지 최대 60초 대기
                    import time
                    deadline = time.time() + 60
                    while time.time() < deadline:
                        try:
                            requests.get(f"{VIDEO_GEN_SERVER_URL}/health", timeout=2)
                            server_ready = True
                            logger.info("[MiniMax] Server started successfully")
                            break
                        except Exception:
                            time.sleep(2)
                except Exception as start_err:
                    logger.warning(f"[MiniMax] Could not auto-start server: {start_err}")
            else:
                logger.info("[MiniMax] Server is offline and start_server_if_needed is False.")

        if not server_ready:
            raise RuntimeError(
                "Video generation server is not running!\n"
                "비디오 생성 서버가 꺼져 있습니다. "
                "노드의 'start_server_if_needed'를 True로 켜거나 video_gen_server를 실행해주세요."
            )

        first_frame_path = ""
        if first_frame:
            try:
                first_frame_path = resolve_file_path(first_frame)
            except Exception as e:
                logger.warning(f"Could not resolve first_frame: {e}")

        last_frame_path = ""
        if last_frame:
            try:
                last_frame_path = resolve_file_path(last_frame)
            except Exception as e:
                logger.warning(f"Could not resolve last_frame: {e}")

        reference_paths = []
        if ref_image:
            try:
                reference_paths.append({"type": "image", "path": resolve_file_path(ref_image)})
            except Exception as e:
                logger.warning(f"Could not resolve ref_image: {e}")
        if ref_video:
            try:
                reference_paths.append({"type": "video", "path": resolve_file_path(ref_video)})
            except Exception as e:
                logger.warning(f"Could not resolve ref_video: {e}")
        if ref_audio_path:
            # Clean path from potential surrounding quotes
            clean_audio_path = ref_audio_path.strip("\"'")
            if os.path.exists(clean_audio_path):
                reference_paths.append({"type": "audio", "path": clean_audio_path})
            else:
                logger.warning(f"Audio reference file not found: {clean_audio_path}")

        flow_shift = self.get_parameter_value("flow_shift")

        # Explicit mode sanitization: ensure unused references/frames are not sent
        mode_upper = generation_mode.upper()
        if mode_upper == "T2V":
            first_frame_path = ""
            last_frame_path = ""
            reference_paths = []
        elif mode_upper == "I2V":
            last_frame_path = ""
            reference_paths = []
        elif mode_upper == "FL2VA":
            reference_paths = []

        payload = {
            "prompt": prompt,
            "generation_mode": generation_mode,
            "first_frame_path": first_frame_path,
            "last_frame_path": last_frame_path,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
        }
        if flow_shift and float(flow_shift) > 0:
            payload["flow_shift"] = float(flow_shift)
        if reference_paths:
            payload["reference_paths"] = reference_paths
        if lora_path and os.path.exists(lora_path):
            payload["lora_path"] = lora_path
            payload["lora_scale"] = float(lora_weight * lora_scale_mult)
            logger.info(f"[MiniMax] LoRA enabled: path={lora_path}, effective_scale={payload['lora_scale']:.2f}")

        logger.info(
            f"MiniMax generate: {width}x{height}, duration={duration_seconds}s, "
            f"steps={num_inference_steps}, seed={seed}"
        )

        try:
            resp = requests.post(f"{VIDEO_GEN_SERVER_URL}/minimax/generate", json=payload, timeout=7200)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            try:
                requests.post(f"{VIDEO_GEN_SERVER_URL}/cancel", timeout=3)
                logger.warning("MiniMax generate timed out")
            except Exception:
                pass
            raise RuntimeError("Video gen server timed out during MiniMax generation.")
        except Exception as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "") if getattr(e, "response", None) is not None else ""
            except Exception:
                pass
            logger.error(f"[MiniMax] Server request failed: {e} | detail: {detail}")
            try:
                requests.post(f"{VIDEO_GEN_SERVER_URL}/cancel", timeout=3)
            except Exception:
                pass
            raise RuntimeError(f"Video gen server request failed: {e}\nServer detail: {detail}")

        result = resp.json()
        out_path = result.get("video_path", "")
        if out_path and os.path.exists(out_path):
            video_artifact = VideoUrlArtifact(out_path)
            self.set_parameter_value("output_video", video_artifact)
            self.publish_update_to_parameter("output_video", video_artifact)
            logger.info(f"MiniMax video & audio generated: {out_path}")

            # auto_stop_server 설정 시 생성 완료 후 서버 안전 종료
            auto_stop = self.get_parameter_value("auto_stop_server") or False
            if auto_stop:
                try:
                    requests.post(f"{VIDEO_GEN_SERVER_URL}/server/stop", timeout=3)
                    logger.info("[MiniMax] Video server stopped successfully (auto_stop_server=True).")
                except Exception as e:
                    logger.warning(f"[MiniMax] Could not stop server: {e}")
        else:
            raise RuntimeError(f"Output video file not found: {out_path}")

    def get_next_control_output(self):
        return self.get_parameter_by_name("exec_out")
