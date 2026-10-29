from __future__ import annotations

import logging
import os
import random
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File

logger = logging.getLogger("griptape_nodes")

__all__ = ["AnimateDiffVideoGeneration"]

VIDEO_GEN_SERVER_URL = "http://127.0.0.1:8090"


def resolve_file_path(param_val) -> str:
    raw_path = None
    if isinstance(param_val, str): raw_path = param_val
    elif hasattr(param_val, "value"): raw_path = param_val.value
    elif isinstance(param_val, dict) and "value" in param_val: raw_path = param_val["value"]
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


class AnimateDiffVideoGeneration(BaseNode):
    """Generate video using AnimateDiff + ControlNet OpenPose (via video gen server)."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Video"
        self.description = "Generate video using AnimateDiff + ControlNet with pose reference."

        self.add_parameter(ParameterString(name="prompt", tooltip="Text description of the video to generate", multiline=True, placeholder_text="A person dancing gracefully...", allow_output=False))
        self.add_parameter(ParameterString(name="negative_prompt", default_value="low quality, blurry, distorted, ugly, bad anatomy", tooltip="Negative prompt", multiline=True, allow_output=False))
        self.add_parameter(ParameterVideo(name="pose_video", tooltip="Pose reference video (from Pose Extractor)", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterInt(name="num_frames", default_value=16, tooltip="Number of frames (max 32)", allow_output=False, min_val=8, max_val=32))
        self.add_parameter(ParameterInt(name="fps", default_value=8, tooltip="Output video FPS", allow_output=False, min_val=1, max_val=30))
        self.add_parameter(ParameterInt(name="width", default_value=512, tooltip="Output width", allow_output=False, min_val=256, max_val=768, step=64))
        self.add_parameter(ParameterInt(name="height", default_value=512, tooltip="Output height", allow_output=False, min_val=256, max_val=768, step=64))
        self.add_parameter(ParameterFloat(name="guidance_scale", default_value=7.5, tooltip="Guidance scale", allow_output=False, min_val=1.0, max_val=20.0))
        self.add_parameter(ParameterInt(name="num_inference_steps", default_value=25, tooltip="Inference steps", allow_output=False, min_val=10, max_val=50))
        self.add_parameter(ParameterBool(name="randomize_seed", default_value=True, tooltip="Randomize seed on each run", allow_output=False))
        self.add_parameter(ParameterInt(name="seed", default_value=42, tooltip="Random seed", allow_output=False))
        self.add_parameter(ParameterVideo(name="output_video", tooltip="Generated video", allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY}, settable=False, pulse_on_run=True))

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output_video"] = None
        import requests

        prompt = self.get_parameter_value("prompt") or ""
        negative_prompt = self.get_parameter_value("negative_prompt") or ""
        pose_video = self.get_parameter_value("pose_video")
        num_frames = self.get_parameter_value("num_frames") or 16
        fps = self.get_parameter_value("fps") or 8
        width = self.get_parameter_value("width") or 512
        height = self.get_parameter_value("height") or 512
        guidance_scale = self.get_parameter_value("guidance_scale") or 7.5
        num_inference_steps = self.get_parameter_value("num_inference_steps") or 25
        randomize = self.get_parameter_value("randomize_seed") or False
        seed = self.get_parameter_value("seed") or 42

        if not prompt:
            raise ValueError("Prompt is required.")
        if randomize:
            seed = random.randint(0, 2**32 - 1)
            self.set_parameter_value("seed", seed)

        try:
            requests.get(f"{VIDEO_GEN_SERVER_URL}/health", timeout=3)
        except Exception:
            raise RuntimeError("Video gen server is not running! Please start video_gen_server first.")

        pose_video_path = ""
        if pose_video:
            try:
                pose_video_path = resolve_file_path(pose_video)
            except Exception as e:
                logger.warning(f"Could not resolve pose video: {e}")

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "pose_video_path": pose_video_path,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
        }

        logger.info(f"AnimateDiff generate: frames={num_frames}, {width}x{height}, steps={num_inference_steps}, seed={seed}")
        try:
            resp = requests.post(f"{VIDEO_GEN_SERVER_URL}/animatediff/generate", json=payload, timeout=1800)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Video gen server request failed: {e}")

        result = resp.json()
        out_path = result.get("video_path", "")
        if out_path and os.path.exists(out_path):
            video_artifact = VideoUrlArtifact(out_path)
            self.set_parameter_value("output_video", video_artifact)
            self.publish_update_to_parameter("output_video", video_artifact)
            logger.info(f"AnimateDiff video done: {out_path}")
        else:
            raise RuntimeError(f"Output video not found: {out_path}")
