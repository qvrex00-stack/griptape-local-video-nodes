from __future__ import annotations

import logging
import os
import random
from typing import Any

from griptape.artifacts import ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.traits.clamp import Clamp

logger = logging.getLogger("griptape_nodes")

__all__ = ["WanVideoGenerationInp"]

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


class WanVideoGenerationInp(BaseNode):
    """
    Wan2.1-Fun-InP: 스타일 이미지(첫 프레임) + depth_video → 스타일 유지 영상 생성

    파이프라인:
      원본 영상 첫 프레임 → FLUX로 스타일 변환 → ref_image (스타일 레퍼런스)
      원본 영상 → Depth Extractor → depth_video (동작 공간감 유지)
      ref_image + depth_video + 프롬프트 → Wan2.1-Fun-InP → 스타일 유지 영상
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Video"
        self.description = (
            "Generate style-consistent video using Wan2.1-Fun-InP. "
            "Provide a style reference image (first frame styled with FLUX) "
            "and a depth video to maintain motion and spatial consistency."
        )

        # ── 입력 ──────────────────────────────────────────────────────────
        self.add_parameter(
            ParameterImage(
                name="ref_image",
                tooltip=(
                    "스타일 레퍼런스 이미지 (원본 영상의 첫 프레임을 FLUX로 변환한 결과)\n"
                    "이 이미지의 스타일/외형이 전체 영상에 유지됩니다."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterVideo(
                name="control_video",
                tooltip=(
                    "Depth 비디오 (Depth Extractor 출력)\n"
                    "원본 영상의 공간감과 동작을 유지하는 가이드로 사용됩니다."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── 프롬프트 ──────────────────────────────────────────────────────
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="생성할 영상의 텍스트 설명",
                multiline=True,
                placeholder_text="A character performing martial arts moves...",
                allow_output=False,
            )
        )
        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                default_value="low quality, blurry, distorted, inconsistent style",
                tooltip="Negative prompt",
                multiline=True,
                allow_output=False,
            )
        )

        # ── 영상 설정 ─────────────────────────────────────────────────────
        self.add_parameter(
            ParameterInt(
                name="num_frames",
                default_value=16,
                tooltip="생성할 프레임 수 (control_video 길이에 자동 맞춤)",
                allow_output=False,
                traits={Clamp(min_val=8, max_val=81)},
            )
        )
        self.add_parameter(
            ParameterInt(
                name="fps",
                default_value=8,
                tooltip="출력 영상 FPS",
                allow_output=False,
                traits={Clamp(min_val=1, max_val=30)},
            )
        )
        self.add_parameter(
            ParameterInt(
                name="width",
                default_value=480,
                tooltip="출력 너비 (32 단위)",
                allow_output=False,
                traits={Clamp(min_val=256, max_val=1280)},
            )
        )
        self.add_parameter(
            ParameterInt(
                name="height",
                default_value=832,
                tooltip="출력 높이 (32 단위)",
                allow_output=False,
                traits={Clamp(min_val=256, max_val=1280)},
            )
        )
        self.add_parameter(
            ParameterFloat(
                name="guidance_scale",
                default_value=6.0,
                tooltip="Guidance scale (높을수록 프롬프트에 더 충실)",
                allow_output=False,
                traits={Clamp(min_val=1.0, max_val=20.0)},
            )
        )
        self.add_parameter(
            ParameterInt(
                name="num_inference_steps",
                default_value=20,
                tooltip="Inference steps (높을수록 품질 좋지만 느림)",
                allow_output=False,
                traits={Clamp(min_val=10, max_val=50)},
            )
        )
        self.add_parameter(
            ParameterBool(
                name="randomize_seed",
                default_value=True,
                tooltip="매 실행마다 시드 무작위화",
                allow_output=False,
            )
        )
        self.add_parameter(
            ParameterInt(
                name="seed",
                default_value=42,
                tooltip="Random seed",
                allow_output=False,
            )
        )

        # ── 출력 ──────────────────────────────────────────────────────────
        self.add_parameter(
            ParameterVideo(
                name="output_video",
                tooltip="생성된 영상",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
            )
        )

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output_video"] = None

        import requests

        # ── 파라미터 수집 ─────────────────────────────────────────────────
        ref_image_param    = self.get_parameter_value("ref_image")
        control_video_param = self.get_parameter_value("control_video")
        prompt             = self.get_parameter_value("prompt") or ""
        negative_prompt    = self.get_parameter_value("negative_prompt") or "low quality, blurry, distorted"
        num_frames         = self.get_parameter_value("num_frames") or 16
        fps                = self.get_parameter_value("fps") or 8
        width              = self.get_parameter_value("width") or 480
        height             = self.get_parameter_value("height") or 832
        guidance_scale     = self.get_parameter_value("guidance_scale") or 6.0
        num_inference_steps = self.get_parameter_value("num_inference_steps") or 20
        randomize_seed     = self.get_parameter_value("randomize_seed")
        seed               = self.get_parameter_value("seed") or 42

        if randomize_seed:
            seed = random.randint(0, 2**32 - 1)

        if not prompt:
            raise ValueError("prompt is required.")

        # ── tool_server 헬스 체크 ─────────────────────────────────────────
        try:
            resp = requests.get(f"{VIDEO_GEN_SERVER_URL}/health", timeout=3)
            resp.raise_for_status()
        except Exception:
            raise RuntimeError(
                "Video gen server is not running! "
                "video_gen_server/server.py를 먼저 실행하세요. (포트 8090)"
            )

        # ── 파일 경로 해석 ────────────────────────────────────────────────
        ref_image_path = ""
        if ref_image_param:
            ref_image_path = resolve_file_path(ref_image_param)

        control_video_path = ""
        if control_video_param:
            control_video_path = resolve_file_path(control_video_param)

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "ref_image_path": ref_image_path,
            "control_video_path": control_video_path,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": num_inference_steps,
            "seed": seed,
        }

        logger.info(
            f"WanVideoGenerationInp: prompt='{prompt[:50]}...', "
            f"ref={ref_image_path}, depth={control_video_path}, seed={seed}"
        )

        # ── video_gen_server 요청 ─────────────────────────────────────────
        try:
            resp = requests.post(
                f"{VIDEO_GEN_SERVER_URL}/wan/generate_inp",
                json=payload,
                timeout=3600,
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Video gen server /wan/generate_inp failed: {e}")

        result = resp.json()
        out_path = result.get("video_path", "")
        info = result.get("info", "")

        logger.info(f"WanVideoGenerationInp done: {out_path} | {info}")

        if not out_path or not os.path.exists(out_path):
            raise RuntimeError(f"Output video not found: {out_path}")

        video_artifact = VideoUrlArtifact(out_path)
        self.set_parameter_value("output_video", video_artifact)
        self.publish_update_to_parameter("output_video", video_artifact)
