"""Model Registry for FreeCloud AI Studio.

Enables modular registration and discovery of video generation models,
continuity encoders, and audio/lip-sync models for seamless future upgrades.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModelCategory(str, Enum):
    VIDEO_GENERATION = "video_generation"
    CHARACTER_ENCODER = "character_encoder"
    MOTION_TRACKER = "motion_tracker"
    AUDIO_SYNTHESIS = "audio_synthesis"
    LIP_SYNC = "lip_sync"
    UPSCALER = "upscaler"


class ModelSpec(BaseModel):
    model_id: str
    name: str
    category: ModelCategory
    provider: str  # e.g., "AnimateDiff", "Wan2.1", "InsightFace", "CosyVoice", "SadTalker"
    version: str
    vram_minimum_gb: int
    comfyui_compatible: bool = True
    installed: bool = False
    download_url: Optional[str] = None
    description: str = ""
    parameters: Dict[str, str] = Field(default_factory=dict)


class ModelRegistry:
    """Registry maintaining available and planned AI foundation models."""

    def __init__(self):
        self._models: Dict[str, ModelSpec] = {}
        self._register_default_models()

    def register(self, spec: ModelSpec) -> None:
        self._models[spec.model_id] = spec

    def get(self, model_id: str) -> Optional[ModelSpec]:
        return self._models.get(model_id)

    def list_models(
        self, category: Optional[ModelCategory] = None
    ) -> List[ModelSpec]:
        if category:
            return [m for m in self._models.values() if m.category == category]
        return list(self._models.values())

    def _register_default_models(self) -> None:
        defaults = [
            ModelSpec(
                model_id="animatediff_v3",
                name="AnimateDiff v3 Motion Module",
                category=ModelCategory.VIDEO_GENERATION,
                provider="AnimateDiff",
                version="3.0",
                vram_minimum_gb=8,
                comfyui_compatible=True,
                installed=False,
                description="Temporal diffusion motion adapter for generating coherent video frames.",
            ),
            ModelSpec(
                model_id="ipadapter_plus_face",
                name="IP-Adapter Plus Face SD15/SDXL",
                category=ModelCategory.CHARACTER_ENCODER,
                provider="TencentARC",
                version="1.5",
                vram_minimum_gb=4,
                comfyui_compatible=True,
                installed=False,
                description="Image prompt adapter preserving facial identities across generated shots.",
            ),
            ModelSpec(
                model_id="dwpose_controlnet",
                name="DWPose Full Body & Face Estimator",
                category=ModelCategory.MOTION_TRACKER,
                provider="IDEA-Research",
                version="1.1",
                vram_minimum_gb=4,
                comfyui_compatible=True,
                installed=False,
                description="Extracts detailed 2D poses, hand gestures, and facial expressions from original serial footage.",
            ),
            ModelSpec(
                model_id="sadtalker_lipsync",
                name="SadTalker Audio-Driven Talking Head",
                category=ModelCategory.LIP_SYNC,
                provider="OpenGVLab",
                version="0.0.2",
                vram_minimum_gb=6,
                comfyui_compatible=True,
                installed=False,
                description="Synchronizes lip motion and facial animation to original actor dialogues.",
            ),
            ModelSpec(
                model_id="cosyvoice_tts",
                name="CosyVoice Multi-lingual Zero-Shot Cloner",
                category=ModelCategory.AUDIO_SYNTHESIS,
                provider="Alibaba",
                version="1.0",
                vram_minimum_gb=8,
                comfyui_compatible=False,
                installed=False,
                description="Zero-shot character voice cloning preserving Indian serial vocal intonations.",
            ),
            ModelSpec(
                model_id="minimax_h3_video",
                name="MiniMax H3 Omni-Modal Video Generator (Low VRAM 8GB/16GB)",
                category=ModelCategory.VIDEO_GENERATION,
                provider="MiniMax / Comfy-Org",
                version="H3-INT8-ConvRot",
                vram_minimum_gb=8,
                comfyui_compatible=True,
                installed=False,
                download_url="https://huggingface.co/Comfy-Org/MiniMax-H3",
                description="High-end open-weights omni-modal video generator with native stereo audio and low-VRAM tiled VAE execution.",
                parameters={
                    "quantization": "int8_convrot / pruned_w4a8",
                    "text_encoder": "qwen3vl_32b_minimax_h3",
                    "video_vae": "minimax_h3_video_vae",
                    "audio_vae": "minimax_h3_audio_vae",
                    "native_audio": "true",
                    "modes": "Text-to-Video (T2V), Image-to-Video (I2V), Reference-to-Video (R2V)",
                },
            ),
            ModelSpec(
                model_id="minimax_h3_image",
                name="MiniMax H3 Image & Keyframe Conditioning",
                category=ModelCategory.CHARACTER_ENCODER,
                provider="MiniMax / Comfy-Org",
                version="H3-Image-W4A8",
                vram_minimum_gb=8,
                comfyui_compatible=True,
                installed=False,
                download_url="https://huggingface.co/Comfy-Org/MiniMax-H3-Image",
                description="High-end image editing, character face swapping, and first/last keyframe conditioning for video sequences.",
                parameters={
                    "quantization": "pruned_w4a8",
                    "modes": "Keyframe conditioning, Character swap, Image editing",
                },
            ),
        ]
        for spec in defaults:
            self.register(spec)


model_registry = ModelRegistry()
