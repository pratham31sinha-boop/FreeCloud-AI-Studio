"""ComfyUI Workflow Graph Builder.

Generates modular ComfyUI node graphs incorporating AnimateDiff for video generation,
IP-Adapter for character appearance consistency, and ControlNet for action preservation.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class ComfyUIWorkflowBuilder:
    """Constructs parameterized ComfyUI API execution graphs."""

    def build_anime_video_workflow(
        self,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, low resolution, deformed anatomy, flicker, noisy",
        checkpoint_name: str = "animagine_xl_v3.safetensors",
        lora_name: Optional[str] = "jagdhatri_anime_v1.safetensors",
        frame_count: int = 24,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        steps: int = 25,
        cfg: float = 7.0,
    ) -> Dict[str, Any]:
        """Build a standard AnimateDiff + Anime checkpoint workflow graph."""
        active_seed = seed if seed is not None else random.randint(1, 100000000)

        nodes: Dict[str, Any] = {
            # Checkpoint Loader
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint_name},
            },
            # Positive Prompt Encode
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{prompt}, masterpiece, highly detailed anime visual, sharp lineart, cinematic lighting",
                    "clip": ["1", 1],
                },
            },
            # Negative Prompt Encode
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                },
            },
            # Empty Latent Image / Batch for Video
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": frame_count,
                },
            },
            # AnimateDiff Model Loader
            "5": {
                "class_type": "AnimateDiffLoaderWithContext",
                "inputs": {
                    "model_name": "mm_sd_v15_v2.ckpt",
                    "beta_schedule": "sqrt_linear",
                    "context_length": 16,
                    "context_stride": 1,
                    "context_overlap": 4,
                },
            },
            # Apply AnimateDiff to Model
            "6": {
                "class_type": "AnimateDiffApply",
                "inputs": {
                    "model": ["1", 0],
                    "motion_model": ["5", 0],
                },
            },
            # KSampler
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["6", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": active_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "karras",
                    "denoise": 1.0,
                },
            },
            # VAE Decode
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["1", 2],
                },
            },
            # Video Output Combine
            "9": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "freecloud_chunk",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }

        # Optional LoRA insertion
        if lora_name:
            nodes["10"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": lora_name,
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                },
            }
            # Rewire AnimateDiff input model to LoRA output
            nodes["6"]["inputs"]["model"] = ["10", 0]
            nodes["2"]["inputs"]["clip"] = ["10", 1]
            nodes["3"]["inputs"]["clip"] = ["10", 1]

        return nodes


workflow_builder = ComfyUIWorkflowBuilder()
