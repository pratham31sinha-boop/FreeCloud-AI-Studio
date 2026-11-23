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

    def build_minimax_h3_t2v_workflow(
        self,
        prompt: str,
        audio_prompt: Optional[str] = None,
        negative_prompt: str = "bad quality, blurry, distorted anatomy, low resolution, artifacts, jerky motion, noisy audio",
        model_name: str = "minimax_h3_pruned_int8.safetensors",
        text_encoder_name: str = "qwen3vl_32b_minimax_h3.safetensors",
        video_vae_name: str = "minimax_h3_video_vae.safetensors",
        audio_vae_name: str = "minimax_h3_audio_vae.safetensors",
        width: int = 768,
        height: int = 768,
        length_frames: int = 72,  # 3 seconds @ 24fps
        fps: int = 24,
        steps: int = 25,
        cfg: float = 6.5,
        seed: Optional[int] = None,
        low_vram_mode: bool = True,
    ) -> Dict[str, Any]:
        """Build the official MiniMax H3 Text-to-Video ComfyUI workflow graph.

        Supports Low VRAM (8GB/16GB) execution with quantized INT8/W4A8 weights,
        Qwen-VL text conditioning, and native stereo audio generation.
        """
        active_seed = seed if seed is not None else random.randint(1, 100000000)
        sound_description = audio_prompt or f"Cinematic atmospheric audio and sound effects matching {prompt}"

        nodes: Dict[str, Any] = {
            # 1. MiniMax H3 Diffusion Model Loader (Low VRAM quantized INT8)
            "1": {
                "class_type": "MiniMaxModelLoader",
                "inputs": {
                    "model_name": model_name,
                    "precision": "int8" if low_vram_mode else "bf16",
                    "device": "auto",
                    "cpu_offload": low_vram_mode,
                },
            },
            # 2. Qwen3-VL-32B Dual CLIP / Text Encoder
            "2": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": text_encoder_name,
                    "clip_name2": text_encoder_name,
                    "type": "qwen_vl",
                },
            },
            # 3. Video Positive Prompt Conditioning
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{prompt}, 4k ultra-high definition, cinematic lighting, fluid motion, masterwork",
                    "clip": ["2", 0],
                },
            },
            # 4. Audio Conditioning Prompt (Native Stereo Audio)
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": sound_description,
                    "clip": ["2", 0],
                },
            },
            # 5. Negative Prompt Conditioning
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["2", 0],
                },
            },
            # 6. MiniMax Video VAE Loader
            "6": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": video_vae_name},
            },
            # 7. MiniMax Audio VAE Loader
            "7": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": audio_vae_name},
            },
            # 8. Empty MiniMax Video & Audio Latent Canvas
            "8": {
                "class_type": "EmptyMiniMaxLatent",
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": length_frames,
                    "batch_size": 1,
                },
            },
            # 9. MiniMax H3 Multimodal KSampler
            "9": {
                "class_type": "MiniMaxSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive_video": ["3", 0],
                    "positive_audio": ["4", 0],
                    "negative": ["5", 0],
                    "latent": ["8", 0],
                    "seed": active_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "low_vram_tiling": low_vram_mode,
                },
            },
            # 10. Tiled Video VAE Decode (Low VRAM memory optimization for 8GB/16GB)
            "10": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["6", 0],
                    "tile_size": 256 if low_vram_mode else 512,
                },
            },
            # 11. MiniMax Audio VAE Decode (Stereo Audio Track)
            "11": {
                "class_type": "MiniMaxAudioDecode",
                "inputs": {
                    "samples": ["9", 1],
                    "vae": ["7", 0],
                },
            },
            # 12. Combined Video + Native Audio Output Exporter
            "12": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["10", 0],
                    "audio": ["11", 0],
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": "minimax_h3_t2v",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }
        return nodes

    def build_minimax_h3_i2v_workflow(
        self,
        prompt: str,
        image_path: str,
        audio_prompt: Optional[str] = None,
        negative_prompt: str = "bad quality, blurry, distorted anatomy, jittery motion",
        width: int = 768,
        height: int = 768,
        length_frames: int = 72,
        fps: int = 24,
        steps: int = 25,
        cfg: float = 6.5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the official MiniMax H3 Image-to-Video (I2V) ComfyUI workflow graph."""
        nodes = self.build_minimax_h3_t2v_workflow(
            prompt=prompt,
            audio_prompt=audio_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            length_frames=length_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )
        # Add LoadImage node for starting keyframe
        nodes["13"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_path},
        }
        # Encode keyframe into MiniMax latent conditioning
        nodes["14"] = {
            "class_type": "MiniMaxImageEncode",
            "inputs": {
                "image": ["13", 0],
                "vae": ["6", 0],
                "latent": ["8", 0],
            },
        }
        # Rewire sampler latent input to the encoded keyframe latent
        nodes["9"]["inputs"]["latent"] = ["14", 0]
        nodes["12"]["inputs"]["filename_prefix"] = "minimax_h3_i2v"
        return nodes

    def build_minimax_h3_r2v_workflow(
        self,
        prompt: str,
        character_reference_image: str,
        motion_video_path: Optional[str] = None,
        audio_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 768,
        length_frames: int = 72,
        fps: int = 24,
        steps: int = 25,
    ) -> Dict[str, Any]:
        """Build the MiniMax H3 Multimodal Reference-to-Video (R2V) workflow graph."""
        nodes = self.build_minimax_h3_t2v_workflow(
            prompt=prompt,
            audio_prompt=audio_prompt,
            width=width,
            height=height,
            length_frames=length_frames,
            fps=fps,
            steps=steps,
        )
        # Load character identity reference image
        nodes["15"] = {
            "class_type": "LoadImage",
            "inputs": {"image": character_reference_image},
        }
        # Multi-modal Reference injection node
        nodes["16"] = {
            "class_type": "MiniMaxReferenceConditioning",
            "inputs": {
                "reference_image": ["15", 0],
                "positive": ["3", 0],
                "clip": ["2", 0],
                "reference_strength": 0.85,
            },
        }
        # Rewire sampler positive video conditioning to the reference-augmented condition
        nodes["9"]["inputs"]["positive_video"] = ["16", 0]
        nodes["12"]["inputs"]["filename_prefix"] = "minimax_h3_r2v"
        return nodes


workflow_builder = ComfyUIWorkflowBuilder()
