"""API routes package for FreeCloud AI Studio."""

from backend.api.routes_comfyui import router as comfyui_router
from backend.api.routes_continuity import router as continuity_router
from backend.api.routes_jobs import router as jobs_router
from backend.api.routes_upload import router as upload_router
from backend.api.routes_workers import router as workers_router

__all__ = [
    "comfyui_router",
    "continuity_router",
    "jobs_router",
    "upload_router",
    "workers_router",
]
