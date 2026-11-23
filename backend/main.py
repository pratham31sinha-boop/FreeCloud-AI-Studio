"""FreeCloud AI Studio - Main FastAPI Application.

Serves the REST API, worker coordination interfaces, and the responsive
frontend studio dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.api import comfyui_router, continuity_router, jobs_router, upload_router, workers_router
from backend.config import settings
from backend.storage.manager import storage_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("freecloud")

app = FastAPI(
    title="FreeCloud AI Studio",
    description="AI-powered anime video generation platform with distributed GitHub Actions worker orchestration and continuity preservation.",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Enable CORS for local development and remote worker callbacks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routers
api_v1_prefix = "/api/v1"
app.include_router(jobs_router, prefix=api_v1_prefix)
app.include_router(workers_router, prefix=api_v1_prefix)
app.include_router(continuity_router, prefix=api_v1_prefix)
app.include_router(upload_router, prefix=api_v1_prefix)
app.include_router(comfyui_router, prefix=api_v1_prefix)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health and diagnostic status."""
    return {
        "status": "healthy",
        "service": "FreeCloud AI Studio",
        "version": "0.1.0",
        "environment": settings.env,
    }


@app.get("/api/v1/status", tags=["Health"])
def system_status():
    """System overview and architecture configuration."""
    return {
        "status": "operational",
        "storage": {
            "data_dir": str(settings.data_dir),
            "uploads_dir": str(settings.uploads_dir),
            "artifacts_dir": str(settings.artifacts_dir),
            "outputs_dir": str(settings.outputs_dir),
        },
        "orchestration": {
            "github_actions_enabled": settings.github_actions_enabled,
            "github_repository": settings.github_repository,
            "worker_heartbeat_timeout_sec": settings.worker_heartbeat_timeout_sec,
            "worker_task_lease_sec": settings.worker_task_lease_sec,
        },
        "comfyui": {
            "host": settings.comfyui_host,
            "port": settings.comfyui_port,
            "enabled": settings.comfyui_enabled,
        },
    }


# Mount frontend static assets
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        return {"message": "FreeCloud AI Studio frontend index.html not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
