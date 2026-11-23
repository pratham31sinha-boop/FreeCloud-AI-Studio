# Setup & Operator's Guide

This guide details how to install, configure, run, and test **FreeCloud AI Studio**.

---

## 1. Prerequisites

- **Python**: 3.10, 3.11, or 3.12 (Python 3.12 verified).
- **Git**: Installed for repository tracking.
- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows WSL2.
- **Optional**: GPU with CUDA support (for local ComfyUI inference) or GitHub repository access (for distributed Actions orchestration).

---

## 2. Quickstart Installation

1. Navigate to the project root directory:
   ```bash
   cd /home/prathamsinha/.gemini/antigravity/scratch/FreeCloud-AI-Studio
   ```

2. Inspect and install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize your environment file:
   ```bash
   cp .env.example .env
   ```

---

## 3. Starting the Backend Server

Launch the FastAPI application with Uvicorn:

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Output:
```
INFO:     Started server process [pid]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Access Points
- **Web Studio Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive OpenAPI Docs**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **ReDoc Documentation**: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 4. Running a Distributed Worker Runner

### Local Development / Offline Simulation
You can execute the worker runner locally to process leased chunks:

```bash
python3 github/worker_entrypoint.py \
  --backend-url http://localhost:8000 \
  --worker-id local_runner_01 \
  --max-idle-seconds 30
```

### GitHub Actions Cloud Provisioning
To enable live GitHub Actions dispatching:
1. In `.env`, set:
   ```bash
   GITHUB_ACTIONS_ENABLED=true
   GITHUB_REPOSITORY=your-org/FreeCloud-AI-Studio
   GITHUB_PAT_TOKEN=ghp_your_personal_access_token_with_workflow_scope
   ```
2. The orchestrator will automatically dispatch the `worker_runner.yml` workflow whenever a new job is created, scaling the worker count according to the planned duration table.

---

## 5. Running the Test Suite

Execute all unit and integration tests using pytest:

```bash
cd /home/prathamsinha/.gemini/antigravity/scratch/FreeCloud-AI-Studio
PYTHONPATH=. python3 -m pytest -v
```

All 20 tests verify:
- Worker allocation table compliance (1m: 4, 2m: 5, 3m: 6, 5m: 7, 10m: 8, 15m: 10).
- Task chunk partitioning and sequence continuity.
- Worker registration, task lease, heartbeat, and fault-recovery retries.
- Character reference profiles (Jagdhatri, Jash) and appearance drift calculations.
- Motion sequence and action choreography preservation.
- Scene history and lighting continuity verification.
- Script beat parsing and viseme lip-sync planning.
- ComfyUI workflow graph construction and offline simulation fallback.
- Storage manager persistence and checksum verification.
- End-to-end FastAPI endpoint integration.
