# Deployment & Operations Guide

## 1. Local Development & Verification

### Prerequisites
- Python 3.10+ (Current environment uses Python 3.13 in `.\.venv\`)
- Node.js 18+ (Current environment uses Node.js 24.16.0)
- PowerShell (Windows) or Bash (Linux/macOS)

### Launching the Application Locally
We provide a unified orchestration script `start_breastcare.ps1`:

```powershell
# Option A: Start both Backend and Frontend in separate windows
.\start_breastcare.ps1 -Mode all

# Option B: Start FastAPI Backend only (:8000)
.\start_breastcare.ps1 -Mode api

# Option C: Start Next.js Frontend only (:3000)
.\start_breastcare.ps1 -Mode web
```

### Manual Command Launch
```bash
# Backend (FastAPI / Uvicorn)
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Frontend (Next.js Production Build)
cd web
npm run build
npm run start -p 3000
```

---

## 2. Docker & Container Deployment

### Standalone Docker Containers
```bash
# Build Backend API Container
docker build -t breastcare-api:latest -f Dockerfile.api .
docker run -p 8000:8000 breastcare-api:latest

# Build Frontend Container
cd web
docker build -t breastcare-web:latest .
docker run -p 3000:3000 breastcare-web:latest
```

### Docker Compose
Run the entire stack with single command:
```bash
docker-compose up --build
```

---

## 3. Health & Readiness Verification
- **Liveness Probe:** `http://127.0.0.1:8000/health` (Returns status `ok`)
- **Readiness Probe:** `http://127.0.0.1:8000/ready` (Returns status `ready`)
- **Models Metadata:** `http://127.0.0.1:8000/api/models`
- **Frontend Web UI:** `http://127.0.0.1:3000`

