# BreastCare AI — Deployment & Infrastructure Guide

## 1. Overview & Architecture

BreastCare AI is architected as a privacy-preserving, decoupled multimodal research platform:
- **Backend API (`backend/app/main.py`)**: Python 3.11 / FastAPI service exposing `/health`, `/ready`, `/api/models`, `/api/references`, `/api/validate-image`, and `/api/analyze`. Operates strictly fail-closed: cancer classification remains hard-disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`) until independent prospective clinical evaluation and multi-category OOD gating are completed.
- **Frontend Web UI (`web/`)**: Next.js 15 (React 19, Tailwind CSS) providing pink-themed healthcare styling, 4-stage validation progress visualizer, Grad-CAM interpretability overlay, model evidence, and clinical research report downloads.
- **Container Infrastructure (`Dockerfile.api`, `web/Dockerfile`, `docker-compose.yml`)**: Multi-container Docker deployment with separate non-root service accounts, automated health checks, and explicit internal networking.
- **Cloud Deployment Manifests**:
  - `render.yaml`: Blueprint configuration for 1-click backend deployment on Render.
  - `railway.json`: Deployment manifest for Docker-based deployment on Railway.
  - `web/vercel.json`: Next.js 15 deployment manifest for Vercel.

---

## 2. Cloud Public Hosting (Render + Vercel)

The recommended production architecture utilizes Vercel for the Next.js frontend and Render or Railway for the containerized FastAPI backend.

### 2.1 Backend Deployment (Render)
1. Navigate to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** ➔ **Blueprint**.
3. Connect repository `https://github.com/salahibrahiem12/breast-cancer-ai.git`.
4. Render detects `render.yaml` automatically and configures:
   - Build: `Dockerfile.api`
   - Healthcheck: `/health`
   - Env: `BREASTCARE_CLASSIFICATION_ENABLED=false`
5. Note your assigned public backend URL (e.g. `https://breastcare-api.onrender.com`).

### 2.2 Frontend Deployment (Vercel)
1. Navigate to [vercel.com/new](https://vercel.com/new).
2. Import repository `salahibrahiem12/breast-cancer-ai`.
3. Set **Root Directory** to `web`.
4. Under **Environment Variables**, add:
   - `NEXT_PUBLIC_API_URL`: Your public backend URL (e.g. `https://breastcare-api.onrender.com`).
5. Click **Deploy**. Vercel will build and assign your production domain.

---

## 3. Local Containerized Deployment (Docker Compose)

### 2.1 Prerequisites
- Docker Engine 24.0+ or Docker Desktop
- Docker Compose v2.20+

### 2.2 Quick Start

1. **Verify Environment Configuration**:
   ```bash
   cp .env.example .env
   ```
   Confirm `BREASTCARE_CLASSIFICATION_ENABLED=false` in `.env`.

2. **Build and Launch Containers**:
   ```bash
   docker compose up --build -d
   ```

3. **Verify Container Status & Health**:
   ```bash
   docker compose ps
   ```
   Both `breastcare-api` and `breastcare-web` should report `(healthy)`.

4. **Access the Applications**:
   - Web Application: [http://localhost:3000](http://localhost:3000)
   - API Health: [http://localhost:8000/health](http://localhost:8000/health)
   - API Model Info: [http://localhost:8000/model-info](http://localhost:8000/model-info)

5. **Inspect Logs**:
   ```bash
   docker compose logs -f api
   docker compose logs -f web
   ```

6. **Stop Services**:
   ```bash
   docker compose down
   ```

---

## 3. Local Bare-Metal / Development Deployment

For local execution without Docker, use the included PowerShell automation script:

```powershell
# 1. Environment & artifact integrity pre-flight check
.\start_breastcare.ps1 -Mode check

# 2. Run both Backend and Frontend in separate windows
.\start_breastcare.ps1 -Mode both

# Or run individually:
.\start_breastcare.ps1 -Mode api   # Port 8000
.\start_breastcare.ps1 -Mode web   # Port 3000
```

---

## 4. Evaluation of Safest Free Hosting Options

Per project instructions, we identify and compare the safest available free hosting options:

| Hosting Option | Type | Free Tier Limits | Privacy & Security Assessment | Clinical Safety Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: Self-Hosted Docker Compose (Recommended)** | On-Premise / Local Workstation / Private LAN | Unlimited (Local Hardware) | **Highest (Optimal)**: Zero data leaves local memory; no public internet exposure; complete network isolation. | **Safest**: Ideal for research, internal review, and clinical investigator audits without risk of public leakage. |
| **Option B: Hugging Face Spaces (Docker SDK)** | Public Cloud Research Platform | 2 vCPU, 16 GB RAM (Free CPU Tier) | **High**: Supports custom `Dockerfile`; free container runtime; zero-cost persistence via ephemeral disks. | **Approved for Research Showcase ONLY**: Permitted exclusively with `BREASTCARE_CLASSIFICATION_ENABLED=false` locked. Public access to unvalidated diagnostic classifiers is strictly forbidden. |
| **Option C: Render / Railway Free / Hobby Tier** | Cloud Container Platform | 512 MB RAM, free tier spins down after inactivity | **Moderate**: Ephemeral storage; supports environment secrets. Lower RAM may cause OOM during heavy PyTorch CPU forward passes. | **Conditionally Approved**: Requires strict environment variable protection and explicit rate-limiting. |

### Why Option A (Self-Hosted) is the Safest Option:
1. **Zero Data Transmission**: Ultrasound images never leave the local workstation.
2. **Deterministic Security Boundary**: No accidental misconfiguration can expose endpoints to unvetted external queries.
3. **Guaranteed Compliance**: Fully adheres to medical device software pre-clearance rules prohibiting the public deployment of unvalidated clinical classifiers.

---

## 5. Security & Fail-Closed Safety Guarantees

1. **Classification Hard Gate**:
   - `BREASTCARE_CLASSIFICATION_ENABLED` defaults to `false`.
   - The browser or client cannot override the server-side setting.
   - Any `/analyze` request returns:
     ```json
     {
       "status": "unsupported_or_unsuitable",
       "input_suitability": "abstain",
       "abstention_reason": "classification_disabled_by_server"
     }
     ```
2. **Strict Upload Bounds**:
   - Max file size: 10 MB (`BREASTCARE_MAX_UPLOAD_BYTES=10485760`).
   - Max image pixels: 25,000,000 (`BREASTCARE_MAX_IMAGE_PIXELS=25000000`).
   - Min image side: 64 px (`BREASTCARE_MIN_IMAGE_SIDE=64`).
3. **Ephemeral Memory Processing**:
   - Uploaded image bytes are processed strictly in RAM for structural and quality verification.
   - No patient images are written to persistent container disks or external databases.
4. **Least-Privilege Containerization**:
   - Both `Dockerfile.api` and `web/Dockerfile` create and run as unprivileged users (`breastcare:1000` and `nextjs:1001`).
   - No root execution in production containers.

