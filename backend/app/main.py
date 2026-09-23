"""
VisionTrust Assurance - Main FastAPI Application
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from backend.app.config import settings
from backend.app.database.database import init_db
from backend.app.core.signatures import generate_or_load_keypair
from backend.app.api import (
    health, datasets, models, inference, distribution, findings, audit, reports, demo
)

app = FastAPI(
    title="VisionTrust Assurance",
    description="Offline Computer-Vision Integrity & Provenance Framework",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Initialize database and offline cryptographic signing keys on startup."""
    init_db()
    generate_or_load_keypair()

# Include API Routers
app.include_router(health.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(distribution.router, prefix="/api")
app.include_router(findings.router, prefix="/api")
app.include_router(audit.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(demo.router, prefix="/api")

# Static files for frontend
FRONTEND_DIR = settings.BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def read_root():
    """Serve the enterprise dashboard frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "VisionTrust Assurance API is running. Visit /docs for API documentation."}

# Additional page routes for direct browser navigation
@app.get("/{page_name}.html")
def read_page(page_name: str):
    target_page = FRONTEND_DIR / f"{page_name}.html"
    if target_page.exists():
        return FileResponse(target_page)
    return RedirectResponse(url="/")
