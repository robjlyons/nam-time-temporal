from pathlib import Path


BASE = Path("webapp_runs")
UPLOADS = BASE / "uploads"
JOBS = BASE / "jobs"


def ensure_dirs():
    UPLOADS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
