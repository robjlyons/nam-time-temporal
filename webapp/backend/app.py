from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse

from .jobs import get_status
from .jobs import list_jobs
from .jobs import read_log
from .jobs import start_training
from .schemas import TrainRequest
from .storage import JOBS
from .storage import UPLOADS
from .storage import ensure_dirs

app = FastAPI(title="NAM Temporal Web API")
ensure_dirs()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    if suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    out = UPLOADS / f"{uuid.uuid4().hex[:12]}_{file.filename}"
    with out.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    return {"path": str(out)}


@app.post("/train")
def train(req: TrainRequest):
    return start_training(req)


@app.get("/jobs")
def jobs():
    return list_jobs()


@app.get("/jobs/{job_id}")
def job(job_id: str):
    try:
        return get_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job id")


@app.get("/jobs/{job_id}/logs")
def job_logs(job_id: str):
    try:
        return {"log": read_log(job_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job id")


@app.get("/jobs/{job_id}/checkpoints")
def job_checkpoints(job_id: str):
    ckpt_dir = JOBS / job_id / "checkpoints"
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    return {"checkpoints": [str(p) for p in sorted(ckpt_dir.glob("*.ckpt"))]}


@app.get("/jobs/{job_id}/model")
def job_model(job_id: str):
    p = JOBS / job_id / "model.nam"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Model not ready")
    return FileResponse(path=str(p), filename="model.nam", media_type="application/json")
