from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import uuid

from .schemas import JobStatus
from .schemas import TrainRequest
from .storage import JOBS


@dataclass
class _Job:
    id: str
    outdir: Path
    command: list[str]
    process: subprocess.Popen
    log_path: Path


_JOBS: dict[str, _Job] = {}


def start_training(req: TrainRequest) -> JobStatus:
    job_id = uuid.uuid4().hex[:12]
    outdir = JOBS / job_id
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"
    cmd = [
        "python",
        "train.py",
        "--input",
        req.input_wav,
        "--output",
        req.output_wav,
        "--outdir",
        str(outdir),
        "--steps",
        str(req.steps),
        "--batch-size",
        str(req.batch_size),
        "--context",
        str(req.context),
        "--target",
        str(req.target),
        "--device",
        req.device,
    ]
    if req.resume_checkpoint:
        cmd.extend(["--resume", req.resume_checkpoint])
    with log_path.open("w") as fp:
        p = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT)
    _JOBS[job_id] = _Job(
        id=job_id,
        outdir=outdir,
        command=cmd,
        process=p,
        log_path=log_path,
    )
    return get_status(job_id)


def get_status(job_id: str) -> JobStatus:
    j = _JOBS[job_id]
    code = j.process.poll()
    if code is None:
        state = "running"
    elif code == 0:
        state = "completed"
    else:
        state = f"failed({code})"
    return JobStatus(
        id=j.id,
        state=state,
        outdir=str(j.outdir),
        pid=j.process.pid,
        command=j.command,
    )


def list_jobs() -> list[JobStatus]:
    return [get_status(job_id) for job_id in _JOBS]


def read_log(job_id: str, max_chars: int = 20000) -> str:
    j = _JOBS[job_id]
    if not j.log_path.exists():
        return ""
    txt = j.log_path.read_text(errors="ignore")
    return txt[-max_chars:]
