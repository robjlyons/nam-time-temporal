from pydantic import BaseModel


class TrainRequest(BaseModel):
    input_wav: str
    output_wav: str
    steps: int = 20000
    batch_size: int = 8
    context: int = 8192
    target: int = 8192
    device: str = "auto"
    resume_checkpoint: str | None = None


class JobStatus(BaseModel):
    id: str
    state: str
    outdir: str
    pid: int | None = None
    command: list[str]
