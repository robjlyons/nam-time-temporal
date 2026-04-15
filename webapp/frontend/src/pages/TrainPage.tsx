import React, { useEffect, useState } from "react";
import { AudioComparePlayer } from "../components/AudioComparePlayer";
import { CheckpointList } from "../components/CheckpointList";
import { LossChart } from "../components/LossChart";
import { TrainingConfig, TrainingConfigForm } from "../components/TrainingConfigForm";

const API = "http://localhost:8000";

export function TrainPage() {
  const [cfg, setCfg] = useState<TrainingConfig>({
    input_wav: "",
    output_wav: "",
    steps: 20000,
    batch_size: 8,
    context: 8192,
    target: 8192,
    device: "auto",
  });
  const [jobId, setJobId] = useState<string>("");
  const [log, setLog] = useState<string>("");
  const [checkpoints, setCheckpoints] = useState<string[]>([]);

  async function start() {
    const r = await fetch(`${API}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    const j = await r.json();
    setJobId(j.id);
  }

  async function refresh() {
    if (!jobId) return;
    const logResp = await fetch(`${API}/jobs/${jobId}/logs`);
    const logJson = await logResp.json();
    setLog(logJson.log ?? "");
    const ckptResp = await fetch(`${API}/jobs/${jobId}/checkpoints`);
    const ckptJson = await ckptResp.json();
    setCheckpoints(ckptJson.checkpoints ?? []);
  }

  useEffect(() => {
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  });

  return (
    <div className="page">
      <h1>NAM Temporal Trainer</h1>
      <TrainingConfigForm value={cfg} onChange={setCfg} onSubmit={start} />
      <LossChart log={log} />
      <CheckpointList checkpoints={checkpoints} onResume={(ckpt) => setCfg({ ...cfg, resume_checkpoint: ckpt })} />
      {jobId ? (
        <a href={`${API}/jobs/${jobId}/model`} target="_blank">
          Download model.nam
        </a>
      ) : null}
      <AudioComparePlayer baseUrl={`${API}/jobs/${jobId}/artifacts`} />
    </div>
  );
}
