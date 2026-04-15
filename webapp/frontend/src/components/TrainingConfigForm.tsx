import React from "react";

export type TrainingConfig = {
  input_wav: string;
  output_wav: string;
  steps: number;
  batch_size: number;
  context: number;
  target: number;
  device: "auto" | "cpu" | "gpu";
  resume_checkpoint?: string;
};

export function TrainingConfigForm({
  value,
  onChange,
  onSubmit,
}: {
  value: TrainingConfig;
  onChange: (v: TrainingConfig) => void;
  onSubmit: () => void;
}) {
  const set = <K extends keyof TrainingConfig>(k: K, v: TrainingConfig[K]) =>
    onChange({ ...value, [k]: v });
  return (
    <div className="card">
      <h3>Training Configuration</h3>
      <label>Input WAV</label>
      <input value={value.input_wav} onChange={(e) => set("input_wav", e.target.value)} />
      <label>Output WAV</label>
      <input value={value.output_wav} onChange={(e) => set("output_wav", e.target.value)} />
      <label>Steps</label>
      <input type="number" value={value.steps} onChange={(e) => set("steps", Number(e.target.value))} />
      <label>Batch size</label>
      <input type="number" value={value.batch_size} onChange={(e) => set("batch_size", Number(e.target.value))} />
      <label>Context samples</label>
      <input type="number" value={value.context} onChange={(e) => set("context", Number(e.target.value))} />
      <label>Target samples</label>
      <input type="number" value={value.target} onChange={(e) => set("target", Number(e.target.value))} />
      <label>Device</label>
      <select value={value.device} onChange={(e) => set("device", e.target.value as "auto" | "cpu" | "gpu")}>
        <option value="auto">Auto</option>
        <option value="cpu">CPU</option>
        <option value="gpu">GPU</option>
      </select>
      <button onClick={onSubmit}>Start Training</button>
    </div>
  );
}
