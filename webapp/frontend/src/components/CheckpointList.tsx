import React from "react";

export function CheckpointList({
  checkpoints,
  onResume,
}: {
  checkpoints: string[];
  onResume: (ckpt: string) => void;
}) {
  return (
    <div className="card">
      <h3>Checkpoints</h3>
      <ul>
        {checkpoints.map((c) => (
          <li key={c}>
            <code>{c}</code>
            <button onClick={() => onResume(c)}>Resume</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
